"""Experiment tracking for proposal recall and annotation-effort studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

from feature_extraction import FeatureMaps
from labeling import ReviewedAnnotation
from region_proposal import ProposalResult, _components


METHOD_NAMES = (
    "contour-only baseline",
    "fixed-threshold baseline",
    "multi-scale fused method",
    "refined contextual method",
)


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    reviewer_id: str
    image_filename: str
    method: str
    final_proposals: int
    accepted: int
    rejected: int
    uncertain: int
    image_outcome: str
    review_start_time: str
    review_completion_time: str
    review_duration_seconds: float
    first_accepted_true_anomaly_rank: int | None
    true_anomaly_found_top_1: bool
    true_anomaly_found_top_3: bool
    true_anomaly_found_top_5: bool
    true_anomaly_found_top_8: bool
    proposals_reviewed_before_first_useful: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_experiment_records(
    experiment_id: str,
    reviewer_id: str,
    image_filename: str,
    image_outcome: str,
    review_start_time: str,
    review_completion_time: str,
    annotations: list[ReviewedAnnotation],
    proposal_result: ProposalResult,
    feature_maps: FeatureMaps,
    overlap_threshold: float = 0.10,
) -> list[ExperimentRecord]:
    """Build actual and counterfactual Top-K records for the four proposal methods."""
    if image_outcome not in {"anomaly present", "no anomaly", "uncertain"}:
        raise ValueError("Image outcome must be anomaly present, no anomaly, or uncertain.")
    if not experiment_id.strip() or not reviewer_id.strip():
        raise ValueError("Experiment ID and reviewer ID are required.")

    accepted_annotations = [annotation for annotation in annotations if annotation.decision == "accept"]
    references = [annotation.bbox for annotation in accepted_annotations]
    duration = _duration_seconds(review_start_time, review_completion_time)
    method_boxes = _ranked_method_boxes(proposal_result, feature_maps)
    records = []

    for method, boxes in method_boxes.items():
        boxes = boxes[:8]
        if method == "refined contextual method":
            accepted = sum(annotation.decision == "accept" for annotation in annotations)
            rejected = sum(annotation.decision == "reject" for annotation in annotations)
            uncertain = sum(annotation.decision == "uncertain" for annotation in annotations)
            accepted_ranks = [
                index for index, proposal in enumerate(proposal_result.proposals, 1)
                if any(annotation.region_id == proposal.region_id and annotation.decision == "accept" for annotation in annotations)
            ]
            first_rank = min(accepted_ranks) if accepted_ranks else None
        else:
            matches = [any(_iou(box, reference) >= overlap_threshold for reference in references) for box in boxes]
            accepted = sum(matches)
            rejected = len(boxes) - accepted
            uncertain = 0
            first_rank = next((index for index, matched in enumerate(matches, 1) if matched), None)

        if image_outcome != "anomaly present":
            first_rank = None
        records.append(ExperimentRecord(
            experiment_id=experiment_id.strip(), reviewer_id=reviewer_id.strip(), image_filename=image_filename,
            method=method, final_proposals=len(boxes), accepted=accepted, rejected=rejected, uncertain=uncertain,
            image_outcome=image_outcome, review_start_time=review_start_time,
            review_completion_time=review_completion_time, review_duration_seconds=duration,
            first_accepted_true_anomaly_rank=first_rank,
            true_anomaly_found_top_1=first_rank is not None and first_rank <= 1,
            true_anomaly_found_top_3=first_rank is not None and first_rank <= 3,
            true_anomaly_found_top_5=first_rank is not None and first_rank <= 5,
            true_anomaly_found_top_8=first_rank is not None and first_rank <= 8,
            proposals_reviewed_before_first_useful=first_rank,
        ))
    return records


def experiment_tables(records: Iterable[ExperimentRecord | dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [record.to_dict() if isinstance(record, ExperimentRecord) else dict(record) for record in records]
    image_table = pd.DataFrame(rows)
    if image_table.empty:
        return image_table, image_table

    summaries = []
    for method, group in image_table.groupby("method", sort=False):
        anomaly_group = group[group["image_outcome"] == "anomaly present"]
        useful_ranks = pd.to_numeric(anomaly_group["proposals_reviewed_before_first_useful"], errors="coerce").dropna()
        total_proposals = max(float(group["final_proposals"].sum()), 1.0)
        summaries.append({
            "method": method,
            "images": int(len(group)),
            "anomaly_images": int(len(anomaly_group)),
            "top_1_proposal_recall": _boolean_mean(anomaly_group, "true_anomaly_found_top_1"),
            "top_3_proposal_recall": _boolean_mean(anomaly_group, "true_anomaly_found_top_3"),
            "top_5_proposal_recall": _boolean_mean(anomaly_group, "true_anomaly_found_top_5"),
            "top_8_proposal_recall": _boolean_mean(anomaly_group, "true_anomaly_found_top_8"),
            "mean_accepted_proposals_per_image": float(group["accepted"].mean()),
            "mean_false_proposals_per_image": float(group["rejected"].mean()),
            "annotation_acceptance_rate": float(group["accepted"].sum()) / total_proposals,
            "mean_review_time_seconds": float(group["review_duration_seconds"].mean()),
            "mean_proposals_reviewed_before_first_useful": float(useful_ranks.mean()) if not useful_ranks.empty else 0.0,
        })
    return image_table, pd.DataFrame(summaries)


def save_experiment_records(
    new_records: Iterable[ExperimentRecord], csv_path: Path, json_path: Path,
) -> tuple[Path, Path]:
    existing = load_experiment_records(json_path)
    rows = existing + [record.to_dict() for record in new_records]
    keys = ("experiment_id", "reviewer_id", "image_filename", "method")
    deduplicated = {tuple(row.get(key) for key in keys): row for row in rows}
    final_rows = list(deduplicated.values())
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(final_rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(final_rows, indent=2), encoding="utf-8")
    return csv_path, json_path


def load_experiment_records(json_path: Path) -> list[dict[str, object]]:
    if not json_path.exists():
        return []
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _ranked_method_boxes(proposal_result: ProposalResult, feature_maps: FeatureMaps) -> dict[str, list[tuple[int, int, int, int]]]:
    contour = _rank_components(feature_maps.contour_map, feature_maps.anomaly_strength)
    fixed = _rank_components((feature_maps.anomaly_strength > 128).astype(np.uint8) * 255, feature_maps.anomaly_strength)
    raw = []
    for proposal in proposal_result.proposals:
        mask = cv2.imread(str(proposal.raw_mask_path), cv2.IMREAD_GRAYSCALE)
        bbox = _bbox_from_mask(mask) if mask is not None else proposal.bbox
        raw.append(bbox)
    refined = [proposal.bbox for proposal in proposal_result.proposals]
    return {
        "contour-only baseline": contour[:8], "fixed-threshold baseline": fixed[:8],
        "multi-scale fused method": raw[:8], "refined contextual method": refined[:8],
    }


def _rank_components(mask: np.ndarray, heatmap: np.ndarray) -> list[tuple[int, int, int, int]]:
    candidates = []
    minimum = max(8, int(mask.size * .0001))
    for component in _components(mask):
        area = cv2.countNonZero(component.mask)
        if area < minimum:
            continue
        score = float(np.mean(heatmap[component.mask > 0])) if area else 0.0
        candidates.append((score, area, component.bbox))
    return [bbox for _, _, bbox in sorted(candidates, reverse=True)]


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    return (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)) if xs.size else (0, 0, 0, 0)


def _duration_seconds(start: str, completion: str) -> float:
    try:
        return max((datetime.fromisoformat(completion) - datetime.fromisoformat(start)).total_seconds(), 0.0)
    except ValueError:
        return 0.0


def _boolean_mean(frame: pd.DataFrame, column: str) -> float:
    return float(frame[column].astype(bool).mean()) if not frame.empty else 0.0


def _iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1)
