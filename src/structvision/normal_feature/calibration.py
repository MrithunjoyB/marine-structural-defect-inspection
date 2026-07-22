"""Validation-only operating points and immutable calibration provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from scientific_contract.evaluation_policy import default_evaluation_policy
from scientific_contract.hashing import canonical_json, is_sha256, sha256_json
from scientific_contract.matching import _maximum_weight_assignment

from .configuration import LearnedProposalConfig
from .errors import CalibrationError
from .proposal_extraction import extract_proposals
from .types import readonly_array


CALIBRATION_ARTIFACT_SCHEMA_VERSION = "normal-feature-calibration-artifact-v1"
DEFAULT_FALSE_PROPOSAL_BUDGETS = (0.25, 0.50, 1.00)


@dataclass(frozen=True)
class CalibrationSample:
    image_id: str
    role: str
    category: str
    image_outcome: str
    anomaly_map: np.ndarray
    ground_truth: np.ndarray

    def __post_init__(self) -> None:
        if self.role != "calibration_validation":
            raise CalibrationError("Calibration accepts validation-role samples only")
        if self.image_outcome not in {"no_anomaly", "anomaly_present"}:
            raise CalibrationError("Calibration outcome is invalid")
        anomaly_map = readonly_array(self.anomaly_map, ndim=2, dtype=np.float32, name="calibration anomaly map")
        truth = readonly_array(self.ground_truth, ndim=2, dtype=np.uint8, name="calibration ground truth")
        if anomaly_map.shape != truth.shape or not np.isfinite(anomaly_map).all():
            raise CalibrationError("Calibration map and truth must be finite and aligned")
        truth = np.ascontiguousarray((truth > 0).astype(np.uint8) * 255)
        truth.setflags(write=False)
        object.__setattr__(self, "anomaly_map", anomaly_map)
        object.__setattr__(self, "ground_truth", truth)
        if self.image_outcome == "no_anomaly" and np.any(truth):
            raise CalibrationError("Clean calibration samples require an empty truth mask")
        if self.image_outcome == "anomaly_present" and not np.any(truth):
            raise CalibrationError("Positive calibration samples require non-empty truth")


@dataclass(frozen=True)
class CalibrationCurvePoint:
    threshold: float
    clean_false_proposals_per_image: float
    clean_images_with_any_proposal: float
    micro_component_sensitivity: float | None
    macro_per_positive_image_recall: float | None
    image_level_detection_sensitivity: float | None
    proposal_precision: float | None
    mean_proposals_per_image: float
    category_component_sensitivity: tuple[tuple[str, float | None], ...]
    pareto_nondominated: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationOperatingPoint:
    operating_point_id: str
    false_proposal_budget: float
    threshold: float
    achieved_clean_false_proposals_per_image: float
    achieved_clean_images_with_any_proposal: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationArtifact:
    schema_version: str
    model_artifact_hash: str
    calibration_manifest_hash: str
    evaluation_policy_hash: str
    proposal_extraction_policy_hash: str
    candidate_thresholds: tuple[float, ...]
    false_proposal_budgets: tuple[float, ...]
    selected_threshold_policy: str
    category_preservation_constraints: tuple[str, ...]
    curve: tuple[CalibrationCurvePoint, ...]
    operating_points: tuple[CalibrationOperatingPoint, ...]
    deterministic_calibration_procedure: str
    artifact_hash: str

    def __post_init__(self) -> None:
        required = (
            self.model_artifact_hash, self.calibration_manifest_hash,
            self.evaluation_policy_hash, self.proposal_extraction_policy_hash,
            self.artifact_hash,
        )
        if any(not is_sha256(item) for item in required):
            raise CalibrationError("Calibration provenance requires SHA-256 identities")
        if self.schema_version != CALIBRATION_ARTIFACT_SCHEMA_VERSION:
            raise CalibrationError("Unsupported calibration-artifact schema")
        if not self.curve or tuple(item.threshold for item in self.curve) != self.candidate_thresholds:
            raise CalibrationError("The complete threshold curve must be preserved in candidate order")
        if tuple(sorted(set(self.false_proposal_budgets))) != self.false_proposal_budgets:
            raise CalibrationError("False-proposal budgets must be unique and increasing")
        thresholds = tuple(item.threshold for item in self.operating_points)
        if any(left < right for left, right in zip(thresholds, thresholds[1:])):
            raise CalibrationError("Thresholds must be monotone non-increasing as budget increases")
        if self.artifact_hash != "0" * 64 and self.artifact_hash != sha256_json(self.to_dict(include_artifact_hash=False)):
            raise CalibrationError("Calibration-artifact hash mismatch")

    @classmethod
    def create(cls, **values: object) -> "CalibrationArtifact":
        initial = cls(artifact_hash="0" * 64, **values)
        return replace(initial, artifact_hash=sha256_json(initial.to_dict(include_artifact_hash=False)))

    def operating_point(self, identity: str) -> CalibrationOperatingPoint:
        try:
            return next(item for item in self.operating_points if item.operating_point_id == identity)
        except StopIteration as error:
            raise KeyError(identity) from error

    def to_dict(self, *, include_artifact_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "model_artifact_hash": self.model_artifact_hash,
            "calibration_manifest_hash": self.calibration_manifest_hash,
            "evaluation_policy_hash": self.evaluation_policy_hash,
            "proposal_extraction_policy_hash": self.proposal_extraction_policy_hash,
            "candidate_thresholds": list(self.candidate_thresholds),
            "false_proposal_budgets": list(self.false_proposal_budgets),
            "selected_threshold_policy": self.selected_threshold_policy,
            "category_preservation_constraints": list(self.category_preservation_constraints),
            "curve": [item.to_dict() for item in self.curve],
            "operating_points": [item.to_dict() for item in self.operating_points],
            "deterministic_calibration_procedure": self.deterministic_calibration_procedure,
        }
        if include_artifact_hash:
            payload["artifact_hash"] = self.artifact_hash
        return payload

    def to_json(self) -> str:
        return canonical_json(self.to_dict())


def _point(
    samples: tuple[CalibrationSample, ...], threshold: float, proposal_config: LearnedProposalConfig,
) -> CalibrationCurvePoint:
    clean_count = 0
    clean_proposals = 0
    clean_any = 0
    proposal_total = 0
    matched_total = 0
    truth_total = 0
    positive_recalls: list[float] = []
    detected_positive = 0
    category_counts: dict[str, list[int]] = {}
    for sample in samples:
        proposals = extract_proposals(
            sample.anomaly_map,
            threshold=threshold,
            operating_point_id="calibration-candidate",
            config=proposal_config,
        )
        proposal_total += len(proposals)
        if sample.image_outcome == "no_anomaly":
            clean_count += 1
            clean_proposals += len(proposals)
            clean_any += bool(proposals)
            continue
        truth_count, truth_labels = cv2.connectedComponents(
            (sample.ground_truth > 0).astype(np.uint8), connectivity=8,
        )
        truth_masks = tuple(truth_labels == index for index in range(1, truth_count))
        proposal_masks = tuple(item.mask > 0 for item in proposals)
        truth_total += len(truth_masks)
        size = max(len(proposal_masks), len(truth_masks))
        weights = np.zeros((size, size), dtype=float)
        cardinality_weight = float(size + 1)
        for row, proposal_mask in enumerate(proposal_masks):
            proposal_area = int(np.count_nonzero(proposal_mask))
            for column, truth_mask in enumerate(truth_masks):
                intersection = int(np.count_nonzero(proposal_mask & truth_mask))
                union = proposal_area + int(np.count_nonzero(truth_mask)) - intersection
                iou = intersection / union if union else 0.0
                if iou + 1e-12 >= 0.25:
                    weights[row, column] = cardinality_weight + iou
        assignment = _maximum_weight_assignment(weights)
        matched = sum(
            column < len(truth_masks) and weights[row, column] > 0
            for row, column in assignment.items() if row < len(proposal_masks)
        )
        matched_total += matched
        positive_recalls.append(matched / len(truth_masks))
        detected_positive += matched > 0
        category = category_counts.setdefault(sample.category, [0, 0])
        category[0] += matched
        category[1] += len(truth_masks)
    positive_count = len(positive_recalls)
    return CalibrationCurvePoint(
        threshold=float(threshold),
        clean_false_proposals_per_image=clean_proposals / clean_count,
        clean_images_with_any_proposal=clean_any / clean_count,
        micro_component_sensitivity=matched_total / truth_total if truth_total else None,
        macro_per_positive_image_recall=float(np.mean(positive_recalls)) if positive_recalls else None,
        image_level_detection_sensitivity=detected_positive / positive_count if positive_count else None,
        proposal_precision=matched_total / proposal_total if proposal_total else None,
        mean_proposals_per_image=proposal_total / len(samples),
        category_component_sensitivity=tuple(
            (category, matched / total if total else None)
            for category, (matched, total) in sorted(category_counts.items())
        ),
    )


def _mark_pareto(curve: tuple[CalibrationCurvePoint, ...]) -> tuple[CalibrationCurvePoint, ...]:
    marked = []
    for point in curve:
        sensitivity = -1.0 if point.micro_component_sensitivity is None else point.micro_component_sensitivity
        dominated = any(
            other.clean_false_proposals_per_image <= point.clean_false_proposals_per_image
            and (-1.0 if other.micro_component_sensitivity is None else other.micro_component_sensitivity) >= sensitivity
            and (
                other.clean_false_proposals_per_image < point.clean_false_proposals_per_image
                or (-1.0 if other.micro_component_sensitivity is None else other.micro_component_sensitivity) > sensitivity
            )
            for other in curve
        )
        marked.append(replace(point, pareto_nondominated=not dominated))
    return tuple(marked)


def calibrate(
    samples: tuple[CalibrationSample, ...] | list[CalibrationSample],
    *,
    model_artifact_hash: str,
    calibration_manifest_hash: str,
    proposal_config: LearnedProposalConfig,
    false_proposal_budgets: tuple[float, ...] = DEFAULT_FALSE_PROPOSAL_BUDGETS,
    quantile_count: int = 101,
) -> CalibrationArtifact:
    prepared = tuple(samples)
    if not prepared or len({item.image_id for item in prepared}) != len(prepared):
        raise CalibrationError("Calibration requires unique validation images")
    if {item.image_outcome for item in prepared} != {"no_anomaly", "anomaly_present"}:
        raise CalibrationError("Calibration requires both clean and anomaly-present validation images")
    budgets = tuple(float(item) for item in false_proposal_budgets)
    if tuple(sorted(set(budgets))) != budgets or any(item <= 0 for item in budgets):
        raise CalibrationError("False-proposal budgets must be positive, unique, and increasing")
    if quantile_count < 3:
        raise CalibrationError("At least three deterministic quantiles are required")
    values = np.concatenate([item.anomaly_map.reshape(-1) for item in prepared])
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, quantile_count), method="linear")
    above_max = np.nextafter(np.float32(np.max(values)), np.float32(np.inf)).item()
    thresholds = tuple(sorted({float(item) for item in quantiles} | {float(above_max)}, reverse=True))
    curve = _mark_pareto(tuple(_point(prepared, threshold, proposal_config) for threshold in thresholds))
    operating_points = []
    prior_threshold = float("inf")
    for budget in budgets:
        selected = curve[0]
        for point in curve:
            if point.clean_false_proposals_per_image > budget:
                break
            selected = point
        threshold = min(prior_threshold, selected.threshold)
        if threshold != selected.threshold:
            selected = next(item for item in curve if item.threshold == threshold)
        prior_threshold = threshold
        operating_points.append(CalibrationOperatingPoint(
            operating_point_id=f"fp-budget-{budget:.2f}",
            false_proposal_budget=budget,
            threshold=selected.threshold,
            achieved_clean_false_proposals_per_image=selected.clean_false_proposals_per_image,
            achieved_clean_images_with_any_proposal=selected.clean_images_with_any_proposal,
        ))
    policy = default_evaluation_policy()
    categories = sorted({item.category for item in prepared})
    return CalibrationArtifact.create(
        schema_version=CALIBRATION_ARTIFACT_SCHEMA_VERSION,
        model_artifact_hash=model_artifact_hash,
        calibration_manifest_hash=calibration_manifest_hash,
        evaluation_policy_hash=policy.configuration_hash,
        proposal_extraction_policy_hash=proposal_config.configuration_hash,
        candidate_thresholds=thresholds,
        false_proposal_budgets=budgets,
        selected_threshold_policy=(
            "descending-threshold first-budget-crossing path; choose the most permissive prefix "
            "that remains within each clean false-proposal budget; no anomaly metric optimisation"
        ),
        category_preservation_constraints=tuple(f"report:{category}" for category in categories),
        curve=curve,
        operating_points=tuple(operating_points),
        deterministic_calibration_procedure=(
            f"{quantile_count} pooled validation-map quantiles plus nextafter(max,+inf); "
            "OpenCV deterministic connected components; structvision-eval-v2 matching"
        ),
    )


class CalibrationArtifactSink(Protocol):
    def write(self, artifact: CalibrationArtifact) -> None:
        ...


class DirectoryCalibrationArtifactSink:
    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def write(self, artifact: CalibrationArtifact) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{artifact.artifact_hash}.json"
        if path.exists():
            raise CalibrationError("Calibration artifact is immutable and refuses overwrite")
        path.write_text(artifact.to_json() + "\n", encoding="utf-8")


def load_calibration_artifact(path: Path) -> CalibrationArtifact:
    text = Path(path).read_text(encoding="utf-8")
    payload = json.loads(text)
    if canonical_json(payload) + "\n" != text:
        raise CalibrationError("Calibration artifact is not canonical")
    return CalibrationArtifact(
        schema_version=str(payload["schema_version"]),
        model_artifact_hash=str(payload["model_artifact_hash"]),
        calibration_manifest_hash=str(payload["calibration_manifest_hash"]),
        evaluation_policy_hash=str(payload["evaluation_policy_hash"]),
        proposal_extraction_policy_hash=str(payload["proposal_extraction_policy_hash"]),
        candidate_thresholds=tuple(float(item) for item in payload["candidate_thresholds"]),
        false_proposal_budgets=tuple(float(item) for item in payload["false_proposal_budgets"]),
        selected_threshold_policy=str(payload["selected_threshold_policy"]),
        category_preservation_constraints=tuple(str(item) for item in payload["category_preservation_constraints"]),
        curve=tuple(CalibrationCurvePoint(
            threshold=float(item["threshold"]),
            clean_false_proposals_per_image=float(item["clean_false_proposals_per_image"]),
            clean_images_with_any_proposal=float(item["clean_images_with_any_proposal"]),
            micro_component_sensitivity=item["micro_component_sensitivity"],
            macro_per_positive_image_recall=item["macro_per_positive_image_recall"],
            image_level_detection_sensitivity=item["image_level_detection_sensitivity"],
            proposal_precision=item["proposal_precision"],
            mean_proposals_per_image=float(item["mean_proposals_per_image"]),
            category_component_sensitivity=tuple((str(a), b) for a, b in item["category_component_sensitivity"]),
            pareto_nondominated=bool(item["pareto_nondominated"]),
        ) for item in payload["curve"]),
        operating_points=tuple(CalibrationOperatingPoint(**item) for item in payload["operating_points"]),
        deterministic_calibration_procedure=str(payload["deterministic_calibration_procedure"]),
        artifact_hash=str(payload["artifact_hash"]),
    )
