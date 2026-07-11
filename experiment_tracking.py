"""Persistent experiment records and annotation-efficiency metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Iterable
from uuid import uuid4

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
EXPERIMENT_STATUSES = ("Development / Test", "Final Research Evaluation")
REVIEW_STATUSES = ("fully_reviewed", "partially_reviewed", "not_reviewed")
GROUND_TRUTH_STATUSES = ("verified ground truth", "reviewer-estimated", "unknown")
UNIQUE_FIELDS = ("experiment_id", "experiment_version", "reviewer_id", "image_filename", "method")

IMAGE_TABLE_COLUMNS = [
    "record_id", "experiment_id", "experiment_version", "reviewer_id", "experiment_status",
    "image_filename", "method", "review_status", "final_proposals", "accepted", "rejected",
    "uncertain", "not_reviewed", "image_outcome", "ground_truth_status",
    "review_duration_seconds", "first_accepted_true_anomaly_rank", "recorded_timestamp",
]

SUMMARY_COLUMNS = [
    "method", "eligible_images", "anomaly_present_images", "reviewed_images", "not_reviewed_images",
    "top_1_proposal_recall", "top_3_proposal_recall", "top_5_proposal_recall",
    "top_8_proposal_recall", "mean_accepted_proposals_per_image",
    "mean_rejected_proposals_per_image", "mean_uncertain_proposals_per_image",
    "mean_not_reviewed_proposals_per_image", "mean_false_proposals_per_image",
    "annotation_acceptance_rate", "mean_review_time_seconds",
    "mean_proposals_reviewed_before_first_useful",
]


class DuplicateRecordError(ValueError):
    pass


@dataclass(frozen=True)
class ExperimentRecord:
    record_id: str
    experiment_id: str
    experiment_version: int
    reviewer_id: str
    experiment_status: str
    image_filename: str
    method: str
    recorded_timestamp: str
    review_status: str
    final_proposals: int
    accepted: int | None
    rejected: int | None
    uncertain: int | None
    not_reviewed: int
    image_outcome: str
    dataset_source: str
    image_provenance: str
    license_status: str
    ground_truth_status: str
    ground_truth_recall_override: bool
    development_notes: str
    review_start_time: str
    review_completion_time: str
    review_duration_seconds: float | None
    first_accepted_true_anomaly_rank: int | None
    true_anomaly_found_top_1: bool | None
    true_anomaly_found_top_3: bool | None
    true_anomaly_found_top_5: bool | None
    true_anomaly_found_top_8: bool | None
    proposals_reviewed_before_first_useful: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ExperimentStore:
    """SQLite repository for method-level experiment records."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path))
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        columns = []
        for name, annotation in ExperimentRecord.__annotations__.items():
            if name in {"experiment_version", "final_proposals", "accepted", "rejected", "uncertain", "not_reviewed", "first_accepted_true_anomaly_rank", "proposals_reviewed_before_first_useful"}:
                sql_type = "INTEGER"
            elif name in {"review_duration_seconds"}:
                sql_type = "REAL"
            elif name.startswith("true_anomaly_found_") or name == "ground_truth_recall_override":
                sql_type = "INTEGER"
            else:
                sql_type = "TEXT"
            suffix = " PRIMARY KEY" if name == "record_id" else ""
            columns.append(f"{name} {sql_type}{suffix}")
        unique = ", ".join(UNIQUE_FIELDS)
        with self.connect() as connection:
            connection.execute(f"CREATE TABLE IF NOT EXISTS experiment_records ({', '.join(columns)}, UNIQUE ({unique}))")
            existing = {row[1] for row in connection.execute("PRAGMA table_info(experiment_records)").fetchall()}
            for definition in columns:
                name = definition.split()[0]
                if name not in existing and name != "record_id":
                    default = " DEFAULT 0" if name == "ground_truth_recall_override" else ""
                    connection.execute(f"ALTER TABLE experiment_records ADD COLUMN {definition}{default}")

    def save(self, records: Iterable[ExperimentRecord], duplicate_action: str = "cancel") -> int:
        records = list(records)
        if duplicate_action not in {"cancel", "overwrite"}:
            raise ValueError("duplicate_action must be cancel or overwrite")
        fields = list(ExperimentRecord.__annotations__)
        placeholders = ", ".join("?" for _ in fields)
        columns = ", ".join(fields)
        key_clause = " AND ".join(f"{field} = ?" for field in UNIQUE_FIELDS)
        saved = 0
        with self.connect() as connection:
            for record in records:
                row = record.to_dict()
                key = tuple(row[field] for field in UNIQUE_FIELDS)
                duplicate = connection.execute(f"SELECT record_id FROM experiment_records WHERE {key_clause}", key).fetchone()
                if duplicate and duplicate_action == "cancel":
                    raise DuplicateRecordError(f"Duplicate experiment row exists for {key}")
                if duplicate:
                    connection.execute(f"DELETE FROM experiment_records WHERE {key_clause}", key)
                values = [_sqlite_value(row[field]) for field in fields]
                connection.execute(f"INSERT INTO experiment_records ({columns}) VALUES ({placeholders})", values)
                saved += 1
        return saved

    def next_version(self, experiment_id: str) -> int:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT MAX(experiment_version) AS version FROM experiment_records WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
        return int(row["version"] or 0) + 1

    def duplicate_count(self, records: Iterable[ExperimentRecord]) -> int:
        key_clause = " AND ".join(f"{field} = ?" for field in UNIQUE_FIELDS)
        count = 0
        with self.connect() as connection:
            for record in records:
                row = record.to_dict()
                key = tuple(row[field] for field in UNIQUE_FIELDS)
                count += int(connection.execute(f"SELECT 1 FROM experiment_records WHERE {key_clause}", key).fetchone() is not None)
        return count

    def dataframe(self, filters: dict[str, object] | None = None) -> pd.DataFrame:
        filters = filters or {}
        clauses, values = [], []
        mapping = {
            "experiment_id": "experiment_id", "reviewer_id": "reviewer_id",
            "image_filename": "image_filename", "method": "method",
            "experiment_status": "experiment_status",
        }
        for key, column in mapping.items():
            value = filters.get(key)
            if value:
                clauses.append(f"{column} = ?")
                values.append(value)
        if filters.get("date_from"):
            clauses.append("date(recorded_timestamp) >= date(?)")
            values.append(str(filters["date_from"]))
        if filters.get("date_to"):
            clauses.append("date(recorded_timestamp) <= date(?)")
            values.append(str(filters["date_to"]))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as connection:
            rows = connection.execute(f"SELECT * FROM experiment_records{where} ORDER BY recorded_timestamp DESC", values).fetchall()
        return pd.DataFrame([dict(row) for row in rows], columns=list(ExperimentRecord.__annotations__))

    def delete_record_ids(self, record_ids: Iterable[str]) -> tuple[int, list[str]]:
        ids = list(dict.fromkeys(record_ids))
        if not ids:
            return 0, []
        placeholders = ",".join("?" for _ in ids)
        with self.connect() as connection:
            affected = connection.execute(
                f"SELECT DISTINCT experiment_id FROM experiment_records WHERE record_id IN ({placeholders})", ids
            ).fetchall()
            cursor = connection.execute(f"DELETE FROM experiment_records WHERE record_id IN ({placeholders})", ids)
        return cursor.rowcount, [row[0] for row in affected]

    def delete_where(self, field: str, value: str) -> tuple[int, list[str]]:
        allowed = {"experiment_id", "image_filename", "experiment_status"}
        if field not in allowed:
            raise ValueError("Unsupported deletion field")
        with self.connect() as connection:
            affected = connection.execute(f"SELECT DISTINCT experiment_id FROM experiment_records WHERE {field} = ?", (value,)).fetchall()
            cursor = connection.execute(f"DELETE FROM experiment_records WHERE {field} = ?", (value,))
        return cursor.rowcount, [row[0] for row in affected]

    def clear(self) -> tuple[int, list[str]]:
        with self.connect() as connection:
            rows = connection.execute("SELECT DISTINCT experiment_id FROM experiment_records").fetchall()
            count = connection.execute("SELECT COUNT(*) FROM experiment_records").fetchone()[0]
            connection.execute("DELETE FROM experiment_records")
        return int(count), [row[0] for row in rows]

    def reset(self, confirmed: bool = False) -> tuple[int, list[str]]:
        if not confirmed:
            raise PermissionError("Evaluation-store reset requires explicit confirmation.")
        result = self.clear()
        with self.connect() as connection:
            connection.execute("DROP TABLE IF EXISTS experiment_records")
        self.initialize()
        return result


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
    experiment_version: int = 1,
    experiment_status: str = "Development / Test",
    dataset_source: str = "",
    image_provenance: str = "",
    license_status: str = "unknown",
    ground_truth_status: str = "unknown",
    ground_truth_recall_override: bool = False,
    development_notes: str = "",
    overlap_threshold: float = 0.10,
) -> list[ExperimentRecord]:
    if image_outcome not in {"anomaly present", "no anomaly", "uncertain"}:
        raise ValueError("Invalid image outcome")
    if experiment_status not in EXPERIMENT_STATUSES or ground_truth_status not in GROUND_TRUTH_STATUSES:
        raise ValueError("Invalid experiment or ground-truth status")
    if not experiment_id.strip() or not reviewer_id.strip():
        raise ValueError("Experiment ID and reviewer ID are required.")

    references = [annotation.bbox for annotation in annotations if annotation.decision == "accept"]
    duration = _duration_seconds(review_start_time, review_completion_time)
    method_boxes = _ranked_method_boxes(proposal_result, feature_maps)
    timestamp = datetime.now().isoformat(timespec="seconds")
    records = []

    for method, boxes in method_boxes.items():
        boxes = boxes[:8]
        first_rank = _first_matching_rank(boxes, references, overlap_threshold) if image_outcome == "anomaly present" else None
        is_reviewed = method == "refined contextual method"
        if is_reviewed:
            accepted = sum(annotation.decision == "accept" for annotation in annotations)
            rejected = sum(annotation.decision == "reject" for annotation in annotations)
            uncertain = sum(annotation.decision == "uncertain" for annotation in annotations)
            not_reviewed = max(len(boxes) - len(annotations), 0)
            if not annotations:
                review_status = "not_reviewed"
            elif len(annotations) < len(boxes):
                review_status = "partially_reviewed"
            else:
                review_status = "fully_reviewed"
            accepted_ranks = [
                index for index, proposal in enumerate(proposal_result.proposals, 1)
                if any(annotation.region_id == proposal.region_id and annotation.decision == "accept" for annotation in annotations)
            ]
            first_rank = min(accepted_ranks) if image_outcome == "anomaly present" and accepted_ranks else None
            row_duration = duration
        else:
            accepted = rejected = uncertain = None
            not_reviewed = len(boxes)
            review_status = "not_reviewed"
            row_duration = None

        top_values = _top_k_values(first_rank, image_outcome, ground_truth_status, ground_truth_recall_override)
        records.append(ExperimentRecord(
            record_id=str(uuid4()), experiment_id=experiment_id.strip(), experiment_version=int(experiment_version),
            reviewer_id=reviewer_id.strip(), experiment_status=experiment_status, image_filename=image_filename,
            method=method, recorded_timestamp=timestamp, review_status=review_status,
            final_proposals=len(boxes), accepted=accepted, rejected=rejected, uncertain=uncertain,
            not_reviewed=not_reviewed, image_outcome=image_outcome, dataset_source=dataset_source.strip(),
            image_provenance=image_provenance.strip(), license_status=license_status.strip(),
            ground_truth_status=ground_truth_status, ground_truth_recall_override=ground_truth_recall_override,
            development_notes=development_notes.strip(),
            review_start_time=review_start_time, review_completion_time=review_completion_time,
            review_duration_seconds=row_duration, first_accepted_true_anomaly_rank=first_rank,
            true_anomaly_found_top_1=top_values[0], true_anomaly_found_top_3=top_values[1],
            true_anomaly_found_top_5=top_values[2], true_anomaly_found_top_8=top_values[3],
            proposals_reviewed_before_first_useful=first_rank,
        ))
    return records


def experiment_tables(
    records: Iterable[ExperimentRecord | dict[str, object]] | pd.DataFrame,
    include_development: bool = False,
    development_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(records, pd.DataFrame):
        image_table = records.copy()
    else:
        rows = [record.to_dict() if isinstance(record, ExperimentRecord) else dict(record) for record in records]
        image_table = pd.DataFrame(rows)
    if image_table.empty:
        return image_table, pd.DataFrame(columns=SUMMARY_COLUMNS)
    if development_only:
        image_table = image_table[image_table["experiment_status"] == "Development / Test"].copy()
    elif not include_development:
        image_table = image_table[image_table["experiment_status"] == "Final Research Evaluation"].copy()
    if image_table.empty:
        return image_table, pd.DataFrame(columns=SUMMARY_COLUMNS)

    summaries = []
    for method, group in image_table.groupby("method", sort=False):
        reviewed = group[group["review_status"].isin(["fully_reviewed", "partially_reviewed"])]
        eligible = group[
            (group["image_outcome"] == "anomaly present")
            & (group["ground_truth_status"].isin(["verified ground truth", "reviewer-estimated"])
               | (pd.to_numeric(group["ground_truth_recall_override"], errors="coerce") == 1))
            & group["first_accepted_true_anomaly_rank"].notna()
        ]
        accepted = pd.to_numeric(reviewed["accepted"], errors="coerce")
        rejected = pd.to_numeric(reviewed["rejected"], errors="coerce")
        uncertain = pd.to_numeric(reviewed["uncertain"], errors="coerce")
        denominator = accepted.sum(min_count=1) + rejected.sum(min_count=1)
        useful = pd.to_numeric(eligible["proposals_reviewed_before_first_useful"], errors="coerce").dropna()
        summaries.append({
            "method": method,
            "eligible_images": len(eligible),
            "anomaly_present_images": int((group["image_outcome"] == "anomaly present").sum()),
            "reviewed_images": len(reviewed),
            "not_reviewed_images": int((group["review_status"] == "not_reviewed").sum()),
            "top_1_proposal_recall": _nullable_boolean_mean(eligible, "true_anomaly_found_top_1"),
            "top_3_proposal_recall": _nullable_boolean_mean(eligible, "true_anomaly_found_top_3"),
            "top_5_proposal_recall": _nullable_boolean_mean(eligible, "true_anomaly_found_top_5"),
            "top_8_proposal_recall": _nullable_boolean_mean(eligible, "true_anomaly_found_top_8"),
            "mean_accepted_proposals_per_image": accepted.mean(),
            "mean_rejected_proposals_per_image": rejected.mean(),
            "mean_uncertain_proposals_per_image": uncertain.mean(),
            "mean_not_reviewed_proposals_per_image": pd.to_numeric(group["not_reviewed"], errors="coerce").mean(),
            "mean_false_proposals_per_image": rejected.mean(),
            "annotation_acceptance_rate": accepted.sum(min_count=1) / denominator if pd.notna(denominator) and denominator > 0 else np.nan,
            "mean_review_time_seconds": pd.to_numeric(reviewed["review_duration_seconds"], errors="coerce").mean(),
            "mean_proposals_reviewed_before_first_useful": useful.mean() if not useful.empty else np.nan,
        })
    summary = pd.DataFrame(summaries, columns=SUMMARY_COLUMNS)
    first = [column for column in IMAGE_TABLE_COLUMNS if column in image_table]
    rest = [column for column in image_table.columns if column not in first]
    return image_table[first + rest], summary


def records_to_csv(records: pd.DataFrame) -> bytes:
    return records.to_csv(index=False).encode("utf-8")


def records_to_json(records: pd.DataFrame) -> bytes:
    return records.to_json(orient="records", indent=2).encode("utf-8")


LEGACY_REQUIRED_FIELDS = {"record_id", "review_status", "not_reviewed", "experiment_status"}


def load_legacy_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        return rows if isinstance(rows, list) else []
    except json.JSONDecodeError:
        return []


def legacy_record_indices(rows: list[dict[str, object]]) -> list[int]:
    return [index for index, row in enumerate(rows) if not LEGACY_REQUIRED_FIELDS.issubset(row)]


def migrate_legacy_records(rows: list[dict[str, object]], indices: Iterable[int], store: ExperimentStore) -> int:
    migrated = []
    for index in indices:
        row = rows[index]
        method = str(row.get("method", "refined contextual method"))
        baseline = method != "refined contextual method"
        final_proposals = int(row.get("final_proposals", 0) or 0)
        record = ExperimentRecord(
            record_id=str(row.get("record_id") or uuid4()), experiment_id=str(row.get("experiment_id", "legacy")),
            experiment_version=int(row.get("experiment_version", 1) or 1), reviewer_id=str(row.get("reviewer_id", "unknown")),
            experiment_status=str(row.get("experiment_status", "Development / Test")),
            image_filename=str(row.get("image_filename", "unknown")), method=method,
            recorded_timestamp=str(row.get("recorded_timestamp", datetime.now().isoformat(timespec="seconds"))),
            review_status="not_reviewed" if baseline else str(row.get("review_status", "partially_reviewed")),
            final_proposals=final_proposals, accepted=None if baseline else _optional_int(row.get("accepted")),
            rejected=None if baseline else _optional_int(row.get("rejected")),
            uncertain=None if baseline else _optional_int(row.get("uncertain")),
            not_reviewed=final_proposals if baseline else int(row.get("not_reviewed", 0) or 0),
            image_outcome=str(row.get("image_outcome", "uncertain")), dataset_source=str(row.get("dataset_source", "legacy")),
            image_provenance=str(row.get("image_provenance", "unknown")), license_status=str(row.get("license_status", "unknown")),
            ground_truth_status=str(row.get("ground_truth_status", "unknown")),
            ground_truth_recall_override=bool(row.get("ground_truth_recall_override", False)),
            development_notes=str(row.get("development_notes", "Migrated legacy record")),
            review_start_time=str(row.get("review_start_time", "")), review_completion_time=str(row.get("review_completion_time", "")),
            review_duration_seconds=_optional_float(row.get("review_duration_seconds")),
            first_accepted_true_anomaly_rank=_optional_int(row.get("first_accepted_true_anomaly_rank")),
            true_anomaly_found_top_1=_optional_bool(row.get("true_anomaly_found_top_1")),
            true_anomaly_found_top_3=_optional_bool(row.get("true_anomaly_found_top_3")),
            true_anomaly_found_top_5=_optional_bool(row.get("true_anomaly_found_top_5")),
            true_anomaly_found_top_8=_optional_bool(row.get("true_anomaly_found_top_8")),
            proposals_reviewed_before_first_useful=_optional_int(row.get("proposals_reviewed_before_first_useful")),
        )
        migrated.append(record)
    return store.save(migrated, duplicate_action="overwrite") if migrated else 0


def delete_legacy_records(path: Path, rows: list[dict[str, object]], indices: Iterable[int]) -> int:
    selected = set(indices)
    remaining = [row for index, row in enumerate(rows) if index not in selected]
    path.write_text(json.dumps(remaining, indent=2), encoding="utf-8")
    return len(rows) - len(remaining)


def with_version(records: Iterable[ExperimentRecord], version: int) -> list[ExperimentRecord]:
    return [replace(record, record_id=str(uuid4()), experiment_version=version) for record in records]


def _ranked_method_boxes(proposal_result: ProposalResult, feature_maps: FeatureMaps) -> dict[str, list[tuple[int, int, int, int]]]:
    contour = _rank_components(feature_maps.contour_map, feature_maps.anomaly_strength)
    fixed = _rank_components((feature_maps.anomaly_strength > 128).astype(np.uint8) * 255, feature_maps.anomaly_strength)
    raw = []
    for proposal in proposal_result.proposals:
        mask = cv2.imread(str(proposal.raw_mask_path), cv2.IMREAD_GRAYSCALE)
        raw.append(_bbox_from_mask(mask) if mask is not None else proposal.bbox)
    return {
        "contour-only baseline": contour[:8], "fixed-threshold baseline": fixed[:8],
        "multi-scale fused method": raw[:8],
        "refined contextual method": [proposal.bbox for proposal in proposal_result.proposals[:8]],
    }


def _rank_components(mask: np.ndarray, heatmap: np.ndarray) -> list[tuple[int, int, int, int]]:
    candidates = []
    minimum = max(8, int(mask.size * .0001))
    for component in _components(mask):
        area = cv2.countNonZero(component.mask)
        if area >= minimum:
            candidates.append((float(np.mean(heatmap[component.mask > 0])), area, component.bbox))
    return [bbox for _, _, bbox in sorted(candidates, reverse=True)]


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    return (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)) if xs.size else (0, 0, 0, 0)


def _first_matching_rank(boxes, references, threshold):
    return next((index for index, box in enumerate(boxes, 1) if any(_iou(box, reference) >= threshold for reference in references)), None)


def _top_k_values(rank, outcome, ground_truth, override=False):
    eligible = outcome == "anomaly present" and (ground_truth != "unknown" or override) and rank is not None
    if not eligible:
        return None, None, None, None
    return rank <= 1, rank <= 3, rank <= 5, rank <= 8


def _duration_seconds(start: str, completion: str) -> float:
    try:
        return max((datetime.fromisoformat(completion) - datetime.fromisoformat(start)).total_seconds(), 0.0)
    except ValueError:
        return 0.0


def _nullable_boolean_mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else np.nan


def _iou(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1)


def _sqlite_value(value):
    return int(value) if isinstance(value, bool) else value


def _optional_int(value):
    return int(value) if value is not None and value != "" else None


def _optional_float(value):
    return float(value) if value is not None and value != "" else None


def _optional_bool(value):
    return bool(value) if value is not None else None
