"""Read-only exact, perceptual, and declared-group dataset overlap audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sqlite3


OVERLAP_AUDIT_ALGORITHM_ID = "structvision-dataset-overlap-v1"
PERCEPTUAL_HASH_ALGORITHM_ID = "legacy-dhash-64-candidate-screen"


@dataclass(frozen=True)
class DatasetImageIdentity:
    dataset_id: str
    dataset_version: str
    image_id: str
    filename: str
    split: str
    category: str
    image_sha256: str
    perceptual_hash: str
    source_group_id: str
    template_group_id: str
    acquisition_group_id: str
    declared_duplicate_status: str


@dataclass(frozen=True)
class OverlapPair:
    left_image_id: str
    right_image_id: str
    left_filename: str
    right_filename: str
    right_split: str
    right_category: str
    evidence: str
    distance: int | None = None


@dataclass(frozen=True)
class DatasetOverlapAudit:
    algorithm_id: str
    perceptual_hash_algorithm_id: str
    perceptual_distance_threshold: int
    left_dataset_id: str
    left_dataset_version: str
    right_dataset_id: str
    right_dataset_version: str
    left_image_count: int
    right_image_count: int
    exact_duplicates: tuple[OverlapPair, ...]
    perceptual_near_duplicate_candidates: tuple[OverlapPair, ...]
    overlap_by_right_split: tuple[tuple[str, int], ...]
    overlap_by_right_category: tuple[tuple[str, int], ...]
    source_group_crossings: tuple[OverlapPair, ...]
    template_group_crossings: tuple[OverlapPair, ...]
    acquisition_group_crossings: tuple[OverlapPair, ...]
    unsupported_unique_statuses: tuple[OverlapPair, ...]
    perceptual_uniqueness_established: bool
    evidence_classification: str
    confirmatory_test_protected: bool
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def hamming_distance(left: str, right: str) -> int:
    if not left or not right:
        raise ValueError("Perceptual hashes are required")
    return sum(a != b for a, b in zip(left, right)) + abs(len(left) - len(right))


def _pair(left: DatasetImageIdentity, right: DatasetImageIdentity, evidence: str, distance: int | None = None) -> OverlapPair:
    return OverlapPair(left.image_id, right.image_id, left.filename, right.filename, right.split, right.category, evidence, distance)


def audit_dataset_overlap(
    left: tuple[DatasetImageIdentity, ...] | list[DatasetImageIdentity],
    right: tuple[DatasetImageIdentity, ...] | list[DatasetImageIdentity],
    *,
    perceptual_distance_threshold: int = 3,
) -> DatasetOverlapAudit:
    left = tuple(left)
    right = tuple(right)
    if not left or not right:
        raise ValueError("Both datasets must contain images")
    if perceptual_distance_threshold < 0:
        raise ValueError("Perceptual distance threshold cannot be negative")
    exact: list[OverlapPair] = []
    perceptual: list[OverlapPair] = []
    source: list[OverlapPair] = []
    template: list[OverlapPair] = []
    acquisition: list[OverlapPair] = []
    unsupported: list[OverlapPair] = []
    for left_image in left:
        for right_image in right:
            is_exact = bool(left_image.image_sha256) and left_image.image_sha256 == right_image.image_sha256
            if is_exact:
                item = _pair(left_image, right_image, "exact_sha256")
                exact.append(item)
                statuses = (left_image.declared_duplicate_status.lower(), right_image.declared_duplicate_status.lower())
                if any("unique" in status for status in statuses):
                    unsupported.append(_pair(left_image, right_image, "declared_unique_despite_cross_dataset_exact_sha256"))
            elif left_image.perceptual_hash and right_image.perceptual_hash:
                distance = hamming_distance(left_image.perceptual_hash, right_image.perceptual_hash)
                if distance <= perceptual_distance_threshold:
                    perceptual.append(_pair(left_image, right_image, "perceptual_candidate_not_duplicate_proof", distance))
                    statuses = (left_image.declared_duplicate_status.lower(), right_image.declared_duplicate_status.lower())
                    if any("unique" in status for status in statuses):
                        unsupported.append(_pair(left_image, right_image, "declared_unique_despite_unresolved_perceptual_candidate", distance))
            if left_image.source_group_id and left_image.source_group_id == right_image.source_group_id:
                source.append(_pair(left_image, right_image, f"source_group:{left_image.source_group_id}"))
            if left_image.template_group_id and left_image.template_group_id == right_image.template_group_id:
                template.append(_pair(left_image, right_image, f"template_group:{left_image.template_group_id}"))
            if left_image.acquisition_group_id and left_image.acquisition_group_id == right_image.acquisition_group_id:
                acquisition.append(_pair(left_image, right_image, f"acquisition_group:{left_image.acquisition_group_id}"))
    split_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    for item in exact:
        split_counts[item.right_split] = split_counts.get(item.right_split, 0) + 1
        category_counts[item.right_category] = category_counts.get(item.right_category, 0) + 1
    left_identity = (left[0].dataset_id, left[0].dataset_version)
    right_identity = (right[0].dataset_id, right[0].dataset_version)
    classification = (
        "historical engineering comparison — not confirmatory"
        if exact else
        "no exact overlap detected; confirmatory status still requires protocol review"
    )
    return DatasetOverlapAudit(
        OVERLAP_AUDIT_ALGORITHM_ID, PERCEPTUAL_HASH_ALGORITHM_ID,
        perceptual_distance_threshold, *left_identity, *right_identity,
        len(left), len(right), tuple(exact), tuple(perceptual),
        tuple(sorted(split_counts.items())), tuple(sorted(category_counts.items())),
        tuple(source), tuple(template), tuple(acquisition), tuple(unsupported),
        False, classification, False,
        (
            "A 64-bit difference hash is only a candidate screen and cannot establish absence of semantic or transformed near duplicates.",
            "Exact or declared-group overlap with a final split invalidates protected confirmatory-test status.",
        ),
    )


def read_registry_dataset(database_path: Path, dataset_id: str, dataset_version: str) -> tuple[DatasetImageIdentity, ...]:
    """Read identities using SQLite read-only mode; never initialize or update a registry."""
    uri = Path(database_path).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT dataset_id,dataset_version,image_id,original_filename,split,
                   anomaly_type,sha256_hash,perceptual_hash,source_group_id,
                   template_group_id,group_id,duplicate_type,duplicate_status
            FROM images WHERE dataset_id=? AND dataset_version=? ORDER BY image_id
            """,
            (dataset_id, dataset_version),
        ).fetchall()
    finally:
        connection.close()
    return tuple(
        DatasetImageIdentity(
            row["dataset_id"], row["dataset_version"], row["image_id"],
            row["original_filename"], row["split"], row["anomaly_type"],
            row["sha256_hash"], row["perceptual_hash"], row["source_group_id"],
            row["template_group_id"], row["group_id"],
            row["duplicate_type"] or row["duplicate_status"],
        )
        for row in rows
    )


def audit_registered_datasets(
    database_path: Path,
    left_dataset_id: str,
    left_dataset_version: str,
    right_dataset_id: str,
    right_dataset_version: str,
    *,
    perceptual_distance_threshold: int = 3,
) -> DatasetOverlapAudit:
    left = read_registry_dataset(database_path, left_dataset_id, left_dataset_version)
    right = read_registry_dataset(database_path, right_dataset_id, right_dataset_version)
    return audit_dataset_overlap(left, right, perceptual_distance_threshold=perceptual_distance_threshold)
