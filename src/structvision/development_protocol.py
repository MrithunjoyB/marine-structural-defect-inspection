"""Prospective development-only cohort selection over the read-only registry."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import sqlite3

import numpy as np

from scientific_contract.dataset_audit import DatasetImageIdentity, audit_dataset_overlap, read_registry_dataset
from scientific_contract.hashing import canonical_json, is_sha256, sha256_json
from structvision.inputs import content_hash

from .normal_feature.errors import DevelopmentProtocolError
from .normal_feature.types import NormalFitSample


DEVELOPMENT_PROTOCOL_VERSION = "structvision-normal-feature-development-v1"
DEVELOPMENT_MANIFEST_SCHEMA_VERSION = "protected-development-manifest-v1"
SOURCE_DATASET_ID = "synthetic-expanded"
SOURCE_DATASET_VERSION = "1.0"


@dataclass(frozen=True)
class DevelopmentImageIdentity:
    image_id: str
    role: str
    split_role: str
    category: str
    image_outcome: str
    image_path: str
    image_sha256: str
    ground_truth_path: str | None
    ground_truth_sha256: str
    ground_truth_kind: str
    source_group_id: str
    template_group_id: str
    acquisition_group_id: str

    def __post_init__(self) -> None:
        if self.role not in {"normal_fit", "calibration_validation"}:
            raise DevelopmentProtocolError("Development roles are fixed; no test role exists")
        required_split = "train" if self.role == "normal_fit" else "validation"
        if self.split_role != required_split:
            raise DevelopmentProtocolError("Development role and source split differ")
        if self.role == "normal_fit" and self.image_outcome != "no_anomaly":
            raise DevelopmentProtocolError("An anomaly-present image entered normal_fit")
        if not is_sha256(self.image_sha256) or not is_sha256(self.ground_truth_sha256):
            raise DevelopmentProtocolError("Selected images require exact image and ground-truth hashes")

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class DevelopmentExclusion:
    image_id: str
    source_split: str
    category: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "image_id": self.image_id,
            "source_split": self.source_split,
            "category": self.category,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class ProtectedDevelopmentManifest:
    schema_version: str
    protocol_version: str
    evidence_classification: str
    source_dataset_id: str
    source_dataset_version: str
    source_registry_sha256: str
    historical_result_store_sha256: str
    selected_images: tuple[DevelopmentImageIdentity, ...]
    exclusions: tuple[DevelopmentExclusion, ...]
    expected_fit_categories: tuple[str, ...]
    expected_validation_categories: tuple[str, ...]
    missing_categories: tuple[str, ...]
    overlap_policy: tuple[str, ...]
    manifest_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != DEVELOPMENT_MANIFEST_SCHEMA_VERSION:
            raise DevelopmentProtocolError("Unsupported development-manifest schema")
        if self.protocol_version != DEVELOPMENT_PROTOCOL_VERSION:
            raise DevelopmentProtocolError("Unsupported development protocol")
        if self.evidence_classification != "development-only — non-confirmatory":
            raise DevelopmentProtocolError("The protected cohort cannot be labelled confirmatory")
        if self.source_dataset_id != SOURCE_DATASET_ID or self.source_dataset_version != SOURCE_DATASET_VERSION:
            raise DevelopmentProtocolError("The protected cohort source identity is fixed")
        if self.missing_categories:
            raise DevelopmentProtocolError("A required category disappeared from the protected cohort")
        if len({item.image_id for item in self.selected_images}) != len(self.selected_images):
            raise DevelopmentProtocolError("Selected development image IDs must be unique")
        fit = tuple(item for item in self.selected_images if item.role == "normal_fit")
        validation = tuple(item for item in self.selected_images if item.role == "calibration_validation")
        if not fit or not validation:
            raise DevelopmentProtocolError("Both fit and calibration-validation roles are required")
        if not {item.image_outcome for item in validation}.issuperset({"no_anomaly", "anomaly_present"}):
            raise DevelopmentProtocolError("Calibration-validation requires clean and anomaly-present images")
        if {item.image_sha256 for item in fit} & {item.image_sha256 for item in validation}:
            raise DevelopmentProtocolError("An exact duplicate crosses fit and validation")
        for field in ("source_group_id", "template_group_id", "acquisition_group_id"):
            fit_groups = {getattr(item, field) for item in fit if getattr(item, field)}
            validation_groups = {getattr(item, field) for item in validation if getattr(item, field)}
            if fit_groups & validation_groups:
                raise DevelopmentProtocolError(f"{field} crosses fit and validation")
        if self.manifest_hash != "0" * 64 and self.manifest_hash != sha256_json(self.to_dict(include_manifest_hash=False)):
            raise DevelopmentProtocolError("Development-manifest hash mismatch")

    @classmethod
    def create(cls, **values: object) -> "ProtectedDevelopmentManifest":
        initial = cls(manifest_hash="0" * 64, **values)
        return replace(initial, manifest_hash=sha256_json(initial.to_dict(include_manifest_hash=False)))

    @property
    def normal_fit(self) -> tuple[DevelopmentImageIdentity, ...]:
        return tuple(item for item in self.selected_images if item.role == "normal_fit")

    @property
    def calibration_validation(self) -> tuple[DevelopmentImageIdentity, ...]:
        return tuple(item for item in self.selected_images if item.role == "calibration_validation")

    def to_dict(self, *, include_manifest_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "evidence_classification": self.evidence_classification,
            "source_dataset_id": self.source_dataset_id,
            "source_dataset_version": self.source_dataset_version,
            "source_registry_sha256": self.source_registry_sha256,
            "historical_result_store_sha256": self.historical_result_store_sha256,
            "selected_images": [item.to_dict() for item in self.selected_images],
            "exclusions": [item.to_dict() for item in self.exclusions],
            "expected_fit_categories": list(self.expected_fit_categories),
            "expected_validation_categories": list(self.expected_validation_categories),
            "missing_categories": list(self.missing_categories),
            "overlap_policy": list(self.overlap_policy),
            "role_counts": {
                "normal_fit": len(self.normal_fit),
                "calibration_validation": len(self.calibration_validation),
            },
        }
        if include_manifest_hash:
            payload["manifest_hash"] = self.manifest_hash
        return payload

    def to_json(self) -> str:
        return canonical_json(self.to_dict())


def _read_rows(database: Path) -> list[sqlite3.Row]:
    connection = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute(
            """
            SELECT image_id,stored_filename,sha256_hash,width,height,annotation_path,split,
                   anomaly_type,image_outcome,source_group_id,template_group_id,group_id
            FROM images WHERE dataset_id=? AND dataset_version=? AND split IN ('train','validation')
            ORDER BY image_id
            """,
            (SOURCE_DATASET_ID, SOURCE_DATASET_VERSION),
        ).fetchall()
    finally:
        connection.close()


def _historical_result_ids(database: Path) -> set[str]:
    connection = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return {str(row[0]) for row in connection.execute("SELECT DISTINCT image_id FROM automatic_results")}
    finally:
        connection.close()


def _reason_map(
    left: tuple[DatasetImageIdentity, ...],
    right: tuple[DatasetImageIdentity, ...],
    prefix: str,
) -> dict[str, set[str]]:
    if not left or not right:
        return {}
    audit = audit_dataset_overlap(left, right, perceptual_distance_threshold=3)
    reasons: dict[str, set[str]] = {}
    families = (
        (audit.exact_duplicates, "exact_sha256"),
        (audit.perceptual_near_duplicate_candidates, "flagged_perceptual_candidate"),
        (audit.source_group_crossings, "source_group_crossing"),
        (audit.template_group_crossings, "template_group_crossing"),
        (audit.acquisition_group_crossings, "acquisition_group_crossing"),
    )
    for pairs, name in families:
        for pair in pairs:
            reasons.setdefault(pair.left_image_id, set()).add(f"{prefix}:{name}")
    return reasons


def _merge_reasons(target: dict[str, set[str]], source: dict[str, set[str]]) -> None:
    for image_id, reasons in source.items():
        target.setdefault(image_id, set()).update(reasons)


def create_protected_development_manifest(
    *,
    repository_root: Path,
    registry_database: Path,
    historical_result_database: Path,
) -> ProtectedDevelopmentManifest:
    """Select only protected train/validation identities; never decode a test image."""
    repository_root = Path(repository_root).resolve()
    registry_database = Path(registry_database).resolve()
    historical_result_database = Path(historical_result_database).resolve()
    if not registry_database.is_file() or not historical_result_database.is_file():
        raise DevelopmentProtocolError("Protected registry and historical result store are required")
    expanded = read_registry_dataset(registry_database, SOURCE_DATASET_ID, SOURCE_DATASET_VERSION)
    pilot = read_registry_dataset(registry_database, "synthetic-expanded-pilot", "1.0")
    controlled = read_registry_dataset(registry_database, "synthetic-controlled", "1.0")
    candidates = tuple(item for item in expanded if item.split in {"train", "validation"})
    historical_tests = tuple(item for item in expanded + controlled if item.split == "test")
    result_ids = _historical_result_ids(historical_result_database)
    registered = {item.image_id: item for item in expanded + controlled + pilot}
    result_identities = tuple(registered[item] for item in sorted(result_ids) if item in registered)
    if result_ids - set(registered):
        raise DevelopmentProtocolError("A historical result image is absent from the registry")
    protected: dict[str, set[str]] = {}
    _merge_reasons(protected, _reason_map(candidates, tuple(pilot), "pilot"))
    _merge_reasons(protected, _reason_map(candidates, historical_tests, "historical_test"))
    _merge_reasons(protected, _reason_map(candidates, result_identities, "historical_verification"))
    rows = {str(row["image_id"]): row for row in _read_rows(registry_database)}
    expected_fit = tuple(sorted({item.category for item in candidates if item.split == "train" and rows[item.image_id]["image_outcome"] == "no_anomaly"}))
    expected_validation = tuple(sorted({item.category for item in candidates if item.split == "validation"}))
    selected: list[DevelopmentImageIdentity] = []
    exclusions: list[DevelopmentExclusion] = []
    for identity in candidates:
        row = rows[identity.image_id]
        reasons = set(protected.get(identity.image_id, set()))
        if identity.split == "train" and str(row["image_outcome"]) != "no_anomaly":
            reasons.add("train_anomaly_not_eligible_for_normal_fit")
        if reasons:
            exclusions.append(DevelopmentExclusion(identity.image_id, identity.split, identity.category, tuple(sorted(reasons))))
            continue
        role = "normal_fit" if identity.split == "train" else "calibration_validation"
        image_path = Path("research_data") / "raw" / SOURCE_DATASET_ID / str(row["stored_filename"])
        absolute_image = repository_root / image_path
        if not absolute_image.is_file() or content_hash(absolute_image) != str(row["sha256_hash"]):
            raise DevelopmentProtocolError(f"Missing or changed selected image: {identity.image_id}")
        if str(row["image_outcome"]) == "no_anomaly":
            truth_path = None
            truth = np.zeros((int(row["height"]), int(row["width"])), dtype=np.uint8)
            truth_hash = content_hash(truth)
            truth_kind = "implicit_verified_zero_mask"
        else:
            raw_truth_path = str(row["annotation_path"])
            if not raw_truth_path:
                raise DevelopmentProtocolError(f"Missing selected ground truth: {identity.image_id}")
            path = Path(raw_truth_path)
            absolute_truth = path if path.is_absolute() else repository_root / path
            if not absolute_truth.is_file():
                raise DevelopmentProtocolError(f"Missing selected ground truth: {identity.image_id}")
            truth_path = absolute_truth.relative_to(repository_root).as_posix()
            truth_hash = content_hash(absolute_truth)
            truth_kind = "registered_binary_mask_file"
        selected.append(DevelopmentImageIdentity(
            identity.image_id, role, identity.split, identity.category, str(row["image_outcome"]),
            image_path.as_posix(), str(row["sha256_hash"]), truth_path, truth_hash, truth_kind,
            identity.source_group_id, identity.template_group_id, identity.acquisition_group_id,
        ))
    selected.sort(key=lambda item: (0 if item.role == "normal_fit" else 1, item.image_id))
    exclusions.sort(key=lambda item: item.image_id)
    present_fit = {item.category for item in selected if item.role == "normal_fit"}
    present_validation = {item.category for item in selected if item.role == "calibration_validation"}
    missing = tuple(sorted((set(expected_fit) - present_fit) | (set(expected_validation) - present_validation)))
    return ProtectedDevelopmentManifest.create(
        schema_version=DEVELOPMENT_MANIFEST_SCHEMA_VERSION,
        protocol_version=DEVELOPMENT_PROTOCOL_VERSION,
        evidence_classification="development-only — non-confirmatory",
        source_dataset_id=SOURCE_DATASET_ID,
        source_dataset_version=SOURCE_DATASET_VERSION,
        source_registry_sha256=content_hash(registry_database),
        historical_result_store_sha256=content_hash(historical_result_database),
        selected_images=tuple(selected),
        exclusions=tuple(exclusions),
        expected_fit_categories=expected_fit,
        expected_validation_categories=expected_validation,
        missing_categories=missing,
        overlap_policy=(
            "exclude exact SHA-256 overlap with pilot, historical tests, or prior verification",
            "exclude legacy dHash candidates at Hamming distance <=3",
            "exclude declared source/template/acquisition group crossings",
            "never decode or inspect a forbidden test image",
        ),
    )


def write_development_manifest(manifest: ProtectedDevelopmentManifest, path: Path) -> None:
    path = Path(path)
    if path.exists():
        raise DevelopmentProtocolError("Development manifest is immutable and refuses overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.to_json() + "\n", encoding="utf-8")


def load_development_manifest(path: Path) -> ProtectedDevelopmentManifest:
    text = Path(path).read_text(encoding="utf-8")
    payload = json.loads(text)
    if canonical_json(payload) + "\n" != text:
        raise DevelopmentProtocolError("Development manifest is not canonical")
    return ProtectedDevelopmentManifest(
        schema_version=str(payload["schema_version"]),
        protocol_version=str(payload["protocol_version"]),
        evidence_classification=str(payload["evidence_classification"]),
        source_dataset_id=str(payload["source_dataset_id"]),
        source_dataset_version=str(payload["source_dataset_version"]),
        source_registry_sha256=str(payload["source_registry_sha256"]),
        historical_result_store_sha256=str(payload["historical_result_store_sha256"]),
        selected_images=tuple(DevelopmentImageIdentity(**item) for item in payload["selected_images"]),
        exclusions=tuple(DevelopmentExclusion(
            str(item["image_id"]), str(item["source_split"]), str(item["category"]),
            tuple(str(reason) for reason in item["reasons"]),
        ) for item in payload["exclusions"]),
        expected_fit_categories=tuple(str(item) for item in payload["expected_fit_categories"]),
        expected_validation_categories=tuple(str(item) for item in payload["expected_validation_categories"]),
        missing_categories=tuple(str(item) for item in payload["missing_categories"]),
        overlap_policy=tuple(str(item) for item in payload["overlap_policy"]),
        manifest_hash=str(payload["manifest_hash"]),
    )


def normal_fit_samples(manifest: ProtectedDevelopmentManifest, repository_root: Path) -> tuple[NormalFitSample, ...]:
    root = Path(repository_root)
    return tuple(
        NormalFitSample(
            image=root / item.image_path,
            image_id=item.image_id,
            image_sha256=item.image_sha256,
            ground_truth_sha256=item.ground_truth_sha256,
            metadata={"category": item.category, "source_group_id": item.source_group_id, "template_group_id": item.template_group_id},
        )
        for item in manifest.normal_fit
    )
