"""Prospective train/validation-only protocol for hybrid development.

The selector reads registry metadata and protected hashes in read-only mode.  It
never decodes a registered test image.  Only paths embedded in the returned
manifest may subsequently be loaded by the hybrid workflow.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from scientific_contract.dataset_audit import (
    DatasetImageIdentity,
    hamming_distance,
    read_registry_dataset,
)
from scientific_contract.hashing import canonical_json, is_sha256, sha256_json
from structvision.development_protocol import _historical_result_ids, _merge_reasons, _reason_map
from structvision.inputs import content_hash
from structvision.normal_feature.types import NormalFitSample

from .errors import HybridProtocolError


HYBRID_PROTOCOL_VERSION = "structvision-hybrid-development-v1"
HYBRID_MANIFEST_SCHEMA_VERSION = "hybrid-development-manifest-v1"
SOURCE_DATASET_ID = "synthetic-expanded"
SOURCE_DATASET_VERSION = "1.0"
EVIDENCE_CLASSIFICATION = "development holdout — non-confirmatory"
DETERMINISTIC_SEED = 73021
NORMAL_FIT_TARGET_FRACTION = 0.75
PRIORITY_POSITIVE_CATEGORIES = ("pitting_cluster", "thin_crack", "weld_disturbance")
ROLE_ORDER = ("hybrid_normal_fit", "hybrid_fusion_fit", "hybrid_development_holdout")


@dataclass(frozen=True)
class HybridImageIdentity:
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
    perceptual_hash: str

    def __post_init__(self) -> None:
        if self.role not in ROLE_ORDER:
            raise HybridProtocolError("Hybrid roles are fixed and contain no test role")
        required_split = "validation" if self.role == "hybrid_development_holdout" else "train"
        if self.split_role != required_split:
            raise HybridProtocolError("Hybrid role and source split differ")
        if self.role == "hybrid_normal_fit" and self.image_outcome != "no_anomaly":
            raise HybridProtocolError("An anomaly-present image entered hybrid_normal_fit")
        if self.image_outcome not in {"no_anomaly", "anomaly_present"}:
            raise HybridProtocolError("Unsupported image outcome")
        if not is_sha256(self.image_sha256) or not is_sha256(self.ground_truth_sha256):
            raise HybridProtocolError("Every selected image and truth requires a SHA-256 identity")
        for value in (self.image_path, self.ground_truth_path):
            if value is not None and (Path(value).is_absolute() or ".." in Path(value).parts):
                raise HybridProtocolError("Manifest paths must remain repository-relative")

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class HybridExclusion:
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


def _role_members(images: tuple[HybridImageIdentity, ...], role: str) -> tuple[HybridImageIdentity, ...]:
    return tuple(item for item in images if item.role == role)


@dataclass(frozen=True)
class HybridDevelopmentManifest:
    schema_version: str
    protocol_version: str
    evidence_classification: str
    source_dataset_id: str
    source_dataset_version: str
    source_registry_sha256: str
    historical_result_store_sha256: str
    allocation_seed: int
    normal_fit_target_fraction: float
    selected_images: tuple[HybridImageIdentity, ...]
    exclusions: tuple[HybridExclusion, ...]
    expected_clean_categories: tuple[str, ...]
    expected_positive_categories: tuple[str, ...]
    priority_positive_categories: tuple[str, ...]
    missing_priority_categories_by_role: tuple[tuple[str, tuple[str, ...]], ...]
    overlap_policy: tuple[str, ...]
    allocation_policy: str
    manifest_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != HYBRID_MANIFEST_SCHEMA_VERSION:
            raise HybridProtocolError("Unsupported hybrid manifest schema")
        if self.protocol_version != HYBRID_PROTOCOL_VERSION:
            raise HybridProtocolError("Unsupported hybrid protocol")
        if self.evidence_classification != EVIDENCE_CLASSIFICATION:
            raise HybridProtocolError("Hybrid holdout cannot be labelled confirmatory")
        if (self.source_dataset_id, self.source_dataset_version) != (SOURCE_DATASET_ID, SOURCE_DATASET_VERSION):
            raise HybridProtocolError("Hybrid source data identity differs")
        if self.allocation_seed != DETERMINISTIC_SEED or self.normal_fit_target_fraction != NORMAL_FIT_TARGET_FRACTION:
            raise HybridProtocolError("Prospective allocation policy differs")
        ids = [item.image_id for item in self.selected_images]
        if len(ids) != len(set(ids)):
            raise HybridProtocolError("An image appears in multiple hybrid roles")
        roles = {role: _role_members(self.selected_images, role) for role in ROLE_ORDER}
        if any(not values for values in roles.values()):
            raise HybridProtocolError("All three hybrid roles are required")
        if {item.image_outcome for item in roles["hybrid_normal_fit"]} != {"no_anomaly"}:
            raise HybridProtocolError("Hybrid normal fit must contain only clean images")
        for role in ("hybrid_fusion_fit", "hybrid_development_holdout"):
            if {item.image_outcome for item in roles[role]} != {"no_anomaly", "anomaly_present"}:
                raise HybridProtocolError(f"{role} requires clean and anomaly-present images")
        for left_index, left_role in enumerate(ROLE_ORDER):
            for right_role in ROLE_ORDER[left_index + 1:]:
                left = roles[left_role]
                right = roles[right_role]
                if {item.image_sha256 for item in left} & {item.image_sha256 for item in right}:
                    raise HybridProtocolError("A content hash crosses hybrid roles")
                for field in ("source_group_id", "template_group_id", "acquisition_group_id"):
                    left_groups = {getattr(item, field) for item in left if getattr(item, field)}
                    right_groups = {getattr(item, field) for item in right if getattr(item, field)}
                    if left_groups & right_groups:
                        raise HybridProtocolError(f"{field} crosses hybrid roles")
                for a in left:
                    for b in right:
                        if a.perceptual_hash and b.perceptual_hash and hamming_distance(a.perceptual_hash, b.perceptual_hash) <= 3:
                            raise HybridProtocolError("A perceptual candidate group crosses hybrid roles")
        reported = dict(self.missing_priority_categories_by_role)
        if set(reported) != {"hybrid_fusion_fit", "hybrid_development_holdout"}:
            raise HybridProtocolError("Priority-category preservation must be explicitly reported for both labelled roles")
        for role in reported:
            observed = {item.category for item in roles[role] if item.image_outcome == "anomaly_present"}
            expected_missing = tuple(sorted(set(self.priority_positive_categories) - observed))
            if reported[role] != expected_missing:
                raise HybridProtocolError("Priority-category disappearance report is incomplete")
        if any(reported.values()):
            raise HybridProtocolError("A priority category disappeared from a protected role")
        if self.manifest_hash != "0" * 64 and self.manifest_hash != sha256_json(self.to_dict(include_manifest_hash=False)):
            raise HybridProtocolError("Hybrid manifest hash mismatch")

    @classmethod
    def create(cls, **values: object) -> "HybridDevelopmentManifest":
        initial = cls(manifest_hash="0" * 64, **values)
        return replace(initial, manifest_hash=sha256_json(initial.to_dict(include_manifest_hash=False)))

    @property
    def normal_fit(self) -> tuple[HybridImageIdentity, ...]:
        return _role_members(self.selected_images, "hybrid_normal_fit")

    @property
    def fusion_fit(self) -> tuple[HybridImageIdentity, ...]:
        return _role_members(self.selected_images, "hybrid_fusion_fit")

    @property
    def development_holdout(self) -> tuple[HybridImageIdentity, ...]:
        return _role_members(self.selected_images, "hybrid_development_holdout")

    def to_dict(self, *, include_manifest_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "evidence_classification": self.evidence_classification,
            "source_dataset_id": self.source_dataset_id,
            "source_dataset_version": self.source_dataset_version,
            "source_registry_sha256": self.source_registry_sha256,
            "historical_result_store_sha256": self.historical_result_store_sha256,
            "allocation_seed": self.allocation_seed,
            "normal_fit_target_fraction": self.normal_fit_target_fraction,
            "selected_images": [item.to_dict() for item in self.selected_images],
            "exclusions": [item.to_dict() for item in self.exclusions],
            "expected_clean_categories": list(self.expected_clean_categories),
            "expected_positive_categories": list(self.expected_positive_categories),
            "priority_positive_categories": list(self.priority_positive_categories),
            "missing_priority_categories_by_role": [
                [role, list(categories)] for role, categories in self.missing_priority_categories_by_role
            ],
            "overlap_policy": list(self.overlap_policy),
            "allocation_policy": self.allocation_policy,
            "role_counts": {role: len(_role_members(self.selected_images, role)) for role in ROLE_ORDER},
        }
        if include_manifest_hash:
            payload["manifest_hash"] = self.manifest_hash
        return payload

    def to_json(self) -> str:
        return canonical_json(self.to_dict())


class _UnionFind:
    def __init__(self, identities: tuple[str, ...]):
        self.parent = {item: item for item in identities}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _components(identities: tuple[DatasetImageIdentity, ...]) -> tuple[tuple[DatasetImageIdentity, ...], ...]:
    union = _UnionFind(tuple(item.image_id for item in identities))
    indexes: tuple[tuple[str, object], ...] = (
        ("image_sha256", defaultdict(list)),
        ("source_group_id", defaultdict(list)),
        ("template_group_id", defaultdict(list)),
        ("acquisition_group_id", defaultdict(list)),
    )
    for field, index in indexes:
        for item in identities:
            value = getattr(item, field)
            if value:
                index[value].append(item.image_id)
        for members in index.values():
            for member in members[1:]:
                union.union(members[0], member)
    for offset, left in enumerate(identities):
        if not left.perceptual_hash:
            continue
        for right in identities[offset + 1:]:
            if right.perceptual_hash and hamming_distance(left.perceptual_hash, right.perceptual_hash) <= 3:
                union.union(left.image_id, right.image_id)
    groups: dict[str, list[DatasetImageIdentity]] = defaultdict(list)
    for item in identities:
        groups[union.find(item.image_id)].append(item)
    return tuple(
        tuple(sorted(values, key=lambda item: item.image_id))
        for _, values in sorted(groups.items())
    )


def _read_rows(database: Path) -> dict[str, sqlite3.Row]:
    connection = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT image_id,stored_filename,sha256_hash,width,height,annotation_path,split,
                   anomaly_type,image_outcome,source_group_id,template_group_id,group_id,perceptual_hash
            FROM images WHERE dataset_id=? AND dataset_version=? AND split IN ('train','validation')
            ORDER BY image_id
            """,
            (SOURCE_DATASET_ID, SOURCE_DATASET_VERSION),
        ).fetchall()
    finally:
        connection.close()
    return {str(row["image_id"]): row for row in rows}


def _allocation_digest(component: tuple[DatasetImageIdentity, ...]) -> str:
    identity = "\0".join(item.image_id for item in component)
    return hashlib.sha256(f"{DETERMINISTIC_SEED}\0{identity}".encode("utf-8")).hexdigest()


def _allocate_clean_components(
    components: tuple[tuple[DatasetImageIdentity, ...], ...],
) -> dict[str, str]:
    assignments: dict[str, str] = {}
    by_category: dict[str, list[tuple[DatasetImageIdentity, ...]]] = defaultdict(list)
    for component in components:
        categories = {item.category for item in component}
        if len(categories) != 1:
            raise HybridProtocolError("A clean allocation group crosses categories and cannot be stratified prospectively")
        by_category[next(iter(categories))].append(component)
    for category, values in sorted(by_category.items()):
        ordered = sorted(values, key=lambda item: (_allocation_digest(item), item[0].image_id))
        if len(ordered) < 2:
            raise HybridProtocolError(f"Clean category {category} lacks two disjoint groups")
        fusion_count = max(1, round(len(ordered) * (1.0 - NORMAL_FIT_TARGET_FRACTION)))
        fusion_count = min(fusion_count, len(ordered) - 1)
        for index, component in enumerate(ordered):
            role = "hybrid_fusion_fit" if index < fusion_count else "hybrid_normal_fit"
            for item in component:
                assignments[item.image_id] = role
    return assignments


def create_hybrid_development_manifest(
    *,
    repository_root: Path,
    registry_database: Path,
    historical_result_database: Path,
) -> HybridDevelopmentManifest:
    root = Path(repository_root).resolve()
    registry = Path(registry_database).resolve()
    historical = Path(historical_result_database).resolve()
    if not registry.is_file() or not historical.is_file():
        raise HybridProtocolError("Protected registry and historical result store are required")
    expanded = read_registry_dataset(registry, SOURCE_DATASET_ID, SOURCE_DATASET_VERSION)
    pilot = read_registry_dataset(registry, "synthetic-expanded-pilot", "1.0")
    controlled = read_registry_dataset(registry, "synthetic-controlled", "1.0")
    candidates = tuple(item for item in expanded if item.split in {"train", "validation"})
    historical_tests = tuple(item for item in expanded + controlled if item.split == "test")
    result_ids = _historical_result_ids(historical)
    registered = {item.image_id: item for item in expanded + controlled + pilot}
    if result_ids - set(registered):
        raise HybridProtocolError("A historical result image is absent from the registry")
    historical_results = tuple(registered[item] for item in sorted(result_ids))
    protected: dict[str, set[str]] = {}
    _merge_reasons(protected, _reason_map(candidates, tuple(pilot), "pilot"))
    _merge_reasons(protected, _reason_map(candidates, historical_tests, "historical_test"))
    _merge_reasons(protected, _reason_map(candidates, historical_results, "historical_verification"))
    rows = _read_rows(registry)
    exclusion_reasons: dict[str, set[str]] = {key: set(value) for key, value in protected.items()}
    externally_eligible = tuple(item for item in candidates if item.image_id not in exclusion_reasons)
    for component in _components(externally_eligible):
        splits = {item.split for item in component}
        if splits == {"train", "validation"}:
            for item in component:
                if item.split == "train":
                    exclusion_reasons.setdefault(item.image_id, set()).add("holdout:identity_or_group_crossing")
    eligible = tuple(item for item in externally_eligible if item.image_id not in exclusion_reasons)
    components = _components(eligible)
    if any({item.split for item in component} == {"train", "validation"} for component in components):
        raise HybridProtocolError("A train/validation identity group remained after holdout protection")
    clean_components = tuple(
        component for component in components
        if {item.split for item in component} == {"train"}
        and all(str(rows[item.image_id]["image_outcome"]) == "no_anomaly" for item in component)
    )
    assignments = _allocate_clean_components(clean_components)
    for component in components:
        if {item.split for item in component} == {"validation"}:
            for item in component:
                assignments[item.image_id] = "hybrid_development_holdout"
        elif any(str(rows[item.image_id]["image_outcome"]) == "anomaly_present" for item in component):
            for item in component:
                assignments[item.image_id] = "hybrid_fusion_fit"
    selected: list[HybridImageIdentity] = []
    for identity in eligible:
        row = rows[identity.image_id]
        role = assignments.get(identity.image_id)
        if role is None:
            raise HybridProtocolError(f"Eligible image lacks a prospective role: {identity.image_id}")
        image_path = Path("research_data") / "raw" / SOURCE_DATASET_ID / str(row["stored_filename"])
        absolute_image = root / image_path
        if not absolute_image.is_file() or content_hash(absolute_image) != str(row["sha256_hash"]):
            raise HybridProtocolError(f"Missing or changed selected image: {identity.image_id}")
        outcome = str(row["image_outcome"])
        if outcome == "no_anomaly":
            truth_path = None
            truth_hash = content_hash(np.zeros((int(row["height"]), int(row["width"])), dtype=np.uint8))
            truth_kind = "implicit_verified_zero_mask"
        else:
            raw_truth_path = str(row["annotation_path"] or "")
            if not raw_truth_path:
                raise HybridProtocolError(f"Missing selected truth: {identity.image_id}")
            candidate_path = Path(raw_truth_path)
            absolute_truth = candidate_path if candidate_path.is_absolute() else root / candidate_path
            if not absolute_truth.is_file():
                raise HybridProtocolError(f"Missing selected truth: {identity.image_id}")
            truth_path = absolute_truth.relative_to(root).as_posix()
            truth_hash = content_hash(absolute_truth)
            truth_kind = "registered_binary_mask_file"
        selected.append(HybridImageIdentity(
            identity.image_id, role, identity.split, identity.category, outcome,
            image_path.as_posix(), str(row["sha256_hash"]), truth_path, truth_hash, truth_kind,
            identity.source_group_id, identity.template_group_id, identity.acquisition_group_id,
            identity.perceptual_hash,
        ))
    selected.sort(key=lambda item: (ROLE_ORDER.index(item.role), item.image_id))
    exclusions = tuple(sorted((
        HybridExclusion(
            item.image_id, item.split, item.category,
            tuple(sorted(exclusion_reasons.get(item.image_id, set()))),
        )
        for item in candidates if item.image_id in exclusion_reasons
    ), key=lambda item: item.image_id))
    clean_categories = tuple(sorted({
        item.category for item in candidates
        if item.split == "train" and str(rows[item.image_id]["image_outcome"]) == "no_anomaly"
    }))
    positive_categories = tuple(sorted({
        item.category for item in candidates
        if str(rows[item.image_id]["image_outcome"]) == "anomaly_present"
    }))
    missing = tuple(
        (role, tuple(sorted(set(PRIORITY_POSITIVE_CATEGORIES) - {
            item.category for item in selected
            if item.role == role and item.image_outcome == "anomaly_present"
        })))
        for role in ("hybrid_fusion_fit", "hybrid_development_holdout")
    )
    return HybridDevelopmentManifest.create(
        schema_version=HYBRID_MANIFEST_SCHEMA_VERSION,
        protocol_version=HYBRID_PROTOCOL_VERSION,
        evidence_classification=EVIDENCE_CLASSIFICATION,
        source_dataset_id=SOURCE_DATASET_ID,
        source_dataset_version=SOURCE_DATASET_VERSION,
        source_registry_sha256=content_hash(registry),
        historical_result_store_sha256=content_hash(historical),
        allocation_seed=DETERMINISTIC_SEED,
        normal_fit_target_fraction=NORMAL_FIT_TARGET_FRACTION,
        selected_images=tuple(selected),
        exclusions=exclusions,
        expected_clean_categories=clean_categories,
        expected_positive_categories=positive_categories,
        priority_positive_categories=PRIORITY_POSITIVE_CATEGORIES,
        missing_priority_categories_by_role=missing,
        overlap_policy=(
            "exclude exact SHA-256 and legacy dHash<=3 candidates against pilot, historical tests, and prior verification identities",
            "exclude declared source/template/acquisition crossings against protected identities",
            "keep exact, dHash<=3, source, template, and acquisition components within one role",
            "validation is holdout-only and train identities linked to validation are excluded",
            "never decode or expose a registered test image",
        ),
        allocation_policy=(
            "deterministic category-stratified allocation of clean train identity components; "
            "one quarter per category (rounded, at least one group) to hybrid_fusion_fit and the remainder to hybrid_normal_fit; "
            "all eligible anomaly-present train components to hybrid_fusion_fit; all eligible validation components to holdout"
        ),
    )


def write_hybrid_manifest(manifest: HybridDevelopmentManifest, path: Path) -> None:
    destination = Path(path)
    if destination.exists():
        raise HybridProtocolError("Hybrid manifest is immutable and refuses overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(manifest.to_json() + "\n", encoding="utf-8")


def load_hybrid_manifest(path: Path) -> HybridDevelopmentManifest:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    payload = json.loads(text)
    if canonical_json(payload) + "\n" != text:
        raise HybridProtocolError("Hybrid manifest is not canonical")
    return HybridDevelopmentManifest(
        schema_version=str(payload["schema_version"]),
        protocol_version=str(payload["protocol_version"]),
        evidence_classification=str(payload["evidence_classification"]),
        source_dataset_id=str(payload["source_dataset_id"]),
        source_dataset_version=str(payload["source_dataset_version"]),
        source_registry_sha256=str(payload["source_registry_sha256"]),
        historical_result_store_sha256=str(payload["historical_result_store_sha256"]),
        allocation_seed=int(payload["allocation_seed"]),
        normal_fit_target_fraction=float(payload["normal_fit_target_fraction"]),
        selected_images=tuple(HybridImageIdentity(**item) for item in payload["selected_images"]),
        exclusions=tuple(HybridExclusion(
            str(item["image_id"]), str(item["source_split"]), str(item["category"]),
            tuple(str(reason) for reason in item["reasons"]),
        ) for item in payload["exclusions"]),
        expected_clean_categories=tuple(str(item) for item in payload["expected_clean_categories"]),
        expected_positive_categories=tuple(str(item) for item in payload["expected_positive_categories"]),
        priority_positive_categories=tuple(str(item) for item in payload["priority_positive_categories"]),
        missing_priority_categories_by_role=tuple(
            (str(role), tuple(str(category) for category in categories))
            for role, categories in payload["missing_priority_categories_by_role"]
        ),
        overlap_policy=tuple(str(item) for item in payload["overlap_policy"]),
        allocation_policy=str(payload["allocation_policy"]),
        manifest_hash=str(payload["manifest_hash"]),
    )


def hybrid_normal_fit_samples(
    manifest: HybridDevelopmentManifest,
    repository_root: Path,
) -> tuple[NormalFitSample, ...]:
    root = Path(repository_root)
    return tuple(NormalFitSample(
        image=root / item.image_path,
        image_id=item.image_id,
        image_sha256=item.image_sha256,
        ground_truth_sha256=item.ground_truth_sha256,
        metadata={
            "hybrid_role": item.role,
            "category": item.category,
            "source_group_id": item.source_group_id,
            "template_group_id": item.template_group_id,
        },
    ) for item in manifest.normal_fit)


class FusionFitView:
    """Capability object that deliberately exposes fusion-fit and not holdout."""

    def __init__(self, manifest: HybridDevelopmentManifest):
        self.manifest_hash = manifest.manifest_hash
        self.identities = manifest.fusion_fit


def fusion_fit_view(manifest: HybridDevelopmentManifest) -> FusionFitView:
    return FusionFitView(manifest)
