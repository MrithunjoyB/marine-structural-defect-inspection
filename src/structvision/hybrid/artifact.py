"""Immutable, content-addressed hybrid fusion artifact."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Protocol

from scientific_contract.hashing import canonical_json, is_sha256, sha256_json

from .errors import HybridFusionError
from .features import CandidateFeatureDefinition, FeatureNormalisation, FEATURE_ORDER


HYBRID_IMPLEMENTATION_ID = "structvision-proposal-guided-hybrid-v1-dev"
HYBRID_IMPLEMENTATION_VERSION = "1.0.0-dev"
HYBRID_ARTIFACT_SCHEMA_VERSION = "hybrid-fusion-artifact-v1"
DECLARED_BUDGETS = (0.25, 0.50, 1.00)
PRIMARY_BUDGET = 0.50


@dataclass(frozen=True)
class FusionSearchConfiguration:
    configuration_id: str
    classical_weight: float
    normality_weight: float
    preservation_floor: float | None

    def __post_init__(self) -> None:
        if self.classical_weight <= 0 or self.normality_weight < 0:
            raise HybridFusionError("Fusion weights must be non-negative and classical cannot be zero")
        if abs(self.classical_weight + self.normality_weight - 1.0) > 1e-12:
            raise HybridFusionError("Fusion weights must sum to one")
        if self.preservation_floor is not None and not 0.0 <= self.preservation_floor <= 1.0:
            raise HybridFusionError("Generic preservation floor must use normalised classical evidence")

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class FusionOperatingPoint:
    false_proposal_budget: float
    threshold: float
    achieved_clean_false_proposals_per_image: float
    achieved_clean_images_with_any_proposal: float
    micro_component_sensitivity: float
    macro_per_positive_image_recall: float
    image_level_sensitivity: float
    proposal_precision: float
    mean_assigned_pair_iou: float
    mean_proposals_per_image: float
    category_sensitivity: tuple[tuple[str, float], ...]
    budget_feasible: bool
    preservation_passed: bool
    preservation_failures: tuple[str, ...]

    def __post_init__(self) -> None:
        numeric = (
            self.false_proposal_budget, self.threshold, self.achieved_clean_false_proposals_per_image,
            self.achieved_clean_images_with_any_proposal, self.micro_component_sensitivity,
            self.macro_per_positive_image_recall, self.image_level_sensitivity,
            self.proposal_precision, self.mean_assigned_pair_iou, self.mean_proposals_per_image,
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise HybridFusionError("Operating-point metrics must be finite")
        expected_budget_status = self.achieved_clean_false_proposals_per_image <= self.false_proposal_budget + 1e-12
        if self.budget_feasible != expected_budget_status:
            raise HybridFusionError("Stored clean-FP budget status is inconsistent")
        if self.preservation_passed != (not self.preservation_failures):
            raise HybridFusionError("Preservation status and reasons disagree")

    def to_dict(self) -> dict[str, object]:
        payload = dict(self.__dict__)
        payload["category_sensitivity"] = [list(item) for item in self.category_sensitivity]
        payload["preservation_failures"] = list(self.preservation_failures)
        return payload


@dataclass(frozen=True)
class EvaluatedFusionConfiguration:
    search: FusionSearchConfiguration
    operating_points: tuple[FusionOperatingPoint, ...]

    def __post_init__(self) -> None:
        if tuple(item.false_proposal_budget for item in self.operating_points) != DECLARED_BUDGETS:
            raise HybridFusionError("Every search configuration must preserve all declared budgets")

    def operating_point(self, budget: float) -> FusionOperatingPoint:
        return next(item for item in self.operating_points if item.false_proposal_budget == budget)

    def to_dict(self) -> dict[str, object]:
        return {
            "search": self.search.to_dict(),
            "operating_points": [item.to_dict() for item in self.operating_points],
        }


@dataclass(frozen=True)
class HybridFusionArtifact:
    schema_version: str
    implementation_identity: str
    implementation_version: str
    hybrid_protocol_hash: str
    normal_feature_model_artifact_hash: str
    frozen_classical_configuration_hash: str
    candidate_feature_definitions: tuple[CandidateFeatureDefinition, ...]
    feature_order_identity: tuple[str, ...]
    high_anomaly_reference: float
    normalisation_statistics: tuple[FeatureNormalisation, ...]
    coefficient_search_space: tuple[FusionSearchConfiguration, ...]
    evaluated_configurations: tuple[EvaluatedFusionConfiguration, ...]
    preservation_constraints: tuple[tuple[str, float], ...]
    selection_status: str
    selected_configuration_id: str | None
    selected_coefficients: tuple[float, float] | None
    selected_preservation_floor: float | None
    selected_operating_points: tuple[FusionOperatingPoint, ...]
    selected_operating_threshold: float | None
    false_proposal_budget: float
    fusion_fit_image_hashes: tuple[tuple[str, str], ...]
    fusion_fit_truth_hashes: tuple[tuple[str, str], ...]
    evaluation_policy_hash: str
    environment_lock_hash: str
    code_commit: str
    git_dirty_state: str
    git_diff_hash: str | None
    deterministic_seed: int
    creation_timestamp: str
    artifact_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != HYBRID_ARTIFACT_SCHEMA_VERSION:
            raise HybridFusionError("Unsupported fusion-artifact schema")
        if (self.implementation_identity, self.implementation_version) != (
            HYBRID_IMPLEMENTATION_ID, HYBRID_IMPLEMENTATION_VERSION,
        ):
            raise HybridFusionError("Hybrid implementation identity differs")
        required_hashes = (
            self.hybrid_protocol_hash, self.normal_feature_model_artifact_hash,
            self.frozen_classical_configuration_hash, self.evaluation_policy_hash,
            self.environment_lock_hash, self.artifact_hash,
        )
        if any(not is_sha256(value) for value in required_hashes):
            raise HybridFusionError("Fusion provenance requires SHA-256 identities")
        if self.feature_order_identity != FEATURE_ORDER:
            raise HybridFusionError("Fusion feature-order identity differs")
        if tuple(item.name for item in self.candidate_feature_definitions) != FEATURE_ORDER:
            raise HybridFusionError("Feature definitions are incomplete or reordered")
        if tuple(item.name for item in self.normalisation_statistics) != FEATURE_ORDER:
            raise HybridFusionError("Normalisation statistics are incomplete or reordered")
        search_ids = tuple(item.configuration_id for item in self.coefficient_search_space)
        evaluated_ids = tuple(item.search.configuration_id for item in self.evaluated_configurations)
        if len(search_ids) != len(set(search_ids)) or search_ids != evaluated_ids:
            raise HybridFusionError("Coefficient enumeration is incomplete or duplicated")
        if self.false_proposal_budget != PRIMARY_BUDGET:
            raise HybridFusionError("Primary clean-FP budget differs from the predeclaration")
        if self.selection_status not in {"selected", "failed_no_preserving_configuration"}:
            raise HybridFusionError("Unknown fusion selection status")
        if self.selection_status == "selected":
            if self.selected_configuration_id not in search_ids or self.selected_coefficients is None:
                raise HybridFusionError("Selected fusion configuration is absent")
            if tuple(item.false_proposal_budget for item in self.selected_operating_points) != DECLARED_BUDGETS:
                raise HybridFusionError("Selected artifact must freeze all declared budgets")
            primary = next(item for item in self.selected_operating_points if item.false_proposal_budget == PRIMARY_BUDGET)
            if not primary.budget_feasible or not primary.preservation_passed or self.selected_operating_threshold != primary.threshold:
                raise HybridFusionError("Selected primary point is not preservation-compliant")
        else:
            if any(value is not None for value in (
                self.selected_configuration_id, self.selected_coefficients,
                self.selected_preservation_floor, self.selected_operating_threshold,
            )) or self.selected_operating_points:
                raise HybridFusionError("Failed selection cannot contain a hidden selected model")
        image_ids = [item[0] for item in self.fusion_fit_image_hashes]
        truth_ids = [item[0] for item in self.fusion_fit_truth_hashes]
        if image_ids != truth_ids or len(image_ids) != len(set(image_ids)):
            raise HybridFusionError("Fusion-fit image/truth identities are incomplete")
        if any(not is_sha256(value) for _, value in self.fusion_fit_image_hashes + self.fusion_fit_truth_hashes):
            raise HybridFusionError("Fusion-fit hashes are invalid")
        if self.git_dirty_state not in {"clean", "dirty"} or (self.git_dirty_state == "dirty") != bool(self.git_diff_hash):
            raise HybridFusionError("Git dirty-state provenance is inconsistent")
        if self.artifact_hash != "0" * 64 and self.artifact_hash != sha256_json(self.to_dict(include_artifact_hash=False)):
            raise HybridFusionError("Hybrid fusion artifact hash mismatch")

    @classmethod
    def create(cls, **values: object) -> "HybridFusionArtifact":
        initial = cls(artifact_hash="0" * 64, **values)
        return replace(initial, artifact_hash=sha256_json(initial.to_dict(include_artifact_hash=False)))

    def operating_point(self, budget: float = PRIMARY_BUDGET) -> FusionOperatingPoint:
        if self.selection_status != "selected":
            raise HybridFusionError("No fusion operating point was selected")
        try:
            return next(item for item in self.selected_operating_points if item.false_proposal_budget == budget)
        except StopIteration as error:
            raise KeyError(budget) from error

    def to_dict(self, *, include_artifact_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "implementation_identity": self.implementation_identity,
            "implementation_version": self.implementation_version,
            "hybrid_protocol_hash": self.hybrid_protocol_hash,
            "normal_feature_model_artifact_hash": self.normal_feature_model_artifact_hash,
            "frozen_classical_configuration_hash": self.frozen_classical_configuration_hash,
            "candidate_feature_definitions": [item.to_dict() for item in self.candidate_feature_definitions],
            "feature_order_identity": list(self.feature_order_identity),
            "high_anomaly_reference": self.high_anomaly_reference,
            "normalisation_statistics": [item.to_dict() for item in self.normalisation_statistics],
            "coefficient_search_space": [item.to_dict() for item in self.coefficient_search_space],
            "evaluated_configurations": [item.to_dict() for item in self.evaluated_configurations],
            "preservation_constraints": [list(item) for item in self.preservation_constraints],
            "selection_status": self.selection_status,
            "selected_configuration_id": self.selected_configuration_id,
            "selected_coefficients": list(self.selected_coefficients) if self.selected_coefficients is not None else None,
            "selected_preservation_floor": self.selected_preservation_floor,
            "selected_operating_points": [item.to_dict() for item in self.selected_operating_points],
            "selected_operating_threshold": self.selected_operating_threshold,
            "false_proposal_budget": self.false_proposal_budget,
            "fusion_fit_image_hashes": [list(item) for item in self.fusion_fit_image_hashes],
            "fusion_fit_truth_hashes": [list(item) for item in self.fusion_fit_truth_hashes],
            "evaluation_policy_hash": self.evaluation_policy_hash,
            "environment_lock_hash": self.environment_lock_hash,
            "code_commit": self.code_commit,
            "git_dirty_state": self.git_dirty_state,
            "git_diff_hash": self.git_diff_hash,
            "deterministic_seed": self.deterministic_seed,
            "creation_timestamp": self.creation_timestamp,
        }
        if include_artifact_hash:
            payload["artifact_hash"] = self.artifact_hash
        return payload

    def to_json(self) -> str:
        return canonical_json(self.to_dict())


class HybridFusionArtifactSink(Protocol):
    def write(self, artifact: HybridFusionArtifact) -> None:
        ...


class DirectoryHybridFusionArtifactSink:
    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def write(self, artifact: HybridFusionArtifact) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{artifact.artifact_hash}.json"
        if path.exists():
            raise HybridFusionError("Fusion artifact is immutable and refuses overwrite")
        path.write_text(artifact.to_json() + "\n", encoding="utf-8")


def _operating_point(payload: dict[str, object]) -> FusionOperatingPoint:
    return FusionOperatingPoint(
        false_proposal_budget=float(payload["false_proposal_budget"]),
        threshold=float(payload["threshold"]),
        achieved_clean_false_proposals_per_image=float(payload["achieved_clean_false_proposals_per_image"]),
        achieved_clean_images_with_any_proposal=float(payload["achieved_clean_images_with_any_proposal"]),
        micro_component_sensitivity=float(payload["micro_component_sensitivity"]),
        macro_per_positive_image_recall=float(payload["macro_per_positive_image_recall"]),
        image_level_sensitivity=float(payload["image_level_sensitivity"]),
        proposal_precision=float(payload["proposal_precision"]),
        mean_assigned_pair_iou=float(payload["mean_assigned_pair_iou"]),
        mean_proposals_per_image=float(payload["mean_proposals_per_image"]),
        category_sensitivity=tuple((str(a), float(b)) for a, b in payload["category_sensitivity"]),
        budget_feasible=bool(payload["budget_feasible"]),
        preservation_passed=bool(payload["preservation_passed"]),
        preservation_failures=tuple(str(item) for item in payload["preservation_failures"]),
    )


def load_hybrid_fusion_artifact(path: Path) -> HybridFusionArtifact:
    text = Path(path).read_text(encoding="utf-8")
    payload = json.loads(text)
    if canonical_json(payload) + "\n" != text:
        raise HybridFusionError("Fusion artifact is not canonical")
    search = tuple(FusionSearchConfiguration(**item) for item in payload["coefficient_search_space"])
    evaluated = tuple(EvaluatedFusionConfiguration(
        search=FusionSearchConfiguration(**item["search"]),
        operating_points=tuple(_operating_point(point) for point in item["operating_points"]),
    ) for item in payload["evaluated_configurations"])
    return HybridFusionArtifact(
        schema_version=str(payload["schema_version"]),
        implementation_identity=str(payload["implementation_identity"]),
        implementation_version=str(payload["implementation_version"]),
        hybrid_protocol_hash=str(payload["hybrid_protocol_hash"]),
        normal_feature_model_artifact_hash=str(payload["normal_feature_model_artifact_hash"]),
        frozen_classical_configuration_hash=str(payload["frozen_classical_configuration_hash"]),
        candidate_feature_definitions=tuple(CandidateFeatureDefinition(**item) for item in payload["candidate_feature_definitions"]),
        feature_order_identity=tuple(str(item) for item in payload["feature_order_identity"]),
        high_anomaly_reference=float(payload["high_anomaly_reference"]),
        normalisation_statistics=tuple(FeatureNormalisation(**item) for item in payload["normalisation_statistics"]),
        coefficient_search_space=search,
        evaluated_configurations=evaluated,
        preservation_constraints=tuple((str(a), float(b)) for a, b in payload["preservation_constraints"]),
        selection_status=str(payload["selection_status"]),
        selected_configuration_id=payload["selected_configuration_id"],
        selected_coefficients=None if payload["selected_coefficients"] is None else tuple(float(item) for item in payload["selected_coefficients"]),
        selected_preservation_floor=None if payload["selected_preservation_floor"] is None else float(payload["selected_preservation_floor"]),
        selected_operating_points=tuple(_operating_point(item) for item in payload["selected_operating_points"]),
        selected_operating_threshold=None if payload["selected_operating_threshold"] is None else float(payload["selected_operating_threshold"]),
        false_proposal_budget=float(payload["false_proposal_budget"]),
        fusion_fit_image_hashes=tuple((str(a), str(b)) for a, b in payload["fusion_fit_image_hashes"]),
        fusion_fit_truth_hashes=tuple((str(a), str(b)) for a, b in payload["fusion_fit_truth_hashes"]),
        evaluation_policy_hash=str(payload["evaluation_policy_hash"]),
        environment_lock_hash=str(payload["environment_lock_hash"]),
        code_commit=str(payload["code_commit"]),
        git_dirty_state=str(payload["git_dirty_state"]),
        git_diff_hash=payload["git_diff_hash"],
        deterministic_seed=int(payload["deterministic_seed"]),
        creation_timestamp=str(payload["creation_timestamp"]),
        artifact_hash=str(payload["artifact_hash"]),
    )
