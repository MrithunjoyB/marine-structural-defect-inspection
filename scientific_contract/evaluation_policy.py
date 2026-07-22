"""Typed definition of the prospective StructVision evaluation policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .hashing import sha256_json


HISTORICAL_EVALUATION_POLICY_ID = "structvision-eval-v1-historical"
EVALUATION_POLICY_ID = "structvision-eval-v2"


@dataclass(frozen=True)
class EvaluationPolicyV2:
    policy_id: str = EVALUATION_POLICY_ID
    policy_version: int = 2
    annotation_semantics: str = (
        "anomaly_present requires immutable, non-empty truth instances; "
        "no_anomaly requires zero truth instances"
    )
    primary_match_metric: str = "mask_iou"
    primary_match_threshold: float = 0.25
    threshold_analyses: tuple[float, ...] = (0.10, 0.25, 0.50)
    assignment_algorithm: str = (
        "deterministic maximum-cardinality then maximum-IoU bipartite assignment v1"
    )
    assignment_tie_break: str = "proposal_id then truth_id lexical order"
    centroid_semantics: str = "diagnostic_only_never_primary"
    proposal_ordering: str = (
        "Top-K requires finite numeric scores, unique contiguous ranks, a declared "
        "ranking definition, and score-descending/proposal-id tie ordering"
    )
    top_k_values: tuple[int, ...] = (1, 3, 5, 8)
    undefined_value: None = None
    clean_image_semantics: str = (
        "no_anomaly has zero truth instances; recall is undefined; all proposals "
        "are false proposals"
    )
    aggregation_modes: tuple[str, ...] = (
        "micro_component_sensitivity",
        "macro_per_positive_image_recall",
        "image_level_detection_sensitivity",
    )
    statistical_grouping: str = (
        "paired by immutable image_id; confidence intervals resample acquisition "
        "groups and category analyses remain stratified"
    )
    metric_definitions: tuple[tuple[str, str], ...] = (
        ("micro_component_sensitivity", "matched_truth_instances/total_truth_instances"),
        ("macro_per_positive_image_recall", "mean per-positive-image component recall"),
        ("image_level_detection_sensitivity", "positive images with >=1 match/positive images"),
        ("proposal_precision", "matched proposals/total proposals"),
        ("clean_false_proposals_per_image", "false proposals/clean images"),
        ("clean_images_with_any_proposal", "clean images with >=1 proposal/clean images"),
        ("localisation", "assigned-pair mask IoU and Dice distributions"),
        ("ranked_sensitivity", "matched truth instances within Top-K/total truth instances"),
        ("efficiency", "processing time and proposal burden with hardware/cache metadata"),
    )
    primary_endpoint: str = (
        "component sensitivity at a predeclared clean-image false-proposal budget"
    )
    co_primary_operational_endpoints: tuple[str, ...] = (
        "clean_false_proposals_per_image",
        "clean_images_with_any_proposal",
    )
    preservation_endpoints: tuple[str, ...] = (
        "critical_category_sensitivity",
        "thin_local_anomaly_sensitivity",
        "predeclared_category_non_inferiority",
    )

    def __post_init__(self) -> None:
        if self.policy_id != EVALUATION_POLICY_ID or self.policy_version != 2:
            raise ValueError("EvaluationPolicyV2 identity is fixed")
        if not 0.0 <= self.primary_match_threshold <= 1.0:
            raise ValueError("Primary matching threshold must be in [0, 1]")
        if tuple(sorted(set(self.threshold_analyses))) != self.threshold_analyses:
            raise ValueError("Threshold analyses must be unique and increasing")
        if self.primary_match_threshold not in self.threshold_analyses:
            raise ValueError("Primary threshold must be one of the named analyses")
        if self.top_k_values != (1, 3, 5, 8):
            raise ValueError("The v2 Top-K contract is fixed at 1, 3, 5, and 8")

    @property
    def configuration_hash(self) -> str:
        return sha256_json(asdict(self))

    @property
    def matching_policy_hash(self) -> str:
        return sha256_json({
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "metric": self.primary_match_metric,
            "thresholds": self.threshold_analyses,
            "algorithm": self.assignment_algorithm,
            "tie_break": self.assignment_tie_break,
            "centroid_semantics": self.centroid_semantics,
        })

    @property
    def metric_definitions_hash(self) -> str:
        return sha256_json(self.metric_definitions)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["configuration_hash"] = self.configuration_hash
        payload["matching_policy_hash"] = self.matching_policy_hash
        payload["metric_definitions_hash"] = self.metric_definitions_hash
        return payload


def default_evaluation_policy() -> EvaluationPolicyV2:
    return EvaluationPolicyV2()
