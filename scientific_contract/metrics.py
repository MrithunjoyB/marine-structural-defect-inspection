"""Metric aggregation with explicit v2 denominators and null semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import statistics

from .evaluation_policy import EvaluationPolicyV2, default_evaluation_policy
from .matching import GroundTruthRecord, MatchingResult, ProposalSet, match_one_to_one


@dataclass(frozen=True)
class ImageEvaluationV2:
    image_id: str
    method_id: str
    category: str
    acquisition_group_id: str
    ground_truth_status: str
    ranking_eligible: bool
    proposal_count: int
    truth_count: int
    threshold_results: tuple[tuple[float, MatchingResult], ...]
    top_k_matched_truths: tuple[tuple[int, int | None], ...]
    processing_time_seconds: float | None = None
    hardware_context: str | None = None
    cache_state: str | None = None

    def result_at(self, threshold: float) -> MatchingResult:
        for value, result in self.threshold_results:
            if value == threshold:
                return result
        raise KeyError(threshold)


def evaluate_image(
    proposal_set: ProposalSet,
    ground_truth: GroundTruthRecord,
    *,
    category: str = "",
    acquisition_group_id: str = "",
    processing_time_seconds: float | None = None,
    hardware_context: str | None = None,
    cache_state: str | None = None,
    policy: EvaluationPolicyV2 | None = None,
) -> ImageEvaluationV2:
    policy = policy or default_evaluation_policy()
    threshold_results = tuple(
        (threshold, match_one_to_one(proposal_set, ground_truth, threshold, policy))
        for threshold in policy.threshold_analyses
    )
    top_k: list[tuple[int, int | None]] = []
    for value in policy.top_k_values:
        if not proposal_set.ranking_eligible:
            top_k.append((value, None))
            continue
        subset = tuple(proposal for proposal in proposal_set.proposals if proposal.rank is not None and proposal.rank <= value)
        ranked = ProposalSet(proposal_set.method_id, subset, True, proposal_set.ranking_definition)
        top_k.append((value, match_one_to_one(ranked, ground_truth, policy.primary_match_threshold, policy).matched_truth_count))
    return ImageEvaluationV2(
        ground_truth.image_id, proposal_set.method_id, category,
        acquisition_group_id, ground_truth.ground_truth_status,
        proposal_set.ranking_eligible, len(proposal_set.proposals),
        len(ground_truth.truth_instances), threshold_results, tuple(top_k),
        processing_time_seconds, hardware_context, cache_state,
    )


@dataclass(frozen=True)
class AggregateMetricsV2:
    method_id: str
    image_count: int
    positive_image_count: int
    clean_image_count: int
    total_truth_instances: int
    total_proposals: int
    matched_truth_instances: int
    matched_proposals: int
    micro_component_sensitivity: float | None
    macro_per_positive_image_recall: float | None
    image_level_detection_sensitivity: float | None
    proposal_precision: float | None
    clean_false_proposals_per_image: float | None
    clean_images_with_any_proposal: float | None
    assigned_pair_ious: tuple[float, ...]
    assigned_pair_dice: tuple[float, ...]
    assigned_pair_iou_mean: float | None
    assigned_pair_dice_mean: float | None
    sensitivity_by_iou_threshold: tuple[tuple[float, float | None], ...]
    top_k_component_sensitivity: tuple[tuple[int, float | None], ...]
    category_component_sensitivity: tuple[tuple[str, float | None], ...]
    nuisance_false_proposals_per_image: tuple[tuple[str, float | None], ...]
    acquisition_group_component_sensitivity: tuple[tuple[str, float | None], ...]
    mean_processing_time_seconds: float | None
    mean_proposals_per_image: float | None
    efficiency_limitation: str
    primary_endpoint_selector: None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _stratified_sensitivity(
    images: tuple[ImageEvaluationV2, ...],
    field: str,
    policy: EvaluationPolicyV2,
) -> tuple[tuple[str, float | None], ...]:
    values = sorted({str(getattr(image, field)) for image in images if str(getattr(image, field))})
    rows = []
    for value in values:
        subset = [image for image in images if str(getattr(image, field)) == value and image.truth_count]
        total = sum(image.truth_count for image in subset)
        matched = sum(image.result_at(policy.primary_match_threshold).matched_truth_count for image in subset)
        rows.append((value, _ratio(matched, total)))
    return tuple(rows)


def aggregate_metrics(
    evaluations: list[ImageEvaluationV2] | tuple[ImageEvaluationV2, ...],
    policy: EvaluationPolicyV2 | None = None,
) -> AggregateMetricsV2:
    policy = policy or default_evaluation_policy()
    images = tuple(evaluations)
    if not images:
        raise ValueError("At least one image evaluation is required")
    methods = {image.method_id for image in images}
    if len(methods) != 1:
        raise ValueError("Aggregate one method at a time")
    eligibility = {image.ranking_eligible for image in images}
    if len(eligibility) != 1:
        raise ValueError("A method cannot mix ranked and unordered outputs")
    primary = [image.result_at(policy.primary_match_threshold) for image in images]
    positives = [image for image in images if image.ground_truth_status == "anomaly_present"]
    clean = [image for image in images if image.ground_truth_status == "no_anomaly"]
    truth_total = sum(image.truth_count for image in positives)
    matched_truth = sum(result.matched_truth_count for result in primary)
    proposal_total = sum(image.proposal_count for image in images)
    matched_proposals = sum(result.matched_proposal_count for result in primary)
    per_positive = [image.result_at(policy.primary_match_threshold).matched_truth_count / image.truth_count for image in positives]
    detected_images = sum(image.result_at(policy.primary_match_threshold).matched_truth_count > 0 for image in positives)
    clean_false = sum(image.proposal_count for image in clean)
    clean_any = sum(image.proposal_count > 0 for image in clean)
    matched_decisions = [decision for result in primary for decision in result.proposal_decisions if decision.matched]
    ious = tuple(decision.mask_iou for decision in matched_decisions)
    dice = tuple(decision.mask_dice for decision in matched_decisions)
    threshold_sensitivity = tuple(
        (threshold, _ratio(sum(image.result_at(threshold).matched_truth_count for image in positives), truth_total))
        for threshold in policy.threshold_analyses
    )
    ranked = next(iter(eligibility))
    top_k = []
    for value in policy.top_k_values:
        if not ranked:
            top_k.append((value, None))
        else:
            matched = sum(dict(image.top_k_matched_truths)[value] or 0 for image in positives)
            top_k.append((value, _ratio(matched, truth_total)))
    nuisance = []
    for category in sorted({image.category for image in clean if image.category}):
        subset = [image for image in clean if image.category == category]
        nuisance.append((category, _ratio(sum(image.proposal_count for image in subset), len(subset))))
    times = [float(image.processing_time_seconds) for image in images if image.processing_time_seconds is not None]
    return AggregateMetricsV2(
        next(iter(methods)), len(images), len(positives), len(clean), truth_total,
        proposal_total, matched_truth, matched_proposals,
        _ratio(matched_truth, truth_total), _mean(per_positive),
        _ratio(detected_images, len(positives)), _ratio(matched_proposals, proposal_total),
        _ratio(clean_false, len(clean)), _ratio(clean_any, len(clean)),
        ious, dice, _mean(list(ious)), _mean(list(dice)), threshold_sensitivity,
        tuple(top_k), _stratified_sensitivity(images, "category", policy), tuple(nuisance),
        _stratified_sensitivity(images, "acquisition_group_id", policy),
        _mean(times), _ratio(proposal_total, len(images)),
        "Timing comparisons require identical recorded hardware and cache state; this aggregator does not infer equivalence.",
    )
