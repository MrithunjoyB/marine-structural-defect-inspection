"""Fusion-fit-only enumeration and preservation-constrained selection."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import math
from pathlib import Path
import time

import cv2
import numpy as np

from scientific_contract.evaluation_policy import default_evaluation_policy
from scientific_contract.matching import GroundTruthRecord, _maximum_weight_assignment
from scientific_contract.metrics import AggregateMetricsV2, aggregate_metrics, evaluate_image
from structvision.api import StructuralAnomalyDetector
from structvision.configuration import DetectorConfig
from structvision.executor import ExperimentSample, _ground_truth_record, _proposal_set
from structvision.inputs import content_hash
from structvision.normal_feature.model_artifact import NormalFeatureModelArtifact
from structvision.normal_feature.patchcore import NormalFeatureAnomalyDetector, _git_metadata
from structvision.normal_feature.types import NormalFeatureScoreResult
from structvision.types import AnalysisResult, Proposal

from .artifact import (
    DECLARED_BUDGETS,
    HYBRID_ARTIFACT_SCHEMA_VERSION,
    HYBRID_IMPLEMENTATION_ID,
    HYBRID_IMPLEMENTATION_VERSION,
    PRIMARY_BUDGET,
    EvaluatedFusionConfiguration,
    FusionOperatingPoint,
    FusionSearchConfiguration,
    HybridFusionArtifact,
    HybridFusionArtifactSink,
)
from .errors import HybridFusionError, HybridProtocolError
from .features import (
    FEATURE_DEFINITIONS,
    FEATURE_ORDER,
    CandidateEvidence,
    FeatureNormalisation,
    candidate_evidence,
    clean_map_reference,
    fit_normalisation,
    normalised_components,
)
from .protocol import DETERMINISTIC_SEED, FusionFitView, HybridImageIdentity


PRESERVATION_CONSTRAINTS = (
    ("overall_micro_max_decrease", 0.02),
    ("thin_crack_max_decrease", 0.00),
    ("pitting_cluster_max_decrease", 0.00),
    ("weld_disturbance_max_decrease", 0.00),
    ("image_level_max_decrease", 0.00),
    ("mean_assigned_pair_iou_max_decrease", 0.02),
)


def coefficient_search_space() -> tuple[FusionSearchConfiguration, ...]:
    """Small, fixed grid; order is part of deterministic tie-breaking."""
    configurations = []
    for classical_weight in (0.90, 0.80, 0.70, 0.60, 0.50):
        for floor in (None, 0.90, 0.80):
            normality_weight = round(1.0 - classical_weight, 10)
            floor_id = "none" if floor is None else f"{floor:.2f}"
            configurations.append(FusionSearchConfiguration(
                configuration_id=f"cw-{classical_weight:.2f}_nw-{normality_weight:.2f}_floor-{floor_id}",
                classical_weight=classical_weight,
                normality_weight=normality_weight,
                preservation_floor=floor,
            ))
    return tuple(configurations)


@dataclass(frozen=True)
class PreparedCandidate:
    proposal: Proposal
    evidence: CandidateEvidence
    classical_component: float
    normality_component: float
    normalised_features: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class PreparedFusionImage:
    identity: HybridImageIdentity
    classical_result: AnalysisResult
    normal_score: NormalFeatureScoreResult
    ground_truth: GroundTruthRecord
    candidates: tuple[PreparedCandidate, ...]
    truth_iou_by_proposal: tuple[tuple[str, tuple[float, ...]], ...]
    elapsed_seconds: float


@dataclass(frozen=True)
class FastMetrics:
    clean_false_proposals_per_image: float
    clean_images_with_any_proposal: float
    micro_component_sensitivity: float
    macro_per_positive_image_recall: float
    image_level_sensitivity: float
    proposal_precision: float
    mean_assigned_pair_iou: float
    mean_proposals_per_image: float
    category_sensitivity: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class FusionFitOutcome:
    artifact: HybridFusionArtifact
    baseline_metrics: AggregateMetricsV2
    normalisation_statistics: tuple[FeatureNormalisation, ...]
    prepared_images: tuple[PreparedFusionImage, ...]


def _truth_object(identity: HybridImageIdentity, root: Path, shape: tuple[int, int]) -> object:
    if identity.ground_truth_kind == "implicit_verified_zero_mask":
        truth = np.zeros(shape, dtype=np.uint8)
        if content_hash(truth) != identity.ground_truth_sha256:
            raise HybridProtocolError(f"Clean truth identity mismatch: {identity.image_id}")
        return truth
    path = root / str(identity.ground_truth_path)
    if content_hash(path) != identity.ground_truth_sha256:
        raise HybridProtocolError(f"Ground-truth identity mismatch: {identity.image_id}")
    return path


def _ground_truth(
    identity: HybridImageIdentity,
    root: Path,
    image_shape: tuple[int, int],
) -> GroundTruthRecord:
    sample = ExperimentSample(
        identity.image_id,
        root / identity.image_path,
        _truth_object(identity, root, image_shape),
        identity.image_outcome,
        category=identity.category,
        acquisition_group_id=identity.acquisition_group_id,
        metadata={"hybrid_role": identity.role},
    )
    return _ground_truth_record(sample, image_shape)


def _truth_ious(proposals: tuple[Proposal, ...], ground_truth: GroundTruthRecord) -> tuple[tuple[str, tuple[float, ...]], ...]:
    truths = tuple(item.mask.to_array().astype(bool) for item in sorted(ground_truth.truth_instances, key=lambda item: item.truth_id))
    rows = []
    for proposal in proposals:
        left = proposal.final_mask > 0
        left_area = int(np.count_nonzero(left))
        values = []
        for right in truths:
            intersection = int(np.count_nonzero(left & right))
            union = left_area + int(np.count_nonzero(right)) - intersection
            values.append(intersection / union if union else 0.0)
        rows.append((proposal.proposal_id, tuple(values)))
    return tuple(rows)


def _prepare_raw(
    view: FusionFitView,
    repository_root: Path,
    classical_detector: StructuralAnomalyDetector,
    normal_detector: NormalFeatureAnomalyDetector,
    model_artifact: NormalFeatureModelArtifact,
) -> tuple[tuple[HybridImageIdentity, AnalysisResult, NormalFeatureScoreResult, GroundTruthRecord, float], ...]:
    if not view.identities or {item.role for item in view.identities} != {"hybrid_fusion_fit"}:
        raise HybridProtocolError("Fusion fitting accepts only the capability-limited fusion-fit view")
    if model_artifact.normal_fit_manifest_hash != view.manifest_hash:
        raise HybridFusionError("Hybrid normal model was not fitted from this protocol manifest")
    root = Path(repository_root).resolve()
    rows = []
    for identity in view.identities:
        image = root / identity.image_path
        if content_hash(image) != identity.image_sha256:
            raise HybridProtocolError(f"Fusion-fit image identity mismatch: {identity.image_id}")
        started = time.perf_counter()
        classical = classical_detector.analyse(
            image, image_id=identity.image_id,
            metadata={"hybrid_role": identity.role, "source_group_id": identity.source_group_id},
        )
        if not classical.provenance.protected_hashes_verified:
            raise HybridFusionError("Frozen classical source identity differs")
        normal = normal_detector.score(image, model_artifact=model_artifact, image_id=identity.image_id)
        if classical.input_hash != normal.input_hash or classical.image_shape != normal.image_shape:
            raise HybridFusionError("Classical and PatchCore analysed coordinate systems differ")
        truth = _ground_truth(identity, root, classical.image_shape[:2])
        rows.append((identity, classical, normal, truth, time.perf_counter() - started))
    return tuple(rows)


def prepare_fusion_fit(
    view: FusionFitView,
    *,
    repository_root: Path,
    classical_detector: StructuralAnomalyDetector,
    normal_detector: NormalFeatureAnomalyDetector,
    model_artifact: NormalFeatureModelArtifact,
) -> tuple[float, tuple[FeatureNormalisation, ...], tuple[PreparedFusionImage, ...]]:
    raw = _prepare_raw(view, repository_root, classical_detector, normal_detector, model_artifact)
    reference = clean_map_reference(tuple(
        normal.anomaly_map for identity, _, normal, _, _ in raw
        if identity.image_outcome == "no_anomaly"
    ))
    extracted: list[tuple[HybridImageIdentity, AnalysisResult, NormalFeatureScoreResult, GroundTruthRecord, float, tuple[CandidateEvidence, ...]]] = []
    all_evidence: list[CandidateEvidence] = []
    for identity, classical, normal, truth, elapsed in raw:
        evidence = tuple(candidate_evidence(
            proposal, normal.anomaly_map, high_anomaly_reference=reference,
        ) for proposal in classical.proposals)
        if tuple(item.proposal_id for item in evidence) != tuple(item.proposal_id for item in classical.proposals):
            raise HybridFusionError("Candidate evidence does not preserve the complete classical candidate order")
        extracted.append((identity, classical, normal, truth, elapsed, evidence))
        all_evidence.extend(evidence)
    normalisation = fit_normalisation(tuple(all_evidence))
    prepared = []
    for identity, classical, normal, truth, elapsed, evidence_rows in extracted:
        candidates = []
        for proposal, evidence in zip(classical.proposals, evidence_rows):
            classical_component, normality_component, scaled = normalised_components(evidence, normalisation)
            candidates.append(PreparedCandidate(proposal, evidence, classical_component, normality_component, scaled))
        prepared.append(PreparedFusionImage(
            identity, classical, normal, truth, tuple(candidates),
            _truth_ious(classical.proposals, truth), elapsed,
        ))
    return reference, normalisation, tuple(prepared)


def _hybrid_score(candidate: PreparedCandidate, search: FusionSearchConfiguration) -> float:
    return (
        search.classical_weight * candidate.classical_component
        + search.normality_weight * candidate.normality_component
    )


def _preserved(candidate: PreparedCandidate, search: FusionSearchConfiguration) -> bool:
    return search.preservation_floor is not None and candidate.classical_component >= search.preservation_floor


def _threshold(
    images: tuple[PreparedFusionImage, ...],
    search: FusionSearchConfiguration,
    budget: float,
) -> float:
    clean = tuple(item for item in images if item.identity.image_outcome == "no_anomaly")
    allowed = math.floor(budget * len(clean) + 1e-12)
    fixed = sum(_preserved(candidate, search) for image in clean for candidate in image.candidates)
    scores = sorted({
        _hybrid_score(candidate, search)
        for image in clean for candidate in image.candidates
        if not _preserved(candidate, search)
    }, reverse=True)
    all_scores = [_hybrid_score(candidate, search) for image in images for candidate in image.candidates]
    maximum = max(all_scores, default=0.0)
    selected_threshold = float(np.nextafter(np.float64(maximum), np.float64(np.inf)))
    if fixed > allowed:
        return selected_threshold
    for score in scores:
        count = fixed + sum(
            (not _preserved(candidate, search)) and _hybrid_score(candidate, search) >= score
            for image in clean for candidate in image.candidates
        )
        if count <= allowed:
            selected_threshold = float(score)
        else:
            break
    return selected_threshold


def _selected(
    image: PreparedFusionImage,
    search: FusionSearchConfiguration,
    threshold: float,
) -> tuple[tuple[PreparedCandidate, float], ...]:
    rows = tuple(
        (candidate, _hybrid_score(candidate, search))
        for candidate in image.candidates
        if _preserved(candidate, search) or _hybrid_score(candidate, search) >= threshold
    )
    return tuple(sorted(rows, key=lambda item: (-item[1], item[0].proposal.proposal_id)))


def _assigned(image: PreparedFusionImage, selected: tuple[tuple[PreparedCandidate, float], ...]) -> tuple[int, tuple[float, ...]]:
    truth_count = len(image.ground_truth.truth_instances)
    if not selected or not truth_count:
        return 0, ()
    lookup = dict(image.truth_iou_by_proposal)
    size = max(len(selected), truth_count)
    weights = np.zeros((size, size), dtype=np.float64)
    cardinality_weight = float(size + 1)
    for row, (candidate, _) in enumerate(selected):
        for column, iou in enumerate(lookup[candidate.proposal.proposal_id]):
            if iou + 1e-12 >= 0.25:
                weights[row, column] = cardinality_weight + iou
    assignment = _maximum_weight_assignment(weights)
    ious = tuple(
        float(weights[row, column] - cardinality_weight)
        for row, column in assignment.items()
        if row < len(selected) and column < truth_count and weights[row, column] > 0
    )
    return len(ious), ious


def _fast_metrics(
    images: tuple[PreparedFusionImage, ...],
    search: FusionSearchConfiguration,
    threshold: float,
) -> FastMetrics:
    clean = [item for item in images if item.identity.image_outcome == "no_anomaly"]
    positive = [item for item in images if item.identity.image_outcome == "anomaly_present"]
    proposal_total = 0
    clean_proposals = 0
    clean_any = 0
    truth_total = 0
    matched_total = 0
    detected = 0
    positive_recalls = []
    assigned_ious: list[float] = []
    categories: dict[str, list[int]] = {}
    for image in images:
        chosen = _selected(image, search, threshold)
        proposal_total += len(chosen)
        if image.identity.image_outcome == "no_anomaly":
            clean_proposals += len(chosen)
            clean_any += bool(chosen)
            continue
        truth_count = len(image.ground_truth.truth_instances)
        matched, ious = _assigned(image, chosen)
        truth_total += truth_count
        matched_total += matched
        detected += matched > 0
        positive_recalls.append(matched / truth_count)
        assigned_ious.extend(ious)
        category = categories.setdefault(image.identity.category, [0, 0])
        category[0] += matched
        category[1] += truth_count
    return FastMetrics(
        clean_false_proposals_per_image=clean_proposals / len(clean),
        clean_images_with_any_proposal=clean_any / len(clean),
        micro_component_sensitivity=matched_total / truth_total,
        macro_per_positive_image_recall=float(np.mean(positive_recalls)),
        image_level_sensitivity=detected / len(positive),
        proposal_precision=matched_total / proposal_total if proposal_total else 0.0,
        mean_assigned_pair_iou=float(np.mean(assigned_ious)) if assigned_ious else 0.0,
        mean_proposals_per_image=proposal_total / len(images),
        category_sensitivity=tuple(
            (category, matched / total) for category, (matched, total) in sorted(categories.items())
        ),
    )


def _baseline_metrics(images: tuple[PreparedFusionImage, ...]) -> AggregateMetricsV2:
    evaluations = []
    for image in images:
        proposal_set = _proposal_set(
            image.classical_result,
            DetectorConfig().implementation_id,
            "priority_score descending; frozen proposal order; proposal_id tie-break",
        )
        evaluations.append(evaluate_image(
            proposal_set, image.ground_truth,
            category=image.identity.category,
            acquisition_group_id=image.identity.acquisition_group_id,
            processing_time_seconds=image.elapsed_seconds,
            hardware_context="fusion-fit deterministic CPU",
            cache_state="single_process_cache",
        ))
    return aggregate_metrics(evaluations)


def _preservation_failures(metrics: FastMetrics, baseline: AggregateMetricsV2) -> tuple[str, ...]:
    failures = []
    baseline_categories = dict(baseline.category_component_sensitivity)
    observed_categories = dict(metrics.category_sensitivity)
    if metrics.micro_component_sensitivity + 1e-12 < float(baseline.micro_component_sensitivity) - 0.02:
        failures.append("overall_micro_sensitivity_decrease_exceeds_0.02")
    for category in ("thin_crack", "pitting_cluster", "weld_disturbance"):
        if category not in baseline_categories or category not in observed_categories:
            failures.append(f"{category}_sensitivity_missing")
        elif observed_categories[category] + 1e-12 < float(baseline_categories[category]):
            failures.append(f"{category}_sensitivity_decreased")
    if metrics.image_level_sensitivity + 1e-12 < float(baseline.image_level_detection_sensitivity):
        failures.append("image_level_sensitivity_decreased")
    if metrics.mean_assigned_pair_iou + 1e-12 < float(baseline.assigned_pair_iou_mean) - 0.02:
        failures.append("mean_assigned_pair_iou_decrease_exceeds_0.02")
    return tuple(failures)


def _operating_point(
    images: tuple[PreparedFusionImage, ...],
    search: FusionSearchConfiguration,
    budget: float,
    baseline: AggregateMetricsV2,
) -> FusionOperatingPoint:
    threshold = _threshold(images, search, budget)
    metrics = _fast_metrics(images, search, threshold)
    budget_feasible = metrics.clean_false_proposals_per_image <= budget + 1e-12
    failures = list(_preservation_failures(metrics, baseline))
    if not budget_feasible:
        failures.insert(0, "clean_fp_budget_exceeded")
    return FusionOperatingPoint(
        false_proposal_budget=budget,
        threshold=threshold,
        achieved_clean_false_proposals_per_image=metrics.clean_false_proposals_per_image,
        achieved_clean_images_with_any_proposal=metrics.clean_images_with_any_proposal,
        micro_component_sensitivity=metrics.micro_component_sensitivity,
        macro_per_positive_image_recall=metrics.macro_per_positive_image_recall,
        image_level_sensitivity=metrics.image_level_sensitivity,
        proposal_precision=metrics.proposal_precision,
        mean_assigned_pair_iou=metrics.mean_assigned_pair_iou,
        mean_proposals_per_image=metrics.mean_proposals_per_image,
        category_sensitivity=metrics.category_sensitivity,
        budget_feasible=budget_feasible,
        preservation_passed=not failures,
        preservation_failures=tuple(failures),
    )


def _select(evaluated: tuple[EvaluatedFusionConfiguration, ...]) -> EvaluatedFusionConfiguration | None:
    eligible = [
        item for item in evaluated
        if item.operating_point(PRIMARY_BUDGET).budget_feasible
        and item.operating_point(PRIMARY_BUDGET).preservation_passed
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda item: (
        -item.operating_point(PRIMARY_BUDGET).micro_component_sensitivity,
        item.operating_point(PRIMARY_BUDGET).achieved_clean_false_proposals_per_image,
        -item.operating_point(PRIMARY_BUDGET).mean_assigned_pair_iou,
        item.search.preservation_floor is not None,
        -item.search.classical_weight,
        item.search.configuration_id,
    ))


def fit_hybrid_fusion(
    view: FusionFitView,
    *,
    repository_root: Path,
    classical_detector: StructuralAnomalyDetector,
    normal_detector: NormalFeatureAnomalyDetector,
    model_artifact: NormalFeatureModelArtifact,
    environment_lock_hash: str,
    artifact_sink: HybridFusionArtifactSink | None,
) -> FusionFitOutcome:
    if artifact_sink is None:
        raise HybridFusionError("Fusion persistence requires an explicit sink")
    if classical_detector.config != DetectorConfig():
        raise HybridFusionError("Fusion requires the exact frozen classical configuration")
    reference, normalisation, images = prepare_fusion_fit(
        view,
        repository_root=repository_root,
        classical_detector=classical_detector,
        normal_detector=normal_detector,
        model_artifact=model_artifact,
    )
    baseline = _baseline_metrics(images)
    search = coefficient_search_space()
    evaluated = tuple(EvaluatedFusionConfiguration(
        item,
        tuple(_operating_point(images, item, budget, baseline) for budget in DECLARED_BUDGETS),
    ) for item in search)
    selected = _select(evaluated)
    commit, dirty_state, diff_hash = _git_metadata(Path(repository_root))
    policy = default_evaluation_policy()
    artifact = HybridFusionArtifact.create(
        schema_version=HYBRID_ARTIFACT_SCHEMA_VERSION,
        implementation_identity=HYBRID_IMPLEMENTATION_ID,
        implementation_version=HYBRID_IMPLEMENTATION_VERSION,
        hybrid_protocol_hash=view.manifest_hash,
        normal_feature_model_artifact_hash=model_artifact.artifact_hash,
        frozen_classical_configuration_hash=classical_detector.config.configuration_hash,
        candidate_feature_definitions=FEATURE_DEFINITIONS,
        feature_order_identity=FEATURE_ORDER,
        high_anomaly_reference=reference,
        normalisation_statistics=normalisation,
        coefficient_search_space=search,
        evaluated_configurations=evaluated,
        preservation_constraints=PRESERVATION_CONSTRAINTS,
        selection_status="selected" if selected is not None else "failed_no_preserving_configuration",
        selected_configuration_id=selected.search.configuration_id if selected is not None else None,
        selected_coefficients=(selected.search.classical_weight, selected.search.normality_weight) if selected is not None else None,
        selected_preservation_floor=selected.search.preservation_floor if selected is not None else None,
        selected_operating_points=selected.operating_points if selected is not None else (),
        selected_operating_threshold=selected.operating_point(PRIMARY_BUDGET).threshold if selected is not None else None,
        false_proposal_budget=PRIMARY_BUDGET,
        fusion_fit_image_hashes=tuple((item.image_id, item.image_sha256) for item in view.identities),
        fusion_fit_truth_hashes=tuple((item.image_id, item.ground_truth_sha256) for item in view.identities),
        evaluation_policy_hash=policy.configuration_hash,
        environment_lock_hash=environment_lock_hash,
        code_commit=commit,
        git_dirty_state=dirty_state,
        git_diff_hash=diff_hash,
        deterministic_seed=DETERMINISTIC_SEED,
        creation_timestamp=datetime.now(timezone.utc).isoformat(),
    )
    artifact_sink.write(artifact)
    return FusionFitOutcome(artifact, baseline, normalisation, images)
