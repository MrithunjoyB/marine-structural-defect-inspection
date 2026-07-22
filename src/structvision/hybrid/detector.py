"""Write-free public proposal-guided hybrid detector."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
from typing import Mapping

import numpy as np

from structvision.api import StructuralAnomalyDetector
from structvision.normal_feature.model_artifact import NormalFeatureModelArtifact
from structvision.normal_feature.patchcore import NormalFeatureAnomalyDetector
from structvision.sinks import ArtifactSink
from structvision.types import Proposal, frozen_mapping

from .artifact import (
    HYBRID_IMPLEMENTATION_ID,
    HYBRID_IMPLEMENTATION_VERSION,
    PRIMARY_BUDGET,
    FusionSearchConfiguration,
    HybridFusionArtifact,
)
from .errors import HybridFusionError
from .features import CandidateEvidence, candidate_evidence, normalised_components


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    prefix = f"{array.dtype.str}\0{','.join(str(item) for item in array.shape)}\0C\0".encode("ascii")
    return hashlib.sha256(prefix + array.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class HybridCandidateDiagnostic:
    classical_proposal_id: str
    classical_rank: int
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    mask_hash: str
    classical_priority_score: float
    classical_evidence_score: float
    heuristic_mask_reliability: float
    patchcore_candidate_evidence: tuple[tuple[str, float], ...]
    normalised_features: tuple[tuple[str, float], ...]
    normalised_classical_evidence: float
    normalised_patchcore_evidence: float
    hybrid_score: float
    final_rank: int | None
    selected: bool
    operating_threshold: float
    generic_preservation_applied: bool
    explanation: tuple[str, ...]

    def __post_init__(self) -> None:
        array = np.ascontiguousarray(self.mask)
        array.setflags(write=False)
        object.__setattr__(self, "mask", array)
        if _array_hash(array) != self.mask_hash:
            raise HybridFusionError("Hybrid diagnostic mask identity differs from classical input")
        if self.selected != (self.final_rank is not None):
            raise HybridFusionError("Hybrid selection and final rank disagree")
        numeric = (
            self.classical_priority_score, self.classical_evidence_score,
            self.heuristic_mask_reliability, self.normalised_classical_evidence,
            self.normalised_patchcore_evidence, self.hybrid_score, self.operating_threshold,
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise HybridFusionError("Hybrid diagnostics require finite scores")

    def to_dict(self, *, include_mask: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "classical_proposal_id": self.classical_proposal_id,
            "classical_rank": self.classical_rank,
            "bbox": list(self.bbox),
            "bbox_convention": "half-open:x_min,y_min,x_max,y_max",
            "mask_hash": self.mask_hash,
            "classical_priority_score": self.classical_priority_score,
            "classical_evidence_score": self.classical_evidence_score,
            "heuristic_mask_reliability": self.heuristic_mask_reliability,
            "patchcore_candidate_evidence": dict(self.patchcore_candidate_evidence),
            "normalised_features": dict(self.normalised_features),
            "normalised_classical_evidence": self.normalised_classical_evidence,
            "normalised_patchcore_evidence": self.normalised_patchcore_evidence,
            "hybrid_score": self.hybrid_score,
            "score_semantics": "explainable_linear_rank_score_not_probability",
            "final_rank": self.final_rank,
            "selected": self.selected,
            "operating_threshold": self.operating_threshold,
            "generic_preservation_applied": self.generic_preservation_applied,
            "explanation": list(self.explanation),
        }
        if include_mask:
            payload["mask"] = self.mask.tolist()
        return payload


@dataclass(frozen=True)
class HybridProposal:
    proposal_id: str
    rank: int
    bbox: tuple[int, int, int, int]
    final_mask: np.ndarray
    hybrid_score: float
    classical_priority_score: float
    patchcore_evidence_score: float
    explanation: tuple[str, ...]
    source_classical_proposal_id: str

    def __post_init__(self) -> None:
        array = np.ascontiguousarray(self.final_mask)
        array.setflags(write=False)
        object.__setattr__(self, "final_mask", array)
        if self.rank <= 0 or not self.proposal_id:
            raise HybridFusionError("Hybrid proposal ID and one-based rank are required")
        ys, xs = np.where(array > 0)
        expected = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        if self.bbox != expected:
            raise HybridFusionError("Hybrid proposal box must remain half-open and mask-derived")

    def to_dict(self, *, include_mask: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "proposal_id": self.proposal_id,
            "rank": self.rank,
            "bbox": list(self.bbox),
            "bbox_convention": "half-open:x_min,y_min,x_max,y_max",
            "mask_hash": _array_hash(self.final_mask),
            "hybrid_score": self.hybrid_score,
            "score_semantics": "explainable_linear_rank_score_not_probability",
            "classical_priority_score": self.classical_priority_score,
            "patchcore_evidence_score": self.patchcore_evidence_score,
            "explanation": list(self.explanation),
            "source_classical_proposal_id": self.source_classical_proposal_id,
        }
        if include_mask:
            payload["final_mask"] = self.final_mask.tolist()
        return payload


@dataclass(frozen=True)
class HybridAnalysisResult:
    image_id: str
    input_hash: str
    image_shape: tuple[int, int, int]
    complete_original_classical_candidate_count: int
    selected_hybrid_proposal_count: int
    complete_candidate_diagnostics: tuple[HybridCandidateDiagnostic, ...]
    proposals: tuple[HybridProposal, ...]
    implementation_id: str
    implementation_version: str
    classical_configuration_hash: str
    classical_implementation_id: str
    normal_feature_model_artifact_hash: str
    normal_feature_implementation_id: str
    fusion_artifact_hash: str
    false_proposal_budget: float
    operating_threshold: float
    timing_breakdown_seconds: tuple[tuple[str, float], ...]
    warnings: tuple[str, ...]
    provenance: tuple[tuple[str, object], ...]

    def __post_init__(self) -> None:
        if self.implementation_id != HYBRID_IMPLEMENTATION_ID:
            raise HybridFusionError("Hybrid result implementation identity differs")
        if self.complete_original_classical_candidate_count != len(self.complete_candidate_diagnostics):
            raise HybridFusionError("Complete pre-threshold diagnostics were not preserved")
        if self.selected_hybrid_proposal_count != len(self.proposals):
            raise HybridFusionError("Selected proposal count differs")
        ranks = [item.rank for item in self.proposals]
        if ranks != list(range(1, len(ranks) + 1)):
            raise HybridFusionError("Hybrid ranks must be unique, contiguous, and one-based")
        selected_diagnostics = sorted(
            (item for item in self.complete_candidate_diagnostics if item.selected),
            key=lambda item: int(item.final_rank),
        )
        if [item.classical_proposal_id for item in selected_diagnostics] != [item.source_classical_proposal_id for item in self.proposals]:
            raise HybridFusionError("Selected proposals and diagnostics differ")

    def to_dict(self, *, include_masks: bool = False) -> dict[str, object]:
        return {
            "image_id": self.image_id,
            "input_hash": self.input_hash,
            "image_shape": list(self.image_shape),
            "complete_original_classical_candidate_count": self.complete_original_classical_candidate_count,
            "selected_hybrid_proposal_count": self.selected_hybrid_proposal_count,
            "complete_candidate_diagnostics": [item.to_dict(include_mask=include_masks) for item in self.complete_candidate_diagnostics],
            "proposals": [item.to_dict(include_mask=include_masks) for item in self.proposals],
            "implementation_id": self.implementation_id,
            "implementation_version": self.implementation_version,
            "classical_configuration_hash": self.classical_configuration_hash,
            "classical_implementation_id": self.classical_implementation_id,
            "normal_feature_model_artifact_hash": self.normal_feature_model_artifact_hash,
            "normal_feature_implementation_id": self.normal_feature_implementation_id,
            "fusion_artifact_hash": self.fusion_artifact_hash,
            "false_proposal_budget": self.false_proposal_budget,
            "operating_threshold": self.operating_threshold,
            "timing_breakdown_seconds": dict(self.timing_breakdown_seconds),
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
        }


class ProposalGuidedHybridDetector:
    """Classical proposals reranked and selected with frozen PatchCore evidence."""

    def __init__(
        self,
        *,
        classical_detector: StructuralAnomalyDetector,
        normal_feature_detector: NormalFeatureAnomalyDetector,
        normal_feature_model_artifact: NormalFeatureModelArtifact,
        fusion_artifact: HybridFusionArtifact,
    ) -> None:
        if fusion_artifact.selection_status != "selected":
            raise HybridFusionError("A rejected fusion artifact cannot be used for inference")
        if classical_detector.config.configuration_hash != fusion_artifact.frozen_classical_configuration_hash:
            raise HybridFusionError("Classical detector and fusion artifact differ")
        if normal_feature_model_artifact.artifact_hash != fusion_artifact.normal_feature_model_artifact_hash:
            raise HybridFusionError("Normal-feature model and fusion artifact differ")
        if normal_feature_detector.environment_lock_hash != fusion_artifact.environment_lock_hash:
            raise HybridFusionError("Learned environment lock and fusion artifact differ")
        self.classical_detector = classical_detector
        self.normal_feature_detector = normal_feature_detector
        self.normal_feature_model_artifact = normal_feature_model_artifact
        self.fusion_artifact = fusion_artifact

    def analyse(
        self,
        image: object,
        *,
        image_id: str,
        false_proposal_budget: float = PRIMARY_BUDGET,
        colour_space: str | None = None,
        alpha_handling: str | None = None,
        metadata: Mapping[str, object] | None = None,
        artifact_sink: ArtifactSink | None = None,
    ) -> HybridAnalysisResult:
        started = time.perf_counter()
        classical_started = time.perf_counter()
        classical = self.classical_detector.analyse(
            image, image_id=image_id, colour_space=colour_space,
            alpha_handling=alpha_handling, metadata=metadata,
        )
        classical_elapsed = time.perf_counter() - classical_started
        normal_started = time.perf_counter()
        normal = self.normal_feature_detector.score(
            image, model_artifact=self.normal_feature_model_artifact,
            image_id=image_id, colour_space=colour_space, alpha_handling=alpha_handling,
        )
        normal_elapsed = time.perf_counter() - normal_started
        if classical.input_hash != normal.input_hash or classical.image_shape != normal.image_shape:
            raise HybridFusionError("Classical and normal-feature coordinate systems differ")
        operating_point = self.fusion_artifact.operating_point(false_proposal_budget)
        classical_weight, normality_weight = self.fusion_artifact.selected_coefficients or (0.0, 0.0)
        search = FusionSearchConfiguration(
            self.fusion_artifact.selected_configuration_id or "missing",
            classical_weight, normality_weight,
            self.fusion_artifact.selected_preservation_floor,
        )
        scored = []
        for proposal in classical.proposals:
            evidence = candidate_evidence(
                proposal, normal.anomaly_map,
                high_anomaly_reference=self.fusion_artifact.high_anomaly_reference,
            )
            classical_component, normality_component, scaled = normalised_components(
                evidence, self.fusion_artifact.normalisation_statistics,
            )
            hybrid_score = classical_weight * classical_component + normality_weight * normality_component
            preserved = search.preservation_floor is not None and classical_component >= search.preservation_floor
            selected = preserved or hybrid_score >= operating_point.threshold
            scored.append((proposal, evidence, scaled, classical_component, normality_component, hybrid_score, preserved, selected))
        ordered_selected = sorted(
            (item for item in scored if item[-1]),
            key=lambda item: (-item[5], item[0].proposal_id),
        )
        rank_by_id = {item[0].proposal_id: rank for rank, item in enumerate(ordered_selected, 1)}
        diagnostics = []
        for proposal, evidence, scaled, classical_component, normality_component, hybrid_score, preserved, selected in scored:
            rank = rank_by_id.get(proposal.proposal_id)
            reason = "selected_by_generic_classical_preservation_floor" if preserved else (
                "selected_by_hybrid_score_threshold" if selected else "not_selected_at_frozen_operating_threshold"
            )
            explanation = (
                f"classical contribution={classical_weight:.2f}×{classical_component:.6f}={classical_weight * classical_component:.6f}",
                f"PatchCore contribution={normality_weight:.2f}×{normality_component:.6f}={normality_weight * normality_component:.6f}",
                f"hybrid rank score={hybrid_score:.6f}; this is not a calibrated probability",
                reason,
            )
            diagnostics.append(HybridCandidateDiagnostic(
                proposal.proposal_id, proposal.rank, proposal.bbox, proposal.final_mask,
                _array_hash(proposal.final_mask), proposal.priority_score, proposal.evidence_score,
                proposal.heuristic_reliability,
                tuple((name, value) for name, value in evidence.feature_values if name.startswith("patchcore_")),
                scaled, classical_component, normality_component, hybrid_score, rank, selected,
                operating_point.threshold, preserved, explanation,
            ))
        proposals = tuple(HybridProposal(
            proposal_id=f"hybrid-{rank:04d}-{item[0].proposal_id}",
            rank=rank,
            bbox=item[0].bbox,
            final_mask=item[0].final_mask,
            hybrid_score=item[5],
            classical_priority_score=item[0].priority_score,
            patchcore_evidence_score=item[4],
            explanation=next(row.explanation for row in diagnostics if row.classical_proposal_id == item[0].proposal_id),
            source_classical_proposal_id=item[0].proposal_id,
        ) for rank, item in enumerate(ordered_selected, 1))
        result = HybridAnalysisResult(
            image_id=image_id,
            input_hash=classical.input_hash,
            image_shape=classical.image_shape,
            complete_original_classical_candidate_count=len(classical.proposals),
            selected_hybrid_proposal_count=len(proposals),
            complete_candidate_diagnostics=tuple(diagnostics),
            proposals=proposals,
            implementation_id=HYBRID_IMPLEMENTATION_ID,
            implementation_version=HYBRID_IMPLEMENTATION_VERSION,
            classical_configuration_hash=classical.configuration_hash,
            classical_implementation_id=classical.implementation_id,
            normal_feature_model_artifact_hash=normal.model_artifact_hash,
            normal_feature_implementation_id=self.normal_feature_detector.config.implementation_id,
            fusion_artifact_hash=self.fusion_artifact.artifact_hash,
            false_proposal_budget=false_proposal_budget,
            operating_threshold=operating_point.threshold,
            timing_breakdown_seconds=(
                ("classical_seconds", classical_elapsed),
                ("patchcore_seconds", normal_elapsed),
                ("fusion_seconds", max(0.0, time.perf_counter() - started - classical_elapsed - normal_elapsed)),
                ("total_seconds", time.perf_counter() - started),
            ),
            warnings=(
                "development holdout — non-confirmatory",
                "hybrid scores and heuristic reliability are not calibrated probabilities",
                "complete classical candidate diagnostics retained pre-threshold",
            ),
            provenance=tuple(frozen_mapping({
                "hybrid_protocol_hash": self.fusion_artifact.hybrid_protocol_hash,
                "classical_protected_hashes_verified": classical.provenance.protected_hashes_verified,
                "normal_feature_model_artifact_hash": normal.model_artifact_hash,
                "fusion_artifact_hash": self.fusion_artifact.artifact_hash,
                "feature_order": list(self.fusion_artifact.feature_order_identity),
            })),
        )
        if artifact_sink is not None:
            artifact_sink.write(result)
        return result

    def reselect(
        self,
        result: HybridAnalysisResult,
        *,
        false_proposal_budget: float,
    ) -> HybridAnalysisResult:
        """Apply another pre-frozen budget without rerunning or inspecting an image."""
        if result.fusion_artifact_hash != self.fusion_artifact.artifact_hash:
            raise HybridFusionError("Cannot reselect a result from another fusion artifact")
        operating_point = self.fusion_artifact.operating_point(false_proposal_budget)
        selected_rows = sorted(
            (
                item for item in result.complete_candidate_diagnostics
                if item.generic_preservation_applied or item.hybrid_score >= operating_point.threshold
            ),
            key=lambda item: (-item.hybrid_score, item.classical_proposal_id),
        )
        ranks = {item.classical_proposal_id: rank for rank, item in enumerate(selected_rows, 1)}
        diagnostics = tuple(HybridCandidateDiagnostic(
            classical_proposal_id=item.classical_proposal_id,
            classical_rank=item.classical_rank,
            bbox=item.bbox,
            mask=item.mask,
            mask_hash=item.mask_hash,
            classical_priority_score=item.classical_priority_score,
            classical_evidence_score=item.classical_evidence_score,
            heuristic_mask_reliability=item.heuristic_mask_reliability,
            patchcore_candidate_evidence=item.patchcore_candidate_evidence,
            normalised_features=item.normalised_features,
            normalised_classical_evidence=item.normalised_classical_evidence,
            normalised_patchcore_evidence=item.normalised_patchcore_evidence,
            hybrid_score=item.hybrid_score,
            final_rank=ranks.get(item.classical_proposal_id),
            selected=item.classical_proposal_id in ranks,
            operating_threshold=operating_point.threshold,
            generic_preservation_applied=item.generic_preservation_applied,
            explanation=item.explanation[:-1] + ((
                "selected_by_generic_classical_preservation_floor" if item.generic_preservation_applied else
                "selected_by_hybrid_score_threshold" if item.classical_proposal_id in ranks else
                "not_selected_at_frozen_operating_threshold"
            ),),
        ) for item in result.complete_candidate_diagnostics)
        by_id = {item.classical_proposal_id: item for item in diagnostics}
        proposals = tuple(HybridProposal(
            proposal_id=f"hybrid-{rank:04d}-{item.classical_proposal_id}",
            rank=rank,
            bbox=item.bbox,
            final_mask=item.mask,
            hybrid_score=item.hybrid_score,
            classical_priority_score=item.classical_priority_score,
            patchcore_evidence_score=item.normalised_patchcore_evidence,
            explanation=by_id[item.classical_proposal_id].explanation,
            source_classical_proposal_id=item.classical_proposal_id,
        ) for rank, item in enumerate(selected_rows, 1))
        return HybridAnalysisResult(
            image_id=result.image_id,
            input_hash=result.input_hash,
            image_shape=result.image_shape,
            complete_original_classical_candidate_count=result.complete_original_classical_candidate_count,
            selected_hybrid_proposal_count=len(proposals),
            complete_candidate_diagnostics=diagnostics,
            proposals=proposals,
            implementation_id=result.implementation_id,
            implementation_version=result.implementation_version,
            classical_configuration_hash=result.classical_configuration_hash,
            classical_implementation_id=result.classical_implementation_id,
            normal_feature_model_artifact_hash=result.normal_feature_model_artifact_hash,
            normal_feature_implementation_id=result.normal_feature_implementation_id,
            fusion_artifact_hash=result.fusion_artifact_hash,
            false_proposal_budget=false_proposal_budget,
            operating_threshold=operating_point.threshold,
            timing_breakdown_seconds=result.timing_breakdown_seconds + (("budget_reselection_seconds", 0.0),),
            warnings=result.warnings,
            provenance=result.provenance,
        )
