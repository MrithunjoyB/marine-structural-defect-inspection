"""Visual anomaly priority scoring before trained classification is available."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class PriorityResult:
    score: float
    label: str
    rationale: str


DEFAULT_EVIDENCE_WEIGHTS = {
    "local_texture_contrast": 0.22, "local_colour_contrast": 0.20,
    "local_entropy_contrast": 0.13, "edge_concentration": 0.14,
    "gradient_contrast": 0.16, "geometric_irregularity": 0.15,
}
DEFAULT_RELIABILITY_WEIGHTS = {
    "perturbation_stability": 0.24, "connectedness": 0.18,
    "boundary_smoothness": 0.18, "scale_agreement": 0.16, "segmentation_coherence": 0.24,
}
DEFAULT_PRIORITY_WEIGHTS = {
    "anomaly_evidence": 0.60, "mask_reliability": 0.20,
    "area_relevance": 0.10, "novelty": 0.10,
}
DEFAULT_SCORE_WEIGHTS = DEFAULT_EVIDENCE_WEIGHTS


def weighted_anomaly_score(components: Mapping[str, float], weights: Mapping[str, float] | None = None) -> float:
    selected = dict(DEFAULT_SCORE_WEIGHTS if weights is None else weights)
    total = sum(max(float(value), 0.0) for value in selected.values()) or 1.0
    return 100.0 * sum(
        max(0.0, min(1.0, float(components.get(name, 0.0)))) * max(float(weight), 0.0)
        for name, weight in selected.items()
    ) / total


def score_architecture(
    evidence: Mapping[str, float], reliability: Mapping[str, float], area_relevance: float, novelty: float,
    evidence_weights: Mapping[str, float] | None = None, reliability_weights: Mapping[str, float] | None = None,
    priority_weights: Mapping[str, float] | None = None,
) -> tuple[float, float, float, dict[str, float]]:
    evidence_score = weighted_anomaly_score(evidence, evidence_weights or DEFAULT_EVIDENCE_WEIGHTS)
    reliability_score = weighted_anomaly_score(reliability, reliability_weights or DEFAULT_RELIABILITY_WEIGHTS)
    priority_inputs = {
        "anomaly_evidence": evidence_score / 100, "mask_reliability": reliability_score / 100,
        "area_relevance": area_relevance, "novelty": novelty,
    }
    priority_score = weighted_anomaly_score(priority_inputs, priority_weights or DEFAULT_PRIORITY_WEIGHTS)
    weights = evidence_weights or DEFAULT_EVIDENCE_WEIGHTS
    weighted = {name: max(evidence.get(name, 0.0), 0.0) * weight for name, weight in weights.items()}
    denominator = sum(weighted.values()) or 1.0
    contributions = {name: 100.0 * value / denominator for name, value in weighted.items()}
    return evidence_score, reliability_score, priority_score, contributions


def score_region(
    relative_area: float,
    edge_density: float,
    texture_score: float,
    color_variation_score: float,
    aspect_ratio: float,
    nearby_count: int = 0,
) -> PriorityResult:
    """Compute a neutral visual priority score for a candidate region."""

    area_component = min(relative_area * 360.0, 24.0)
    edge_component = min(edge_density * 28.0, 22.0)
    texture_component = min(texture_score * 22.0, 22.0)
    color_component = min(color_variation_score * 20.0, 18.0)
    elongation_component = min(max(aspect_ratio - 2.0, 0.0) * 3.0, 10.0)
    cluster_component = min(nearby_count * 2.0, 8.0)
    score = min(
        100.0,
        area_component
        + edge_component
        + texture_component
        + color_component
        + elongation_component
        + cluster_component,
    )

    label = priority_label(score)
    rationale = (
        "Score combines relative area, edge concentration, texture discontinuity, "
        "color variation, elongation, and nearby candidate density."
    )
    return PriorityResult(round(score, 1), label, rationale)


def priority_label(score: float) -> str:
    if score >= 72:
        return "Review Required"
    if score >= 52:
        return "High"
    if score >= 28:
        return "Moderate"
    return "Low"
