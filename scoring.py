"""Visual anomaly priority scoring before trained classification is available."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class PriorityResult:
    score: float
    label: str
    rationale: str


DEFAULT_SCORE_WEIGHTS = {
    "edge_density": 0.16,
    "texture_variation": 0.20,
    "colour_difference": 0.14,
    "gradient_strength": 0.14,
    "entropy": 0.10,
    "area_relevance": 0.10,
    "mask_stability": 0.16,
}


def weighted_anomaly_score(components: Mapping[str, float], weights: Mapping[str, float] | None = None) -> float:
    selected = dict(DEFAULT_SCORE_WEIGHTS if weights is None else weights)
    total = sum(max(float(value), 0.0) for value in selected.values()) or 1.0
    return 100.0 * sum(
        max(0.0, min(1.0, float(components.get(name, 0.0)))) * max(float(weight), 0.0)
        for name, weight in selected.items()
    ) / total


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
