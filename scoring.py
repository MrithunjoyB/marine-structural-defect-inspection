"""Visual anomaly priority scoring before trained classification is available."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PriorityResult:
    score: float
    label: str
    rationale: str


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
