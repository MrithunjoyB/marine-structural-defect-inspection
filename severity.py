"""Severity scoring for marine structural visual defects."""

from __future__ import annotations

from dataclasses import dataclass

from config import RISK_WEIGHTS
from detect import Defect


@dataclass(frozen=True)
class SeverityResult:
    score: float
    label: str
    drivers: list[str]


def estimate_overall_severity(defects: list[Defect], image_shape: tuple[int, int]) -> SeverityResult:
    if not defects:
        return SeverityResult(0.0, "Low", ["No visible defect was detected by the selected mode."])

    defect_count = len(defects)
    total_relative_area = min(sum(defect.relative_area for defect in defects), 1.0)
    max_component = 0.0
    drivers: list[str] = []

    for defect in defects:
        risk_weight = RISK_WEIGHTS.get(defect.defect_type, 0.4)
        area_component = min(defect.relative_area * 420, 35)
        confidence_component = defect.confidence * 25
        risk_component = risk_weight * 40
        defect_score = area_component + confidence_component + risk_component
        max_component = max(max_component, defect_score)
        if risk_weight >= 0.72:
            drivers.append(f"{defect.defect_type} is structurally significant in marine service.")

    count_component = min(defect_count * 4, 18)
    spread_component = min(total_relative_area * 120, 22)
    score = min(100.0, max_component + count_component + spread_component)
    label = _label_from_score(score)

    if total_relative_area > 0.05:
        drivers.append("The affected visual area is non-trivial relative to the uploaded image.")
    if defect_count >= 4:
        drivers.append("Multiple defect regions increase inspection priority.")
    if not drivers:
        drivers.append("Severity is driven mainly by visual confidence and affected area.")

    return SeverityResult(round(score, 1), label, drivers)


def _label_from_score(score: float) -> str:
    if score >= 82:
        return "Critical"
    if score >= 62:
        return "High"
    if score >= 35:
        return "Medium"
    return "Low"
