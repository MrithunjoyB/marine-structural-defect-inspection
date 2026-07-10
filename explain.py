"""Engineering explanations and recommended actions for detected defects."""

from __future__ import annotations

from collections import Counter

from detect import Defect
from severity import SeverityResult


EXPLANATIONS = {
    "corrosion": (
        "Corrosion-like surface degradation was detected. In marine environments, corrosion can reduce "
        "effective plate thickness, weaken structural members, and accelerate fatigue damage under cyclic loading."
    ),
    "crack": (
        "A crack-like discontinuity was detected. Cracks in hull plates, weld zones, or structural members may "
        "indicate fatigue damage, stress concentration, or local material failure."
    ),
    "weld_defect": (
        "A possible weld defect was detected. Weld discontinuities can affect load transfer, reduce fatigue "
        "strength, and compromise structural integrity."
    ),
    "coating_damage": (
        "Coating damage was detected. Protective coating failure can expose the substrate to seawater, increasing "
        "the risk of corrosion initiation and propagation."
    ),
    "dent": (
        "Dent-like local indentation was detected. Local geometric damage can alter stress distribution and should "
        "be checked for associated yielding or coating loss."
    ),
    "scratch": (
        "Scratch-like surface damage was detected. Scratches are often low severity, but in coated marine surfaces "
        "they can become corrosion initiation points."
    ),
    "deformation": (
        "A deformation-like region was detected. Distortion of marine structural components can indicate overload, "
        "impact, buckling, or permanent set."
    ),
    "pitting": (
        "Pitting-like localized damage was detected. Pitting can produce sharp local section loss and may act as a "
        "stress raiser under cyclic loading."
    ),
    "surface_anomaly": (
        "A visually anomalous surface region was detected. The region should be reinspected because it may represent "
        "staining, roughness, coating failure, or early-stage defect formation."
    ),
}

ACTION_MAP = {
    "corrosion": ["Surface cleaning and corrosion mapping", "Ultrasonic thickness measurement", "Recoating assessment"],
    "crack": ["Close visual reinspection", "Dye penetrant testing", "Magnetic particle testing", "Structural repair assessment"],
    "weld_defect": ["Weld inspection", "Magnetic particle testing", "Ultrasonic testing of weld zone"],
    "coating_damage": ["Surface cleaning and recoating", "Check for under-film corrosion"],
    "dent": ["Dimensional survey", "Structural repair assessment"],
    "scratch": ["Visual reinspection", "Touch-up coating or surface protection"],
    "deformation": ["Structural alignment check", "Repair or reinforcement assessment"],
    "pitting": ["Pit depth measurement", "Ultrasonic thickness measurement", "Corrosion protection review"],
    "surface_anomaly": ["Visual reinspection", "Surface cleaning before re-evaluation"],
}


def build_engineering_summary(defects: list[Defect], severity: SeverityResult) -> str:
    if not defects:
        return (
            "The selected detection mode did not identify a strong visible defect region. This should be treated "
            "as a preliminary image-based result only; inaccessible faces, poor lighting, and subsurface damage "
            "still require conventional inspection."
        )

    counts = Counter(defect.defect_type for defect in defects)
    leading_types = ", ".join(f"{count} {name}" for name, count in counts.most_common(3))
    explanations = []
    for defect_type in counts:
        explanations.append(EXPLANATIONS.get(defect_type, EXPLANATIONS["surface_anomaly"]))

    driver_text = " ".join(severity.drivers)
    return (
        f"The inspection identified {leading_types}. "
        f"Overall severity is estimated as {severity.label} with a score of {severity.score}/100. "
        + " ".join(explanations)
        + " "
        + driver_text
    )


def recommend_actions(defects: list[Defect], severity: SeverityResult) -> list[str]:
    actions: list[str] = []
    for defect in defects:
        for action in ACTION_MAP.get(defect.defect_type, ACTION_MAP["surface_anomaly"]):
            if action not in actions:
                actions.append(action)

    if severity.label in {"High", "Critical"}:
        actions.append("Restrict service or escalate to certified marine/structural inspector before continued operation.")
    elif severity.label == "Medium":
        actions.append("Schedule follow-up inspection and compare with maintenance history.")
    else:
        actions.append("Document condition and monitor during the next planned inspection interval.")

    if not defects:
        return ["Document image result", "Repeat inspection with better lighting or closer view if defect suspicion remains."]
    return actions
