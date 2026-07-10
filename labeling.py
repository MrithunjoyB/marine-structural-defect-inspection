"""Human-in-the-loop annotation models and helpers."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime

from config import DEFAULT_LABEL_CLASSES
from region_proposal import RegionProposal


@dataclass(frozen=True)
class ReviewedAnnotation:
    image_name: str
    region_id: str
    accepted: bool
    decision: str
    label: str
    bbox: tuple[int, int, int, int]
    mask_path: str
    mask_source: str
    priority_score: float
    priority_label: str
    notes: str
    reviewed_at: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_annotation(
    image_name: str,
    proposal: RegionProposal,
    accepted: bool,
    label: str,
    notes: str = "",
    decision: str | None = None,
    corrected_bbox: tuple[int, int, int, int] | None = None,
    corrected_mask_path: str | None = None,
    mask_source: str = "refined",
) -> ReviewedAnnotation:
    decision = decision or ("accept" if accepted else "reject")
    if decision not in {"accept", "reject", "uncertain"}:
        raise ValueError("Review decision must be accept, reject, or uncertain.")
    accepted = decision == "accept"
    label = label.strip() or "unassigned"
    return ReviewedAnnotation(
        image_name=image_name,
        region_id=proposal.region_id,
        accepted=accepted,
        decision=decision,
        label=label.strip(),
        bbox=corrected_bbox or proposal.bbox,
        mask_path=corrected_mask_path or str(proposal.mask_path),
        mask_source=mask_source,
        priority_score=proposal.priority.score,
        priority_label=proposal.priority.label,
        notes=notes.strip(),
        reviewed_at=datetime.now().isoformat(timespec="seconds"),
    )
