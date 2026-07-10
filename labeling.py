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
    label: str
    bbox: tuple[int, int, int, int]
    mask_path: str
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
) -> ReviewedAnnotation:
    if label not in DEFAULT_LABEL_CLASSES and not label.strip():
        label = "other_surface_anomaly"
    return ReviewedAnnotation(
        image_name=image_name,
        region_id=proposal.region_id,
        accepted=accepted,
        label=label.strip(),
        bbox=proposal.bbox,
        mask_path=str(proposal.mask_path),
        priority_score=proposal.priority.score,
        priority_label=proposal.priority.label,
        notes=notes.strip(),
        reviewed_at=datetime.now().isoformat(timespec="seconds"),
    )
