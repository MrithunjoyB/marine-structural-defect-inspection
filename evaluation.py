"""Quantitative evaluation helpers for manually reviewed proposal regions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from labeling import ReviewedAnnotation
from region_proposal import RegionProposal


@dataclass(frozen=True)
class EvaluationMetrics:
    proposal_recall: float
    average_best_iou: float
    false_proposals_per_image: float
    region_coverage: float
    annotation_acceptance_rate: float

    def to_dict(self) -> dict[str, float]:
        return {key: round(value, 4) for key, value in self.__dict__.items()}


def evaluate_proposals(
    proposals: list[RegionProposal], references: list[tuple[int, int, int, int]],
    annotations: list[ReviewedAnnotation] | None = None, image_count: int = 1, iou_threshold: float = 0.3,
) -> EvaluationMetrics:
    """Compare candidate boxes with reviewed reference boxes."""
    best = [max((_iou(reference, proposal.bbox) for proposal in proposals), default=0.0) for reference in references]
    recall = sum(value >= iou_threshold for value in best) / max(len(references), 1)
    average = float(np.mean(best)) if best else 0.0
    matched = {index for index, proposal in enumerate(proposals) if any(_iou(proposal.bbox, ref) >= iou_threshold for ref in references)}
    false_per_image = (len(proposals) - len(matched)) / max(image_count, 1)
    coverage = _coverage(proposals, references)
    reviewed = annotations or []
    acceptance = sum(item.accepted for item in reviewed) / max(len(reviewed), 1)
    return EvaluationMetrics(recall, average, false_per_image, coverage, acceptance)


def _coverage(proposals: list[RegionProposal], references: list[tuple[int, int, int, int]]) -> float:
    if not references:
        return 0.0
    values = []
    for ref in references:
        ref_area = max((ref[2] - ref[0]) * (ref[3] - ref[1]), 1)
        intersections = [max(0, min(ref[2], p.bbox[2]) - max(ref[0], p.bbox[0])) * max(0, min(ref[3], p.bbox[3]) - max(ref[1], p.bbox[1])) for p in proposals]
        values.append(min(sum(intersections) / ref_area, 1.0))
    return float(np.mean(values))


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / max(union, 1)
