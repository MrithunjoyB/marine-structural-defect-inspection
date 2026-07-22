"""Ground-truth-free deterministic components from calibrated anomaly maps."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .configuration import LearnedProposalConfig
from .types import LearnedProposal, array_hash


def extract_proposals(
    anomaly_map: np.ndarray,
    *,
    threshold: float,
    operating_point_id: str,
    config: LearnedProposalConfig,
) -> tuple[LearnedProposal, ...]:
    array = np.ascontiguousarray(anomaly_map, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Proposal extraction requires one finite anomaly map")
    if not math.isfinite(threshold) or not operating_point_id:
        raise ValueError("A finite calibration threshold and operating-point ID are required")
    binary = np.ascontiguousarray((array >= np.float32(threshold)).astype(np.uint8))
    count, labels, statistics, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=config.connectivity,
    )
    candidates: list[dict[str, object]] = []
    for label in range(1, count):
        area = int(statistics[label, cv2.CC_STAT_AREA])
        if area < config.minimum_area_pixels:
            continue
        mask = np.ascontiguousarray((labels == label).astype(np.uint8) * 255)
        y_values, x_values = np.where(mask > 0)
        bbox = (
            int(x_values.min()), int(y_values.min()),
            int(x_values.max()) + 1, int(y_values.max()) + 1,
        )
        score = float(np.max(array[labels == label]))
        candidates.append({
            "bbox": bbox,
            "mask": mask,
            "score": score,
            "area": area,
            "centroid": (float(centroids[label, 0]), float(centroids[label, 1])),
        })
    candidates.sort(key=lambda item: (-float(item["score"]), -int(item["area"]), item["bbox"]))
    selected = candidates[:config.maximum_proposal_count]
    return tuple(
        LearnedProposal(
            proposal_id=f"NF-P{rank:03d}",
            rank=rank,
            bbox=item["bbox"],
            mask=item["mask"],
            component_anomaly_score=float(item["score"]),
            area=int(item["area"]),
            centroid_xy=item["centroid"],
            threshold=float(threshold),
            operating_point_id=operating_point_id,
            mask_hash=array_hash(item["mask"]),
            extraction_policy_hash=config.configuration_hash,
        )
        for rank, item in enumerate(selected, start=1)
    )
