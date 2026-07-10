"""Classical CV region proposal for unlabeled structural or surface images."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from config import MASK_DIR, OUTPUT_DIR
from feature_extraction import FeatureMaps
from scoring import PriorityResult, score_region


@dataclass(frozen=True)
class RegionProposal:
    region_id: str
    bbox: tuple[int, int, int, int]
    pixel_area: int
    relative_area: float
    aspect_ratio: float
    perimeter: float
    edge_density: float
    texture_score: float
    color_variation_score: float
    priority: PriorityResult
    mask_path: Path

    def to_row(self) -> dict[str, object]:
        x1, y1, x2, y2 = self.bbox
        return {
            "Region ID": self.region_id,
            "BBox": f"({x1}, {y1}) - ({x2}, {y2})",
            "Pixel Area": self.pixel_area,
            "Relative Area (%)": round(self.relative_area * 100, 3),
            "Aspect Ratio": round(self.aspect_ratio, 2),
            "Perimeter": round(self.perimeter, 1),
            "Edge Density": round(self.edge_density, 3),
            "Texture Score": round(self.texture_score, 3),
            "Color Variation": round(self.color_variation_score, 3),
            "Priority Score": self.priority.score,
            "Priority Label": self.priority.label,
        }


@dataclass(frozen=True)
class ProposalResult:
    proposals: list[RegionProposal]
    overlay_path: Path
    combined_mask_path: Path


def propose_regions(
    image: np.ndarray,
    feature_maps: FeatureMaps,
    image_stem: str,
    min_area: int = 250,
    max_regions: int = 20,
) -> ProposalResult:
    """Create visual anomaly candidates from edge, texture, and color features."""

    height, width = image.shape[:2]
    image_area = max(height * width, 1)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_mask = _combined_binary_mask(feature_maps)
    contours, _ = cv2.findContours(base_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw: list[tuple[np.ndarray, tuple[int, int, int, int], int]] = []
    for contour in contours:
        area = int(cv2.contourArea(contour))
        if area < min_area or area > image_area * 0.45:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 4 or h < 4:
            continue
        raw.append((contour, (x, y, w, h), area))

    raw = sorted(raw, key=lambda item: item[2], reverse=True)[: max_regions * 3]
    centers = [(x + w / 2, y + h / 2) for _, (x, y, w, h), _ in raw]

    proposals: list[RegionProposal] = []
    overlay = image.copy()
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    for idx, (contour, (x, y, w, h), area) in enumerate(raw[:max_regions], start=1):
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        region_id = f"R{idx:03d}"
        mask_path = MASK_DIR / f"{image_stem}_{region_id}_mask.png"
        cv2.imwrite(str(mask_path), mask)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

        bbox = (x, y, x + w, y + h)
        aspect_ratio = max(w / max(h, 1), h / max(w, 1))
        perimeter = float(cv2.arcLength(contour, True))
        edge_density = _masked_mean(feature_maps.canny_edges, mask) / 255.0
        texture_score = _masked_mean(feature_maps.texture_variation, mask) / 255.0
        color_score = _masked_mean(feature_maps.color_variation, mask) / 255.0
        nearby_count = _nearby_count(centers, centers[idx - 1], max(width, height) * 0.08)
        priority = score_region(area / image_area, edge_density, texture_score, color_score, aspect_ratio, nearby_count)

        proposal = RegionProposal(
            region_id=region_id,
            bbox=bbox,
            pixel_area=area,
            relative_area=area / image_area,
            aspect_ratio=aspect_ratio,
            perimeter=perimeter,
            edge_density=edge_density,
            texture_score=texture_score,
            color_variation_score=color_score,
            priority=priority,
            mask_path=mask_path,
        )
        proposals.append(proposal)
        _draw_proposal(overlay, proposal)

    combined_mask_path = MASK_DIR / f"{image_stem}_combined_mask.png"
    overlay_path = OUTPUT_DIR / f"{image_stem}_{uuid4().hex[:8]}_region_proposals.png"
    cv2.imwrite(str(combined_mask_path), combined_mask)
    cv2.imwrite(str(overlay_path), overlay)

    proposals = _recompute_cluster_scores(proposals)
    return ProposalResult(proposals=proposals, overlay_path=overlay_path, combined_mask_path=combined_mask_path)


def _combined_binary_mask(feature_maps: FeatureMaps) -> np.ndarray:
    combined = cv2.bitwise_or(feature_maps.canny_edges, feature_maps.texture_variation)
    combined = cv2.bitwise_or(combined, feature_maps.color_variation)
    combined = cv2.bitwise_or(combined, feature_maps.threshold_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    return combined


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    selected = values[mask > 0]
    if selected.size == 0:
        return 0.0
    return float(np.mean(selected))


def _nearby_count(centers: list[tuple[float, float]], center: tuple[float, float], radius: float) -> int:
    count = 0
    cx, cy = center
    for ox, oy in centers:
        if (ox, oy) == center:
            continue
        if ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5 <= radius:
            count += 1
    return count


def _recompute_cluster_scores(proposals: list[RegionProposal]) -> list[RegionProposal]:
    centers = [((p.bbox[0] + p.bbox[2]) / 2, (p.bbox[1] + p.bbox[3]) / 2) for p in proposals]
    if not centers:
        return proposals
    span = max(max(p.bbox[2] for p in proposals), max(p.bbox[3] for p in proposals))
    updated = []
    for proposal, center in zip(proposals, centers):
        nearby = _nearby_count(centers, center, span * 0.08)
        priority = score_region(
            proposal.relative_area,
            proposal.edge_density,
            proposal.texture_score,
            proposal.color_variation_score,
            proposal.aspect_ratio,
            nearby,
        )
        updated.append(replace(proposal, priority=priority))
    return sorted(updated, key=lambda item: item.priority.score, reverse=True)


def _draw_proposal(image: np.ndarray, proposal: RegionProposal) -> None:
    x1, y1, x2, y2 = proposal.bbox
    color = (0, 180, 255) if proposal.priority.label in {"High", "Review Required"} else (70, 190, 80)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    text = f"{proposal.region_id} {proposal.priority.label} {proposal.priority.score:.1f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    top = max(0, y1 - th - 8)
    cv2.rectangle(image, (x1, top), (x1 + tw + 8, top + th + 8), color, -1)
    cv2.putText(image, text, (x1 + 4, top + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 2)


def create_region_crop(image: np.ndarray, proposal: RegionProposal) -> np.ndarray:
    x1, y1, x2, y2 = proposal.bbox
    return image[y1:y2, x1:x2].copy()
