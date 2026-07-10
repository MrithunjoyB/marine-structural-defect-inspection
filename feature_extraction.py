"""Classical computer vision feature maps for anomaly region proposal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import FEATURE_DIR


@dataclass(frozen=True)
class FeatureMaps:
    grayscale: np.ndarray
    canny_edges: np.ndarray
    sobel_gradient: np.ndarray
    laplacian: np.ndarray
    threshold_mask: np.ndarray
    contour_map: np.ndarray
    texture_variation: np.ndarray
    color_variation: np.ndarray
    anomaly_heatmap: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "Grayscale": self.grayscale,
            "Canny Edge Map": self.canny_edges,
            "Sobel Gradient Map": self.sobel_gradient,
            "Laplacian Map": self.laplacian,
            "Threshold Mask": self.threshold_mask,
            "Contour Map": self.contour_map,
            "Texture Variation Map": self.texture_variation,
            "Color Variation Mask": self.color_variation,
            "Combined Anomaly Heatmap": self.anomaly_heatmap,
        }


def extract_feature_maps(
    image: np.ndarray,
    edge_sensitivity: int = 100,
    texture_sensitivity: int = 35,
    color_sensitivity: int = 35,
    threshold_level: int = 128,
) -> FeatureMaps:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    low = max(10, int(edge_sensitivity * 0.55))
    high = max(low + 20, int(edge_sensitivity * 1.65))
    canny = cv2.Canny(blur, low, high)

    sobel_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    sobel = cv2.normalize(cv2.magnitude(sobel_x, sobel_y), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    lap = cv2.Laplacian(blur, cv2.CV_32F)
    laplacian = cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, threshold_mask = cv2.threshold(blur, threshold_level, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_map = np.zeros_like(gray)
    cv2.drawContours(contour_map, contours, -1, 255, 1)

    mean = cv2.blur(gray, (15, 15))
    texture = cv2.absdiff(gray, mean)
    texture = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, texture_mask = cv2.threshold(texture, texture_sensitivity, 255, cv2.THRESH_BINARY)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    local_mean = cv2.blur(lab, (21, 21))
    color_delta = cv2.absdiff(lab, local_mean)
    color_strength = np.sqrt(np.sum(color_delta.astype(np.float32) ** 2, axis=2))
    color_strength = cv2.normalize(color_strength, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, color_mask = cv2.threshold(color_strength, color_sensitivity, 255, cv2.THRESH_BINARY)

    heat_raw = (
        0.25 * sobel.astype(np.float32)
        + 0.25 * laplacian.astype(np.float32)
        + 0.20 * canny.astype(np.float32)
        + 0.18 * texture_mask.astype(np.float32)
        + 0.12 * color_mask.astype(np.float32)
    )
    heatmap = cv2.normalize(heat_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)

    return FeatureMaps(
        grayscale=gray,
        canny_edges=canny,
        sobel_gradient=sobel,
        laplacian=laplacian,
        threshold_mask=threshold_mask,
        contour_map=contour_map,
        texture_variation=texture_mask,
        color_variation=color_mask,
        anomaly_heatmap=heatmap,
    )


def save_feature_maps(feature_maps: FeatureMaps, image_stem: str) -> dict[str, Path]:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for name, fmap in feature_maps.as_dict().items():
        safe_name = name.lower().replace(" ", "_")
        path = FEATURE_DIR / f"{image_stem}_{safe_name}.png"
        cv2.imwrite(str(path), fmap)
        saved[name] = path
    return saved
