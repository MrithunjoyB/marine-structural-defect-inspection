"""Defect detection using YOLO when available and OpenCV fallback otherwise."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import cv2
import numpy as np

from config import DEFECT_CLASSES, MODEL_PATH


@dataclass(frozen=True)
class Defect:
    defect_type: str
    confidence: float
    bbox: tuple[int, int, int, int]
    area_px: int
    relative_area: float

    def to_table_row(self) -> dict[str, str | float | int]:
        x1, y1, x2, y2 = self.bbox
        return {
            "Defect Type": self.defect_type,
            "Confidence": round(self.confidence, 3),
            "Bounding Box": f"({x1}, {y1}) - ({x2}, {y2})",
            "Area (px)": self.area_px,
            "Relative Area (%)": round(self.relative_area * 100, 2),
        }


@dataclass(frozen=True)
class DetectionResult:
    mode: str
    defects: list[Defect]
    annotated_image_path: Path
    message: str = ""


def run_detection(
    image: np.ndarray,
    original_name: str,
    output_dir: Path,
    confidence_threshold: float = 0.35,
    force_classical: bool = False,
) -> DetectionResult:
    """Run YOLO detection when weights exist, otherwise use classical CV fallback."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and not force_classical:
        try:
            return _run_yolo(image, original_name, output_dir, confidence_threshold)
        except Exception as exc:
            message = f"YOLO inference failed, switched to classical CV demo mode. Reason: {exc}"
            return _run_classical_cv(image, original_name, output_dir, message)

    message = (
        "models/best.pt was not found. Running Classical Computer Vision Demo Mode. "
        "Accurate defect classification requires a trained YOLO model."
    )
    if force_classical:
        message = (
            "Classical Computer Vision Demo Mode selected. Accurate defect classification "
            "requires a trained YOLO model."
        )
    return _run_classical_cv(image, original_name, output_dir, message)


def _run_yolo(image: np.ndarray, original_name: str, output_dir: Path, confidence_threshold: float) -> DetectionResult:
    from ultralytics import YOLO

    model = YOLO(str(MODEL_PATH))
    results = model.predict(image, conf=confidence_threshold, verbose=False)
    annotated = image.copy()
    defects: list[Defect] = []
    height, width = image.shape[:2]
    image_area = max(height * width, 1)

    for result in results:
        names = result.names or {idx: name for idx, name in enumerate(DEFECT_CLASSES)}
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)
            area = max(0, x2 - x1) * max(0, y2 - y1)
            defect_type = str(names.get(cls_id, DEFECT_CLASSES[cls_id] if cls_id < len(DEFECT_CLASSES) else "surface_anomaly"))
            if defect_type == "no_defect":
                continue
            defects.append(Defect(defect_type, confidence, (x1, y1, x2, y2), area, area / image_area))
            _draw_box(annotated, (x1, y1, x2, y2), defect_type, confidence, (35, 70, 220))

    output_path = _write_output(annotated, original_name, output_dir)
    return DetectionResult("YOLO Object Detection", defects, output_path)


def _run_classical_cv(image: np.ndarray, original_name: str, output_dir: Path, message: str) -> DetectionResult:
    annotated = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 160)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = image.shape[:2]
    image_area = max(height * width, 1)
    defects: list[Defect] = []

    for contour in _rank_contours(contours, image_area)[:12]:
        x, y, w, h = cv2.boundingRect(contour)
        contour_area = int(cv2.contourArea(contour))
        if contour_area <= 0:
            continue
        aspect_ratio = max(w / max(h, 1), h / max(w, 1))
        fill_ratio = contour_area / max(w * h, 1)
        defect_type = _classify_classical_region(aspect_ratio, fill_ratio, contour_area / image_area)
        confidence = min(0.85, 0.35 + (aspect_ratio / 12.0) + min(contour_area / image_area, 0.2))
        bbox = (x, y, x + w, y + h)
        defects.append(Defect(defect_type, confidence, bbox, contour_area, contour_area / image_area))
        _draw_box(annotated, bbox, defect_type, confidence, (0, 140, 255))

    if not defects:
        cv2.putText(annotated, "No strong surface anomaly detected", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 2)

    output_path = _write_output(annotated, original_name, output_dir)
    return DetectionResult("Classical Computer Vision Demo Mode", defects, output_path, message)


def _rank_contours(contours: Iterable[np.ndarray], image_area: int) -> list[np.ndarray]:
    candidates = []
    min_area = max(80, int(image_area * 0.00025))
    max_area = int(image_area * 0.35)
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            candidates.append(contour)
    return sorted(candidates, key=cv2.contourArea, reverse=True)


def _classify_classical_region(aspect_ratio: float, fill_ratio: float, relative_area: float) -> str:
    if aspect_ratio > 5.5 and fill_ratio < 0.45:
        return "crack"
    if relative_area > 0.035 and fill_ratio > 0.35:
        return "corrosion"
    if fill_ratio < 0.22:
        return "pitting"
    if aspect_ratio > 3.0:
        return "scratch"
    return "surface_anomaly"


def _draw_box(image: np.ndarray, bbox: tuple[int, int, int, int], label: str, confidence: float, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y_text = max(0, y1 - th - 8)
    cv2.rectangle(image, (x1, y_text), (x1 + tw + 8, y_text + th + 8), color, -1)
    cv2.putText(image, text, (x1 + 4, y_text + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def _write_output(image: np.ndarray, original_name: str, output_dir: Path) -> Path:
    stem = Path(original_name).stem.replace(" ", "_")[:48] or "inspection"
    output_path = output_dir / f"{stem}_{uuid4().hex[:8]}_annotated.png"
    cv2.imwrite(str(output_path), image)
    return output_path
