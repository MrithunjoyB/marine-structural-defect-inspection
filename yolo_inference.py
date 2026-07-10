"""Optional trained YOLO inference for comparison with region proposals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from config import MODEL_PATH, OUTPUT_DIR, YOLO_CLASS_NAMES


@dataclass(frozen=True)
class YoloPrediction:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]

    def to_row(self) -> dict[str, object]:
        return {"Label": self.label, "Confidence": round(self.confidence, 3), "BBox": self.bbox}


@dataclass(frozen=True)
class YoloInferenceResult:
    available: bool
    predictions: list[YoloPrediction]
    annotated_path: Path | None
    message: str


def run_yolo_inference(image: np.ndarray, image_stem: str, confidence_threshold: float = 0.35) -> YoloInferenceResult:
    if not MODEL_PATH.exists():
        return YoloInferenceResult(
            available=False,
            predictions=[],
            annotated_path=None,
            message="No trained YOLO model found at models/best.pt. Classical proposals remain fully available.",
        )

    try:
        from ultralytics import YOLO
    except Exception as exc:
        return YoloInferenceResult(
            available=False,
            predictions=[],
            annotated_path=None,
            message=f"Ultralytics is not available in this environment: {exc}",
        )

    model = YOLO(str(MODEL_PATH))
    results = model.predict(image, conf=confidence_threshold, verbose=False)
    annotated = image.copy()
    predictions: list[YoloPrediction] = []
    height, width = image.shape[:2]

    for result in results:
        names = result.names or {idx: name for idx, name in enumerate(YOLO_CLASS_NAMES)}
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            bbox = (max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2))
            label = str(names.get(cls_id, YOLO_CLASS_NAMES[cls_id] if cls_id < len(YOLO_CLASS_NAMES) else "trained_prediction"))
            predictions.append(YoloPrediction(label=label, confidence=confidence, bbox=bbox))
            _draw_prediction(annotated, bbox, label, confidence)

    output_path = OUTPUT_DIR / f"{image_stem}_{uuid4().hex[:8]}_yolo_predictions.png"
    cv2.imwrite(str(output_path), annotated)
    return YoloInferenceResult(True, predictions, output_path, "Trained YOLO inference completed.")


def _draw_prediction(image: np.ndarray, bbox: tuple[int, int, int, int], label: str, confidence: float) -> None:
    x1, y1, x2, y2 = bbox
    color = (220, 80, 30)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {confidence:.2f}"
    cv2.putText(image, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
