"""YOLO training entry point for a custom marine structural defect dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def train(data_yaml: Path, model_name: str, epochs: int, image_size: int, batch: int, project: str) -> None:
    model = YOLO(model_name)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        project=project,
        name="marine_defect_yolo",
        patience=25,
        plots=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO for marine structural defect detection.")
    parser.add_argument("--data", type=Path, default=Path("data.yaml"), help="Path to YOLO dataset YAML.")
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLO model checkpoint.")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Training batch size.")
    parser.add_argument("--project", default="runs", help="Training output directory.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data, args.model, args.epochs, args.imgsz, args.batch, args.project)
