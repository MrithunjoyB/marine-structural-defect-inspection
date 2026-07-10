"""Future-ready Ultralytics YOLO training entry point.

Examples:
    python train.py --data datasets/data.yaml --task detect --model yolo11n.pt --epochs 80 --imgsz 640
    python train.py --data datasets/data.yaml --task segment --model yolo11n-seg.pt --epochs 80 --imgsz 640
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def train_yolo(data_yaml: Path, task: str, model_name: str, epochs: int, image_size: int, batch: int) -> None:
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}. Export reviewed annotations first.")

    from ultralytics import YOLO

    model = YOLO(model_name)
    results = model.train(
        data=str(data_yaml),
        task=task,
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        project="runs",
        name=f"structvision_yolo_{task}",
        patience=25,
        plots=True,
    )

    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    if best.exists():
        target = Path("models") / "best.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, target)
        print(f"Saved trained model to {target}")
    else:
        print("Training finished, but best.pt was not found in the expected output directory.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train future YOLO detection/segmentation models for StructVision-AI.")
    parser.add_argument("--data", type=Path, default=Path("datasets/data.yaml"), help="Path to exported dataset YAML.")
    parser.add_argument("--task", choices=["detect", "segment"], default="detect", help="YOLO task type.")
    parser.add_argument("--model", default="yolo11n.pt", help="Base Ultralytics model checkpoint.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_yolo(args.data, args.task, args.model, args.epochs, args.imgsz, args.batch)
