"""Export reviewed candidate annotations to future training dataset formats."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from config import DATASET_DIR, DATASET_IMAGE_DIR, DATASET_LABEL_DIR, DATASET_MASK_DIR, DEFAULT_LABEL_CLASSES
from labeling import ReviewedAnnotation


def export_dataset(
    image_path: Path,
    image_shape: tuple[int, int],
    annotations: list[ReviewedAnnotation],
    label_classes: list[str] | None = None,
) -> dict[str, Path]:
    """Export accepted reviewed annotations in CSV, JSON, YOLO box, and YOLO segment formats."""

    label_classes = label_classes or [name for name in DEFAULT_LABEL_CLASSES if name not in {"ignore", "unassigned"}]
    invalid = [ann.region_id for ann in annotations if ann.accepted and ann.label in {"", "unassigned", "ignore"}]
    if invalid:
        raise ValueError(f"Accepted regions require an intentional label before export: {', '.join(invalid)}")
    DATASET_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_MASK_DIR.mkdir(parents=True, exist_ok=True)

    copied_image = DATASET_IMAGE_DIR / image_path.name
    shutil.copy2(image_path, copied_image)

    accepted = [ann for ann in annotations if ann.accepted and ann.label not in {"ignore", "unassigned"}]
    image_height, image_width = image_shape
    stem = image_path.stem
    yolo_box_path = DATASET_LABEL_DIR / f"{stem}.txt"
    yolo_seg_path = DATASET_LABEL_DIR / f"{stem}_segments.txt"

    box_lines = []
    seg_lines = []
    for ann in accepted:
        if ann.label not in label_classes:
            label_classes.append(ann.label)
        class_id = label_classes.index(ann.label)
        x1, y1, x2, y2 = ann.bbox
        xc = ((x1 + x2) / 2) / image_width
        yc = ((y1 + y2) / 2) / image_height
        bw = (x2 - x1) / image_width
        bh = (y2 - y1) / image_height
        box_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        seg_lines.append(
            f"{class_id} {x1 / image_width:.6f} {y1 / image_height:.6f} "
            f"{x2 / image_width:.6f} {y1 / image_height:.6f} "
            f"{x2 / image_width:.6f} {y2 / image_height:.6f} "
            f"{x1 / image_width:.6f} {y2 / image_height:.6f}"
        )
        mask_source = Path(ann.mask_path)
        if mask_source.exists():
            shutil.copy2(mask_source, DATASET_MASK_DIR / f"{stem}_{ann.region_id}_mask.png")

    yolo_box_path.write_text("\n".join(box_lines) + ("\n" if box_lines else ""), encoding="utf-8")
    yolo_seg_path.write_text("\n".join(seg_lines) + ("\n" if seg_lines else ""), encoding="utf-8")

    annotations_json = DATASET_DIR / "annotations.json"
    existing = []
    if annotations_json.exists():
        try:
            existing = json.loads(annotations_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []
    existing.extend([ann.to_dict() for ann in annotations])
    annotations_json.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    summary_path = DATASET_DIR / "dataset_summary.csv"
    summary_rows = [ann.to_dict() for ann in existing]
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    data_yaml = DATASET_DIR / "data.yaml"
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(label_classes))
    data_yaml.write_text(
        f"path: {DATASET_DIR.as_posix()}\ntrain: images\nval: images\n\nnames:\n{names}\n",
        encoding="utf-8",
    )

    return {
        "image": copied_image,
        "yolo_boxes": yolo_box_path,
        "yolo_segments": yolo_seg_path,
        "annotations_json": annotations_json,
        "summary_csv": summary_path,
        "data_yaml": data_yaml,
    }
