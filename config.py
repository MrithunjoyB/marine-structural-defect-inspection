"""Project configuration constants for StructVision-AI."""

from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_DIR = BASE_DIR / "reports"
MASK_DIR = OUTPUT_DIR / "masks"
FEATURE_DIR = OUTPUT_DIR / "feature_maps"
DATASET_DIR = BASE_DIR / "datasets"
DATASET_IMAGE_DIR = DATASET_DIR / "images"
DATASET_LABEL_DIR = DATASET_DIR / "labels"
DATASET_MASK_DIR = DATASET_DIR / "masks"

PROJECT_TITLE = "StructVision-AI"
PROJECT_SUBTITLE = "Foundation-Model-Assisted Visual Inspection and Dataset Generation for Structural Surface Anomalies"

DEFAULT_LABEL_CLASSES = [
    "corrosion_candidate",
    "crack_candidate",
    "coating_damage_candidate",
    "weld_irregularity_candidate",
    "pitting_candidate",
    "dent_candidate",
    "scratch_candidate",
    "other_surface_anomaly",
    "ignore",
]

YOLO_CLASS_NAMES = DEFAULT_LABEL_CLASSES[:-1]

for directory in [
    UPLOAD_DIR,
    OUTPUT_DIR,
    REPORT_DIR,
    MASK_DIR,
    FEATURE_DIR,
    DATASET_IMAGE_DIR,
    DATASET_LABEL_DIR,
    DATASET_MASK_DIR,
    MODEL_PATH.parent,
]:
    directory.mkdir(parents=True, exist_ok=True)
