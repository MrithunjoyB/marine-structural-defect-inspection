"""Project configuration constants."""

from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_DIR = BASE_DIR / "reports"

DEFECT_CLASSES = [
    "corrosion",
    "crack",
    "coating_damage",
    "weld_defect",
    "dent",
    "scratch",
    "deformation",
    "pitting",
    "surface_anomaly",
    "no_defect",
]

RISK_WEIGHTS = {
    "crack": 1.0,
    "weld_defect": 0.95,
    "deformation": 0.9,
    "corrosion": 0.75,
    "pitting": 0.72,
    "dent": 0.58,
    "coating_damage": 0.48,
    "scratch": 0.30,
    "surface_anomaly": 0.40,
    "no_defect": 0.0,
}

for directory in [UPLOAD_DIR, OUTPUT_DIR, REPORT_DIR, MODEL_PATH.parent]:
    directory.mkdir(parents=True, exist_ok=True)
