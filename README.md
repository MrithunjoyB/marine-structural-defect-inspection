# AI-Based Visual Inspection and Defect Severity Analysis for Marine Structural Components

Professional computer vision prototype for detecting and analyzing visual defects in marine and industrial structural components such as ship hull plates, offshore structures, pipelines, welded joints, metallic panels, and coated surfaces.

## Problem Statement

Manual inspection of marine structures is time-consuming, subjective, and difficult in corrosive or offshore environments. This project demonstrates an AI-assisted workflow that takes an inspection image, highlights suspicious defect regions, estimates severity, and generates a technical inspection report.

## Why This Matters in Ocean Engineering and Naval Architecture

Marine structures operate under corrosion, cyclic wave loading, impact, coating degradation, fatigue, and harsh inspection conditions. Early recognition of corrosion, crack-like indications, weld defects, coating damage, dents, deformation, pitting, and scratches supports maintenance planning for hulls, offshore platforms, pipelines, and welded structural details.

## Features

- Streamlit inspection dashboard with image upload, preprocessing controls, detection output, and severity metrics.
- YOLO inference through `models/best.pt` when trained weights are available.
- Classical OpenCV demo mode when trained YOLO weights are missing.
- Modular preprocessing: resize, denoise, sharpen, grayscale preview, edge preview, and CLAHE contrast enhancement.
- Severity scoring from defect type, confidence, affected area, defect count, and marine structural risk weight.
- Engineering interpretation and recommended inspection actions.
- Downloadable annotated image and PDF inspection report.

## Tech Stack

Python, OpenCV, Ultralytics YOLO, Streamlit, NumPy, Pandas, Pillow, ReportLab, and optional Matplotlib.

## Folder Structure

```text
marine-structural-defect-inspection/
├── app.py
├── config.py
├── detect.py
├── explain.py
├── preprocess.py
├── report.py
├── severity.py
├── train.py
├── data.yaml
├── requirements.txt
├── README.md
├── models/
│   └── README.md
├── sample_images/
│   └── README.md
├── uploads/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
└── reports/
    └── .gitkeep
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Upload a marine or industrial structure image. If `models/best.pt` is unavailable, the app automatically switches to Classical Computer Vision Demo Mode.

## Detection Classes

`corrosion`, `crack`, `coating_damage`, `weld_defect`, `dent`, `scratch`, `deformation`, `pitting`, `surface_anomaly`, `no_defect`

## Training Instructions

Prepare a YOLO-format dataset:

```text
data/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Then run:

```bash
python train.py --data data.yaml --model yolov8n.pt --epochs 80 --imgsz 640 --batch 8
```

Copy the best trained weight to:

```text
models/best.pt
```

## Example Output Explanation

For corrosion, the report explains that corrosion-like surface degradation may reduce effective plate thickness, weaken structural members, and accelerate fatigue damage under cyclic loading. For crack-like regions, it recommends closer inspection using dye penetrant testing, magnetic particle testing, or structural repair assessment depending on severity.

## CV-Relevant Project Highlights

- Developed an AI-assisted visual inspection prototype for marine structural defect detection.
- Integrated YOLO-based object detection with OpenCV preprocessing and fallback anomaly localization.
- Built severity scoring logic using defect type, confidence, relative affected area, and marine structural risk weights.
- Generated automated engineering inspection reports with annotated imagery and recommended actions.
- Applied computer vision to corrosion, cracks, coating damage, weld defects, deformation, pitting, dents, and scratches in marine or industrial components.

## How to Present to a Faculty Advisor

Open the Streamlit app, upload a representative image, show the annotated detection output, explain the severity score, and download the PDF report. Emphasize that the prototype separates the AI detection module, engineering severity logic, and report generation pipeline, so it can be extended with a real labeled dataset.

## CDC CV Description

AI-Based Visual Inspection and Defect Severity Analysis for Marine Structural Components: built a Streamlit and OpenCV prototype integrating YOLO-based defect detection, fallback classical CV anomaly localization, severity scoring, engineering interpretation, and automated PDF inspection reports for marine structural inspection workflows.

## Dataset Suggestions for Future Training

- Collect ship hull, welded joint, coated surface, corroded plate, pipeline, and offshore structure images.
- Label bounding boxes in YOLO format using tools such as CVAT, Roboflow, or LabelImg.
- Include both defect and no-defect examples across lighting, viewing angle, scale, and surface condition variation.
- Validate model predictions against inspection notes or expert review where possible.

## Limitations and Disclaimer

The OpenCV fallback highlights suspicious visual regions but does not provide certified defect classification. YOLO performance depends on dataset quality, annotation consistency, and field validation. This is an AI-assisted visual inspection prototype and should not replace certified marine or structural inspection.

## Future Improvements

- Train on a domain-specific labeled dataset.
- Add segmentation masks for affected area estimation.
- Calibrate pixel area to physical area using scale markers.
- Include corrosion grade estimation and thickness-loss integration.
- Add database storage for inspection history and trend analysis.
