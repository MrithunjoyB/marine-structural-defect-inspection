# StructVision-AI

**Foundation-Model-Assisted Visual Inspection and Dataset Generation for Structural Surface Anomalies**

StructVision-AI is a computer vision prototype for analyzing structural, component, product, and surface images before labeled training data is available. Instead of pretending to be a trained defect classifier, it uses classical feature extraction and visual anomaly region proposal to help a human reviewer build a labeled dataset for future YOLO or segmentation-model training.

The system remains suitable for Ocean Engineering, Naval Architecture, industrial inspection, and surface quality workflows, while staying general enough for many component or material images.

## Why This Is Not Just a YOLO Detector

Most beginner projects start and end with a detector. StructVision-AI is built around the earlier and more important stage: creating useful training data and inspection evidence when no domain-specific model exists yet.

Current stage:

```text
raw image
→ preprocessing
→ feature extraction
→ visual anomaly candidate proposal
→ segmentation-ready mask output
→ region quantification
→ visual priority scoring
→ human review and candidate labeling
→ dataset export
→ report generation
```

Future stage:

```text
reviewed dataset
→ YOLO detection or segmentation training
→ optional SAM/SAM2 mask refinement
→ trained inference comparison
→ improved inspection reporting
```

## Current Capabilities

- Streamlit tab-based UI with Overview, Image Analysis, Feature Maps, Region Proposals, Human Review, Dataset Export, Report Generation, and Future Model Training.
- Single-image and batch-image upload UI. The first selected image is analyzed; video upload is recognized as a future-ready input path.
- Preprocessing: resizing, denoising, CLAHE contrast enhancement, sharpening, grayscale conversion, and brightness/contrast adjustment.
- Feature maps: grayscale, Canny edge map, Sobel gradient map, Laplacian map, threshold mask, contour map, texture variation map, color variation mask, and combined anomaly heatmap.
- Classical CV region proposals from edge concentration, texture discontinuity, color variation, threshold masks, and contour grouping.
- Per-region measurements: region ID, bounding box, pixel area, relative image area, aspect ratio, perimeter, edge density, texture score, color variation score, and visual anomaly priority score.
- Segmentation-ready binary masks saved to `outputs/masks/`.
- Human-in-the-loop review with accept/reject, candidate label assignment, custom label entry, and reviewer notes.
- Dataset export to YOLO bounding-box text, YOLO-style segmentation polygon text, JSON annotations, CSV summary, copied images, and masks.
- Optional trained YOLO inference if `models/best.pt` exists, shown separately from classical proposals.
- Professional PDF report with preprocessing settings, feature thumbnails, region proposal overlay, region table, review labels, limitations, and future training note.

## Neutral Candidate Labels

Before a trained model is available, the app uses neutral terms such as:

- visual anomaly candidate
- extracted region
- texture discontinuity
- edge concentration
- color variation region
- surface irregularity candidate

Default human-review labels:

- `corrosion_candidate`
- `crack_candidate`
- `coating_damage_candidate`
- `weld_irregularity_candidate`
- `pitting_candidate`
- `dent_candidate`
- `scratch_candidate`
- `other_surface_anomaly`
- `ignore`

These are candidate labels for dataset creation, not trained predictions.

## Folder Structure

```text
.
├── app.py
├── config.py
├── preprocess.py
├── feature_extraction.py
├── region_proposal.py
├── scoring.py
├── labeling.py
├── dataset_export.py
├── yolo_inference.py
├── report.py
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
│   ├── .gitkeep
│   ├── feature_maps/
│   │   └── .gitkeep
│   └── masks/
│       └── .gitkeep
├── reports/
│   └── .gitkeep
└── datasets/
    ├── images/
    │   └── .gitkeep
    ├── labels/
    │   └── .gitkeep
    └── masks/
        └── .gitkeep
```

Legacy modules from the earlier prototype may remain for backward compatibility, but the main architecture is the StructVision-AI module set above.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The app works even when:

- no trained YOLO model exists
- SAM/SAM2 is not installed
- no labeled dataset exists

The default working mode is classical CV feature extraction, anomaly region proposal, human review, and dataset export.

## Annotation And Export Workflow

1. Upload one or more images in the sidebar.
2. Tune preprocessing and proposal filters.
3. Click **Analyze Selected Image**.
4. Inspect feature maps and region proposals.
5. Open **Human Review / Labeling**.
6. Accept or reject candidate regions.
7. Assign candidate labels or custom labels.
8. Save review metadata.
9. Export dataset files from **Dataset Export**.

Exported dataset structure:

```text
datasets/
├── images/
├── labels/
├── masks/
├── annotations.json
├── dataset_summary.csv
└── data.yaml
```

## YOLO Training Workflow

After enough reviewed annotations are exported:

```bash
python train.py --data datasets/data.yaml --task detect --model yolo11n.pt --epochs 80 --imgsz 640
```

For segmentation-style training:

```bash
python train.py --data datasets/data.yaml --task segment --model yolo11n-seg.pt --epochs 80 --imgsz 640
```

The script copies the best trained checkpoint to:

```text
models/best.pt
```

When that file exists, the Streamlit app enables trained YOLO inference and displays those predictions separately from classical region proposals.

## SAM/SAM2 Future Integration

SAM or SAM2 can be added later by using proposed bounding boxes as prompts and replacing rectangular or contour masks with refined segmentation masks. The current app does not require SAM and will not fail if SAM is absent.

## Report Generation

The PDF report includes:

- project title
- image filename
- analysis timestamp
- preprocessing settings
- feature map thumbnails
- highlighted region proposal image
- region summary table
- visual anomaly priority scores
- accepted/rejected review labels when available
- limitations
- future model training note

## Demo Screenshot Placeholders

Add screenshots here after running the app:

```text
docs/screenshots/overview.png
docs/screenshots/feature_maps.png
docs/screenshots/region_proposals.png
docs/screenshots/dataset_export.png
```

## Limitations

- Classical CV proposals identify visually significant candidate regions, not certified defects.
- Region quality depends on lighting, viewpoint, texture, surface cleanliness, and filter settings.
- YOLO inference requires a trained `models/best.pt`.
- Segmentation polygon export currently uses bounding-box polygons unless refined masks are added later.
- Video frame extraction is a planned extension, not an automatic processing path in the current app.

## Future Roadmap

- Add video frame extraction and frame sampling.
- Add SAM/SAM2 prompt-based mask refinement.
- Add active-learning loops for repeated review and retraining.
- Add multi-image batch processing and dataset merge tools.
- Add physical scale calibration for area estimates.
- Add inspection-history database support.
- Add side-by-side YOLO-vs-proposal matching metrics.

## CV-Relevant Project Description

Built StructVision-AI, a visual inspection and dataset-generation system that performs preprocessing, classical feature extraction, anomaly candidate proposal, segmentation-ready mask creation, visual priority scoring, human-in-the-loop labeling, YOLO-format dataset export, optional trained YOLO inference, and automated PDF reporting for structural or surface inspection images.
