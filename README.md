# StructVision-AI

**Foundation-Model-Assisted Visual Inspection and Dataset Generation for Structural Surface Anomalies**

StructVision-AI is a computer vision prototype for analyzing structural, component, product, and surface images before labeled training data is available. Instead of pretending to be a trained defect classifier, it uses classical feature extraction and visual anomaly region proposal to help a human reviewer build a labeled dataset for future YOLO or segmentation-model training.

## Problem Definition

The research problem is proposal generation under weak prior knowledge: identify compact, reviewable surface regions that differ from their local structural context without claiming a defect class. Visual saliency and anomaly detection are not equivalent. A weld bead, plate boundary, or repeating texture may be highly salient but normal; an anomaly must differ statistically from an appropriate surrounding context.

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
- Feature maps: grayscale, Canny, Sobel, Laplacian, local variance, local binary pattern (LBP), Lab color deviation, foreground threshold, contours, and a continuous fused anomaly heatmap.
- Overlapping patch scoring at three spatial scales using edge density, gradient, Laplacian response, texture variance, Lab deviation, entropy, and local contrast difference.
- Adaptive percentile thresholds, morphological processing, connected components, similarity-aware region merging, overlap suppression, and configurable relative-area filtering.
- Per-region stability under brightness, contrast, Gaussian-noise, and resize perturbations.
- Per-region measurements and explanations, padded review crops, mask/heatmap crops, automatic diagnostics, and baseline comparison overlays.
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
├── evaluation.py
├── experiment_tracking.py
├── research_evaluation.py
├── synthetic_benchmark.py
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

## Contextual Region-Proposal Methodology

The proposal engine keeps feature evidence separate long enough to avoid dependence on one contour mask. It creates independent percentile masks for Canny edges, Sobel magnitude, Laplacian response, local intensity variance, LBP deviation, Lab color difference, foreground segmentation, and the fused heatmap. Overlapping tiles are evaluated at approximately 8%, 16%, and 28% of the shorter image dimension. Their normalized measurements are projected back to a dense patch-score map.

The dense feature and tile evidence is thresholded by image-specific percentiles. Three morphological scales preserve small spots, elongated lines, and broad irregular regions. Connected components are filtered by relative area and border/specular evidence, then merged using bounding-box IoU, center distance, morphological connectivity, and texture/color similarity. Large low-coherence masks are split from internal heatmap peaks. Every surviving region is compared with a dilated context ring using local texture, Lab colour, entropy, gradient, and internal-versus-boundary edge contrasts.

The candidate lifecycle is explicit and monotonic where required:

```text
raw components
→ area and border filtering
→ coherence splitting and mask refinement
→ overlap/nesting merge
→ non-maximum overlap suppression
→ contextual ranking sanity filters
→ top-K selection (default: 8)
```

Diagnostics report every stage separately. Runtime assertions enforce that merging cannot increase the split count, the final count cannot exceed top-K, every final mask is non-empty, and every exported proposal box is derived from its final refined mask. The optional debug panel shows stage overlays, rejection reasons, and counts removed by area, border, and overlap rules.

### Three-Score Architecture

Anomaly evidence is deliberately separate from segmentation reliability:

```text
E = 100 * (w_t ΔT + w_c ΔC + w_h ΔH + w_e E_i + w_g ΔG + w_q Q_g) / Σw_E
R = 100 * (v_s S_p + v_c C_n + v_b B_s + v_a A_s + v_q Q_s) / Σv_R
P = 100 * (u_e E/100 + u_r R/100 + u_a A_r + u_n N) / Σu_P
```

`ΔT`, `ΔC`, `ΔH`, and `ΔG` are candidate-to-context differences; `E_i` is internal edge concentration and `Q_g` is geometric irregularity. Reliability uses perturbation stability `S_p`, connectedness `C_n`, boundary smoothness `B_s`, scale agreement `A_s`, and segmentation coherence `Q_s`. Priority combines evidence, reliability, area relevance `A_r`, and contextual novelty `N`. Candidate features are robustly calibrated by median/IQR with clipping. Configurable defaults live in `scoring.py`; stability cannot directly dominate anomaly evidence.

### Border Suppression And Mask Refinement

The valid-image model detects black/near-uniform letterbox bands and applies a configurable exclusion margin. Boundary occupancy produces a border penalty, while thin frame-parallel regions are suppressed. Genuine edge-touching regions can remain when their internal evidence is coherent.

Raw masks are refined with bilateral heatmap filtering, adaptive/local percentile thresholds, opening/closing, hole filling, and small-component removal. Reports include area reduction, scale agreement, coherence, fragmentation, solidity-derived smoothness, raw masks, refined masks, and context rings. Reviewers can adjust boxes, select raw/refined masks, erode or dilate, remove small components, and invert a bounded mask before saving the corrected reference.

Final proposal masks retain the dominant connected component, fill only small enclosed holes, smooth short boundary irregularities, and recompute their bounding boxes after cleanup. Broad regions with low internal heatmap coherence are split or rejected. Strong containment and overlap suppression prevents nested duplicates from reaching the final ranking.

## Baseline Comparison

The Region Proposals tab reports four definitions: contour-only (Canny contours), fixed-threshold (`heatmap > 128`), raw multi-scale fused masks, and refined contextual multi-scale masks. It also displays component counts, adaptive threshold, score distributions, evidence/reliability/priority scores, feature contributions, border penalty, and coherence.

## Proposal Evaluation

`evaluation.py` treats accepted, intentionally labelled, manually corrected regions as references. Per-image and dataset tables include recall at IoU 0.10/0.25/0.50, average best IoU, mask Dice/IoU, false and accepted proposals per image, acceptance rate, area over/under-coverage, correction count, and estimated review time. CSV export is available in Dataset Export.

## Annotation-Efficiency Experiment Protocol

The **Research Evaluation** tab measures whether ranked proposals reduce reviewer effort. Review timing starts when image analysis completes and stops when review metadata is saved. Each record includes experiment ID, reviewer ID, image, method, final proposal count, accept/reject/uncertain counts, image-level outcome, timestamps, duration, first useful rank, and Top-1/3/5/8 indicators.

Method-level records are stored persistently in `outputs/research_evaluation.sqlite3`. Each row has a UUID `record_id` and is unique by experiment ID, experiment version, reviewer ID, image filename, and method. Duplicate previews can be cancelled, overwritten deliberately, or saved as a new experiment version. New records default to `Development / Test`; final dashboards exclude development records unless **Include development records** is enabled.

Only the refined contextual method reviewed in **Human Review / Labeling** receives accepted, rejected, and uncertain counts. Baselines use `review_status = not_reviewed`, null decision counts, and `not_reviewed = final_proposals`. A not-reviewed proposal is never treated as rejected or false. Review status is one of `fully_reviewed`, `partially_reviewed`, or `not_reviewed`.

An accepted reviewed region is treated as a reference true-anomaly proposal for the experiment. For a method with ranked proposals `p_1 ... p_K`, a reference is found at rank `k` when `IoU(p_k, reference) >= 0.10`. Top-K proposal recall is:

```text
Top-K recall = anomaly-present images with first useful rank <= K
               -------------------------------------------------
                         all anomaly-present images
```

Images marked `no anomaly` or `uncertain` are excluded from the Top-K recall denominator. They remain in proposal-burden and timing summaries.

Final quantitative recall also excludes records with `ground_truth_status = unknown` unless the reviewer explicitly enables the caution-labelled recall override. Undefined recall, acceptance, review-time, and first-useful metrics are displayed as N/A rather than zero. Annotation acceptance is `accepted / (accepted + rejected)` and excludes uncertain and not-reviewed proposals. False proposals per image are explicitly rejected proposals from manually reviewed methods only.

Dataset-level annotation-efficiency metrics include mean accepted proposals per image, mean false/rejected proposals per image, annotation acceptance rate, mean review time, and mean proposals reviewed before the first useful region. Results compare contour-only, fixed-threshold, raw multi-scale fused, and refined contextual methods and can be exported as CSV or JSON.

Recommended procedure:

1. Choose a stable experiment ID for one protocol/configuration and a consistent reviewer ID.
2. Analyze one image without inspecting baseline comparison results first.
3. Review proposals in rank order and mark each accept, reject, or uncertain.
4. Save review metadata to close the timer.
5. Set the image outcome to anomaly present, no anomaly, or uncertain.
6. Record the Research Evaluation row set.
7. Repeat across the dataset and reviewers without changing proposal parameters mid-experiment.
8. Export CSV/JSON and compare Top-K recall together with review time and false-proposal burden.

The baseline rankings use mean anomaly-heatmap evidence and are capped at eight proposals, matching the default refined-method review budget. This is an annotation-efficiency experiment, not a clinical or structural-safety validation.

### Manage Experiments

The Research Evaluation page supports filtering by experiment, reviewer, image, method, status, and date range. Selected or filtered records can be exported as CSV or JSON before deletion. Confirmation is required to delete selected rows, one experiment, one image, all development records, all records, or reset the SQLite store. Every deletion reports the row count and affected experiment IDs, then recomputes summaries and charts.

The **Legacy Record Migration** section detects historical JSON rows missing `record_id`, `review_status`, `not_reviewed`, or `experiment_status`. Reviewers may keep them unchanged, explicitly migrate selected rows into SQLite with corrected baseline semantics, or confirm deletion from the legacy JSON. Historical records are never changed silently.

## Ablation Design

The sidebar can disable edge, texture, colour, entropy, stability, contextual contrast, multi-scale fusion, region merging, or mask refinement. Each run appends its switches and diagnostics to `outputs/ablation_results.csv`, enabling controlled comparisons without changing the export contract.

## Synthetic Benchmark

`synthetic_benchmark.py` generates known masks for cracks, weld disturbances, pitting clusters, colour-only and texture-only anomalies, normal texture, illumination gradients, black borders, specular highlights, and noise/blur. It evaluates all four proposal methods automatically:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python synthetic_benchmark.py
```

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
- Percentile thresholds are image-relative, so a uniformly damaged image or a frame with no normal background can still be ambiguous.
- Perturbation stability measures repeatability of classical saliency, not physical defect persistence.
- Synthetic tests exercise geometry and nuisance conditions but do not replace evaluation on reviewed field imagery.
- Context rings can contain another anomaly or cross a material boundary, biasing local contrast.
- Heatmap-peak splitting is heuristic and may divide one heterogeneous defect or retain a weak bridge.
- Review-time estimates require multiple reliable timestamps and do not measure reviewer cognitive effort.
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
