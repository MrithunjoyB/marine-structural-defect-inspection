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
├── research_dataset.py
├── dataset_intake.py
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

## Research Dataset Intake

The **Research Dataset Intake** page registers and validates real inspection data before an experiment. It supports one image, image batches, ZIP archives, optional annotations, controlled synthetic generation, reference-ground-truth review, deterministic split preparation, validation exports, and dataset manifest export.

Required registration captures dataset ID/name/version, source type and reference, provider, licence and usage permissions, citation, acquisition date, domain, ground-truth quality, annotation format, and notes. Supported annotation formats are YOLO boxes, YOLO segmentation, COCO JSON, Pascal VOC, binary masks, CSV regions, and custom formats. The current automated bounds checks cover normalized YOLO coordinates, non-empty masks, and duplicate COCO annotation IDs; other formats remain visible for explicit validation and review.

**Do not commit professor-provided, private, restricted, or unlicensed images to the public GitHub repository.**

Professor-provided data should be registered as `professor-provided`, stored only through the ignored `research_data/raw/` runtime area, assigned its supplied licence/restrictions, and used in Development/Test mode until provenance and ground truth permit final evaluation. Unknown or restricted licences are blocked from public research export unless a reviewer records an explicit warning override.

### Research Data Structure

```text
research_data/
├── registry/
│   ├── datasets.sqlite          # ignored runtime registry
│   ├── dataset_manifest.json    # ignored generated manifest
│   └── schema.json              # Git-tracked schema
├── raw/<dataset_id>/
├── processed/<dataset_id>/
├── annotations/<dataset_id>/
├── splits/<dataset_id>/
├── reports/<dataset_id>/
└── exports/
```

Runtime datasets, SQLite files, annotations, reports, split manifests, exports, and image formats are ignored by Git. Only documentation, schemas, templates, and deliberately reviewed small synthetic assets may be tracked.

### Manifest And Validation

Every image receives an immutable image ID plus dataset ID, original/stored names, SHA-256, dimensions, channels, format, byte size, source, licence, ground-truth status, annotation path, split, exact/near-duplicate status, corruption status, import timestamp, and notes. SHA-256 detects renamed exact duplicates; perceptual hashes flag possible near duplicates.

Validation reports include total/valid/corrupt files, exact and possible near duplicates, annotation coverage, class and size distributions, missing annotations, and invalid annotations. Reports are downloadable as CSV and JSON.

### Splits And Leakage

The default split requests 70/15/15 with a deterministic seed, but allocation is group-aware and stratified by anomaly type, positive/clean outcome, and clean artefact subtype. Exact duplicates are excluded; near-duplicate, source, and template groups remain indivisible. Finalization previews image/outcome/category/group composition and blocks leakage or a test set without positive, negative, and category diversity unless an explicit warning override is recorded.

**Balanced Synthetic Benchmark** uses constrained allocation for the 33-image controlled dataset. With seed 42 it produces 15 train, 6 validation, and 12 test images: test contains `thin_crack`, `pitting_cluster`, `normal_texture`, and `specular_highlights` template groups. The larger-than-15% test split is intentional so representative positive and clean categories remain present without splitting related templates.

### Experiment Reproducibility

Research Evaluation can create an experiment from a registered dataset/version/split with subset size, reviewer, methods, status, parameters, and random seed. The saved JSON records the manifest hash, selected image IDs, preprocessing/proposal/weight/threshold settings, border margin, maximum regions, ablations, Git commit, Python and package versions, operating system, and creation time.

Creating a plan freezes selection and configuration but does not analyze images. Use **Execute Registered Dataset Experiment** to batch-run the selected contour-only, fixed-threshold, multi-scale fused, and refined contextual methods without a sidebar upload. Execution calls the existing feature extraction and proposal pipeline directly, loads registered exact masks, and saves one automatic row for every image-method pair.

Automatic matching accepts a proposal when it reaches the configured bounding-mask IoU or ground-truth overlap threshold. A centroid-inside-ground-truth fallback supports very thin anomalies. Top-K proposal recall counts a positive image as a hit when at least one of its first K ranked proposals matches a verified anomaly. Proposal precision is matched proposals divided by all proposals; proposal recall is matched ground-truth instances divided by all ground-truth instances; false proposals are unmatched final proposals. Clean images are excluded from positive Top-K denominators, and every proposal on a clean image is false.

Automatic rows use `review_status = automatically_evaluated` and remain separate from human-review records. Executions move through planned, running, completed, partially completed, failed, or cancelled states. Resume skips completed image-method pairs, retry targets failed pairs, overwrite explicitly replaces completed pairs, and create-new-version preserves prior runs. Results, configuration, selected manifest, and method summaries are downloadable as CSV or JSON; **Open Selected Test Image** displays the original, exact mask, matched/unmatched proposals, ranks, and IoU values.

Experiment plans may select anomaly-present images, clean images, specific anomaly or clean artefact types, or a balanced positive/negative subset. Clean-only runs are labeled **False-positive robustness evaluation**; precision and false-proposal metrics remain valid, while recall is explicitly undefined and empty recall charts are suppressed.

### Research Analysis Browser

Manage Experiments provides persistent free-text search, categorical and numeric filters, quick comparisons, sorting, column presets, selected-image method comparison, expandable visual evidence, and filtered CSV/JSON exports. Filters use stable Streamlit session keys and remain active across reruns and page navigation. Category-wise Evaluation separates anomaly recall from clean false-alarm robustness and never substitutes zero for undefined recall.

**Multi-scale Fused vs Refined Contextual** pairs methods on identical experiment images, reports per-image metric differences and category-specific win/tie/loss counts, and generates a conservative interpretation. Deterministic bootstrap intervals are descriptive; the interface warns that controlled synthetic results do not establish real-world marine inspection performance.

### Ablation Study

The ablation framework defines stable IDs for the full refined method and removals of texture, colour, entropy, stability, boundary-edge, border, coherence, contextual reranking, multi-scale fusion, and mask refinement components. Configurations are thin `AblationConfig` overlays on the existing proposal pipeline. The full configuration equals the normal default exactly, so existing experiments and proposal behavior are unchanged.

Each ablation snapshot records enabled components, thresholds, weights, seed, code commit, dataset manifest hash, experiment identity, selected images, and matching thresholds. Comparisons require aligned manifests, images, splits, thresholds, and seeds. Leaderboard and contribution tables describe empirical benchmark differences rather than causal effects.

Development/Test experiments may use local development images and unknown ground truth and remain excluded from final summaries by default. Final Research Evaluation requires a registered dataset, verified or reviewer-estimated ground truth, source/licence metadata, and a configuration snapshot; unknown provenance or licence is blocked unless explicitly overridden with a warning.

### Synthetic Intake

The controlled generator creates balanced thin cracks, elongated weld disturbances, pitting clusters, colour-only and texture-only anomalies, clean texture, illumination gradients, border artefacts, specular highlights, blur, and Gaussian noise. A deterministic master seed derives a unique seed for every image; geometry, intensity, orientation, texture, background, and noise vary per sample. Each image is registered with an exact mask and complete generation parameters. Generation asserts that image-file SHA-256 values are unique unless duplication is explicitly requested.

Image SHA-256 is computed only from encoded image-file bytes. Annotation hashes, metadata, and filenames do not determine image duplicates. The intake dashboard reports exact image duplicates, near duplicates, duplicate groups, unique images, and split-eligible images separately. Exact duplicates are excluded by default; perceptually similar and same-template synthetic variants remain available but are forced into one split to prevent leakage.

### Dataset Management And Deletion

Use **Manage Research Datasets** to preview and delete one registered version, every version of a Dataset ID, generated files, synthetic datasets, or the Development/Test store. Metadata-only deletion removes registry and manifest records while preserving owned files. Generated-files-only deletion preserves registration and raw data but removes processed outputs, splits, and reports. Complete deletion removes registry records and moves all dataset-owned files to `research_data/.trash/<timestamp>/<dataset_id>/` first.

Every destructive operation requires a confirmation checkbox, the exact Dataset ID, and a final delete button. Cleanup previews list files, bytes, database rows, manifests, splits, and linked experiment IDs. Linked experiments block complete deletion by default; development experiments may be explicitly unlinked or cascade deleted, while final research experiments are never silently deleted or unlinked.

Trash audit records preserve the reviewer, timestamp, mode, moved files, and removed records. Use **Restore deleted dataset** to restore files and metadata, or explicitly confirm **Permanently empty dataset trash**. Synthetic data can be cleared and regenerated with replacement or a new version. Streamlit cache clearing does not delete registered datasets or their files.

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
