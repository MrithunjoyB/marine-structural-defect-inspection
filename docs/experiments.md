# Experiments

StructVision-AI separates experiment planning, execution, persistent results, and analysis.

## Experiment Identity

A registered experiment plan contains:

- experiment and plan IDs;
- experiment version and status;
- dataset ID, version, and split;
- selected image IDs;
- manifest hash;
- proposal methods;
- preprocessing and proposal parameters;
- matching thresholds;
- subset filters;
- random seed;
- code commit hash;
- Python, package, and operating-system metadata; and
- reviewer identity.

An experiment plan freezes intent and selection but performs no image analysis. Execution loads the selected registered images and runs each configured image-method pair. One automatic result row is stored per experiment/version/image/method.

## Registered Batch Execution

The batch engine calls the existing feature extraction and proposal pipeline. Synthetic registered images load exact masks from their annotation paths. Baselines and refined proposals are evaluated through a common matching function.

Execution states are `planned`, `running`, `completed`, `partially_completed`, `failed`, and `cancelled`. Progress reports the current image and method, completed and total pairs, elapsed time, and estimated remaining time.

## Resume And Duplicate Protection

Completed image-method pairs are identified by stable keys and skipped during resume. Failed pairs can be retried separately. Existing pairs are not silently recomputed; reviewers choose resume, overwrite, or a new execution version. Cancellation takes effect at a pair boundary.

Deleting automatic results removes the selected execution rows but preserves the registered plan. Manual human-review records remain in a separate store and use different review-status semantics.

## Subset Selection

Plans may select:

- all images in a registered split;
- anomaly-present images only;
- clean/no-anomaly images only;
- selected anomaly categories;
- selected clean artefact types; or
- a balanced positive/negative subset.

Subset selection and random sampling are recorded in the plan. Scientific comparison should use identical selected image IDs, dataset version, split, thresholds, and seed.

## Automatic Ground-Truth Evaluation

Proposal masks are matched to connected ground-truth instances using configurable mask IoU and ground-truth overlap thresholds. A centroid-inside fallback supports thin anomalies where small geometric differences can produce low IoU. Matching details retain rank, matched status, and IoU for visual inspection.

Automatic rows use `review_status = automatically_evaluated`; they are never presented as manually reviewed annotations.

## Research Analysis Scope

The analysis browser and scientific analysis are deliberately separate. Browser search and filters do not change scientific scope unless **Analyse currently filtered result rows** is explicitly selected. The default mode selects one experiment ID/version and validates dataset identity.

Advanced-method pairing requires unique rows for experiment ID, experiment version, image ID, and method. Duplicate or unmatched fused/contextual rows block paired analysis. The audit panel reports source rows, unique images, methods, category counts, duplicates, and unmatched images.

## Exports

Available outputs include filtered CSV/JSON, experiment configuration JSON, selected-image manifest, summary report, proposal visualisations, category summaries, paired comparisons, and ablation reproducibility files. Exports are derived copies; the SQLite row remains the persistent record.

## Controlled Benchmark Reproduction

The currently documented benchmark uses:

```text
dataset: synthetic-controlled
dataset version: 1.0
split mode: Balanced Synthetic Benchmark
split seed: 42
experiment: SYN-BALANCED-001
experiment version: 1
methods: contour-only, fixed-threshold, multi-scale fused, refined contextual
```

The generated dataset and SQLite stores are runtime artefacts and are not committed. Reproduction therefore requires generating/registering the controlled dataset, finalising the seed-42 split, creating the plan, executing all pairs, and exporting the resulting configuration and manifest hashes.

## Ablation Experiments

Ablation plans derive from a validated source experiment. Execution is blocked when source rows contain duplicates, mixed versions, dataset/split mismatch, or a selected-image set that cannot be reproduced. Each configuration stores enabled components, thresholds and weights, seed, code commit, manifest hash, and matching thresholds.

The full configuration equals the default refined contextual method. Leaderboard differences are empirical benchmark observations, not causal proof of component value.
