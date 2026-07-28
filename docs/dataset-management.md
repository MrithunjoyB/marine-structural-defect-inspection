# Dataset Management

Research Dataset Intake registers image collections, provenance, annotation status, licensing constraints, and split assignments. Runtime datasets and registries are intentionally excluded from Git.

## Registration Metadata

A dataset registration records a stable dataset ID and version, source type and reference, provider, acquisition date, domain, licence, redistribution and commercial-use permissions, citation text, annotation format, and ground-truth status. Duplicate dataset ID/version combinations require an explicit cancel, replacement, or new-version decision.

Private data should retain redistribution-disabled status unless permission is documented. Unknown licensing prevents public export without a recorded override.

## Intake And Validation

The page accepts individual images, image batches, and ZIP archives. Supported annotation inputs include YOLO boxes and segmentation, COCO JSON, Pascal VOC, binary masks, CSV regions, and custom files. Validation checks image decoding, supported formats, dimensions, zero-byte files, annotation presence, and format-specific bounds where implemented.

The image manifest records:

- immutable image and dataset identifiers;
- original and stored filenames;
- image-byte SHA-256;
- dimensions, channels, format, and byte size;
- source, licence, annotation, and ground-truth status;
- exact and near-duplicate provenance;
- anomaly type and positive/clean outcome;
- source, template, and near-duplicate group IDs;
- split eligibility and assignment; and
- synthetic generation seed and parameters where applicable.

SHA-256 is calculated only from encoded image bytes. Annotation files, metadata, and filenames do not determine exact image duplication.

## Duplicate Handling

Exact duplicates share an image-byte SHA-256 and are excluded from split eligibility by default. The manifest distinguishes exact image duplicates, near duplicates, duplicate annotations, and synthetic samples sharing a template. Duplicate details identify the canonical image and group.

Near-duplicate and source/template relationships are grouping constraints rather than reasons for automatic deletion. Reviewers may exclude or restore individual items. Physical file deletion requires separate confirmation.

## Group-Aware Splitting

Split allocation operates on indivisible connected groups. Exact duplicates remain excluded; near-duplicate, source, sequence, component, and template groups cannot cross train, validation, and test. Allocation is stratified by anomaly category, anomaly-present versus no-anomaly outcome, and clean artefact subtype.

The **Balanced Synthetic Benchmark** mode uses constrained allocation when ordinary target percentages cannot preserve useful coverage. For the 33-image controlled dataset with seed 42, the resulting counts are 15 train, 6 validation, and 12 test. The test size is deliberately larger than 15% so four template groups remain represented without leakage.

The historical **Expanded Synthetic Benchmark** mode allocated deterministic groups within the 500-image registration, producing a 300/100/100 split. A later cross-dataset audit found that `synthetic-expanded-pilot` is not independent: all 80 pilot files recur in the 500-image dataset and 13 occur in its test split. The expanded test is therefore not confirmatory. See the [overlap audit](audits/historical-dataset-overlap.md).

For future v2 registration, clean images use explicit `ground_truth_status = no_anomaly`, zero truth instances, and no required empty mask. Anomaly-present images require immutable truth IDs and valid non-empty annotation objects. Legacy empty clean masks may be read only with an explicit warning; inconsistent status/annotation combinations are rejected.

Before finalisation, the preview reports image and group counts, positive/clean counts, category distributions, missing categories, and leakage checks. Finalisation is blocked when the test split lacks positive images, clean images, category diversity, or required benchmark categories. An override requires an explicit warning acknowledgement.

## Safe Deletion

The management page supports three principal modes:

- **Metadata only:** remove registry and manifest entries while preserving owned files.
- **Generated files only:** preserve registration and raw data while removing selected processed outputs, split manifests, or reports.
- **Complete deletion:** remove registry rows and move dataset-owned files to trash.

Destructive actions require a confirmation checkbox, the exact dataset ID, and a final action. Cleanup previews report affected files, bytes, database rows, manifests, and linked experiment IDs. Paths are resolved beneath `research_data/`; traversal, symlink escape, and source-repository paths are rejected.

Final research experiments are never silently deleted or unlinked. Development experiment links may be preserved, explicitly unlinked with audit metadata, or cascade-deleted after confirmation.

## Trash And Restore

Complete file deletion first moves owned files into:

```text
research_data/.trash/<timestamp>/<dataset_id>/
```

The deletion audit records reviewer, timestamp, mode, moved files, and removed rows. Restore reconstitutes moved files and saved registry rows. Permanently emptying trash requires a separate confirmation. Clearing Streamlit caches does not remove datasets or registries.

## Runtime Layout

```text
research_data/
├── registry/       # SQLite registry and generated manifest
├── raw/            # registered source images
├── processed/      # derived image products
├── annotations/    # imported and reviewed ground truth
├── splits/         # split manifests
├── reports/        # validation and experiment evidence
├── exports/        # reproducibility and result exports
└── .trash/         # reversible deletion staging
```

Only schemas and documentation are intended for version control.

## External And Restricted Data

**Do not commit externally provided, private, restricted, or unlicensed images to the public repository.**

Register restricted collections locally, preserve source and licence metadata, and use ignored runtime directories. Ground truth should be marked verified, reviewer-estimated, unavailable, or unknown as appropriate. Public redistribution requires permission independent of the repository's code status.
