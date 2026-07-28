# Model Card: StructVision PatchCore Development Baseline

## Identity and intended use

- Method: `structvision-patchcore-baseline-v1-dev` `1.0.0-dev1`
- Upstream: official Anomalib `2.5.1` PatchCore
- Intended use: development-only normal-feature anomaly baseline and failure analysis
- Not intended for: defect classification, engineering diagnosis, private-data evaluation, safety decisions, confirmatory comparison, probability estimation, or real-world deployment

## Model and data

Frozen official timm Wide-ResNet-50-2 features from layers 2/3 form 1,536-dimensional patch embeddings. A 0.001 deterministic k-center-greedy coreset is fitted from 91 protected clean synthetic training images. Thresholds come separately from 72 protected synthetic validation images. See the [baseline](normal-feature-baseline.md), [data protocol](development-data-protocol.md), and [results](results/normal-feature-development.md).

## Outputs

The model returns a raw image anomaly distance and full-resolution float32 anomaly map. A separate calibrated operating point produces unlabelled, ranked binary components. Scores are neither probabilities nor comparable in scale to classical heuristic priorities.

## Evaluation and limitations

Development pixel AP is 0.7569 and primary IoU-0.25 component sensitivity is 0.6875 at 0.1765 clean false proposals/image. Thin-crack sensitivity is 0.0 and pitting sensitivity is 0.2667 at that point. All values are same-cohort synthetic diagnostics and may be optimistic. Domain shift, small/thin structures, synthetic templates, memory representativeness, threshold instability, and upstream training-data/licensing constraints remain material risks.

## Reproducibility and licence

The environment, weight, manifest, model, coreset, calibration, and v2 specification are content-addressed. The official weight is declared Apache-2.0 by timm; Anomalib is Apache-2.0. Review ImageNet-derived provenance and all deployment licences independently. Weight/cache/model-memory files are not distributed in Git.
