# Ablation Study Results

## Scope And Reproducibility

`ABL-SYN-BALANCED-001` version 1 evaluates 12 identical test images under 10 configurations, producing 120 completed automatic rows. Configurations use the same dataset version, split, selected images, matching thresholds, and seed. `ABL-FULL` equals the default refined contextual configuration.

The balanced score is the repository's configured weighted combination of precision, recall, Top-1 recall, mean IoU, and normalised false-proposal suppression. Rankings are specific to this benchmark and weighting.

## Aggregate Leaderboard

| Configuration | Top-1 | Precision | Recall | False/image | Mean time (s) | Balanced score |
|---|---:|---:|---:|---:|---:|---:|
| `ABL-RERANK-ONLY` | 1.0000 | 0.5000 | 0.8500 | 0.75 | 0.4080 | 0.6459 |
| `ABL-FULL` | 1.0000 | 0.5000 | 0.7944 | 0.75 | 0.4233 | 0.6305 |
| `ABL-FUSED-ONLY` | 1.0000 | 0.5000 | 0.7944 | 0.75 | 0.3888 | 0.6305 |
| `ABL-NO-BORDER` | 1.0000 | 0.5000 | 0.7944 | 0.75 | 0.4301 | 0.6305 |
| `ABL-NO-BOUNDARY-EDGE` | 1.0000 | 0.5000 | 0.7944 | 0.75 | 0.4169 | 0.6305 |
| `ABL-NO-COHERENCE` | 1.0000 | 0.5000 | 0.7944 | 0.75 | 0.4034 | 0.6305 |
| `ABL-NO-COLOUR` | 1.0000 | 0.5000 | 0.7944 | 0.75 | 0.3978 | 0.6305 |
| `ABL-NO-ENTROPY` | 1.0000 | 0.5000 | 0.7944 | 0.75 | 0.4795 | 0.6305 |
| `ABL-NO-STABILITY` | 1.0000 | 0.5000 | 0.7944 | 0.75 | 0.3713 | 0.6305 |
| `ABL-NO-TEXTURE` | 1.0000 | 0.5000 | 0.7944 | 0.75 | 0.4656 | 0.6305 |

Times are recorded means from one execution environment and are not portable performance guarantees.

## Category Observations

| Configuration group | Thin-crack recall | Pitting recall | Normal-texture false/image | Specular-highlight false/image |
|---|---:|---:|---:|---:|
| `ABL-RERANK-ONLY` | 1.0000 | 0.7000 | 0.0 | 3.0 |
| `ABL-FULL` | 1.0000 | 0.5889 | 0.0 | 3.0 |
| Other recorded configurations | 1.0000 | 0.5889 | 0.0 | 3.0 |

The reranking-only configuration has the highest current balanced score because pitting-cluster recall increases while the displayed aggregate precision, false-proposal burden, and thin-crack recall remain unchanged. The result is based on three pitting images and should be treated as a candidate for further controlled evaluation, not as evidence of universal superiority.

## Interpretation Limits

Removing texture, colour, entropy, stability, border, boundary-edge, or coherence terms does not change aggregate detection metrics in this small benchmark. This may reflect redundant evidence, insensitive categories, limited sample size, or threshold effects. It does not prove that the components are unnecessary on real images. Specular-highlight false alarms remain unresolved across all recorded configurations.

The next ablation phase should expand categories and seeds, retain strict image-level pairing, and report confidence intervals for configuration differences.
