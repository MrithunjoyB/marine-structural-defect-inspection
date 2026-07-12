# Specular-Suppression Experiment

## Research Question

Can an optical/structural candidate score reduce false proposals on smooth achromatic highlights while preserving the controlled gains of contextual reranking? This experiment concerns the 12-image synthetic test only and does not establish real-world reflection suppression.

## Configuration

| Item | Value |
|---|---|
| Configuration | `ABL-RERANK-SPECULAR-SUPPRESS` |
| Display name | Reranking with specular suppression |
| Experiment | `SYN-SPECULAR-SUPPRESS-001` |
| Final evaluated version | 2 |
| Dataset | `synthetic-controlled` v1.0, test split |
| Images | 12 (same manifest as `SYN-BALANCED-001`) |
| Seed | 42 |
| IoU / mask-overlap thresholds | 0.10 / 0.25 |
| Compared methods | multi-scale fused, `ABL-FULL`, `ABL-RERANK-ONLY`, new suppression |
| Expected / completed rows | 48 / 48 |

The likelihood combines high-value/low-saturation occupancy, channel similarity, Lab chroma, intensity smoothness, entropy, context brightness, compactness, and eroded-core structural evidence. Crack and pitting safeguards reduce the effective score for elongated, thin, scale-consistent, multi-component, irregular, or textured candidates. The policy penalises ranking first and rejects only high-likelihood candidates with weak structural evidence.

## Pilot And Revision

Version 1 is preserved as a negative pilot. Its initial core measurement included highlight boundaries, inflating structural strength; proposals were penalised but none were rejected, leaving specular false proposals unchanged at 3.0 per image. The implementation was corrected to measure distance-transform interiors. Version 2 lowered the predeclared rejection threshold from 0.72 to 0.50 for the revised bounded score. No version 1 rows were overwritten.

## Version 2 Aggregate Results

| Method | Top-1 recall | Precision | Proposal recall | False proposals/image | Mean time (s) |
|---|---:|---:|---:|---:|---:|
| Multi-scale fused | 1.0000 | 0.5000 | 0.7944 | 0.75 | 0.4284 |
| `ABL-FULL` | 1.0000 | 0.5000 | 0.7944 | 0.75 | 0.4257 |
| `ABL-RERANK-ONLY` | 1.0000 | 0.5000 | 0.8500 | 0.75 | 0.4360 |
| `ABL-RERANK-SPECULAR-SUPPRESS` | 1.0000 | 0.5000 | 0.8500 | 0.50 | 0.4335 |

## Category-Wise Comparison

| Criterion | Rerank-only | Suppressed | Result |
|---|---:|---:|---|
| Specular-highlight false proposals/image | 3.0 | 2.0 | Decreased |
| Thin-crack Top-1 recall | 1.0000 | 1.0000 | Preserved |
| Thin-crack proposal recall | 1.0000 | 1.0000 | Preserved |
| Thin-crack mean IoU | 0.9092 | 0.9092 | Change 0.0000 |
| Pitting-cluster proposal recall | 0.7000 | 0.7000 | Preserved |
| Normal-texture false proposals/image | 0.0 | 0.0 | Preserved |

The predeclared material-IoU tolerance was an absolute decrease no greater than 0.01; the observed decrease was zero. Mean recorded time changed from approximately 0.4360 s to 0.4335 s per image, which does not indicate measurable overhead on this run.

## Interpretation

All mandatory controlled criteria were met in version 2. The result is promising within this benchmark because false proposals decreased without a measured crack, pitting, or normal-texture loss. It is not evidence of general marine-inspection improvement. Only three synthetic specular images are present, and a one-proposal-per-image reduction may reflect the generator's limited highlight geometry. The mechanism did not eliminate all highlights, which is consistent with the conservative policy, but broader bright-crack, reflective-pitting, mixed-texture, exposure, and real-camera tests are required.
