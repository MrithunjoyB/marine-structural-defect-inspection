# Controlled Benchmark Results

## Scope

These results are preliminary and based on a small synthetic dataset. They do not establish real-world marine or structural inspection performance.

| Item | Value |
|---|---:|
| Dataset | `synthetic-controlled` v1.0 |
| Total images | 33 |
| Train / validation / test | 15 / 6 / 12 |
| Split seed | 42 |
| Test anomaly-present / clean | 6 / 6 |
| Test categories | 3 thin crack, 3 pitting cluster, 3 normal texture, 3 specular highlight |
| Detected split leakage | 0 |
| Experiment | `SYN-BALANCED-001` v1 |
| Stored result rows | 48 |

## Aggregate Results

| Method | Top-1 | Top-3 | Top-5 | Top-8 | Precision | Recall | False/image | Mean time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Contour-only | 0.8333 | 0.8333 | 0.8333 | 0.8333 | 0.4167 | 0.8333 | 0.75 | 0.4101 |
| Fixed-threshold | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.3715 | 0.9667 | 3.25 | 0.0731 |
| Multi-scale fused | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 0.7944 | 0.75 | 0.0078 |
| Refined contextual | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 0.7944 | 0.75 | 0.0081 |

Top-\(K\) recall uses six positive test images. Processing time is an approximate recorded mean and should not be generalised across hardware.

## Paired Advanced-Method Analysis

The 12 fused and 12 contextual rows pair exactly by experiment/version/image ID. Detection-level outcomes are tied on all 12 images: precision, recall, proposal counts, false proposals, and first-hit ranks are unchanged. Refined contextual masks have higher localisation IoU on the six eligible anomaly images. Mean IoU across the stored image-level rows is approximately 0.2017 for fused and 0.4095 for contextual; mean best IoU is approximately 0.2250 and 0.4530, respectively.

Paired bootstrap inputs contain 12 images for precision and false proposals, and 6 positive images for recall and localisation IoU. Confidence intervals are descriptive because of the sample size.

## Observed Failure Modes

Normal-texture images produce zero false proposals under the recorded advanced methods. Specular-highlight images remain a persistent false-positive condition. Pitting-cluster ground-truth instance recall is incomplete, while thin-crack recall is preserved in the controlled test. These observations motivate benchmark expansion and real-data validation rather than deployment claims.

An opt-in follow-up, `ABL-RERANK-SPECULAR-SUPPRESS`, reduced the recorded specular-highlight burden from 3.0 to 2.0 proposals per image in `SYN-SPECULAR-SUPPRESS-001` version 2 while preserving the recorded crack, pitting, and normal-texture criteria. The complete protocol, negative pilot, and limitations are documented in [Specular Suppression](specular-suppression.md).

The subsequent 100-image frozen test did not validate the method because one weld-disturbance detection was lost and aggregate recall decreased. See [Expanded Synthetic Benchmark](expanded-synthetic-benchmark.md). This later negative result supersedes any interpretation of the 12-image result as sufficient validation.
