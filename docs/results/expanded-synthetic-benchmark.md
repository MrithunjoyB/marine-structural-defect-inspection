# Expanded Synthetic Benchmark

## Scope And Integrity

This is synthetic validation, not real-world marine-inspection evidence. `synthetic-expanded` v1.0 contains 500 deterministic images across ten categories. An 80-image pilot (`synthetic-expanded-pilot`) first passed mask, metadata, uniqueness, balance, and visual-diversity checks. The final group-aware seed-42 split is exactly 300 train, 100 validation, and 100 test, with 30/10/10 images from every category, 50 positive and 50 clean test images, and zero exact-hash, near-duplicate, source, or template leakage.

`SYN-EXPANDED-VALIDATION-001` v1 freezes algorithm commit `71964778475d551444d356bdaa126f06c86bb0ef`, IoU 0.10, mask overlap 0.25, and maximum regions 8. All six methods completed the identical 100-image manifest: 600 expected rows, 600 completed rows, 600 unique image-method pairs, zero missing, unexpected, duplicate, or failed pairs.

## Aggregate Results

Recall and IoU use only the 50 anomaly-present images. Clean-image recall is undefined.

| Method | Top-1 | Top-3 | Top-5 | Top-8 | Precision | Recall | Mean IoU | Best IoU | False/image | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Contour-only | 0.54 | 0.54 | 0.54 | 0.54 | 0.2700 | 0.5400 | 0.0522 | 0.0594 | 0.30 | 0.4930 |
| Fixed-threshold | 0.56 | 0.68 | 0.68 | 0.68 | 0.2178 | 0.6617 | 0.1954 | 0.2912 | 5.23 | 0.0805 |
| Multi-scale fused | 0.88 | 0.88 | 0.88 | 0.88 | 0.4350 | 0.7960 | 0.3669 | 0.3819 | 0.84 | 0.0089* |
| Refined contextual | 0.88 | 0.88 | 0.88 | 0.88 | 0.4350 | 0.7960 | 0.6540 | 0.6654 | 0.84 | 0.0088* |
| `ABL-RERANK-ONLY` | 0.88 | 0.88 | 0.88 | 0.88 | 0.4400 | 0.8200 | 0.6543 | 0.6763 | 0.64 | 0.6503 |
| `ABL-RERANK-SPECULAR-SUPPRESS` | 0.86 | 0.86 | 0.86 | 0.86 | 0.4300 | 0.8000 | 0.6371 | 0.6590 | 0.50 | 0.5638 |

*The standard methods shared cached proposal computation; their per-row times exclude the shared computation on later rows and should not be compared with independently executed ablations.

## Frozen Candidate Comparison

| Category | Rerank-only Top-1 / recall / IoU / false | Suppressed Top-1 / recall / IoU / false |
|---|---:|---:|
| Thin crack | 0.90 / 0.90 / 0.6816 / 0.0 | 0.90 / 0.90 / 0.6816 / 0.0 |
| Pitting cluster | 1.00 / 0.70 / 0.6062 / 0.0 | 1.00 / 0.70 / 0.6062 / 0.0 |
| Weld disturbance | 0.60 / 0.60 / 0.4739 / 0.0 | 0.50 / 0.50 / 0.3877 / 0.0 |
| Colour-only anomaly | 1.00 / 1.00 / 0.7763 / 0.0 | 1.00 / 1.00 / 0.7763 / 0.0 |
| Texture-only anomaly | 0.90 / 0.90 / 0.7335 / 0.2 | 0.90 / 0.90 / 0.7335 / 0.2 |
| Normal texture | N/A / N/A / N/A / 0.3 | N/A / N/A / N/A / 0.3 |
| Specular highlights | N/A / N/A / N/A / 2.9 | N/A / N/A / N/A / 1.5 |
| Illumination gradient | N/A / N/A / N/A / 2.1 | N/A / N/A / N/A / 2.1 |
| Border artifact | N/A / N/A / N/A / 0.6 | N/A / N/A / N/A / 0.6 |
| Blur/noise | N/A / N/A / N/A / 0.3 | N/A / N/A / N/A / 0.3 |

Bright-crack and reflective-pitting test subsets were unchanged between the two configurations. The only lost positive was one medium-difficulty weld-disturbance image. No threshold or weight was changed in response.

## Paired Differences And Decision

Suppressed minus rerank-only paired means were -0.02 for Top-1/3/5/8 and proposal recall (bootstrap 95% interval -0.06 to 0.00), -0.14 false proposals/image (interval -0.27 to -0.04), and -0.0865 seconds/image (interval -0.1677 to -0.0114). The timing result is descriptive and order/hardware dependent.

| Predeclared criterion | Result | Pass |
|---|---:|---:|
| Specular false proposals decrease | 2.9 to 1.5 | Yes |
| Thin-crack Top-1 does not decrease | 0.90 to 0.90 | Yes |
| Thin-crack recall does not decrease | 0.90 to 0.90 | Yes |
| Thin-crack mean-IoU decrease <= 0.01 | change 0.0000 | Yes |
| Pitting recall does not decrease | 0.70 to 0.70 | Yes |
| Normal-texture false proposals do not increase | 0.3 to 0.3 | Yes |
| Aggregate proposal recall does not decrease | 0.82 to 0.80 | **No** |
| Processing-time increase below 15% | no increase observed | Yes |

**Expanded synthetic validation failed.** The specular reduction generalized within this generator, but the mandatory aggregate-recall preservation criterion did not. The stored negative result is retained without test-set retuning. Further investigation must use development data, additional generator seeds, independently generated synthetic data, and eventually licensed expert-reviewed real imagery.
