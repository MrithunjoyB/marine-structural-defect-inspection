# Research Evidence Summary

## Scope

These are exact stored values from the 72-image synthetic development holdout. They are development-only, non-confirmatory, and do not establish real-world performance, transferability, deployment readiness, publication readiness, or global superiority.

| Metric at primary IoU 0.25 point | Frozen classical | PatchCore baseline | Proposal-guided hybrid |
|---|---:|---:|---:|
| Status | stable frozen baseline | protected development baseline | **rejected development candidate** |
| Micro component sensitivity | `0.770833` | `0.687500` | `0.750000` |
| Macro positive-image recall | `0.842105` | `0.815789` | `0.815789` |
| Image-level sensitivity | `0.894737` | `0.868421` | `0.868421` |
| Proposal precision | `0.168950` | `0.673469` | `0.720000` |
| Clean false proposals/image | `4.411765` | `0.176471` | `0.323529` |
| Clean images with any proposal | `0.882353` | `0.117647` | `0.235294` |
| Mean proposals/image | `3.041667` | `0.680556` | `0.694444` |
| Mean assigned-pair IoU | `0.621954` | `0.542742` | `0.631250` |
| Mean assigned-pair Dice | `0.750398` | `0.695452` | `0.758844` |
| Thin-crack sensitivity | `0.750000` | `0.000000` | `0.750000` |
| Pitting sensitivity | `0.533333` | `0.266667` | `0.533333` |

## Interpretation

The frozen classical baseline retains the strongest current aggregate and thin-structure sensitivity evidence, but its clean proposal burden is high.

PatchCore substantially reduces clean burden and increases precision. It performs poorly on the current thin-crack and pitting categories, and its lower assigned-pair IoU does not support replacing the classical default.

The hybrid demonstrates complementary evidence: relative to classical it reduces clean false proposals/image from `4.411765` to `0.323529`, increases precision from `0.168950` to `0.720000`, and slightly increases assigned-pair IoU from `0.621954` to `0.631250`. However:

```text
classical micro sensitivity = 0.770833
hybrid micro sensitivity    = 0.750000
observed loss               = 0.020833
predeclared allowed loss    = 0.020000
margin exceeded by          ≈ 0.000833
```

Image-level sensitivity also decreases from `0.894737` to `0.868421`. The outcome is therefore **development candidate rejected under the predeclared protocol**. The failed endpoints are not offset by the burden, precision, or localisation improvements.

## Research contribution supported by current evidence

- complementary-method evidence;
- a transparent sensitivity/burden trade-off;
- disciplined retention of a negative decision;
- reusable, content-addressed local architecture;
- an explicit protected path for future private-data work.

## Sources

- [Normal-Feature Development Results](results/normal-feature-development.md)
- [Proposal-Guided Hybrid Development Results](results/proposal-guided-hybrid-development.md)
- [PatchCore Model Card](model-card-normal-feature-patchcore.md)
- [Hybrid Model Card](model-card-proposal-guided-hybrid.md)
