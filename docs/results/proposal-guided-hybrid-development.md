# Proposal-Guided Hybrid Development Results

## Decision

`SYN-PROPOSAL-HYBRID-DEV-001` version 1 completed one 72-image, three-method `structvision-eval-v2` holdout attempt. The outcome is **development candidate rejected under the predeclared protocol**. It is **development holdout — non-confirmatory** and supports no real-world, transferability, publication, novelty, deployment, professor-data, winner, or global-superiority claim.

At the primary 0.50 budget the hybrid materially reduced clean proposal burden and preserved thin-crack, pitting, weld, and assigned-pair localisation. It failed because micro sensitivity decreased from `0.770833` to `0.750000`, a decrease of `0.020833` that narrowly exceeded the fixed `0.02` margin, and image-level sensitivity decreased from `0.894737` to `0.868421`. The holdout was not rerun or used for modification.

## Identities and pre-task protection

| Item | Identity |
|---|---|
| Baseline HEAD at execution | `6688d2f43b8a514f435d8df87c87861b478756de` on `main` |
| Hybrid protocol/manifest | `structvision-hybrid-development-v1`; `a1e6f9a83e5e8d73275236e6dc4fafd985e6e1ef2c4aef21fd4156dc821829a4` |
| Hybrid normal model | `ef275b0a853231a239eebcccab6c920667616695296450d2d44453d922c341e7` |
| Hybrid memory bank | 116×1,536 float32; `a593d144049b7ac12d6f074464d632ac16724636cd7510510b46609235e631b1` |
| Fusion artifact | `a21b5880c5d8f16d3869227455279ddbf18815d92ae7862e262cc2560de3d8d1` |
| Experiment specification | `97cca2a29164daf63fb86d316817f2a997834f5dc34fb542d136fff75b031b9e` |
| Exact environment lock | `be3a00936219aedbcc397f0b3e8c0af6d901489a06550f3b148c72e22cea87b8` |
| Official backbone weight | `03b71d65fb2c73bb0de079a1781009f27a782ec481d2f64ab3bde9b1cdec3000` |
| Historical databases | `9a77d748dbf9780f5f0e104bea3412ddaadcad10b54a2c1fceed0e532acef640`; `1ebde1de1f065b5b220366798147beb67dd10a446b7cd8840f988c9aeda9ce92` |
| Historical automatic rows | 888 before and after |

All seven protected classical source hashes, the prior 144-row PatchCore development store hash `3a0200e75fde0633587f961d297d91259868df7120f176f5abfa2af9e73febf1`, prior model/calibration files, registry database/manifest, and environment lock were identical before and after. No historical test or professor image was decoded by this task.

## Roles and category counts

| Role | Clean | Positive | Total |
|---|---:|---:|---:|
| `hybrid_normal_fit` | 70 | 0 | 70 |
| `hybrid_fusion_fit` | 19 | 107 | 126 |
| `hybrid_development_holdout` | 34 | 38 | 72 |

Normal-fit clean counts were blur/noise 15, border artefact 12, illumination gradient 9, normal texture 16, and specular highlights 18. Fusion-fit clean counts were 5, 4, 2, 5, and 3 respectively; positive counts were colour-only 20, pitting 24, texture-only 15, thin crack 29, and weld 19. Holdout counts were blur/noise 10, border artefact 9, illumination gradient 3, normal texture 9, specular highlights 3, colour-only 10, pitting 5, texture-only 10, thin crack 4, and weld 9. All priority categories remained in both labelled roles.

The manifest excludes 132 unique train/validation identities after exact, dHash-candidate, protected-history, pilot, and declared-group checks, including two train identities linked to the fixed validation holdout. No selected ID, content hash, dHash≤3 component, source group, template group, or non-empty acquisition group crosses roles.

## Fusion-fit selection

The classical fusion-fit reference had micro sensitivity `0.717647`, image sensitivity `0.915888`, assigned-pair IoU `0.609296`, and `3.789474` clean FP/image. The selected `0.60 classical + 0.40 normality`, no-floor configuration preserved those three performance values exactly while reducing clean FP/image to `0.473684`, clean-any to `0.315789`, and mean proposals/image to `1.174603`; precision was `0.824324`.

All 15 primary-budget candidates are retained:

| Classical / normality | Floor | Micro sens. | Image sens. | Precision | Mean IoU | Preservation |
|---:|---:|---:|---:|---:|---:|---|
| 0.90 / 0.10 | none, 0.90, 0.80 | 0.558824 | 0.719626 | 0.826087 | 0.602945 | fail: overall, thin, pitting, weld, image |
| 0.80 / 0.20 | none, 0.90, 0.80 | 0.605882 | 0.766355 | 0.830645 | 0.615040 | fail: overall, thin, pitting, image |
| 0.70 / 0.30 | none, 0.90, 0.80 | 0.682353 | 0.869159 | 0.822695 | 0.612442 | fail: overall, pitting, image |
| 0.60 / 0.40 | none, 0.90, 0.80 | 0.717647 | 0.915888 | 0.824324 | 0.609296 | pass |
| 0.50 / 0.50 | none, 0.90, 0.80 | 0.717647 | 0.915888 | 0.818792 | 0.609296 | pass |

Every row achieved `0.473684` clean FP/image. Floors did not change selection at this budget. The deterministic simplicity tie-break chose 0.60/0.40 with no floor. The selected configuration's fusion-fit 0.25 point did not satisfy preservation; this is retained rather than hidden.

## Primary holdout comparison

| Metric | Frozen classical | PatchCore baseline | Hybrid @ 0.50 |
|---|---:|---:|---:|
| Micro component sensitivity | 0.770833 | 0.687500 | 0.750000 |
| Macro positive-image recall | 0.842105 | 0.815789 | 0.815789 |
| Image-level sensitivity | 0.894737 | 0.868421 | 0.868421 |
| Proposal precision | 0.168950 | 0.673469 | 0.720000 |
| Clean FP/image | 4.411765 | 0.176471 | 0.323529 |
| Clean images with any proposal | 0.882353 | 0.117647 | 0.235294 |
| Mean proposals/image | 3.041667 | 0.680556 | 0.694444 |
| Mean assigned-pair IoU | 0.621954 | 0.542742 | 0.631250 |
| Mean assigned-pair Dice | 0.750398 | 0.695452 | 0.758844 |
| Sensitivity @ IoU 0.10 / 0.25 / 0.50 | 0.812500 / 0.770833 / 0.541667 | 0.791667 / 0.687500 / 0.416667 | 0.791667 / 0.750000 / 0.541667 |
| Top-1 / 3 / 5 / 8 sensitivity | 0.708333 / 0.770833 / 0.770833 / 0.770833 | 0.666667 / 0.687500 / 0.687500 / 0.687500 | 0.687500 / 0.750000 / 0.750000 / 0.750000 |
| Mean recorded seconds/image | 0.760675 | 0.223192 | 0.961674 |

The hybrid was about 1.26× the classical recorded time, consistent with executing both evidence paths, and the process peaked at 1,420,771,328 bytes resident memory. Timing is same-process development evidence, not a cross-platform performance guarantee.

## Frozen-budget holdout table

| Nominal fusion-fit budget | Frozen threshold | Holdout clean FP/image | Clean-any | Micro sens. | Image sens. | Precision | Proposals/image | Mean IoU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.654096 | 0.147059 | 0.088235 | 0.625000 | 0.710526 | 0.789474 | 0.527778 | 0.628195 |
| 0.50 | 0.470456 | 0.323529 | 0.235294 | 0.750000 | 0.868421 | 0.720000 | 0.694444 | 0.631250 |
| 1.00 | 0.374374 | 1.058824 | 0.676471 | 0.770833 | 0.894737 | 0.486842 | 1.055556 | 0.621954 |

The 1.00 threshold exceeded its nominal burden on holdout; no threshold was adjusted. It descriptively recovered classical sensitivity but is not the primary policy and cannot rescue the failed primary decision.

## Category and paired effects

| Positive category | Classical | PatchCore | Hybrid @ 0.50 |
|---|---:|---:|---:|
| colour-only anomaly | 0.900000 | 1.000000 | 0.900000 |
| pitting cluster | 0.533333 | 0.266667 | 0.533333 |
| texture-only anomaly | 1.000000 | 1.000000 | 0.900000 |
| thin crack | 0.750000 | 0.000000 | 0.750000 |
| weld disturbance | 0.777778 | 1.000000 | 0.777778 |

The hybrid preserved every predeclared critical-category sensitivity and missed one classical-matched texture-only component in one image; matched-truth count was unchanged on the other 71 images. Proposal counts decreased on 41 images and were unchanged on 31. The missed texture-only component caused both failed aggregate preservation endpoints. This is a retained failure, not a rationale for post-hoc category logic.

## Execution and integrity

The expected and actual result counts were 72 images × 3 methods = 216. Each image has one row for each exact method identity, all rows use `structvision-eval-v2`, pairing is complete, and the new append-only store contains one completed attempt. The ledger records one start and one completion. Auxiliary budgets reselected cached full candidate diagnostics within that attempt; they did not rerun the holdout or refit fusion.

No historical store, previous v2 store, prior learned artifact, dataset, image, mask, split, frozen classical file, or protected PatchCore file was written. No model weight, memory bank, cache, private image, dense map, or historical database is committed.

## Remaining risks

The synthetic generator, prior validation exposure, only 19 fusion-fit clean images, fixed PatchCore resolution, threshold transport across clean cohorts, one missed texture-only component, specular burden, and absence of uncertainty calibration limit interpretation. Professor data remains future work under a new intake and evaluation protocol.
