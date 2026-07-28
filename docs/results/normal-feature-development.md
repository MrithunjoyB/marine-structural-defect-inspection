# Normal-Feature Development Results

## Status

`SYN-NORMAL-FEATURE-DEV-001` version 1 is **development-only — non-confirmatory**. It used the same 72 protected validation images for calibration and development diagnostics. It is synthetic, does not estimate out-of-sample performance, and supports no winning-method, transferability, publication-readiness, real-world, hybrid, or private-data claim.

The reference run completed 144/144 `scientific-result-v2` rows: 72 images × the frozen classical method and the predeclared PatchCore method. Every image has exactly one row per method, all rows use `structvision-eval-v2`, and the append-only attempt has zero failures/skips.

## Artifact identities

| Artifact | Identity |
|---|---|
| Development manifest | `2aa40b9db145a37522775b7ac605ae201b91e564cde881528fd6d41f449f3d58` |
| Environment lock | `be3a00936219aedbcc397f0b3e8c0af6d901489a06550f3b148c72e22cea87b8` |
| Official weight | `03b71d65fb2c73bb0de079a1781009f27a782ec481d2f64ab3bde9b1cdec3000` |
| Model artifact | `4542d063a64eb22d795f7a7faabb3cad592f69bd1fe753abdda0e5428f4961e7` |
| Memory bank | 151×1,536 float32; `d63536c87f55cebc1871ea8a46bbf5d6832a7fa9e7db689a993e58e55a645897` |
| Coreset indices | `75e9dbce4c058761cd737607ff9bac4e538cbb0743e7a8e8ead8a01e8068a489` |
| Calibration artifact | `a5a434281d7e16ffb5c0a9af65f5b27d100cd447f1d024b7cbc5199805a21a6f` |
| Experiment specification | `33d12aeba4258720a0f8cf27b4b1410290fd7fde3b42e11432625fe9262510b2` |

Runtime artifacts, the 87 MB v2 database, weight cache, and dense maps are ignored and not committed. Offline replay of a stored learned row reproduced its anomaly-map and proposal-mask hashes exactly.

## Dense PatchCore diagnostics

| Metric | Development value |
|---|---:|
| Pixel average precision | 0.7569 |
| AU-PRO, clean-pixel FPR ≤ 0.30 | 0.9848 |
| Image average precision | 1.0000 |
| Image ROC AUC | 1.0000 |

The perfect image-level separation is an in-cohort synthetic diagnostic after validation exposure, not generalization evidence. Raw anomaly maps average 38.41 on clean pixels and 40.91 on positive-image pixels; these are uncalibrated distances, not probabilities.

## Declared false-proposal budgets

| Budget | Threshold | Achieved clean FP/image | Clean images with any proposal | Component sensitivity @ IoU 0.25 | Proposal precision | Mean proposals/image |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 48.2263 | 0.1765 | 0.1176 | 0.6875 | 0.6735 | 0.6806 |
| 0.50 | 48.2263 | 0.1765 | 0.1176 | 0.6875 | 0.6735 | 0.6806 |
| 1.00 | 44.9375 | 1.0000 | 0.4706 | 0.6042 | 0.2636 | 1.5278 |

The 0.25 and 0.50 budgets select the same candidate threshold. The 1.00-budget point is worse at IoU 0.25 because the more permissive map produces larger/merged components and poorer one-to-one localization. This failure is retained in the full curve and shows why proposal burden alone does not guarantee localized sensitivity.

## Primary paired development description

The immutable v2 matrix uses the predeclared 0.50-budget learned operating point.

| Metric | Frozen classical | PatchCore development baseline |
|---|---:|---:|
| Micro component sensitivity @ IoU 0.25 | 0.7708 | 0.6875 |
| Macro positive-image recall | 0.8421 | 0.8158 |
| Image-level sensitivity | 0.8947 | 0.8684 |
| Proposal precision | 0.1689 | 0.6735 |
| Clean false proposals/image | 4.4118 | 0.1765 |
| Clean images with any proposal | 0.8824 | 0.1176 |
| Assigned-pair mean IoU | 0.6220 | 0.5427 |
| Assigned-pair mean Dice | 0.7504 | 0.6955 |
| Sensitivity @ IoU 0.10 / 0.25 / 0.50 | 0.8125 / 0.7708 / 0.5417 | 0.7917 / 0.6875 / 0.4167 |
| Top-1 / 3 / 5 / 8 sensitivity | 0.7083 / 0.7708 / 0.7708 / 0.7708 | 0.6667 / 0.6875 / 0.6875 / 0.6875 |
| Mean proposals/image | 3.0417 | 0.6806 |
| Mean recorded processing seconds/image | 0.6844 | 0.1545 |

Timing was recorded in one controlled process on the same CPU, but cache and implementation differences still limit causal speed interpretation.

## Category outcomes at the primary point

| Positive category | Frozen classical sensitivity | PatchCore sensitivity |
|---|---:|---:|
| colour-only anomaly | 0.9000 | 1.0000 |
| pitting cluster | 0.5333 | 0.2667 |
| texture-only anomaly | 1.0000 | 1.0000 |
| thin crack | 0.7500 | 0.0000 |
| weld disturbance | 0.7778 | 1.0000 |

PatchCore reduces nuisance proposal burden and shows development complementarity on colour and weld changes, but completely misses thin cracks at the selected component/IoU operating point and is weaker on pitting. The frozen method retains higher aggregate localization and sensitivity but generates substantially more clean proposals. These are failure characteristics, not a winner declaration.

## Integrity audit

Historical automatic rows remain exactly 888. Historical database SHA-256 values remain `1ebde1de…ce92` and `9a77d748…f640`; registry database/manifest hashes remain `50513870…4632` and `bc266fca…7ba4`. All seven protected classical source hashes remain unchanged. No historical test image, specular-retuning cohort, private collaborator data, hybrid fusion, deprecated balanced score, or MPS scientific result was used.
