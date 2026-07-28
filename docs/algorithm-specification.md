# StructVision-AI Algorithm Specification

## 1. Problem definition

StructVision-AI produces ranked visual-anomaly region proposals from one structural or surface image. A proposal identifies pixels that differ from their image context; it is not a defect class, probability, engineering diagnosis, or disposition decision.

The accepted software component is the reusable, write-free Python API. Its stable method is `structvision-classical-baseline-v1-frozen`. `structvision-patchcore-baseline-v1-dev` is a protected development baseline. `structvision-proposal-guided-hybrid-v1-dev` is a **development candidate rejected under the predeclared protocol**. Private-data validation is future work.

## 2. Inputs

The base detector accepts a filesystem image path or `uint8` NumPy array. Arrays must declare `RGB` or `BGR` for three channels and `RGBA` or `BGRA` plus `drop`, `composite_black`, or `composite_white` for four channels. Grayscale is unambiguous.

The demonstration facade accepts one PNG, JPEG, or safely supported TIFF as bytes. It checks encoded size, decoded dimensions, format/suffix agreement, channel semantics, alpha semantics, malformed input, and decompression-bomb warnings before producing an in-memory BGR array. High-bit-depth inputs are rejected by the current `uint8` contract.

The learned methods additionally require an exact Python 3.12 environment, a verified local backbone weight, the exact environment-lock identity, and method-specific immutable artifacts. No component downloads a weight automatically.

## 3. Outputs

Classical analysis returns an immutable `AnalysisResult` containing:

- image, implementation, configuration, and provenance identities;
- analysed `H×W×3` BGR shape;
- anomaly heatmap when exposed;
- ordered `Proposal` records;
- final and raw binary masks;
- half-open mask-derived boxes;
- evidence, reliability, priority, contributions, contextual diagnostics, and timing.

PatchCore returns a `NormalFeatureAnalysisResult` with a raw full-resolution distance map, image distance, immutable model/calibration identities, and thresholded connected-component proposals.

The hybrid returns a `HybridAnalysisResult` with selected proposals plus complete selected/rejected candidate diagnostics, unchanged classical masks, candidate-level PatchCore evidence, fixed fusion output, and fusion identity.

## 4. Assumptions

- Input pixels are a valid `uint8` optical image after explicit colour/alpha handling.
- Pixel coordinates have no physical-scale interpretation.
- A visually anomalous region may be benign.
- Current evidence is synthetic and development-only.
- Learned replay is valid only under the recorded environment and artifact identities.
- The detector does not infer vessel, component, damage class, severity, or engineering fitness.

## 5. Coordinate conventions

All result masks use returned analysed-image coordinates. A box is:

```text
(x_min, y_min, x_max, y_max)
```

and is half-open: `x_min ≤ x < x_max`, `y_min ≤ y < y_max`. `x_max` and `y_max` are one past the final included pixel. Every box is recomputed from its final mask. The demonstration renderer records original-to-analysed `x` and `y` scale factors and draws `(x_max - 1, y_max - 1)` as the visible final box pixel.

## 6. Frozen classical baseline

Identity: `structvision-classical-baseline-v1-frozen`, version `1.0.0`.

The public call is `structvision.api.StructuralAnomalyDetector.analyse`. It normalises input through `structvision.inputs.normalise_input` and delegates to `structvision.classical.run_frozen_classical`. That adapter:

1. imports the protected preprocessing, feature, scoring, and proposal modules;
2. redirects unavoidable legacy mask/diagnostic writes to one temporary directory;
3. runs the unchanged configured implementation;
4. reads final/raw masks back into immutable memory;
5. verifies protected source hashes in provenance;
6. removes the temporary directory before return.

Configuration is an immutable `DetectorConfig`. The default includes width cap 1024, denoising and CLAHE, fixed feature sensitivities, fixed candidate area/border/top-eight rules, contextual scoring, multi-scale fusion, merging, and mask refinement. Hidden kernels, weights, percentiles, scales, merge/NMS rules, and refinement constants are bound by `implementation_constants_hash`; this task does not expose or change them.

Status: stable frozen baseline and recommended demonstration default. It has the strongest current sensitivity evidence and a high false-proposal burden.

## 7. PatchCore development baseline

Identity: `structvision-patchcore-baseline-v1-dev`, version `1.0.0-dev1`.

The implementation uses Anomalib 2.5.1 official `PatchcoreModel` and `KCenterGreedy` components. Frozen timm Wide-ResNet-50-2 layer2/layer3 features produce 1,536-dimensional embeddings. Normal-only fitting stores a deterministic 0.001 k-center-greedy coreset. Analysis uses:

- aspect-preserving 256×416 letterbox input;
- exact nearest-normal Euclidean distance;
- full-resolution inverse-projected anomaly map;
- separately frozen development-calibration threshold;
- 8-connected components;
- no morphology;
- minimum 16 pixels and maximum 8 proposals.

Raw distances are not probabilities. The demonstration does not fit, calibrate, download, or mutate a PatchCore artifact.

Status: protected development baseline; optional research comparison only. Development evidence shows lower clean burden and higher precision, but weaker pitting and zero thin-crack component sensitivity at the selected point.

## 8. Proposal-guided hybrid

Identity: `structvision-proposal-guided-hybrid-v1-dev`, version `1.0.0-dev`.

The hybrid preserves every classical candidate mask, then computes eight candidate features in this exact order:

1. classical priority;
2. classical evidence;
3. classical heuristic mask reliability;
4. PatchCore interior mean;
5. PatchCore interior 0.95 quantile;
6. PatchCore high-support fraction;
7. PatchCore local-context contrast;
8. PatchCore local spatial agreement.

Frozen fusion-fit 0.05/0.95 quantiles scale each feature. Classical and normality components are their respective feature means. The selected fixed score is:

```text
hybrid_score = 0.60 × normalised_classical + 0.40 × normalised_normality
```

At the primary 0.50 budget, selection uses the frozen threshold `0.4704560134385654`; selected candidates are reranked by descending hybrid score then proposal ID. Complete pre-threshold diagnostics remain available.

Status: **rejected development candidate**. It reduced nuisance burden and improved precision/localisation but failed the fixed overall and image-level sensitivity-preservation decision. It is not the recommended method.

## 9. Computational stages

| Method | Stages |
|---|---|
| Classical | normalisation → protected preprocessing → feature evidence → candidate generation → contextual scoring → ranking → mask refinement → immutable proposal adaptation |
| PatchCore fit | protected normal selection → letterbox → official embeddings → deterministic coreset → immutable memory artifact |
| PatchCore analyse | normalisation → letterbox → embedding → nearest-normal distance → inverse map projection → frozen threshold → components |
| Hybrid | classical candidates + PatchCore map → candidate features → frozen normalisation → fixed fusion → frozen threshold → reranking |

Only exposed artifacts are shown. A classical internal stage image that the frozen API does not return is labelled “Not exposed by the current frozen API.”

## 10. Score semantics

| Field | Semantics |
|---|---|
| Classical evidence | heuristic contextual anomaly evidence |
| Classical mask reliability | heuristic stability/geometry measure; not calibrated confidence |
| Classical priority | heuristic review-order score |
| PatchCore image/candidate score | raw nearest-normal distance |
| Hybrid score | fixed explainable linear rank score |

No score is a probability. Missing fields are `N/A`/`null`, never fabricated as zero.

## 11. Determinism

The classical adapter uses immutable configuration, deterministic mode, ordered single-worker execution, source-hash provenance, and stable proposal order. PatchCore fixes Python/NumPy/Torch seeds, CPU execution, one Torch thread, deterministic algorithms, stored coreset indices, exact artifact hashes, and fixed proposal extraction. The hybrid fixes feature order, normalisation, weights, threshold, tie-breaks, and artifact identity.

Wall-clock timing and export timestamps are observational and are not result-determinism identities.

## 12. Failure conditions

Analysis fails explicitly for invalid input type, unsupported dtype/channel layout, ambiguous colour/alpha semantics, malformed or oversized content, empty image ID, configuration mismatch, protected source drift, missing learned dependency, changed environment lock, missing/changed weight, missing/changed model/calibration/fusion artifact, map/mask coordinate mismatch, or sink/output failure.

No method silently substitutes another method. PatchCore and hybrid do not fall back to random weights or classical output.

## 13. Complexity and resources

Classical execution allocates several image-sized feature maps and protected temporary visualisations; memory grows approximately linearly with analysed pixels, with additional connected-component costs. It is deliberately single-worker.

PatchCore feature extraction scales with resized input patches; exact nearest-neighbour scoring scales with query patches × memory-bank entries × embedding width, implemented in chunks. The recorded memory banks are 151×1,536 for the PatchCore baseline and 116×1,536 for the hybrid. Hybrid execution adds one classical and one PatchCore pass plus per-candidate feature extraction.

The demonstration facade rejects images above 40 million decoded pixels and warns above 12 million pixels or a 4,096-pixel edge. Those are demonstration safeguards, not validated detector limits.

## 14. Known limitations

- Synthetic development evidence only; no private collaborator or real marine data.
- No transfer, deployment, cross-platform, compression, or public-benchmark validation.
- No physical scale, uncertainty calibration, class prediction, or engineering diagnosis.
- Classical clean proposal burden is high.
- PatchCore can under-resolve thin cracks and pitting.
- Hybrid coverage cannot exceed its upstream classical candidates and failed the acceptance gate.
- The existing learned environment is platform-specific and CPU-referenced.
- Large-image memory behavior requires separate validation.

## 15. Data requirements

Future data require immutable sample IDs and content hashes, explicit colour/resolution/acquisition metadata, anomaly/annotation semantics, acquisition groups, reviewer/version identity, licence and confidentiality status, role assignment, and a split-lock hash. The detailed boundary is in [Private Dataset Adapter](private-dataset-adapter.md).

## 16. Evaluation contract

`structvision-eval-v2` requires immutable selected-image identities and order, content-verified image/truth hashes, exact executable configuration, fixed matching/metric policy, complete method pairing, explicit attempt identity, and sink-controlled append-only persistence. The technical demonstration never invokes an experiment executor or result sink.

Current method comparison uses the stored 72-image synthetic development holdout only. It must not be reinterpreted as real-world, confirmatory, publication, or global-superiority evidence.

## 17. Implementation identities

| Component | Identity |
|---|---|
| Stable classical method | `structvision-classical-baseline-v1-frozen` |
| PatchCore development method | `structvision-patchcore-baseline-v1-dev` |
| Rejected hybrid method | `structvision-proposal-guided-hybrid-v1-dev` |
| Evaluation | `structvision-eval-v2` |
| Classical default configuration | `16554f8a7a362dc223c9546925f6b0362314cd58becf6a6665c330f48eec9bfd` |
| Environment lock | `be3a00936219aedbcc397f0b3e8c0af6d901489a06550f3b148c72e22cea87b8` |
| PatchCore model | `4542d063a64eb22d795f7a7faabb3cad592f69bd1fe753abdda0e5428f4961e7` |
| PatchCore calibration | `a5a434281d7e16ffb5c0a9af65f5b27d100cd447f1d024b7cbc5199805a21a6f` |
| Hybrid normal model | `ef275b0a853231a239eebcccab6c920667616695296450d2d44453d922c341e7` |
| Hybrid fusion | `a21b5880c5d8f16d3869227455279ddbf18815d92ae7862e262cc2560de3d8d1` |
