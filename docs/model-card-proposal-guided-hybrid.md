# Model Card: Proposal-Guided Hybrid v1 Development Candidate

## Status

- Identity: `structvision-proposal-guided-hybrid-v1-dev` version `1.0.0-dev`
- Artifact: `a21b5880c5d8f16d3869227455279ddbf18815d92ae7862e262cc2560de3d8d1`
- Evidence: synthetic **development holdout — non-confirmatory**
- Decision: **development candidate rejected under the predeclared protocol**

This is not a deployed model, calibrated probability system, diagnostic classifier, safety instrument, globally best method, validated product, or publication-ready result.

## Intended use

The implementation is an explainable research candidate for ordering and filtering the complete frozen classical proposal list. It may be used to replay this development experiment and inspect component-level evidence. It must not be used for operational marine inspection, repair decisions, autonomous rejection, professor-data inference, or real-world performance claims.

## Components

The detector combines unchanged classical proposals with a separate PatchCore memory fitted from 70 protected clean training images. Eight predeclared features produce two normalised components. The selected rank score is `0.60 × classical + 0.40 × normality` with threshold `0.4704560134385654` at the primary budget. Scores and mask reliability are heuristics, not probabilities.

Every returned candidate exposes classical scores, PatchCore evidence, normalised features, weighted contributions, threshold decision, complete pre-threshold diagnostics, half-open box, and an unchanged classical mask.

## Development performance and failure

At the primary holdout point the hybrid achieved 0.7500 micro component sensitivity, 0.3235 clean FP/image, 0.7200 precision, and 0.6313 assigned-pair IoU. It preserved thin-crack, pitting, weld, and localisation quantities relative to classical, but exceeded the overall sensitivity loss margin by 0.000833 and decreased image-level sensitivity. It is therefore rejected even though nuisance burden improved.

## Risks and limitations

The memory and fusion are synthetic-cohort dependent. PatchCore may under-resolve thin structures; classical candidates cap structural coverage; specular regions remain burdensome; thresholded ranking is not uncertainty calibration; and the validation cohort had prior PatchCore exposure. No transfer, real-world, cross-platform, compression, deployment, or novelty analysis has been completed.

## Artifacts and reproducibility

The runtime model memory, official weight, v2 store, and maps are ignored. The committed manifest, exact environment lock, source implementation, artifact identities, and [development results](results/proposal-guided-hybrid-development.md) define the replay boundary. No API key is required.
