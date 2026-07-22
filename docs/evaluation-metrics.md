# Evaluation Metrics

This repository has two explicitly different evaluation contracts. Historical automatic rows retain `structvision-eval-v1-historical` semantics. Future experiments must use `structvision-eval-v2`. Historical rows are not recalculated or reinterpreted.

## Historical V1 Semantics

The historical matcher in `registered_experiment.py` accepts a proposal when mask IoU, truth-area overlap, or an optional centroid-inside rule succeeds. It iterates proposals in supplied order, does not enforce one-to-one assignment, and does not preserve a complete proposal-by-truth decision matrix. Historical Top-K values may also reflect connected-component order for baselines that did not have a documented numeric ranking.

Those rows remain engineering evidence under their original semantics. Their precision, recall, Top-K, IoU, and balanced-score fields are not publication-valid v2 estimates. In particular, historical baseline Top-K values are not comparable to properly ranked outputs under the v2 contract.

## V2 Primary Matching

`structvision-eval-v2` builds the complete proposal-by-truth mask-similarity matrix and performs deterministic one-to-one bipartite assignment. Eligible edges are mask IoU at or above the named threshold; assignment maximises the number of valid matches and then their total IoU. Proposal IDs and truth IDs provide deterministic tie ordering.

Each proposal and truth can be assigned at most once. Unassigned proposals are false proposals and unassigned truths are false negatives. Stored evidence includes proposal and truth mask encodings and hashes, every pair's IoU, Dice, truth overlap, proposal overlap, the assigned truth ID, threshold, metric, and decision reason. The centroid-inside result is stored only as a diagnostic and never changes primary matching.

Named analyses are mask IoU at 0.10, 0.25, and 0.50. The default primary v2 analysis is mask IoU at 0.25. Truth overlap is descriptive and is not combined with IoU into an ambiguous success rule.

## Ranking And Top-K

A method is Top-K eligible only when every proposal has a finite numeric score, a unique contiguous one-based rank, and a documented ranking definition. Ranks must equal score-descending order, with proposal ID as the deterministic score-tie rule. Missing ranks, duplicates, gaps, and inconsistent score order fail validation.

Unordered methods report Top-1, Top-3, Top-5, and Top-8 as `N/A`. Connected-component scan order is never substituted for rank. Ranked component sensitivity at Top-K is:

$$
\operatorname{TopKComponentSensitivity}
=\frac{\text{truth instances assigned using proposals ranked }\le K}
{\text{all truth instances}}.
$$

## Detection And Localisation Denominators

- **Micro component sensitivity:** total matched truth instances divided by total truth instances.
- **Macro per-positive-image recall:** mean of each positive image's matched-truth fraction.
- **Image-level detection sensitivity:** positive images with at least one match divided by positive images.
- **Proposal precision:** matched proposals divided by all proposals.
- **Clean false proposals per image:** proposals on clean images divided by clean images.
- **Clean images with any proposal:** clean images with one or more proposals divided by clean images.
- **Localisation:** the full assigned-pair IoU and Dice distributions, plus component sensitivity at mask IoU 0.10, 0.25, and 0.50.

Micro, macro, and image-level quantities are never combined under the label “aggregate recall.” Precision is null when no proposal exists. Recall is null for clean images and for a dataset with no positive truth instances. Clean-image rates are null when no clean images exist. Undefined values remain `null`/`N/A`; aggregation never converts them to zero.

## Ground-Truth Semantics

An anomaly-present image requires one or more immutable truth-instance IDs with valid non-empty masks or another approved annotation representation. A clean image has `ground_truth_status = no_anomaly` and zero truth instances; it does not require an empty mask file. A legacy empty clean mask may be read only with an explicit warning. Future v2 registration rejects clean images with non-empty anomaly masks, anomaly-present images without truth objects, and ambiguous empty anomaly annotations.

## Endpoints And Statistical Grouping

The v2 primary endpoint is component sensitivity at a predeclared clean-image false-proposal budget. Co-primary operational endpoints are clean false proposals per image and the proportion of clean images with any proposal. Preservation endpoints cover critical categories, thin/local anomalies, and predeclared category non-inferiority. Localisation and valid ranked sensitivity are separate endpoints.

The historical balanced score is retained only as an exploratory v1 field; it is not a v2 method selector. Future comparisons require immutable image pairing, category stratification, effect sizes, and confidence intervals that resample declared acquisition groups. Timing claims must state hardware and cache state.

Implementation details are in [Scientific Contract V2](scientific-contract-v2.md).
