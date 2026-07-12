# Evaluation Metrics

This document defines the automatic proposal metrics and their denominators.

## Matching Rule

A proposal matches a connected ground-truth instance when at least one configured condition is satisfied:

1. proposal-to-ground-truth mask IoU reaches the IoU threshold;
2. intersection relative to ground-truth area reaches the overlap threshold; or
3. the proposal centroid lies inside a thin ground-truth instance when the fallback is enabled.

The matching configuration belongs to the experiment identity. Results from different thresholds should not be treated as directly paired observations.

## Top-K Proposal Recall

For an anomaly-present image, Top-\(K\) is a hit when at least one of the first \(K\) ranked proposals matches verified ground truth:

$$
\operatorname{TopKRecall}=\frac{\text{eligible positive images with a match at rank }\le K}{\text{eligible positive images}}.
$$

Clean and unknown-outcome images do not enter the denominator. If no positive image is eligible, recall is undefined and displayed as `N/A`, not zero.

## Proposal Precision And Recall

$$
\operatorname{Precision}=\frac{\text{matched proposals}}{\text{all final proposals}},
$$

$$
\operatorname{Recall}=\frac{\text{matched ground-truth instances}}{\text{all ground-truth instances}}.
$$

For a clean image, proposal recall is undefined. Every proposal is a false proposal, while zero proposals constitute a correct zero-proposal outcome. Aggregate precision in mixed datasets includes clean-image proposal burden according to the stored per-image values.

## Localisation

Mean IoU averages IoU over matched proposals for an image. Best IoU is the maximum matched IoU. When no localisation-eligible ground truth exists, these metrics are not used as positive-image evidence. False-negative anomalies count unmatched ground-truth connected components.

## Review Burden

False proposals are unmatched final proposals. Processing time is measured per image-method execution and is hardware- and load-dependent. First true-anomaly rank is the first ranked matched proposal and is undefined for clean or missed images.

Manual annotation-efficiency records use accepted, rejected, uncertain, and not-reviewed states. Not-reviewed proposals are not equivalent to rejected proposals. Automatic ground-truth rows and manual records are stored and interpreted separately.

## Category-Wise Analysis

Category tables report image count, positive/clean counts, Top-1/3/5/8 recall, precision, recall, IoU, false proposals, false negatives, first-hit rank, and processing time by method. Positive recall denominators include anomaly-present images only. Clean categories report false-proposal robustness and correct zero-proposal outcomes.

## Strict Paired Comparison

Multi-scale fused and refined contextual rows are paired by experiment ID, experiment version, and image ID. Dataset ID/version/split must be consistent. Duplicate image-method rows or unmatched advanced-method images block comparison.

Metric-specific outcomes use a numerical tolerance of \(10^{-9}\). Higher is better for precision, recall, and IoU; lower is better for false proposals, processing time, and first-hit rank. The composite outcome uses detection and false-alarm metrics. A localisation-only gain is reported as localisation improvement while the overall detection outcome remains tied.

## Bootstrap Intervals

Bootstrap confidence intervals resample paired image-level differences with a deterministic seed. Precision and false-proposal differences may use all strict pairs. Recall uses anomaly-present pairs with defined recall. IoU uses eligible anomaly/localisation pairs. The report states eligible and excluded counts and never replaces undefined values with zero.

Intervals from the current 12-image controlled test are descriptive. They do not support broad significance or real-world performance claims.
