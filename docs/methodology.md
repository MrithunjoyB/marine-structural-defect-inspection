# Methodology

This document describes the implemented visual anomaly-proposal pipeline. The central current algorithm is a classical and contextual image-analysis pipeline, not a foundation-model-assisted detector. Optional downstream learned integrations are separate future capabilities. The method is not a certified defect detector or engineering-diagnosis system.

## Input And Preprocessing

The application accepts common raster image formats. Configurable preprocessing includes resize, brightness and contrast adjustment, denoising, CLAHE enhancement, and sharpening. The processed BGR image is the common input for feature extraction and proposal generation; experiment configurations record the selected settings.

## Feature Evidence

`feature_extraction.py` derives complementary evidence rather than relying on one segmentation mask:

- grayscale intensity and a smoothed grayscale reference;
- Canny edges;
- Sobel gradient magnitude;
- absolute Laplacian response;
- foreground thresholding;
- local intensity variance;
- local binary-pattern deviation;
- Lab-colour deviation from a local mean; and
- a weighted continuous anomaly-strength map and colour heatmap.

The feature maps represent visual evidence, not defect probabilities. High response may correspond to valid boundaries, illumination effects, weld geometry, texture, or imaging artefacts.

## Multi-Scale Proposal Generation

Independent percentile masks are produced from feature channels. Overlapping patches at several fractions of the shorter image dimension are scored from edge, gradient, texture, colour, entropy, and contrast evidence. Patch measurements are projected into a dense tile-score field and combined with feature votes and anomaly strength.

Morphological processing at several scales preserves compact spots, elongated lines, and broader irregular regions. Connected components form the raw candidate set. The diagnostic sequence is:

1. raw connected components;
2. area and border filtering;
3. coherence splitting and optional mask refinement;
4. similarity-aware merging;
5. overlap suppression;
6. contextual metric calculation and ranking; and
7. configurable top-\(K\) selection, with a default maximum of eight.

Assertions enforce non-empty masks, bounding boxes derived from final masks, a final count within the configured limit, and a merge count no greater than the pre-merge count.

## Candidate Filtering And Grouping

Candidates are checked against absolute and relative area limits. A valid-image mask detects near-uniform or black border bands, and border occupancy contributes to rejection and score penalties. Compact candidates with weak contextual evidence are removed. Large candidates with heterogeneous internal heatmap response are split around connected peaks or rejected when coherence remains low.

Merging uses overlap, containment, centre distance, morphological connectivity, and compatible local evidence. Non-maximum suppression removes nested or substantially overlapping duplicates before ranking.

## Contextual Evidence

Every candidate is compared with a dilated context ring. Implemented terms include candidate-to-context differences in texture, Lab colour, entropy, and gradient, together with internal-versus-boundary edge concentration and geometric irregularity. Robust calibration across candidates uses median and interquartile-range statistics with clipping.

The evidence score is

$$
E = 100\frac{\sum_i w_i e_i}{\sum_i w_i},
$$

where \(e_i\) are calibrated contextual and geometric terms. The reliability score is

$$
R = 100\frac{\sum_j v_j r_j}{\sum_j v_j},
$$

where \(r_j\) includes perturbation stability, connectedness, boundary smoothness, scale agreement, and segmentation coherence. Review priority is

$$
P = 100\frac{u_E(E/100)+u_R(R/100)+u_A A+u_N N}{u_E+u_R+u_A+u_N}.
$$

Here \(A\) is area relevance and \(N\) is novelty. Default weights are declared in `scoring.py`. Reliability cannot independently create high anomaly evidence.

## Stability And Mask Refinement

Mask stability is assessed under controlled brightness, contrast, Gaussian-noise, and resize perturbations. Refinement uses heatmap-aware thresholding, morphology, small-component removal, limited hole filling, and boundary smoothing. Final masks retain coherent connected content, and region boxes are recomputed after cleanup.

Saved evidence includes raw masks, refined masks, context rings, combined masks, proposal overlays, and stage diagnostics. Reviewers may create corrected reference masks without overwriting the proposal outputs.

## Baselines

The repository evaluates four proposal definitions:

- **contour-only baseline:** connected components from Canny contour evidence;
- **fixed-threshold baseline:** components from anomaly strength above 128;
- **multi-scale fused method:** ranked raw masks from fused multi-scale candidates; and
- **refined contextual method:** final masks after contextual scoring and refinement.

The baseline definitions are executed on identical registered images for paired analysis.

## Ablation Controls

`AblationConfig` provides opt-in switches for feature families, local contextual terms, stability, internal/boundary-edge evidence, border penalty, coherence, multi-scale fusion, merging, and refinement. `ABL-FULL` equals the normal default configuration. Ablation configurations reuse the proposal pipeline and store stable IDs and reproducibility snapshots; they are not independent reimplementations.

### Experimental Specular Suppression

`ABL-RERANK-SPECULAR-SUPPRESS` inherits the stored historical configuration ID `ABL-RERANK-ONLY`, displayed descriptively as the **single-scale contextual classical baseline**, and enables a continuous candidate-level specular likelihood. The score combines high-value/low-saturation occupancy, Lab chroma, RGB channel agreement, intensity smoothness, entropy, candidate-to-context brightness, compactness, and eroded-core texture/gradient evidence. It does not use category labels, filenames, or ground truth.

The effective likelihood is reduced by two structural safeguards. The crack safeguard uses rotated elongation, thin mask occupancy, and scale agreement. The pitting safeguard uses multiple connected components and irregular, textured core evidence. The historical policy applies a ranking penalty and may reject candidates above its configured effective-likelihood threshold with weak structural evidence. Diagnostics retain component values, safeguards, before/after scores, decision, and rejection reason. Suppression is experimental, disabled by default, and not validated by the current historical evidence.

## Implementation Boundaries

The proposal method does not estimate structural capacity, defect severity, material loss, or repair priority. Reported priority is a visual review order. Domain claims require licensed real data, expert ground truth, calibrated acquisition, and an appropriate engineering validation protocol.
