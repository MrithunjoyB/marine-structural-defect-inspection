# Data Card: Hybrid Development Protocol

## Identity

- Protocol: `structvision-hybrid-development-v1`
- Manifest: `a1e6f9a83e5e8d73275236e6dc4fafd985e6e1ef2c4aef21fd4156dc821829a4`
- Source: eligible `synthetic-expanded` v1.0 train and validation records only
- Classification: **development holdout — non-confirmatory**

## Composition

`hybrid_normal_fit` contains 70 clean training images. `hybrid_fusion_fit` contains 126 training images: 19 clean and 107 positive. `hybrid_development_holdout` contains 72 validation images: 34 clean and 38 positive. The positive fusion/holdout roles both contain colour-only, pitting, texture-only, thin-crack, and weld cases; all three priority categories are present.

## Allocation and protection

Clean training identity components are allocated prospectively and deterministically by category using seed 73021: one quarter of eligible groups, rounded with at least one group, goes to fusion fit and the remainder to normal fit. All eligible positive train components go to fusion fit and all eligible validation components go to holdout.

The selector reads registry metadata in SQLite read-only mode, excludes pilot/history/test exact matches, legacy dHash≤3 candidates, declared group crossings, unresolved provenance, and train identities connected to validation. Image and positive-truth files are hash verified; clean truth uses an exact zero-mask identity. The manifest contains paths and hashes, not images.

## Appropriate use

Normal fit may construct only the hybrid-specific clean memory. Fusion fit may define normalisation, coefficients, thresholds, and preservation checks. Holdout may be loaded once after freezing and may not tune this implementation. The data must not be relabelled as independent test, confirmatory, real-world, professor-provided, or transferable evidence.

## Limitations

All records come from one synthetic generator family; acquisition groups are empty; the validation cohort was previously inspected during PatchCore development; perceptual screening cannot prove semantic independence; and fusion fit has only 19 clean images. Images, model weights, memories, and runtime result stores remain ignored.
