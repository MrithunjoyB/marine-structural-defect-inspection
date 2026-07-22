# Proposal-Guided Hybrid Development Candidate

## Identity and status

`structvision-proposal-guided-hybrid-v1-dev` version `1.0.0-dev` is a separate development implementation. It does not modify `structvision-classical-baseline-v1-frozen` or `structvision-patchcore-baseline-v1-dev`. The completed experiment is **development holdout — non-confirmatory**, and the candidate is **rejected under the predeclared protocol**. This status must not be changed by inspecting or rerunning the holdout.

The development hypothesis was that PatchCore normality evidence could reduce the frozen classical method's clean proposal burden at 0.50 clean false proposals per image while preserving overall, image-level, thin-crack, pitting, weld, and localisation performance. The holdout reduced burden but missed two preservation criteria; see [Proposal-Guided Hybrid Development Results](results/proposal-guided-hybrid-development.md).

## Protected protocol

Protocol `structvision-hybrid-development-v1` uses only eligible `synthetic-expanded` v1.0 train and validation identities. Its committed [manifest](../development_data/hybrid-development-manifest-v1.json) has logical SHA-256 `a1e6f9a83e5e8d73275236e6dc4fafd985e6e1ef2c4aef21fd4156dc821829a4` and three disjoint roles:

- `hybrid_normal_fit`: 70 clean training images used only to fit a hybrid-specific PatchCore memory;
- `hybrid_fusion_fit`: 126 training images (19 clean, 107 positive) used for feature normalisation, coefficient/threshold selection, and preservation checks; and
- `hybrid_development_holdout`: 72 validation images (34 clean, 38 positive), loaded once only after artifact and policy freezing.

Exact hashes, legacy dHash candidates at distance at most three, and source/template/acquisition groups cannot cross roles. Pilot overlap, historical-test overlap, prior-verification overlap, unresolved provenance, test roles, and anomaly-positive normal-fit records fail closed. The fusion-fit capability object contains no holdout identity or path.

## Candidate evidence

The hybrid starts from the complete frozen classical candidate set and never changes its masks, boxes, preprocessing, scoring, or proposal generation. For each candidate it calculates eight fixed, finite features in this order:

1. frozen classical priority;
2. frozen classical anomaly/evidence score;
3. frozen heuristic mask reliability;
4. PatchCore mean distance inside the mask;
5. PatchCore 0.95 quantile inside the mask;
6. fraction of mask pixels above the clean fusion-fit map reference;
7. inside mean minus deterministic context-ring mean; and
8. local spatial agreement: high-distance pixels inside divided by high-distance pixels in mask plus ring.

The context-ring radius is `max(2, round(sqrt(mask area) / 8))`. The high-distance reference is the median clean fusion-fit image q95, `42.226993560791016`. Each feature is scaled by its fusion-fit empirical q05/q95 and clipped to `[0,1]`. Category, filename, truth, and ground-truth class are not inference inputs.

## Fusion and selection

Let (C) be the mean of the three normalised classical features and (N) the mean of the five normalised PatchCore features. The selected formula is

\[
S = 0.60C + 0.40N.
\]

`S` is an explainable ranking heuristic, not a calibrated probability. The complete search enumerated classical weights `0.90, 0.80, 0.70, 0.60, 0.50`, complementary normality weights, and optional category-blind classical preservation floors `none, 0.90, 0.80`—15 configurations total. All weights are non-negative, sum to one, and retain a positive classical contribution. The selected configuration uses no preservation-floor branch. At the primary 0.50 budget its fusion-fit threshold is `0.4704560134385654`.

Selection used only fusion-fit data. Eligible configurations had to meet the clean-FP budget and every predeclared preservation margin. Ties maximised micro sensitivity, minimised clean burden, maximised assigned-pair IoU, then preferred no floor and the more conservative classical weight. Every accepted and rejected configuration remains in fusion artifact `a21b5880c5d8f16d3869227455279ddbf18815d92ae7862e262cc2560de3d8d1` in the ignored reference-run directory.

## Public API

```python
from structvision import DetectorConfig, StructuralAnomalyDetector
from structvision.hybrid import ProposalGuidedHybridDetector

hybrid = ProposalGuidedHybridDetector(
    classical_detector=StructuralAnomalyDetector(DetectorConfig()),
    normal_feature_detector=normal_detector,
    normal_feature_model_artifact=hybrid_model_artifact,
    fusion_artifact=fusion_artifact,
)
result = hybrid.analyse("frame.png", image_id="frame-001")
```

`HybridAnalysisResult` retains the original classical candidate count, every pre-threshold candidate diagnostic, selected proposals, unique contiguous ranks, half-open boxes, byte-identical classical masks, raw/normalised component evidence, score contributions, threshold reason, artifact identities, warnings, provenance, and timing. Persistence requires an explicit sink. No Streamlit module, API key, paid service, historical database, or implicit output path is required.

## Reproduction boundary

Run `hybrid_development.py` only in the exact locked Python 3.12 environment and only into a new empty output directory. The script fits the normal artifact and fusion artifact before creating the holdout ledger, records exactly one primary attempt, writes a new append-only v2 store, and reselects the already-computed candidate list for auxiliary budgets without rerunning inference.

Do not rerun the completed holdout to improve this candidate. A new scientific question requires a new protocol and implementation/artifact identity.

## Limitations

The data are deterministic synthetic generator output, the validation cohort was previously inspected in PatchCore work, and the holdout is not independent or confirmatory. The hybrid retains the classical proposal ceiling, depends on normal-memory representativeness and PatchCore spatial resolution, is not uncertainty calibrated, and has no real-world, professor-data, deployment, transferability, novelty, or global-superiority evidence.
