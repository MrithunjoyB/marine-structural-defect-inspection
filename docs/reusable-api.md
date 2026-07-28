# Reusable Local API

The `structvision` package exposes the frozen classical anomaly-proposal baseline without requiring Streamlit, a database, an output directory, a model download, or a commercial API key. It is an in-memory proposal interface, not a defect classifier or engineering diagnostic system.

## Public Interface

```python
from structvision import DetectorConfig, StructuralAnomalyDetector

config = DetectorConfig()
detector = StructuralAnomalyDetector(config)
result = detector.analyse("inspection.png", image_id="frame-001")
```

For arrays, colour order is explicit:

```python
result = detector.analyse(rgb_array, image_id="frame-002", colour_space="RGB")
```

Ordered batches use typed samples and default to a single worker:

```python
from structvision import AnalysisSample

batch = detector.analyse_batch([
    AnalysisSample(first_bgr, "first", "BGR"),
    AnalysisSample(second_gray, "second"),
])
```

Batch input order and successful result order are stable. Duplicate image IDs fail before execution. `fail_fast=False` isolates per-image failures in typed `BatchFailure` records; failures are never silently skipped. Worker counts other than one are rejected by the frozen adapter.

## Input Contract

- Paths are decoded by OpenCV as grayscale, BGR, or BGRA. Three-channel path inputs therefore need no colour declaration; if supplied, it must agree with BGR decoding.
- `HxW` and `HxWx1` `uint8` arrays are grayscale and unambiguous.
- `HxWx3` `uint8` arrays require `colour_space="RGB"` or `"BGR"`.
- `HxWx4` `uint8` arrays require `RGBA` or `BGRA` plus `alpha_handling="drop"`, `"composite_black"`, or `"composite_white"`.
- Other dtypes, empty arrays, and unsupported channel layouts fail explicitly.

The input hash covers the normalised BGR pixel shape, dtype, order, and bytes. Filesystem provenance separately records the SHA-256 of encoded source bytes. Optional image metadata must be JSON-compatible and becomes immutable in the result.

## Configuration Contract

`DetectorConfig` and its nested preprocessing, feature, proposal, and scoring records are frozen dataclasses. Construction rejects missing/unknown serialised fields, NaN, infinity, invalid ranges, mutable scoring collections, and a different implementation identity. `to_json()` is canonical, round-trippable JSON; `configuration_hash` is its SHA-256.

Defaults reproduce the frozen legacy arguments: resize width 1024, denoising and CLAHE enabled, sharpening disabled, neutral brightness/contrast, the existing feature sensitivities and threshold, the existing area/border/top-eight proposal settings, every default contextual/refinement switch enabled, and experimental specular suppression disabled. Protected internal kernels, feature weights, percentiles, scale ratios, merge/NMS thresholds, and refinement constants remain immutable v1 implementation constants and are represented by `implementation_constants_hash`.

## Result And Mask Contract

`AnalysisResult` contains image identity/hash, analysed `H×W×3` shape, ordered proposals, anomaly heatmap, preprocessing metadata, configuration/implementation identity, deterministic state, timings, warnings, diagnostics, provenance, and optional image metadata. It contains no output path.

Each immutable `Proposal` contains:

- proposal ID and contiguous one-based rank;
- half-open `(x_min, y_min, x_max, y_max)` box recomputed from the final mask;
- final and raw masks in analysed-image coordinates;
- proposal/evidence/priority scores and heuristic mask reliability;
- feature contributions, area, centroid, contextual diagnostics, warnings, and implementation identity.

Masks are read-only contiguous `uint8` arrays with values 0 and 255. Their serialised representation records shape, dtype, C order, raw-byte base64, and SHA-256. Heuristic or mask reliability is not calibrated confidence.

## Write Contract And Explicit Sinks

`analyse` and `analyse_batch` create no reports, databases, repository caches, output folders, thumbnails, or caller-directory files. The frozen legacy function's unavoidable intermediate files are confined to a controlled temporary directory and deleted before return.

An explicitly injected `ArtifactSink` may receive an analysis result. `NullArtifactSink` discards it. V2 execution similarly persists only through an injected `ResultSink`; `MemoryResultSink` is process-local, and `V2SQLiteResultSink` uses a caller-selected append-only v2 database.

## Limitations

The adapter intentionally preserves the protected v1 algorithm and its memory behavior. It does not provide parallel execution, physical scale, calibrated uncertainty, learned classification, real-world validation, or private-data parsing. Large images can require several full-resolution feature maps and temporary visualisations. A future private-data adapter must remain outside the detector core and supply explicit colour, image identity, provenance, and ground-truth semantics.

## Optional Normal-Feature API

The learned API is a separate, versioned development baseline:

```python
from structvision.normal_feature import NormalFeatureAnomalyDetector, NormalFeatureConfig

detector = NormalFeatureAnomalyDetector(
    NormalFeatureConfig(),
    weight_file=verified_weight_path,
    environment_lock_hash=lock_sha256,
)
model = detector.fit_normal(fit_samples, normal_fit_manifest_hash=manifest_hash)
result = detector.analyse(
    image,
    image_id="development-image-001",
    model_artifact=model,
    calibration_artifact=calibration,
    operating_point_id="fp-budget-0.50",
)
```

The default call path is write-free. Explicit directory sinks are required to persist model or calibration artifacts. The input contract is RGB with deterministic aspect-preserving letterboxing; outputs contain a raw full-resolution PatchCore distance map and separately calibrated, unlabelled connected-component proposals. Learned distances are not probabilities and are not comparable to classical review-priority scores. See [the full learned contract](normal-feature-baseline.md).
