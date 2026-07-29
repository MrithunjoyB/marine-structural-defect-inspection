# Reusable Package Architecture

The extraction adds one focused `src` package while retaining the legacy application and protected algorithm files in place.

```text
src/structvision/
├── __init__.py       stable exports
├── api.py            direct and ordered-batch orchestration
├── configuration.py immutable executable configuration and hashing
├── types.py          immutable proposal, result, and batch records
├── inputs.py         explicit path/array/colour/alpha normalisation
├── classical.py      protected legacy compatibility adapter
├── executor.py       prospective ExperimentSpecificationV2 executor
├── provenance.py     source/runtime provenance records
├── storage.py        named roots, translations, private resource bindings
├── operational_storage.py shared no-config/external operational context
├── resources.py      hash-verified protected-resource discovery
├── protected_access.py immutable registry and experiment-store readers
├── legacy_paths.py   read-only role-scoped immutable-path translation
├── sinks.py          explicit artifact/result sink boundaries
└── errors.py         typed fail-closed exceptions
```

The optional learned path adds isolated modules without changing that classical dependency graph:

```text
src/structvision/
├── development_protocol.py       read-only protected-cohort selection
├── learned_executor.py           two-method v2 matrix adaptation
├── hybrid/
│   ├── protocol.py               three-role group-aware protected manifest
│   ├── features.py               aligned candidate-level evidence
│   ├── selection.py              fusion-fit-only constrained enumeration
│   ├── artifact.py               immutable fusion identity and replay
│   ├── detector.py               separate public hybrid detector
│   └── experiment.py             one-shot three-method v2 holdout
└── normal_feature/
    ├── configuration.py          one fixed development configuration
    ├── preprocessing.py          aspect-preserving letterbox/inverse projection
    ├── patchcore.py               lazy official Anomalib adapter
    ├── model_artifact.py          immutable memory/coreset artifact
    ├── calibration.py             separate clean-burden calibration artifact
    ├── proposal_extraction.py     deterministic map-to-component policy
    └── evaluation.py              dense development diagnostics
```

## Dependency Direction

`api.py` depends on input normalisation, immutable configuration/types, and the classical adapter. The adapter imports only the protected preprocessing, feature, scoring, and proposal modules at execution time; it does not import the Streamlit application. Core analysis has no dataset registry, historical database, report writer, session state, sidebar state, or global output path.

`executor.py` depends on the already-established prospective `scientific_contract` specification, matching, metrics, and append-only row types. It calls `StructuralAnomalyDetector` rather than a second algorithm path. `sinks.py` contains the optional explicit adapter to `V2ResultStore`; constructing that sink is what authorises creation of the caller-selected v2 database.

## Frozen Adapter Boundary

The protected implementation writes masks and diagnostic overlays through module-level paths before returning. Changing that implementation would alter protected hashes and widen algorithm risk. The compatibility adapter therefore serialises calls under one lock, redirects the path globals to a `TemporaryDirectory`, calls the same protected functions, loads exact final/raw mask bytes into immutable arrays, removes path-only diagnostics, restores the globals, and deletes the temporary directory. Provenance reports the observed protected source hashes and the number of contained temporary artifacts.

This design is deliberately single-worker. It protects proposal order and module-global redirection, but it is not intended as a high-throughput service boundary.

## Parity Contract

Deterministic generated fixtures cover clean texture, thin crack, pitting, weld disturbance, specular highlights, border artifacts, illumination gradients, grayscale, RGBA with explicit alpha composition, non-square input, a small valid image, and a bounded large image. Direct and API paths must have identical proposal counts/order/IDs/ranks, integer half-open boxes, final/raw mask bytes, accessible scores, component contributions, diagnostic keys/values, and warnings. No field uses a broad tolerance. Candidate rejection identities cannot be compared because the protected function exposes only aggregate rejection reasons; that limitation is retained rather than inferred.

## Package And UI Boundaries

The supported Streamlit interface is `apps/structvision_demo.py`. It consumes
the public operational storage context, retains upload bytes in memory, and
offers exports only through explicit browser controls. The old root-level
`app.py` is a disabled legacy research interface and stops before importing any
mutable path or store. Its mutable monolithic workflow is not an external
storage consumer. External protected evidence is available only through the
read-only readers; the legacy registered-experiment executor is refused in
external mode. See [Portable Storage and Immutable Legacy Paths](storage-portability.md).

## Normal-Feature Isolation Boundary

Importing `structvision` or `structvision.normal_feature` does not import Torch, Anomalib, timm, Streamlit, or a database driver. Heavy learned dependencies are resolved lazily only when fitting or scoring is requested. The lightweight classical install is therefore unchanged. `NormalFeatureAnomalyDetector` verifies exact dependency versions, the pinned pretrained-weight hash, CPU reference policy, environment-lock hash, manifest identity, and artifact identity before calling the official Anomalib `PatchcoreModel` and `KCenterGreedy` components.

Upstream owns embedding, coreset selection, nearest-neighbour distances, image scoring, and anomaly maps. StructVision owns protected cohort construction, letterboxing, provenance, immutable persistence, inverse projection, calibration, component extraction, and v2 row adaptation. Scores and proposal types never cross the frozen classical implementation boundary. See [Normal-Feature Baseline](normal-feature-baseline.md).

## Hybrid Isolation Boundary

The hybrid package depends on the two public detector paths but neither baseline depends on it. It consumes complete immutable classical proposals and a full-resolution PatchCore map, verifies coordinate and input-hash equality, extracts features without truth/category/filename inputs, and emits new typed records under `structvision-proposal-guided-hybrid-v1-dev`. Classical masks remain byte-identical. Fusion fitting receives a capability object exposing fusion-fit identities only; holdout identities are loaded later by a one-shot executor. Runtime persistence remains sink-controlled and uses a new v2 store. See [Proposal-Guided Hybrid](proposal-guided-hybrid.md).

See [Reusable API](reusable-api.md), [Methodology](methodology.md), and [Scientific Contract V2](scientific-contract-v2.md).
