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
├── sinks.py          explicit artifact/result sink boundaries
└── errors.py         typed fail-closed exceptions
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

The existing Streamlit UI remains operational and unchanged. It continues to own uploads, feature-map persistence, ablation CSV output, manual review, reports, exports, optional YOLO integration, and historical registered-experiment controls. Those paths are not dependencies of the reusable package. Future UI migration can call the public API incrementally after a separate compatibility plan; it is not part of this extraction.

See [Reusable API](reusable-api.md), [Methodology](methodology.md), and [Scientific Contract V2](scientific-contract-v2.md).
