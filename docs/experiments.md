# Experiments

StructVision-AI now separates the historical experiment mechanism from the prospective immutable scientific contract.

## Historical V1 Plans And Rows

The existing registry plans and the 888 automatic rows are preserved under `structvision-eval-v1-historical`. Historical plans recorded useful identity fields, but some contain empty placeholder configuration dictionaries, do not hash image and ground-truth content into every result, and are not immutably tied to the actual executed method matrix. Resume, overwrite, result deletion, and post-hoc experiment reassignment are available in those engineering paths.

Historical matching is permissive and non-one-to-one. Historical baseline Top-K may use unordered component output. These records remain inspectable engineering evidence; they are not upgraded, migrated, or made publication-valid by the v2 implementation.

## Immutable V2 Specification

Every future scientific run begins with a frozen `ExperimentSpecificationV2`. It contains or hashes:

- experiment, dataset, split-manifest, and split-lock identity;
- ordered selected image IDs, image hashes, and ground-truth hashes;
- selected method IDs, implementation versions, full method settings, and ranking definitions;
- complete preprocessing, proposal, feature, scoring, threshold, maximum-proposal, and seed settings;
- deterministic-mode state and the complete evaluation-policy hash;
- allowed fitting/calibration splits and an explicit test-access prohibition;
- Git commit, clean/dirty state, and a diff-content hash when dirty;
- Python, dependency, operating-system, hardware, OpenCV, and backend metadata; and
- a deterministic content-addressed specification hash.

Empty configuration dictionaries are invalid. Expected execution pairs always equal selected images multiplied by selected methods.

## Fail-Closed Execution Boundary

Future execution must load the immutable specification and construct the executable configuration only from it. The executable configuration hash is compared with the method-specific expected hash before any result is accepted. A changed preprocessing option, proposal limit, method parameter, scoring value, seed, deterministic flag, or evaluation setting fails closed.

Attempt counters record expected, attempted, completed, failed, skipped, and unique stored pairs. Each result persists the specification hash, executed-configuration hash, method implementation identity, evaluation-policy identity and hash, image and truth hashes, proposal artifact hash or complete proposal details, matching-policy hash, attempt ID, deterministic state, timestamp, and completion status.

## Append-Only V2 Storage

`scientific_contract.result_store.V2ResultStore` creates a caller-selected side-by-side database. It uses transactions, foreign keys, uniqueness constraints, schema-version records, migration history, and plain append-only inserts. It has no overwrite, delete, result-reassignment, or in-place historical migration operation. Corrections are new rows linked by explicit supersession records.

The v2 store is not wired to the historical Streamlit executor in this work package because no new benchmark execution is authorised. Tests exercise the complete specification, validation, and storage boundary only in temporary directories.

## Historical Protocols

The existing controlled studies remain available exactly as recorded:

- `SYN-BALANCED-001` used 12 `synthetic-controlled` test images and four historical methods.
- `ABL-SYN-BALANCED-001` stored the historical ablation rows.
- `SYN-SPECULAR-SUPPRESS-001` retained both a negative first version and a second exploratory version.
- `SYN-EXPANDED-VALIDATION-001` stored 600 rows across six methods on 100 expanded test images, although its registry plan listed four base methods.

`ABL-RERANK-ONLY` remains the stored method ID. Its descriptive display label is **single-scale contextual classical baseline**. The v1 balanced score and metric differences do not prove the causal value of reranking or justify choosing this method under v2.

The expanded comparison is classified as **historical engineering comparison — not confirmatory** because the 80-image pilot is contained byte-for-byte in the 500-image dataset and includes 13 final-test images. See [Historical Dataset Overlap Audit](audits/historical-dataset-overlap.md).

## Future Execution Requirements

A future rerun must create a new v2 specification and new v2 result database, use one-to-one matching, mark unordered methods Top-K-ineligible, validate canonical clean-image annotations, lock non-overlapping development and confirmatory data, and predeclare the false-proposal budget and category preservation margins. Historical rows must not be copied into the v2 schema as if they had been generated under v2 semantics.
