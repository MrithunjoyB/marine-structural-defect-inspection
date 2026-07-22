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

The v2 store is not wired to the historical Streamlit executor. The separately authorised normal-feature development runner uses a caller-selected ignored v2 store and does not touch historical execution paths or databases.

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

## Reusable V2 Executor

`structvision.ExperimentExecutorV2` is the prospective execution path. It accepts an `ExperimentSpecificationV2` and ordered `ExperimentSample` records, verifies the specification hash and every selected image/ground-truth content hash, reconstructs `DetectorConfig` only from the frozen method payload, cross-checks the preprocessing/proposal/feature-scoring partitions, verifies the expected executed-configuration hash, and calls `StructuralAnomalyDetector.analyse` for every image-method pair.

The executor converts returned masks to the existing v2 one-to-one mask-IoU evaluation policy. Ordering is image-major and then specification-method order. The current deterministic implementation requires one worker. Fail-fast is explicit; when disabled, analysis failures produce auditable failed rows and processing continues. Expected, attempted, completed, failed, and skipped counts are immutable and reconciled.

No result is persisted unless a `ResultSink` is supplied. `MemoryResultSink` supports temporary lifecycle checks; `V2SQLiteResultSink` adapts the existing append-only v2 store at a caller-selected path. Neither executor path reads or writes the historical v1 databases, and neither uses `INSERT OR REPLACE`.

The original extraction work exercised this lifecycle only with temporary images, masks, specifications, and stores. The protected normal-feature development work below is a separate, explicitly non-confirmatory v2 execution.

## Protected Normal-Feature Development Experiment

`SYN-NORMAL-FEATURE-DEV-001` version 1 is the only new execution in this work. It is explicitly classified **development-only — non-confirmatory** and uses no historical test data. Its immutable manifest selects 91 clean `normal_fit` images and a disjoint 72-image `calibration_validation` role from train/validation metadata after exact, perceptual-candidate, source-group, and template-group exclusions.

The run sequence is fixed:

1. verify repository, registry, manifest, protected-file, environment-lock, and weight identities;
2. fit the official PatchCore memory/coreset from `normal_fit` only;
3. score validation maps and build a separate clean-false-proposal calibration artifact;
4. freeze the `fp-budget-0.50` point in an `ExperimentSpecificationV2`;
5. execute exactly one classical and one learned row for each validation image; and
6. audit row pairing, counters, artifact replay, historical-store immutability, and protected-file hashes.

Runtime artifacts are ignored. The committed manifest, lock, protocol, model/data cards, and descriptive results contain sufficient identities to reconstruct the run after separately acquiring the verified official weight. Commands and limitations are in [Normal-Feature Baseline](normal-feature-baseline.md); quantitative development diagnostics are in [Normal-Feature Development Results](results/normal-feature-development.md).
