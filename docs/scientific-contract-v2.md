# Scientific Evaluation And Provenance Contract V2

The prospective contract is implemented in the Streamlit-independent `scientific_contract/` package. It changes no proposal generation, preprocessing, scoring, thresholds, mask refinement, or specular-suppression behaviour.

## Evidence Status

| Item | Policy / status | Permitted interpretation | Required action |
|---|---|---|---|
| Historical automatic rows | `structvision-eval-v1-historical` | Engineering evidence under original permissive matching and ordering semantics | Preserve unchanged; do not recalculate claims |
| Future evaluation | `structvision-eval-v2` | Scientifically defined metrics after a valid immutable specification and execution | Use strict one-to-one assignment and append-only v2 storage |
| Current expanded comparison | Historical engineering comparison — not confirmatory | Negative/exploratory generator evidence only | Rebuild a non-overlapping development/confirmatory protocol before rerun |
| Confirmatory evidence | None currently | No confirmatory performance claim is supported | Acquire and lock independent data, then run a predeclared v2 experiment |
| Zero near-duplicate leakage | Unsupported for the expanded study | The legacy 64-bit hash screen cannot prove absence | Use exact, perceptual-candidate, source, template, and acquisition-group audit evidence |
| Historical balanced score | Exploratory v1 field | Describes one arbitrary weighting only | Do not use as the v2 primary method selector |

## Module Boundaries

- `evaluation_policy.py` defines policy identity, denominators, thresholds, endpoints, and hashes.
- `matching.py` stores lossless masks, validates ranking and annotation semantics, and performs deterministic one-to-one matching.
- `metrics.py` separates micro, macro, image-level, clean, localisation, ranked, category, nuisance, acquisition-group, and efficiency quantities.
- `provenance.py` captures Git and runtime state without writing files.
- `specification.py` freezes the complete selected data, methods, configuration, evaluation identity, code state, and environment.
- `result_store.py` supplies the explicit append-only v2 SQLite sink.
- `dataset_audit.py` performs read-only exact, perceptual-candidate, and declared-group comparison.

Imports have no filesystem side effects. Analysis functions accept explicit inputs and return typed records. Only construction of an explicit `V2ResultStore` writes to its caller-selected path.

## Matching And Reconstruction

The matcher records the complete similarity matrix and lossless proposal/truth mask encodings. Valid edges require the named mask-IoU threshold. Deterministic assignment maximises valid match count and then total IoU. Every proposal decision includes an assigned truth ID or unmatched reason, IoU, Dice, truth overlap, proposal overlap, threshold, metric, score/rank, and a centroid diagnostic that cannot grant a match. The stored evidence can reconstruct evaluation without running inference.

## Result Identity And Counters

The specification hash is deterministic over every scientific field, including ordered image/method identity and expected executed-configuration hashes. Results that disagree with the specification's image hash, truth hash, method version, configuration hash, evaluation policy, or deterministic state are rejected. One transaction appends an immutable attempt and its unique image-method rows. Expected pairs equal selected images multiplied by selected methods; counters must reconcile exactly.

Schema-version and migration-history rows document the v2 database layout. Foreign keys bind results to registered specifications and attempts. Explicit supersession links preserve both prior and successor rows.

## Scope Limit

This contract makes future evaluation defensible and future evidence reproducible. It does not repair historical metrics, establish publication readiness, validate any algorithm, or authorise a new experiment.

The normal-feature reference matrix is a prospective v2 execution but remains **development-only — non-confirmatory** because its 72-image validation role supplies both calibration and diagnostics. Contract compliance establishes traceability and metric semantics; it does not establish independence, external validity, or a confirmatory claim.

## Executable API Binding

The prospective executor binds the v2 contract to `structvision-classical-baseline-v1-frozen` version `1.0.0`. The method's frozen configuration payload is a complete canonical `DetectorConfig`; the separate preprocessing, proposal, and feature/scoring partitions must reproduce the same values. The executor rejects a different implementation identity, configuration hash, maximum proposal count, `structvision` seed, or deterministic-mode state before analysis.

Selected files are verified against their immutable encoded-content SHA-256 values before decoding. Ground-truth masks are resized only after encoded provenance verification, using nearest-neighbour interpolation to match the analysed image coordinate space. Returned proposal masks then enter `structvision-eval-v2` through the existing ranked `ProposalSet` and deterministic one-to-one matching policy.

Execution has no implicit persistence. A null or absent result sink leaves no store. The explicit SQLite sink registers the immutable specification and appends one attempt plus its unique image-method rows through the existing v2 store. Reusing an attempt identity fails under append-only constraints. Historical v1 method IDs, databases, rows, and metric meanings are unchanged.

`LearnedExperimentExecutorV2` extends this boundary for a strict paired matrix containing exactly the frozen classical method and the predeclared PatchCore baseline. It preserves the existing one-to-one matcher and row schema, records learned operating-point, calibration, and model identities in executed configuration, and writes through the existing append-only attempt store. It neither fuses the methods nor changes historical execution code. The completed development attempt contains 144 rows (72 per method), with no failures or skips; its immutable identities are listed in [Normal-Feature Development Results](results/normal-feature-development.md).

`HybridDevelopmentExperimentExecutorV2` adds a separate strict matrix containing exactly the frozen classical, protected PatchCore, and proposal-guided hybrid identities. Its specification binds the three-role manifest, hybrid-specific model, complete fusion artifact, primary 0.50 budget, and unchanged v2 policy. A separate append-only ledger rejects a second primary holdout attempt. The completed matrix contains 216 rows (72 per method); auxiliary budgets only reselect the already-computed complete candidate diagnostics. Contract compliance does not override the rejected preservation decision or make this cohort confirmatory.

See [Experiments](experiments.md) for the lifecycle and [Reusable API](reusable-api.md) for direct usage.
