# Portable Storage and Immutable Legacy Paths

## Scope and status

This document defines the operational path-portability boundary for the clean
StructVision-AI installation. The supported CLI, modern Streamlit client,
technical-handoff builder, and read-only protected-evidence readers share one
typed operational storage context. This source repair does not create or
activate a real local configuration.

The retired whole-tree iCloud relocation procedure must not be reused. Its
cross-FileProvider rename verifier was not a valid content-transfer design.
A future installation is instead a clean Git checkout with separately
configured runtime roots.

## Storage architecture

The intended long-term layout separates version-controlled source from runtime
and protected content:

```text
clean Git checkout
└── source_root

~/StructVision/
├── Runs/                 runs_root
├── Trash/                trash_root
├── Cache/                artifact_cache_root
├── Releases/             release_root
├── PrivateData/          private_data_root
└── Protected/            protected_root
    ├── Registry/         registry_root
    ├── ResearchData/     research_data_root
    ├── ExperimentStores/ experiment_store_root
    ├── LearnedArtifacts/ learned_artifact_root
    └── HistoricalReports/historical_report_root
```

`source_root` is discovered from the installed package/checkout or supplied
explicitly. No source file contains an account-specific path. `~/StructVision/`
is a proposal only: this task does not create any of these directories.

The typed access policy is conservative. Source, protected, registry, research,
experiment-store, learned-artifact, and historical-report roots are read-only.
Runs, trash, cache, release, and private-data roots are writable. A caller must
still explicitly perform any creation or write; resolving configuration never
does so.

## Configuration location and schema

The preferred local macOS file is:

```text
~/Library/Application Support/StructVision/config.toml
```

An explicit absolute path can be supplied for tests or controlled launches.
The configuration is local-only and must not be committed or placed in a
public handoff. The technical-handoff verifier rejects a bundled
`config.toml`.

Schema version 2 has fixed top-level identity, migration state, all twelve
named roots, multiple non-overlapping translation rules per immutable reference
role, and optional private hash-bound resource bindings:

```toml
schema = "org.structvision.storage"
schema_version = 2
migration_state = "external"

[roots]
source_root = "/absolute/path/to/Developer/StructVision-AI"
runs_root = "/absolute/path/to/StructVision/Runs"
trash_root = "/absolute/path/to/StructVision/Trash"
protected_root = "/absolute/path/to/StructVision/Protected"
registry_root = "/absolute/path/to/StructVision/Protected/Registry"
research_data_root = "/absolute/path/to/StructVision/Protected/ResearchData"
experiment_store_root = "/absolute/path/to/StructVision/Protected/ExperimentStores"
learned_artifact_root = "/absolute/path/to/StructVision/Protected/LearnedArtifacts"
historical_report_root = "/absolute/path/to/StructVision/Protected/HistoricalReports"
artifact_cache_root = "/absolute/path/to/StructVision/Cache"
release_root = "/absolute/path/to/StructVision/Releases"
private_data_root = "/absolute/path/to/StructVision/PrivateData"

[[translations]]
role = "historical_report"
identity = "historical-report-prefix-v1"
stored_prefix = "/retired/source/research_data/reports"
target_root = "historical_report_root"
destination_subpath = ""
public_safe = false
redistribution_allowed = false

[[translations]]
role = "registry_annotation"
identity = "registry-annotation-relative-v1"
stored_prefix = "research_data/annotations"
target_root = "research_data_root"
destination_subpath = "annotations"
public_safe = false
redistribution_allowed = false

[resources.registry_database]
logical_root = "registry_root"
relative_path = "datasets.sqlite"
expected_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"
redistribution_allowed = false
```

The example prefixes and digest are placeholders, not active bindings. Real
activation must derive them from the reviewed immutable migration manifest.

Loading rejects missing required keys, unknown keys, malformed TOML, relative
roots, parent traversal, filesystem or mount roots, the home directory,
repository-local roots in external mode, repository ancestors, symlink roots,
existing symlinked ancestors, and conflicting overlaps. The deliberate
`protected_root`/typed-protected-child hierarchy is the only accepted nesting.
Loading performs no mkdir, copy, database access, or other write.

The deterministic configuration identity is a SHA-256 of canonical local
configuration fields. It is an identity, not an integrity or validity claim.
Public serialization emits logical names, access types, and the identity while
redacting every absolute path and omitting the private-data root entry.
Credentials and arbitrary extension fields are not accepted by this schema.
Resource bindings are local-only mappings from a fixed role to a contained
relative path and expected SHA-256. Duplicate roles, missing required roles,
hash mismatches, paths outside the role root, and redistribution grants are
refused.

## Migration states

`external` is required for new portable operations. `structvision-live-demo`
and `structvision-analyse` discover the preferred configuration automatically;
an explicit `--storage-config` overrides it. Filesystem inputs must be inside
`private_data_root`, console/CLI outputs inside `runs_root`, handoff outputs
inside `release_root`, and learned resources inside `learned_artifact_root`.
The environment lock is a protected learned/runtime resource, not source.
External-mode command-line and environment selections must also match the
role-specific resource binding and its expected SHA-256; being merely inside
the learned-artifact root is insufficient.
`structvision-analyse` remains no-write-by-default.

`legacy_repository_compatibility` is available only through the explicitly
named `StorageConfig.legacy_repository_compatibility(...)` constructor.
Configuration loading never invents that state as a fallback. Portable run
paths reject it, and a repository-local write additionally requires
`allow_legacy_write=True` at the exact call site. This state exists only while
the staged migration is incomplete.

The supported local inspection interface is
`apps/structvision_demo.py`. Uploads remain in memory and browser downloads
remain explicit. The root-level `app.py` is a disabled legacy research
interface: it stops before importing any mutable registry, store, upload,
output, report, dataset, mask, or model path. The legacy registered-experiment
executor is also refused before payload access or status/output mutation when
external mode is selected.

The protected root-level `config.py` is deliberately unchanged. It contains
frozen classical identities and legacy path globals used by the compatibility
adapter. Portable code uses `structvision.storage`; importing that module does
not import the legacy path globals or create a runtime directory.

## Read-only legacy reference resolution

`LegacyPathResolver` has two separate entry points:

- `resolve_historical_report(...)`;
- `resolve_registry_annotation(...)`.

The resolver accepts one stored string and performs path metadata checks only.
It never opens, hashes, decodes, copies, edits, or deletes the referenced
payload and never updates a database or JSON manifest.

Resolution is fail-closed:

1. Refuse empty, malformed, traversal-bearing, ambiguous, or unexpected-prefix
   values.
2. Permit absolute or repository-relative stored prefixes through separate,
   stable rules.
3. Retain the matched suffix and place it beneath the rule's fixed target root
   and explicit destination subpath.
4. Return `translated` only if that physical regular file exists and no path
   component is a symlink.
5. Return `unavailable` when an approved direct or translated target is absent.
6. Return `refused` for overlapping/wrong-role prefixes, symlink escape,
   directories, or any unapproved location.

In external mode, a matched legacy reference is always translated to the
external target even while an old rollback file still exists. Existing legacy
direct behavior is available only in explicit
`legacy_repository_compatibility` mode.

Historical reports can translate only to `historical_report_root`; registry
annotations can translate only to `research_data_root`. There is no
general-purpose remapper.

Typed provenance retains the original stored string, logical root,
configuration identity, role, status, translation identity, resolved path, and
public-safety/redistribution flags. Public exports replace both stored and
resolved absolute values with `[redacted]`. Every record explicitly states
`scientific_validity_claimed = false`: path availability is not evidence of
annotation quality, experimental validity, licensing, or real-world
performance.

## Database and manifest immutability

All 888 historical visualization strings (168 absolute and 720 relative) and
all 323 non-empty registry annotation strings (33 absolute and 290 relative)
remain unchanged. Compatibility is achieved at the read boundary, not through
a database migration.

`ReadOnlyRegistry` keeps `registry_root` and `research_data_root` separate and
opens the registry with SQLite `mode=ro&immutable=1`. It performs no directory
creation, schema initialization, migration, or write statement.
`ProtectedExperimentStoreReader` provides hash-verified read-only access to the
historical, research-evaluation, PatchCore, and hybrid stores. External
registered-experiment execution remains intentionally disabled pending a
future API-based benchmark runner.

Applications adopting the resolver must retain the stored value as provenance
and use the returned path only for the explicitly authorized read. Dataset
intake, deletion, experiment execution, and result registration must not use
the resolver as write authority.

## Private-data policy

Private content belongs only under a locally configured `private_data_root`.
An absolute private path is never a public identifier. Public JSON, handoff
files, screenshots, logs, and report metadata should use a logical root plus a
content/record identity, with the local path redacted. A true
`redistribution_allowed` flag must come from a separately reviewed licensing
decision; successful resolution does not grant it.

## Future clean-checkout procedure

This is a future controlled operation, not an instruction to move the active
workspace now:

1. Make a clean Git checkout in the intended developer source directory.
2. Verify the exact commit and protected identities against signed/read-only
   records.
3. Create the external root hierarchy through a dedicated, reviewed
   provisioning command.
4. Copy or rehydrate only manifest-approved artifacts into their named roots;
   verify before switching configuration.
5. Install a fresh virtual environment. A moved virtual environment must not be
   reused because launcher shebangs and environment metadata contain the old
   location.
6. In a separate activation task, write the local configuration atomically
   outside Git with exact role-specific translations and hash-bound resources.
7. Run metadata-only resolution and full regression checks before enabling any
   write workflow.
8. Retain rollback inputs until the new installation and protected stores are
   independently verified.

No compatibility symlink is part of this architecture.
