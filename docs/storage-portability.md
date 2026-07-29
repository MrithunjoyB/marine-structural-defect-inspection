# Portable Storage and Immutable Legacy Paths

## Scope and status

This document defines the path-portability foundation for a future clean
StructVision-AI installation. It does not record a completed relocation. The
active repository, historical databases, registry manifest, learned artifacts,
and runtime files remain where they were before this change.

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

Schema version 1 has fixed top-level identity, migration state, all twelve
named roots, and at most one translation rule for each immutable reference
role:

```toml
schema = "org.structvision.storage"
schema_version = 1
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

[translations.historical_report]
identity = "historical-report-prefix-v1"
stored_prefix = "/retired/source/research_data/reports"
target_root = "historical_report_root"
public_safe = false
redistribution_allowed = false

[translations.registry_annotation]
identity = "registry-annotation-prefix-v1"
stored_prefix = "/retired/source/research_data/annotations"
target_root = "research_data_root"
public_safe = false
redistribution_allowed = false
```

The example prefixes are placeholders, not active rules. A future migration
must generate rules from its reviewed manifest and exact source prefixes.

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

## Migration states

`external` is required for new portable operations. A configured CLI output
must be inside `runs_root`; learned artifacts must be inside
`learned_artifact_root`, and an environment lock must be inside `source_root`.
The ordinary CLI remains no-write-by-default.

`legacy_repository_compatibility` is available only through the explicitly
named `StorageConfig.legacy_repository_compatibility(...)` constructor.
Configuration loading never invents that state as a fallback. Portable run
paths reject it, and a repository-local write additionally requires
`allow_legacy_write=True` at the exact call site. This state exists only while
the staged migration is incomplete.

The legacy Streamlit client retains its historical behavior when no portable
configuration is selected. When `STRUCTVISION_STORAGE_CONFIG` explicitly
selects an external configuration, new uploaded bytes go beneath
`runs_root/legacy-streamlit/uploads` instead of the repository. Other legacy UI
artifact paths remain compatibility-only until their protected path-global
dependencies can be replaced in a separately reviewed change. The technical
Streamlit demonstration remains in-memory and write-free.

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

1. Refuse empty, relative, malformed, traversal-bearing, or unexpected-prefix
   values.
2. If the path is an existing regular file inside the current role root or its
   exact configured historical prefix, return `direct`.
3. If the exact configured old prefix matches, retain the relative suffix and
   place it beneath that role's fixed target root.
4. Return `translated` only if that physical regular file exists and no path
   component is a symlink.
5. Return `unavailable` when an approved direct or translated target is absent.
6. Return `refused` for wrong-role prefixes, symlink escape, directories, or
   any unapproved absolute location.

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

The 168 historical visualization strings, 33 registry annotation strings, and
the same 33 JSON-manifest strings remain unchanged. Compatibility is achieved
at the read boundary, not through a database migration. No resolver API
contains an update, insert, delete, commit, or write-back operation.

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
6. Write the local configuration atomically outside Git and add exact
   role-specific legacy translation rules.
7. Run metadata-only resolution and full regression checks before enabling any
   write workflow.
8. Retain rollback inputs until the new installation and protected stores are
   independently verified.

No compatibility symlink is part of this architecture.
