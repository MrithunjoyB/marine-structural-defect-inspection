# StructVision-AI Technical Handoff

## 1. What the system currently does

StructVision-AI analyses one local image and returns ranked visual-anomaly proposals with binary masks, half-open boxes, evidence diagnostics, timing, and implementation/configuration identities. The stable demonstration method is `structvision-classical-baseline-v1-frozen`.

## 2. What it does not yet prove

It does not prove real marine performance, transferability, calibrated confidence, defect classification, physical severity, engineering fitness, deployment readiness, publication readiness, or superiority. Current comparative evidence is synthetic and development-only.

## 3. Install the base system

```bash
cd "/path/to/marine-structural-defect-inspection"
python3 -m venv venv
source venv/bin/activate
python -m pip install '.[demo]'
```

No API key or paid service is required. After installation, classical analysis is offline.

## 4. Run the dedicated live console

Use the stable presentation command:

```bash
structvision-live-demo \
  --input inspection.png \
  --output-dir build/live-demo-runs/demo-run
```

It validates the image, declares colour handling and dimensions, executes the
frozen classical detector once, prints measured processing stages, and creates
only:

```text
demo-run/
├── INPUT/original.png, input-metadata.json
├── PROCESSING/pipeline-stages.json, stage-timings.csv,
│   anomaly-evidence.png, README.txt
├── OUTPUT/overlay.png, proposals.csv, result.json,
│   technical-summary.txt, masks/*.png
├── RUN_MANIFEST.json
├── .structvision-live-console-owner.json
└── CONSOLE_LOG.txt
```

Existing directories are never replaced merely because `--overwrite` is
explicit. An existing-run update is allowed only when the exact directory has the
fixed-schema ownership marker from a completed live-console run and the
marker, manifest, payload hashes, and complete directory contents all validate.
The new run is fully generated in a private sibling staging directory; a
failed analysis, serialization, validation, or installation preserves the
previous valid run. Dangerous, symlinked, changed, unmarked, or mismatched
targets are refused. Persistent files and protected temporary artifacts stay
inside the selected run directory. The general `structvision-analyse` CLI
remains unchanged and no-write by default.

The live ownership identities are
`.structvision-live-console-owner.json`,
`structvision-live-console-run-owner-v1`, `structvision-live-demo`, and
`structvision-live-run-manifest-v1`. Outputs from the retired naming scheme are
not silently accepted: they remain preserved as unowned directories and require
manual removal if no longer needed.

## 5. Inspect the algorithm

- Public API: `src/structvision/api.py`
- Immutable configuration: `src/structvision/configuration.py`
- Frozen compatibility boundary: `src/structvision/classical.py`
- Typed results: `src/structvision/types.py`
- Live console wrapper: `src/structvision/live_console.py`
- Protected classical implementation: `preprocess.py`, `feature_extraction.py`, `region_proposal.py`, `scoring.py`
- Learned baseline: `src/structvision/normal_feature/`
- Rejected hybrid: `src/structvision/hybrid/`
- V2 evaluation: `scientific_contract/`
- Formal detail: [Algorithm Specification](algorithm-specification.md), [Pseudocode](algorithm-pseudocode.md), [Code Guide](code-structure-guide.md)

## 6. Run the StructVision Streamlit alternative

```bash
python -m streamlit run apps/structvision_demo.py
```

Upload bytes are decoded and retained in session memory only. The client does not use the broad legacy research UI or its persistence paths.

## 7. Build and verify the clean drive handoff

```bash
python scripts/build_technical_handoff.py \
  --output "/path/to/StructVision-AI-Technical-Handoff"
python scripts/build_technical_handoff.py \
  --verify "/path/to/StructVision-AI-Technical-Handoff"
```

The builder requires a clean worktree, uses `git archive HEAD` for the source
ZIP, builds one deterministic synthetic-fixture example, records commit/version/
platform/timestamp/count/size, writes SHA-256 values in deterministic order, and
then verifies the result. It excludes virtual environments, Git internals,
caches, databases, historical stores, learned weights and memories, bulk
datasets, private images, private collaborator data, absolute user-home paths, OS metadata,
and unrelated projects.

The drive supports four different contexts: live demonstration on the prepared
Mac; source review from the drive; fresh installation on another machine; and a
future independently audited macOS-arm64 offline wheelhouse. No wheelhouse or
prebuilt environment is included here.

## 8. Output definitions

- `proposal_score`, classical evidence, reliability, and priority are heuristics.
- PatchCore scores are raw nearest-normal distances.
- Hybrid scores are fixed linear rank scores.
- None is a probability or calibrated confidence.
- Binary masks contain values 0/255 in analysed-image coordinates.
- Boxes use half-open `(x_min, y_min, x_max, y_max)`.
- Missing method-specific values are `N/A`, not zero.
- Exposed anomaly evidence is a presentation visualisation, not a new stage or
  changed proposal mask.
- Candidate-generation and mask-refinement intermediate images are not exposed
  by the current frozen API and are not fabricated.

## 9. Current method statuses

| Method | Status | Operational policy |
|---|---|---|
| Frozen classical | stable frozen baseline | recommended demonstration default; base environment |
| PatchCore | protected development baseline | optional exact learned environment/artifacts; comparison only |
| Proposal-guided hybrid | **rejected development candidate** | optional research comparison; never recommended default |

## 10. Current evidence

At the primary synthetic development point, classical micro sensitivity is `0.770833` with `4.411765` clean false proposals/image. PatchCore lowers clean burden to `0.176471` and raises precision to `0.673469`, but micro sensitivity is `0.687500`, thin-crack sensitivity is `0.0`, and pitting sensitivity is `0.266667`. The hybrid reaches `0.323529` clean false proposals/image, `0.720000` precision, `0.631250` mean IoU, and `0.750000` micro sensitivity.

See [Research Evidence Summary](research-evidence-summary.md).

## 11. Why the hybrid was rejected

The fixed protocol allowed at most a `0.02` overall sensitivity loss and required image-level sensitivity preservation. Hybrid micro sensitivity fell from `0.770833` to `0.750000`: loss `0.020833`, exceeding the margin by about `0.000833`. Image-level sensitivity fell from `0.894737` to `0.868421`. Burden and precision improvements do not override those failures.

## 12. Information required from a domain collaborator

- intended inspection question and acceptable proposal role;
- image formats, bit depth, colour, resolution, cameras, and acquisition conditions;
- vessel/component/session grouping;
- anomaly definitions and uncertain/normal semantics;
- existing annotation types and reviewer expertise;
- data ownership, licence, confidentiality, access, retention, and publication constraints;
- minimum meaningful feature size and any known physical scale;
- desired pilot decision criteria.

## 13. Future data-integration sequence

1. Agree written data authority and security controls.
2. Implement a private `PrivateDatasetAdapter`.
3. Validate identities, formats, grouping, and annotations.
4. Lock a group-aware split before method access.
5. Run non-performance intake smoke checks.
6. Predeclare a prospective pilot.
7. Execute with an explicit private result sink.
8. Review results jointly without post-hoc retuning.

## 14. Confidentiality and licensing

Private paths and metadata remain outside Git. Images are not automatically uploaded. Access, encryption, retention, deletion, derived-artifact ownership, model/data licences, and public-example approval must be documented before intake.

## 15. Safe copy and post-meeting retention

- verify the handoff before and after copying to the external drive;
- copy the handoff directory, not the working clone or virtual environment;
- keep any non-fixture live run outside the immutable handoff;
- confirm no private or private collaborator image was substituted into `INPUT/`;
- safely eject the drive;
- delete temporary authorised-image runs after the meeting under the agreed
  retention policy, including trash if required;
- retain reviewed source history and scientific stores under project governance.

## 16. Proposed pilot protocol

Use a small, representative, group-aware prospective cohort with a locked sample manifest; blinded or independently adjudicated annotations where feasible; predeclared method identities and acceptance/failure rules; exact environment/artifact hashes; separate intake, development, and held-back roles; complete failure reporting; and no retuning after held-back access.

The pilot must first establish data validity and operating constraints. It is not automatically a confirmatory study.

## 17. Maintainer

Repository: `MrithunjoyB/marine-structural-defect-inspection`
Maintainer from package/repository metadata: Mrithunjoy Basumatary
