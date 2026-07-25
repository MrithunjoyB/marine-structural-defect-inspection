# Professor Technical Handoff

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

## 4. Run one image

No-write default:

```bash
structvision-analyse --input inspection.png --method classical
```

Explicit outputs:

```bash
structvision-analyse \
  --input inspection.png \
  --method classical \
  --json-out result.json \
  --csv-out proposals.csv \
  --overlay-out overlay.png
```

The command reads the caller-selected input. It writes nothing unless an output path is supplied.

## 5. Inspect the algorithm

- Public API: `src/structvision/api.py`
- Immutable configuration: `src/structvision/configuration.py`
- Frozen compatibility boundary: `src/structvision/classical.py`
- Typed results: `src/structvision/types.py`
- Protected classical implementation: `preprocess.py`, `feature_extraction.py`, `region_proposal.py`, `scoring.py`
- Learned baseline: `src/structvision/normal_feature/`
- Rejected hybrid: `src/structvision/hybrid/`
- V2 evaluation: `scientific_contract/`
- Formal detail: [Algorithm Specification](algorithm-specification.md), [Pseudocode](algorithm-pseudocode.md), [Code Guide](code-structure-guide.md)

## 6. Run the professor client

```bash
python -m streamlit run apps/professor_demo.py
```

Upload bytes are decoded and retained in session memory only. The client does not use the broad legacy research UI or its persistence paths.

## 7. Output definitions

- `proposal_score`, classical evidence, reliability, and priority are heuristics.
- PatchCore scores are raw nearest-normal distances.
- Hybrid scores are fixed linear rank scores.
- None is a probability or calibrated confidence.
- Binary masks contain values 0/255 in analysed-image coordinates.
- Boxes use half-open `(x_min, y_min, x_max, y_max)`.
- Missing method-specific values are `N/A`, not zero.

## 8. Current method statuses

| Method | Status | Operational policy |
|---|---|---|
| Frozen classical | stable frozen baseline | recommended demonstration default; base environment |
| PatchCore | protected development baseline | optional exact learned environment/artifacts; comparison only |
| Proposal-guided hybrid | **rejected development candidate** | optional research comparison; never recommended default |

## 9. Current evidence

At the primary synthetic development point, classical micro sensitivity is `0.770833` with `4.411765` clean false proposals/image. PatchCore lowers clean burden to `0.176471` and raises precision to `0.673469`, but micro sensitivity is `0.687500`, thin-crack sensitivity is `0.0`, and pitting sensitivity is `0.266667`. The hybrid reaches `0.323529` clean false proposals/image, `0.720000` precision, `0.631250` mean IoU, and `0.750000` micro sensitivity.

See [Research Evidence Summary](research-evidence-summary.md).

## 10. Why the hybrid was rejected

The fixed protocol allowed at most a `0.02` overall sensitivity loss and required image-level sensitivity preservation. Hybrid micro sensitivity fell from `0.770833` to `0.750000`: loss `0.020833`, exceeding the margin by about `0.000833`. Image-level sensitivity fell from `0.894737` to `0.868421`. Burden and precision improvements do not override those failures.

## 11. Information required from the professor

- intended inspection question and acceptable proposal role;
- image formats, bit depth, colour, resolution, cameras, and acquisition conditions;
- vessel/component/session grouping;
- anomaly definitions and uncertain/normal semantics;
- existing annotation types and reviewer expertise;
- data ownership, licence, confidentiality, access, retention, and publication constraints;
- minimum meaningful feature size and any known physical scale;
- desired pilot decision criteria.

## 12. Future data-integration sequence

1. Agree written data authority and security controls.
2. Implement a private `ProfessorDatasetAdapter`.
3. Validate identities, formats, grouping, and annotations.
4. Lock a group-aware split before method access.
5. Run non-performance intake smoke checks.
6. Predeclare a prospective pilot.
7. Execute with an explicit private result sink.
8. Review results jointly without post-hoc retuning.

## 13. Confidentiality and licensing

Private paths and metadata remain outside Git. Images are not automatically uploaded. Access, encryption, retention, deletion, derived-artifact ownership, model/data licences, and public-example approval must be documented before intake.

## 14. Proposed pilot protocol

Use a small, representative, group-aware prospective cohort with a locked sample manifest; blinded or independently adjudicated annotations where feasible; predeclared method identities and acceptance/failure rules; exact environment/artifact hashes; separate intake, development, and held-back roles; complete failure reporting; and no retuning after held-back access.

The pilot must first establish data validity and operating constraints. It is not automatically a confirmatory study.

## 15. Maintainer

Repository: `MrithunjoyB/marine-structural-defect-inspection`
Maintainer from package/repository metadata: Mrithunjoy Basumatary
