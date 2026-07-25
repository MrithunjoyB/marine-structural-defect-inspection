# Professor Demonstration Runbook

Target duration: 10–15 minutes. Use the base environment and the frozen classical method. Learned execution is optional.

## Before the meeting

```bash
git branch --show-current
git rev-parse HEAD
git status --short
source venv/bin/activate
python -m pip install '.[demo]'
structvision-analyse --help
python -m streamlit run apps/professor_demo.py
```

Expected: branch/commit are the intended reviewed state; worktree is clean; CLI help states classical default and no-write behavior; the Streamlit client opens with three non-hidden status cards.

Do not open historical/protected images. Use a caller-authorised image or a labelled deterministic demonstration fixture.

## 1. State the problem — 1 minute

Screen: **System Overview**.

Say: “The system proposes visually anomalous regions for expert review. It does not yet classify damage or make an engineering diagnosis.”

Show that classical is the stable frozen default, PatchCore is a development baseline, and the hybrid is rejected.

## 2. Show the architecture — 1 minute

Screen: **Source-Code Architecture**.

Explain:

```text
thin client → public API → protected implementation → typed in-memory result
```

Point to [Code Guide](code-structure-guide.md). Emphasise that detector packages never import Streamlit and the new client does not import the legacy app.

## 3. Show the reusable API — 1 minute

Open `src/structvision/api.py` and show:

```python
result = StructuralAnomalyDetector(DetectorConfig()).analyse(
    image,
    image_id="review-image",
    colour_space="BGR",
)
```

Explain that no database, output directory, API key, or remote service is required. A sink is optional and explicit.

## 4. Run the frozen classical method — 2 minutes

Screen: **Analyse an Image**.

Select one deterministic fixture or upload one authorised PNG/JPEG/TIFF. Keep:

```text
Method: Frozen classical baseline — stable default
```

Click **Run local analysis**.

Expected: completed status, dimensions, proposal count, input image, direct mask overlay, configuration hash, and warnings. If the image has alpha, state the selected composite/drop policy.

CLI alternative:

```bash
structvision-analyse --input inspection.png --method classical
```

Expected: human-readable identity/status/hash/proposal summary and no new file.

## 5. Inspect masks and ranked evidence — 2 minutes

Open **Proposal Evidence** then **Candidate detail**.

Show:

- contiguous rank and proposal ID;
- half-open box;
- area;
- proposal/evidence/priority scores;
- heuristic mask reliability;
- contextual diagnostics;
- full-image single-candidate overlay;
- half-open crop;
- exact binary mask.

Say: “These values order review; none is a probability. The mask directly defines the displayed boundary.”

## 6. Explain PatchCore — 1 minute

Screen: **Algorithm Pipeline**, select PatchCore.

Explain normal-only memory, frozen patch embedding, nearest-normal distance, dense full-resolution map, frozen development operating point, and connected-component extraction.

State the observed weakness: thin-crack component sensitivity `0.0` and pitting `0.266667` at the selected synthetic development point.

## 7. Explain the hybrid and its rejection — 2 minutes

State first that it is a **rejected development candidate** under the predeclared protocol.

Screen: **Method Comparison** and hybrid pipeline.

Explain fixed `0.60` classical / `0.40` normality fusion and threshold `0.4704560134385654`.

State:

- clean false proposals/image: `4.411765 → 0.323529`;
- precision: `0.168950 → 0.720000`;
- mean IoU: `0.621954 → 0.631250`;
- micro sensitivity: `0.770833 → 0.750000`;
- margin miss: about `0.000833`;
- image sensitivity also decreased.

Conclude: “The predeclared decision is rejected. The improvements do not reverse that status.”

## 8. Show scientific-contract v2 — 1 minute

Open `docs/scientific-contract-v2.md` or `scientific_contract/specification.py`.

Explain immutable selection, content hashes, fixed configuration/evaluation policy, complete method pairing, and explicit append-only sinks. State that the demonstration never invokes the experiment executor.

## 9. Show the professor-data boundary — 1 minute

Screen: **Data Integration Contract** and [Professor Data Adapter](professor-data-adapter.md).

Explain that a private adapter supplies content-verified images, annotation semantics, acquisition groups, confidentiality/licence fields, and split-lock identity without changing detector code or putting private paths in Git.

## 10. Close with the next pilot — 1 minute

Propose: written authority → private adapter → format/group audit → split lock → intake smoke → predeclared prospective pilot → joint review.

Do not promise performance. Ask for acquisition metadata, normal/anomaly semantics, reviewer process, grouping, confidentiality, and desired pilot criteria.

## Learned-environment fallback

If PatchCore or hybrid shows “Execution disabled”:

1. Keep classical operational.
2. Show the exact missing requirement displayed by the client.
3. Explain the pipeline and stored evidence only.
4. Do not install a package, download a weight, or substitute a method during the demonstration.

If learned replay is planned in advance, launch the client inside the exact Python 3.12 lock and configure:

```text
STRUCTVISION_ENVIRONMENT_LOCK
STRUCTVISION_PATCHCORE_WEIGHT
STRUCTVISION_PATCHCORE_MODEL_ARTIFACT
STRUCTVISION_PATCHCORE_CALIBRATION_ARTIFACT
STRUCTVISION_HYBRID_MODEL_ARTIFACT
STRUCTVISION_HYBRID_FUSION_ARTIFACT
```

All values must point to already-present verified local files.

## Troubleshooting

| Symptom | Response |
|---|---|
| Malformed/oversized input | Use a valid PNG/JPEG/TIFF within the displayed limits; do not bypass validation |
| Alpha error | Select an explicit white/black composite or drop policy |
| High-bit-depth TIFF rejected | Convert only under a separately recorded intake procedure; do not improvise in-session |
| No proposals | This is a valid frozen-method output, not an execution failure |
| Learned environment unavailable | Use classical; present learned evidence descriptively |
| Artifact mismatch | Stop learned execution; verify exact artifact/lock/weight identities offline |
| Large image warning | Explain that memory and small-region behavior need separate validation |
| Download needed | Use one explicit download button; no automatic export occurs |

## Likely questions

**Is a proposal a defect?**

No. It is an image-context anomaly candidate requiring expert interpretation.

**Are the scores confidence values?**

No. Classical scores are heuristics, PatchCore scores are distances, and hybrid scores are linear rank scores.

**Why is classical the default when it has more false proposals?**

It is the stable reusable base method and retains the strongest current sensitivity evidence. Current use prioritises transparent candidate coverage over a development-only burden reduction.

**Why not use the hybrid?**

It failed the predeclared overall and image-level sensitivity-preservation rules.

**Did you tune on the holdout?**

No. The failed result was retained and the holdout was not rerun for modification.

**Can it use my data?**

Yes through a future private adapter and predeclared protocol; no professor data has been accessed yet.

**Does it require cloud or an API key?**

No. Base analysis is local and offline after installation.

**Where are intermediate classical stages?**

Only exposed diagnostics and returned artifacts are shown. Unexposed stages are explicitly labelled, not fabricated.
