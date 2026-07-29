# Protected Normal-Feature Development Data Protocol

## Classification and immutable identity

Protocol `structvision-normal-feature-development-v1` creates a **development-only — non-confirmatory** cohort from `synthetic-expanded` v1.0 train and validation metadata. It creates no test role. The committed [manifest](../development_data/normal-feature-development-manifest-v1.json) has logical SHA-256 identity `2aa40b9db145a37522775b7ac605ae201b91e564cde881528fd6d41f449f3d58`, calculated over its canonical payload before the self-identifying `manifest_hash` field; the complete JSON file SHA-256 is `deba66884d7769ce016666838a7114a988a8f50b389a2a54ef5d59e1a30e8c7a`.

The generator reads the registry in SQLite read-only mode. It compares metadata and hashes but never decodes a forbidden test image. Every selected development image is verified against its registered encoded-file hash. Positive ground truth uses the registered mask-file hash; verified clean images use a deterministic full-size zero-mask identity because the registry intentionally has no clean annotation file.

## Protection rules

Candidates are rejected when the existing overlap infrastructure finds an exact SHA-256 match, a legacy difference-hash candidate at Hamming distance at most three, or a declared source/template/acquisition group crossing with:

- `synthetic-expanded-pilot`;
- any registered historical test split; or
- any image ID present in a prior historical verification/result row.

The difference hash is only a conservative candidate screen, not duplicate proof. Excluding candidates is protective; absence from the screen does not establish semantic independence.

Additional fail-closed checks prohibit test roles, anomaly-present fit images, missing image/truth hashes, duplicate IDs/hashes across roles, declared group crossings between fit and validation, and disappearance of a required category.

## Counts

| Stage | Count |
|---|---:|
| Source train + validation candidates | 400 |
| Train anomalies ineligible for normal fit | 150 |
| Otherwise role-eligible train-clean + validation | 250 |
| Protected overlap/group exclusions among role-eligible images | 87 |
| `normal_fit` | 91 |
| `calibration_validation` | 72 |
| Total selected | 163 |

Exclusion reasons may overlap. Across all 400 candidates, the manifest records 237 unique excluded images and the following reason occurrences: 150 train-anomaly exclusions; 67 exact pilot matches; 57 pilot perceptual candidates; 85 pilot source-group crossings; 85 pilot template-group crossings; and 55 perceptual candidates against both historical-test and historical-verification identity sets.

`normal_fit` contains only eligible clean train images: blur/noise 20, border artefact 16, illumination gradient 13, normal texture 21, and specular highlights 21.

`calibration_validation` contains 34 clean and 38 anomaly-present validation images:

| Category | Clean | Positive | Total |
|---|---:|---:|---:|
| blur/noise | 10 | 0 | 10 |
| border artefact | 9 | 0 | 9 |
| colour-only anomaly | 0 | 10 | 10 |
| illumination gradient | 3 | 0 | 3 |
| normal texture | 9 | 0 | 9 |
| pitting cluster | 0 | 5 | 5 |
| specular highlights | 3 | 0 | 3 |
| texture-only anomaly | 0 | 10 | 10 |
| thin crack | 0 | 4 | 4 |
| weld disturbance | 0 | 9 | 9 |

All ten expected validation categories remain. Template and source groups are disjoint between fit, validation, and the registered final split.

## Fit/calibration separation

Only `normal_fit` constructs the feature memory and coreset; no anomaly mask or validation score enters fitting. `calibration_validation` supplies clean false-proposal budgets and development diagnostics. Threshold selection follows a fixed descending-threshold path and uses only clean false-proposal burden, not anomaly-label optimization. All candidate points and poor operating points remain in the calibration artifact.

No image in this protocol is described as independent, confirmatory, real-world, or externally provided.

## Protected integration tests

The ordinary source-only test invocation does not require ignored protected
stores. Committed manifest schemas, identities, role separation, and tamper
checks remain active in the default suite. A test that genuinely needs a local
registry or historical result store is marked `protected_integration` and skips
with the exact missing repository-relative store name when that store is absent.

In an authorised installation that already contains the exact protected stores,
run the integrations read-only with:

```bash
python -m pytest -m protected_integration -ra
```

To execute the same tests from a separate source checkout against an authorised
installation, provide its root explicitly without adding it to `PYTHONPATH`:

```bash
STRUCTVISION_PROTECTED_TEST_ROOT=/path/to/authorised/installation \
  python -m pytest -m protected_integration -ra
```

The selected root must be absolute. This mechanism does not copy, migrate, or
symlink protected material and does not authorise image or annotation decoding.

## Proposal-guided hybrid protocol

The newer `structvision-hybrid-development-v1` protocol is separate from the normal-feature baseline protocol above. Its committed [manifest](../development_data/hybrid-development-manifest-v1.json) has logical SHA-256 `a1e6f9a83e5e8d73275236e6dc4fafd985e6e1ef2c4aef21fd4156dc821829a4`. It retains eligible train anomalies for fusion fitting and deterministically divides clean train identity groups between 70-image `hybrid_normal_fit` and the clean portion of 126-image `hybrid_fusion_fit`; the 72 eligible validation images form `hybrid_development_holdout`.

Exact, perceptual-candidate, source, template, and acquisition components cannot cross the three roles. Fusion fit contains 19 clean and 107 positive images; holdout contains 34 clean and 38 positive images. Both contain every positive category and all priority categories. The capability-limited fusion-fit loader cannot return holdout paths. The holdout was loaded once only after fusion freezing, remains non-confirmatory because of prior validation exposure, and cannot be used to revise the rejected candidate. Composition and limitations are in the [Hybrid Data Card](data-card-hybrid-development.md).
