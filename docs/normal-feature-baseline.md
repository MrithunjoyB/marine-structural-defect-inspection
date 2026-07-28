# Normal-Feature PatchCore Baseline

## Scope

`structvision-patchcore-baseline-v1-dev` is a recognised modern baseline, not the proposed StructVision hybrid contribution. It was fitted and calibrated only on the protected development cohort. It has not used a historical test split, externally provided data, or real marine imagery, and it supports no confirmatory or transferability claim.

PatchCore was selected before development scoring because its published method directly matches this work package's needs: normal-only fitting, frozen pretrained patch features, an explicit representative memory bank/coreset, dense localisation distances, and an image anomaly score without a task-specific defect classifier. It is therefore an informative reference family against the frozen classical proposal path while remaining conceptually distinct from the later hybrid. The paper reference is Roth et al., [“Towards Total Recall in Industrial Anomaly Detection,” CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html).

## Official implementation and fixed configuration

The adapter calls Anomalib `2.5.1`'s official `PatchcoreModel` and `KCenterGreedy` components. It does not copy or alter PatchCore mathematics. Upstream responsibilities are pretrained feature extraction, multi-layer embedding, k-center-greedy coreset selection, exact chunked Euclidean nearest-neighbour distance, PatchCore image scoring, and anomaly-map generation. StructVision responsibilities are protected cohort selection, input letterboxing, provenance validation, immutable artifacts, inverse map projection, calibration, deterministic components, and v2 adaptation.

The single predeclared configuration is:

| Field | Value |
|---|---|
| Backbone | `wide_resnet50_2` |
| Layers | `layer2`, `layer3` |
| Embedding width | 1,536 |
| Input | RGB float32, ImageNet mean/std |
| Spatial policy | aspect-preserving 256×416 letterbox; no tiling |
| Coreset | official Anomalib k-center-greedy, ratio 0.001 |
| Neighbours | 9 |
| Distance | Euclidean; exact chunked brute force |
| Reference device | CPU, one Torch and inter-op thread |
| Seed | 42 for Python, NumPy, and Torch |
| Proposals | thresholded full-resolution map, no morphology, 8-connectivity, minimum 16 pixels, maximum 8 |

The 256×416 policy was frozen from normal-only resource evidence: registered images are 500×300, so letterboxing retains their aspect ratio and about 83% linear resolution without the memory multiplication of overlapping tiles. Tiling and anomaly-label performance were not compared. On the 8 GB M1 reference host, the completed process peaked at 2,079,162,368 bytes resident memory.

## Environment

The lightweight reference base install on Python 3.9 still contains only NumPy and desktop OpenCV. The package marker uses API-compatible headless OpenCV on Python 3.12, allowing the learned environment to remain a separate, conflict-free Python `3.12.13` macOS arm64 environment because maintained Anomalib releases require Python 3.10 or newer and Anomalib uses headless OpenCV.

The complete PEP 751 lock is [pylock.normal-feature-macos-arm64.toml](../requirements/pylock.normal-feature-macos-arm64.toml), SHA-256 `be3a00936219aedbcc397f0b3e8c0af6d901489a06550f3b148c72e22cea87b8`. The resolved report is [normal-feature-macos-arm64-environment.json](../requirements/normal-feature-macos-arm64-environment.json).

| Direct learned dependency | Exact version |
|---|---:|
| Python | 3.12.13 |
| Anomalib | 2.5.1 |
| Torch | 2.9.1 |
| torchvision | 0.24.1 |
| timm | 1.0.28 |
| safetensors | 0.8.0 |
| NumPy | 2.2.6 |
| opencv-python-headless | 4.12.0.88 |
| scikit-learn | 1.7.2 |

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/pylock.normal-feature-macos-arm64.toml
python -m pip install --no-deps .
```

This separation and the Python-version marker avoid installing both `opencv-python` and `opencv-python-headless` in one environment. No API key or commercial inference service is required.

## Pretrained-weight provenance

| Field | Value |
|---|---|
| Distributor | official timm Hugging Face repository |
| Model | `timm/wide_resnet50_2.racm_in1k` |
| Revision | `30f73aceaaa1911830a9795b83ab1908dba18719` |
| File | `model.safetensors` |
| SHA-256 | `03b71d65fb2c73bb0de079a1781009f27a782ec481d2f64ab3bde9b1cdec3000` |
| Licence declared by timm | Apache-2.0 |
| Preprocessing | RGB; mean 0.485/0.456/0.406; standard deviation 0.229/0.224/0.225 |

Download is explicit through `cache_normal_feature_weight.py`; fitting and inference reject a missing or changed file and force Hugging Face offline mode. The adapter constructs the official backbone without pretrained resolution and then strictly loads this exact verified safetensors state, preventing timm from substituting a different cached or network weight. A missing key, unexpected key, shape mismatch, random fallback, or implicit download fails closed. Weights and caches are ignored and are not committed. The upstream ImageNet-derived training provenance remains a licensing/deployment consideration even though timm declares this weight Apache-2.0.

## API and artifacts

```python
from structvision.normal_feature import NormalFeatureAnomalyDetector, NormalFeatureConfig

detector = NormalFeatureAnomalyDetector(
    NormalFeatureConfig(),
    weight_file=verified_weight_path,
    environment_lock_hash=lock_sha256,
)
model_artifact = detector.fit_normal(normal_fit_samples, normal_fit_manifest_hash=manifest_hash)
result = detector.analyse(
    "inspection.png",
    model_artifact=model_artifact,
    calibration_artifact=calibration_artifact,
    operating_point_id="fp-budget-0.50",
    image_id="sample-001",
)
```

The default path writes nothing. `DirectoryModelArtifactSink`, `DirectoryCalibrationArtifactSink`, or a result sink must be injected explicitly. `NormalFeatureModelArtifact` binds selected image hashes, configuration, weight, memory bank, coreset indices, preprocessing, environment, hardware, Git state, and creation time. `CalibrationArtifact` separately binds the validation manifest, v2 policy, complete threshold curve, declared budgets, operating points, and selection rule. Loaders recompute identities and reject tampering.

Dense anomaly scores are raw PatchCore distances, not probabilities and not classical heuristic scores. Learned proposals retain the threshold, operating-point ID, component distance, rank, area, half-open box, full-resolution mask hash, map summary, and extraction-policy hash.

## Determinism and limitations

The reference enables deterministic Torch algorithms, fixes all seeds, uses one thread and batch size one, records coreset indices, and fails closed on incompatible runtime state. Repeated official mini-fit, memory-bank, anomaly-map, proposal, and offline artifact replay tests are byte-identical. MPS was built into Torch but unavailable in the sandbox; it is never a scientific reference and CPU/MPS numerical equality is not assumed.

PatchCore can miss thin anomalies, conflate synthetic template changes with defects, and depend strongly on the representativeness of clean memory. The current validation cohort is also the calibration cohort, so its metrics are development diagnostics only. A separate proposal-guided hybrid has since been developed without changing this baseline; it failed its predeclared holdout preservation decision and is documented in [Proposal-Guided Hybrid](proposal-guided-hybrid.md).
