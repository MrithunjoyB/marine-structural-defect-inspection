"""Thin, optional wrapper around the pinned official Anomalib PatchCore model."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import os
from pathlib import Path
import platform
import random
import subprocess
import time
from typing import Iterable

import cv2
import numpy as np

from scientific_contract.hashing import canonical_json, is_sha256, sha256_json
from structvision.inputs import content_hash
from structvision.sinks import ArtifactSink
from structvision.types import AnalysisSample

from .calibration import CalibrationArtifact
from .configuration import NormalFeatureConfig
from .errors import DeterminismError, ModelArtifactError, OptionalDependencyError, WeightProvenanceError
from .model_artifact import ModelArtifactSink, NormalFeatureModelArtifact
from .preprocessing import prepare_input, restore_anomaly_map
from .proposal_extraction import extract_proposals
from .types import (
    NormalFeatureAnalysisResult,
    NormalFeatureScoreResult,
    NormalFitSample,
    array_hash,
)


EXACT_RUNTIME_VERSIONS = {
    "anomalib": "2.5.1",
    "torch": "2.9.1",
    "torchvision": "0.24.1",
    "numpy": "2.2.6",
    "opencv-python-headless": "4.12.0.88",
    "scikit-learn": "1.7.2",
    "safetensors": "0.8.0",
    "timm": "1.0.28",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_runtime_versions() -> None:
    mismatches = []
    for distribution, expected in EXACT_RUNTIME_VERSIONS.items():
        try:
            actual = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            mismatches.append(f"{distribution}=missing (expected {expected})")
            continue
        if actual != expected:
            mismatches.append(f"{distribution}={actual} (expected {expected})")
    if mismatches:
        raise OptionalDependencyError("Pinned normal-feature environment mismatch: " + "; ".join(mismatches))


def _load_backend(config: NormalFeatureConfig) -> tuple[object, object, object]:
    os.environ.setdefault("OMP_NUM_THREADS", str(config.torch_num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(config.torch_num_threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["HF_HUB_OFFLINE"] = "1"
    _verify_runtime_versions()
    try:
        import torch
        from anomalib.models.components import KCenterGreedy
        from anomalib.models.image.patchcore.torch_model import PatchcoreModel
    except Exception as error:
        raise OptionalDependencyError(
            "Install the exact optional normal-feature environment; the lightweight base remains independent"
        ) from error
    try:
        torch.set_num_threads(config.torch_num_threads)
        if torch.get_num_interop_threads() != config.torch_num_interop_threads:
            torch.set_num_interop_threads(config.torch_num_interop_threads)
        torch.use_deterministic_algorithms(True)
    except RuntimeError as error:
        raise DeterminismError("Could not enforce the predeclared deterministic Torch state") from error
    if torch.get_num_threads() != config.torch_num_threads or torch.get_num_interop_threads() != config.torch_num_interop_threads:
        raise DeterminismError("Torch thread state differs from the reference contract")
    return torch, PatchcoreModel, KCenterGreedy


def _set_seeds(torch: object, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _git_metadata(root: Path) -> tuple[str, str, str | None]:
    def run(*arguments: str) -> bytes:
        completed = subprocess.run(
            ("git", *arguments), cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return completed.stdout

    commit = run("rev-parse", "HEAD").decode("ascii").strip()
    status = run("status", "--porcelain=v1", "-z")
    if not status:
        return commit, "clean", None
    digest = hashlib.sha256()
    digest.update(status)
    digest.update(run("diff", "--binary", "HEAD"))
    untracked = run("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
    for raw in sorted(item for item in untracked if item):
        relative = raw.decode("utf-8")
        path = root / relative
        digest.update(relative.encode("utf-8") + b"\0")
        if path.is_file():
            digest.update(_sha256_file(path).encode("ascii"))
    return commit, "dirty", digest.hexdigest()


class NormalFeatureAnomalyDetector:
    """Development-only detector; construction performs no I/O and imports no Torch."""

    def __init__(
        self,
        config: NormalFeatureConfig | None = None,
        *,
        weight_file: Path,
        environment_lock_hash: str,
        repository_root: Path | None = None,
    ) -> None:
        self.config = config or NormalFeatureConfig()
        self.weight_file = Path(weight_file)
        if not is_sha256(environment_lock_hash):
            raise ValueError("The exact environment-lock SHA-256 is required")
        self.environment_lock_hash = environment_lock_hash
        self.repository_root = Path(repository_root or Path.cwd()).resolve()
        self._runtime_models: dict[str, object] = {}

    def _verify_weight(self) -> dict[str, object]:
        if not self.weight_file.is_file():
            raise WeightProvenanceError(
                "The official pretrained weight is not cached; random fallback and implicit download are forbidden"
            )
        actual = _sha256_file(self.weight_file)
        if actual != self.config.pretrained_weight_sha256:
            raise WeightProvenanceError("Cached pretrained-weight SHA-256 differs from the fixed official identity")
        return {
            "source": self.config.pretrained_weight_source,
            "model_id": self.config.pretrained_weight_model_id,
            "revision": self.config.pretrained_weight_revision,
            "filename": self.config.pretrained_weight_filename,
            "sha256": actual,
            "licence": self.config.pretrained_weight_licence,
            "cache_path_recorded_as": self.weight_file.name,
            "download_timestamp_utc": datetime.fromtimestamp(
                self.weight_file.stat().st_mtime, timezone.utc,
            ).isoformat(),
            "preprocessing_contract": self.config.specification_sections()["preprocessing"],
            "feature_layers": list(self.config.extracted_layers),
        }

    def _new_model(self) -> tuple[object, object, object]:
        self._verify_weight()
        torch, PatchcoreModel, KCenterGreedy = _load_backend(self.config)
        _set_seeds(torch, self.config.random_seed)
        # Build the official architecture without permitting timm to resolve a
        # different cache entry or network resource, then strictly load the
        # exact safetensors file that was verified above.
        model = PatchcoreModel(
            layers=self.config.extracted_layers,
            backbone=self.config.backbone,
            pre_trained=False,
            num_neighbors=self.config.nearest_neighbour_count,
        ).to("cpu")
        try:
            import timm
            from safetensors.torch import load_file

            state = load_file(str(self.weight_file), device="cpu")
            # Validate the complete upstream checkpoint against the complete
            # official architecture before copying exactly the subset retained
            # by timm's official features-only wrapper (through layer3).
            complete_backbone = timm.create_model(self.config.backbone, pretrained=False).to("cpu")
            complete_backbone.load_state_dict(state, strict=True)
            target = model.feature_extractor.feature_extractor
            target_keys = tuple(target.state_dict())
            if any(key not in state for key in target_keys):
                raise KeyError("Verified checkpoint is missing a retained feature tensor")
            target.load_state_dict({key: state[key] for key in target_keys}, strict=True)
            del complete_backbone, state
        except Exception as error:
            raise WeightProvenanceError(
                "The verified official safetensors state did not strictly match the official backbone"
            ) from error
        if tuple(model.feature_extractor.out_dims) != (512, 1024):
            raise OptionalDependencyError("Official feature-layer dimensions differ from the frozen configuration")
        return torch, model, KCenterGreedy

    def fit_normal(
        self,
        samples: Iterable[NormalFitSample],
        *,
        normal_fit_manifest_hash: str,
        artifact_sink: ModelArtifactSink | None = None,
    ) -> NormalFeatureModelArtifact:
        prepared_samples = tuple(samples)
        if not prepared_samples or len({item.image_id for item in prepared_samples}) != len(prepared_samples):
            raise ModelArtifactError("Fitting requires ordered, unique normal-fit samples")
        if not is_sha256(normal_fit_manifest_hash):
            raise ModelArtifactError("Normal-fit manifest hash is required")
        for sample in prepared_samples:
            if content_hash(sample.image) != sample.image_sha256:
                raise ModelArtifactError(f"Normal-fit image hash mismatch: {sample.image_id}")
        torch, model, KCenterGreedy = self._new_model()
        model.train()
        preprocessing_identities = []
        for sample in prepared_samples:
            item = prepare_input(
                sample.image, self.config,
                colour_space=sample.colour_space, alpha_handling=sample.alpha_handling,
            )
            tensor = torch.from_numpy(np.asarray(item.tensor_chw).copy()).unsqueeze(0).to("cpu")
            model(tensor)
            preprocessing_identities.append((sample.image_id, item.preprocessing_hash))
        complete_memory = torch.vstack(model.embedding_store)
        model.embedding_store.clear()
        _set_seeds(torch, self.config.random_seed)
        sampler = KCenterGreedy(embedding=complete_memory, sampling_ratio=self.config.coreset_sampling_ratio)
        indices = tuple(int(item) for item in sampler.select_coreset_idxs())
        memory_bank = complete_memory[list(indices)].detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
        del complete_memory
        model.memory_bank = torch.from_numpy(memory_bank.copy()).to("cpu")
        model.eval()
        commit, dirty_state, diff_hash = _git_metadata(self.repository_root)
        artifact = NormalFeatureModelArtifact.create(
            config=self.config,
            selected_normal_fit_ids=tuple(item.image_id for item in prepared_samples),
            normal_fit_image_hashes=tuple((item.image_id, item.image_sha256) for item in prepared_samples),
            normal_fit_manifest_hash=normal_fit_manifest_hash,
            weight_provenance=self._verify_weight(),
            memory_bank=memory_bank,
            selected_coreset_indices=indices,
            preprocessing_hash=sha256_json(preprocessing_identities),
            environment_lock_hash=self.environment_lock_hash,
            hardware_runtime_metadata={
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python": platform.python_version(),
                "cpu_threads": self.config.torch_num_threads,
                "mps_scientific_reference": False,
            },
            git_commit=commit,
            git_dirty_state=dirty_state,
            git_diff_hash=diff_hash,
            creation_timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._runtime_models[artifact.artifact_hash] = model
        if artifact_sink is not None:
            artifact_sink.write(artifact)
        return artifact

    def _model_for(self, artifact: NormalFeatureModelArtifact) -> tuple[object, object]:
        if artifact.configuration_hash != self.config.configuration_hash:
            raise ModelArtifactError("Detector and fitted-model configuration hashes differ")
        if artifact.backbone_weight_hash != self.config.pretrained_weight_sha256:
            raise ModelArtifactError("Detector and fitted-model weight identities differ")
        cached = self._runtime_models.get(artifact.artifact_hash)
        if cached is not None:
            torch, _, _ = _load_backend(self.config)
            return torch, cached
        torch, model, _ = self._new_model()
        model.memory_bank = torch.from_numpy(np.asarray(artifact.memory_bank).copy()).to("cpu")
        model.eval()
        self._runtime_models[artifact.artifact_hash] = model
        return torch, model

    def score(
        self,
        image: object,
        *,
        model_artifact: NormalFeatureModelArtifact,
        image_id: str,
        colour_space: str | None = None,
        alpha_handling: str | None = None,
    ) -> NormalFeatureScoreResult:
        if not image_id:
            raise ValueError("image_id is required")
        torch, model = self._model_for(model_artifact)
        prepared = prepare_input(
            image, self.config, colour_space=colour_space, alpha_handling=alpha_handling,
        )
        tensor = torch.from_numpy(np.asarray(prepared.tensor_chw).copy()).unsqueeze(0).to("cpu")
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(tensor)
        elapsed = time.perf_counter() - started
        raw_map = output.anomaly_map.detach().cpu().numpy()[0, 0]
        anomaly_map = restore_anomaly_map(raw_map, prepared.geometry)
        score = float(output.pred_score.detach().cpu().reshape(-1)[0].item())
        return NormalFeatureScoreResult(
            image_id=image_id,
            input_hash=prepared.normalised_input.input_hash,
            image_shape=tuple(int(item) for item in prepared.normalised_input.image_bgr.shape),
            image_anomaly_score=score,
            anomaly_map=anomaly_map,
            anomaly_map_hash=array_hash(anomaly_map),
            model_artifact_hash=model_artifact.artifact_hash,
            configuration_hash=self.config.configuration_hash,
            preprocessing_metadata=tuple(sorted({
                "preprocessing_hash": prepared.preprocessing_hash,
                "source_hash": prepared.normalised_input.source_hash,
                "source_type": prepared.normalised_input.source_type,
                "geometry": canonical_json(prepared.geometry.to_dict()),
            }.items())),
            deterministic_mode=True,
            device="cpu",
            inference_seconds=elapsed,
        )

    def analyse(
        self,
        image: object,
        *,
        model_artifact: NormalFeatureModelArtifact,
        calibration_artifact: CalibrationArtifact,
        operating_point_id: str,
        image_id: str,
        colour_space: str | None = None,
        alpha_handling: str | None = None,
        artifact_sink: ArtifactSink | None = None,
    ) -> NormalFeatureAnalysisResult:
        if calibration_artifact.model_artifact_hash != model_artifact.artifact_hash:
            raise ModelArtifactError("Calibration and fitted-model artifact identities differ")
        if calibration_artifact.proposal_extraction_policy_hash != self.config.proposal.configuration_hash:
            raise ModelArtifactError("Calibration and proposal-extraction policy identities differ")
        operating_point = calibration_artifact.operating_point(operating_point_id)
        scored = self.score(
            image,
            model_artifact=model_artifact,
            image_id=image_id,
            colour_space=colour_space,
            alpha_handling=alpha_handling,
        )
        proposals = extract_proposals(
            scored.anomaly_map,
            threshold=operating_point.threshold,
            operating_point_id=operating_point_id,
            config=self.config.proposal,
        )
        result = NormalFeatureAnalysisResult(
            image_id=image_id,
            input_hash=scored.input_hash,
            image_shape=scored.image_shape,
            image_anomaly_score=scored.image_anomaly_score,
            anomaly_map=scored.anomaly_map,
            anomaly_map_hash=scored.anomaly_map_hash,
            anomaly_map_coordinate_system="full_resolution_analysed_image_pixels",
            proposals=proposals,
            model_artifact_hash=model_artifact.artifact_hash,
            calibration_artifact_hash=calibration_artifact.artifact_hash,
            configuration_hash=self.config.configuration_hash,
            implementation_id=self.config.implementation_id,
            implementation_version=self.config.implementation_version,
            preprocessing_metadata=scored.preprocessing_metadata,
            deterministic_mode=True,
            device="cpu",
            inference_seconds=scored.inference_seconds,
            warnings=("development-only — non-confirmatory", "raw PatchCore distances are not probabilities"),
            provenance=tuple(sorted({
                "upstream": f"anomalib=={self.config.upstream_version}",
                "weight_sha256": self.config.pretrained_weight_sha256,
                "environment_lock_hash": self.environment_lock_hash,
                "mps_reference": False,
            }.items())),
        )
        if artifact_sink is not None:
            artifact_sink.write(result)
        return result

    def analyse_batch(
        self,
        samples: Iterable[AnalysisSample],
        *,
        model_artifact: NormalFeatureModelArtifact,
        calibration_artifact: CalibrationArtifact,
        operating_point_id: str,
        artifact_sink: ArtifactSink | None = None,
    ) -> tuple[NormalFeatureAnalysisResult, ...]:
        return tuple(
            self.analyse(
                sample.image,
                model_artifact=model_artifact,
                calibration_artifact=calibration_artifact,
                operating_point_id=operating_point_id,
                image_id=sample.image_id,
                colour_space=sample.colour_space,
                alpha_handling=sample.alpha_handling,
                artifact_sink=artifact_sink,
            )
            for sample in samples
        )
