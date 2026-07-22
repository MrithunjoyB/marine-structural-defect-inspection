"""Content-addressed fitted PatchCore memory with explicit persistence only."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol

import numpy as np

from scientific_contract.hashing import canonical_json, is_sha256, sha256_json

from .configuration import (
    IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION,
    WEIGHT_FILENAME,
    WEIGHT_LICENCE,
    WEIGHT_MODEL_ID,
    WEIGHT_REVISION,
    WEIGHT_SHA256,
    WEIGHT_SOURCE,
    NormalFeatureConfig,
)
from .errors import ModelArtifactError
from .types import array_hash, readonly_array


MODEL_ARTIFACT_SCHEMA_VERSION = "normal-feature-model-artifact-v1"


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain(item) for item in value)
    return value


@dataclass(frozen=True)
class NormalFeatureModelArtifact:
    schema_version: str
    method_identity: str
    method_version: str
    model_configuration: Mapping[str, object]
    configuration_hash: str
    selected_normal_fit_ids: tuple[str, ...]
    normal_fit_image_hashes: tuple[tuple[str, str], ...]
    normal_fit_manifest_hash: str
    backbone_weight_hash: str
    weight_provenance: Mapping[str, object]
    extracted_layer_identities: tuple[str, ...]
    memory_bank: np.ndarray
    memory_bank_shape: tuple[int, ...]
    memory_bank_dtype: str
    memory_bank_hash: str
    selected_coreset_indices: tuple[int, ...]
    coreset_hash: str
    nearest_neighbour_index_configuration: Mapping[str, object]
    preprocessing_hash: str
    seed: int
    deterministic_state: Mapping[str, object]
    environment_lock_hash: str
    hardware_runtime_metadata: Mapping[str, object]
    git_commit: str
    git_dirty_state: str
    git_diff_hash: str | None
    creation_timestamp: str
    artifact_hash: str

    def __post_init__(self) -> None:
        memory = readonly_array(self.memory_bank, ndim=2, dtype=np.float32, name="memory bank")
        object.__setattr__(self, "memory_bank", memory)
        if self.schema_version != MODEL_ARTIFACT_SCHEMA_VERSION:
            raise ModelArtifactError("Unsupported fitted-model artifact schema")
        if self.method_identity != IMPLEMENTATION_ID or self.method_version != IMPLEMENTATION_VERSION:
            raise ModelArtifactError("Fitted-model method identity differs from the protected baseline")
        if sha256_json(_plain(self.model_configuration)) != self.configuration_hash:
            raise ModelArtifactError("Fitted-model configuration payload and hash differ")
        try:
            expected_configuration = NormalFeatureConfig().to_dict()
        except ValueError as error:  # pragma: no cover - guards a programmer error in fixed constants
            raise ModelArtifactError("Invalid protected baseline constants") from error
        if self.configuration_hash != sha256_json(expected_configuration):
            raise ModelArtifactError("Fitted-model configuration is not the predeclared baseline")
        expected_weight = {
            "source": WEIGHT_SOURCE,
            "model_id": WEIGHT_MODEL_ID,
            "revision": WEIGHT_REVISION,
            "filename": WEIGHT_FILENAME,
            "sha256": WEIGHT_SHA256,
            "licence": WEIGHT_LICENCE,
        }
        if any(self.weight_provenance.get(key) != value for key, value in expected_weight.items()):
            raise ModelArtifactError("Fitted-model weight provenance is incomplete or unofficial")
        if self.backbone_weight_hash != WEIGHT_SHA256:
            raise ModelArtifactError("Fitted-model backbone weight differs from the official identity")
        if memory.shape != self.memory_bank_shape or str(memory.dtype) != self.memory_bank_dtype:
            raise ModelArtifactError("Stored memory-bank shape or dtype differs from metadata")
        if array_hash(memory) != self.memory_bank_hash:
            raise ModelArtifactError("Memory-bank hash mismatch")
        if sha256_json(list(self.selected_coreset_indices)) != self.coreset_hash:
            raise ModelArtifactError("Coreset-index hash mismatch")
        if len(self.selected_normal_fit_ids) != len(self.normal_fit_image_hashes):
            raise ModelArtifactError("Normal-fit ID and hash counts differ")
        if len(set(self.selected_normal_fit_ids)) != len(self.selected_normal_fit_ids):
            raise ModelArtifactError("Normal-fit image IDs must be unique")
        if tuple(item[0] for item in self.normal_fit_image_hashes) != self.selected_normal_fit_ids:
            raise ModelArtifactError("Normal-fit IDs and image-hash records differ")
        if any(not is_sha256(item[1]) for item in self.normal_fit_image_hashes):
            raise ModelArtifactError("Normal-fit image records require SHA-256 identities")
        required_hashes = (
            self.configuration_hash, self.normal_fit_manifest_hash, self.backbone_weight_hash,
            self.memory_bank_hash, self.coreset_hash, self.preprocessing_hash,
            self.environment_lock_hash, self.artifact_hash,
        )
        if any(not is_sha256(value) for value in required_hashes):
            raise ModelArtifactError("Artifact provenance requires SHA-256 identities")
        if self.artifact_hash != "0" * 64 and self.artifact_hash != sha256_json(self.to_dict(include_artifact_hash=False)):
            raise ModelArtifactError("Fitted-model artifact hash mismatch")
        for field_name in (
            "model_configuration", "weight_provenance", "nearest_neighbour_index_configuration",
            "deterministic_state", "hardware_runtime_metadata",
        ):
            object.__setattr__(self, field_name, _freeze(getattr(self, field_name)))

    @classmethod
    def create(
        cls,
        *,
        config: NormalFeatureConfig,
        selected_normal_fit_ids: tuple[str, ...],
        normal_fit_image_hashes: tuple[tuple[str, str], ...],
        normal_fit_manifest_hash: str,
        weight_provenance: Mapping[str, object],
        memory_bank: np.ndarray,
        selected_coreset_indices: tuple[int, ...],
        preprocessing_hash: str,
        environment_lock_hash: str,
        hardware_runtime_metadata: Mapping[str, object],
        git_commit: str,
        git_dirty_state: str,
        git_diff_hash: str | None,
        creation_timestamp: str,
    ) -> "NormalFeatureModelArtifact":
        memory = np.ascontiguousarray(memory_bank, dtype=np.float32)
        initial = cls(
            MODEL_ARTIFACT_SCHEMA_VERSION,
            config.implementation_id,
            config.implementation_version,
            config.to_dict(),
            config.configuration_hash,
            selected_normal_fit_ids,
            normal_fit_image_hashes,
            normal_fit_manifest_hash,
            config.pretrained_weight_sha256,
            dict(weight_provenance),
            config.extracted_layers,
            memory,
            tuple(memory.shape),
            str(memory.dtype),
            array_hash(memory),
            selected_coreset_indices,
            sha256_json(list(selected_coreset_indices)),
            {
                "implementation": config.nearest_neighbour_index,
                "distance_metric": config.distance_metric,
                "num_neighbors": config.nearest_neighbour_count,
            },
            preprocessing_hash,
            config.random_seed,
            {
                "device": config.device,
                "torch_deterministic_algorithms": config.deterministic_mode,
                "torch_num_threads": config.torch_num_threads,
                "torch_num_interop_threads": config.torch_num_interop_threads,
            },
            environment_lock_hash,
            dict(hardware_runtime_metadata),
            git_commit,
            git_dirty_state,
            git_diff_hash,
            creation_timestamp,
            "0" * 64,
        )
        return replace(initial, artifact_hash=sha256_json(initial.to_dict(include_artifact_hash=False)))

    def to_dict(self, *, include_artifact_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "method_identity": self.method_identity,
            "method_version": self.method_version,
            "model_configuration": _plain(self.model_configuration),
            "configuration_hash": self.configuration_hash,
            "selected_normal_fit_ids": list(self.selected_normal_fit_ids),
            "normal_fit_image_hashes": [list(item) for item in self.normal_fit_image_hashes],
            "normal_fit_manifest_hash": self.normal_fit_manifest_hash,
            "backbone_weight_hash": self.backbone_weight_hash,
            "weight_provenance": _plain(self.weight_provenance),
            "extracted_layer_identities": list(self.extracted_layer_identities),
            "memory_bank_shape": list(self.memory_bank_shape),
            "memory_bank_dtype": self.memory_bank_dtype,
            "memory_bank_hash": self.memory_bank_hash,
            "selected_coreset_indices": list(self.selected_coreset_indices),
            "coreset_hash": self.coreset_hash,
            "nearest_neighbour_index_configuration": _plain(self.nearest_neighbour_index_configuration),
            "preprocessing_hash": self.preprocessing_hash,
            "seed": self.seed,
            "deterministic_state": _plain(self.deterministic_state),
            "environment_lock_hash": self.environment_lock_hash,
            "hardware_runtime_metadata": _plain(self.hardware_runtime_metadata),
            "git_commit": self.git_commit,
            "git_dirty_state": self.git_dirty_state,
            "git_diff_hash": self.git_diff_hash,
            "creation_timestamp": self.creation_timestamp,
        }
        if include_artifact_hash:
            payload["artifact_hash"] = self.artifact_hash
        return payload

    def to_json(self) -> str:
        return canonical_json(self.to_dict())


class ModelArtifactSink(Protocol):
    def write(self, artifact: NormalFeatureModelArtifact) -> None:
        ...


class DirectoryModelArtifactSink:
    """Explicitly persist metadata and memory bank under the artifact hash."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def write(self, artifact: NormalFeatureModelArtifact) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        metadata_path = self.directory / f"{artifact.artifact_hash}.json"
        bank_path = self.directory / f"{artifact.artifact_hash}.npz"
        if metadata_path.exists() or bank_path.exists():
            raise ModelArtifactError("Artifact persistence is immutable and refuses overwrite")
        np.savez_compressed(bank_path, memory_bank=artifact.memory_bank)
        metadata_path.write_text(artifact.to_json() + "\n", encoding="utf-8")


def load_model_artifact(metadata_path: Path) -> NormalFeatureModelArtifact:
    metadata_path = Path(metadata_path)
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelArtifactError("Could not read fitted-model metadata") from error
    if canonical_json(payload) + "\n" != metadata_path.read_text(encoding="utf-8"):
        raise ModelArtifactError("Fitted-model metadata is not canonical")
    artifact_hash = str(payload["artifact_hash"])
    bank_path = metadata_path.with_name(f"{artifact_hash}.npz")
    try:
        with np.load(bank_path, allow_pickle=False) as stored:
            memory_bank = stored["memory_bank"]
    except (OSError, KeyError, ValueError) as error:
        raise ModelArtifactError("Could not read fitted memory bank") from error
    return NormalFeatureModelArtifact(
        schema_version=str(payload["schema_version"]),
        method_identity=str(payload["method_identity"]),
        method_version=str(payload["method_version"]),
        model_configuration=dict(payload["model_configuration"]),
        configuration_hash=str(payload["configuration_hash"]),
        selected_normal_fit_ids=tuple(str(item) for item in payload["selected_normal_fit_ids"]),
        normal_fit_image_hashes=tuple((str(a), str(b)) for a, b in payload["normal_fit_image_hashes"]),
        normal_fit_manifest_hash=str(payload["normal_fit_manifest_hash"]),
        backbone_weight_hash=str(payload["backbone_weight_hash"]),
        weight_provenance=dict(payload["weight_provenance"]),
        extracted_layer_identities=tuple(str(item) for item in payload["extracted_layer_identities"]),
        memory_bank=memory_bank,
        memory_bank_shape=tuple(int(item) for item in payload["memory_bank_shape"]),
        memory_bank_dtype=str(payload["memory_bank_dtype"]),
        memory_bank_hash=str(payload["memory_bank_hash"]),
        selected_coreset_indices=tuple(int(item) for item in payload["selected_coreset_indices"]),
        coreset_hash=str(payload["coreset_hash"]),
        nearest_neighbour_index_configuration=dict(payload["nearest_neighbour_index_configuration"]),
        preprocessing_hash=str(payload["preprocessing_hash"]),
        seed=int(payload["seed"]),
        deterministic_state=dict(payload["deterministic_state"]),
        environment_lock_hash=str(payload["environment_lock_hash"]),
        hardware_runtime_metadata=dict(payload["hardware_runtime_metadata"]),
        git_commit=str(payload["git_commit"]),
        git_dirty_state=str(payload["git_dirty_state"]),
        git_diff_hash=payload.get("git_diff_hash"),
        creation_timestamp=str(payload["creation_timestamp"]),
        artifact_hash=artifact_hash,
    )
