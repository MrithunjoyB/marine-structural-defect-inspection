"""Immutable, content-addressed future experiment specification."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from typing import Mapping

from .hashing import canonical_json, is_sha256, sha256_json


SPECIFICATION_SCHEMA_VERSION = "experiment-specification-v2"


@dataclass(frozen=True)
class FrozenConfiguration:
    canonical_payload: str
    configuration_hash: str

    @classmethod
    def from_value(cls, value: Mapping[str, object]) -> "FrozenConfiguration":
        if not isinstance(value, Mapping) or not value:
            raise ValueError("A complete non-empty configuration mapping is required")
        payload = canonical_json(value)
        return cls(payload, sha256_json(value))

    def __post_init__(self) -> None:
        try:
            value = json.loads(self.canonical_payload)
        except json.JSONDecodeError as error:
            raise ValueError("Frozen configuration is not valid JSON") from error
        if not isinstance(value, dict) or not value:
            raise ValueError("Frozen configuration cannot be empty")
        if canonical_json(value) != self.canonical_payload or sha256_json(value) != self.configuration_hash:
            raise ValueError("Frozen configuration is not canonical or its hash is invalid")

    @property
    def value(self) -> dict[str, object]:
        return json.loads(self.canonical_payload)

    def to_dict(self) -> dict[str, object]:
        return {"payload": self.value, "configuration_hash": self.configuration_hash}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FrozenConfiguration":
        value = payload["payload"]
        return cls(canonical_json(value), str(payload["configuration_hash"]))


@dataclass(frozen=True)
class SelectedImageIdentity:
    image_id: str
    image_sha256: str
    ground_truth_sha256: str

    def __post_init__(self) -> None:
        if not self.image_id.strip() or not is_sha256(self.image_sha256) or not is_sha256(self.ground_truth_sha256):
            raise ValueError("Selected image identity requires an ID and SHA-256 image/truth hashes")

    def to_dict(self) -> dict[str, str]:
        return {"image_id": self.image_id, "image_sha256": self.image_sha256, "ground_truth_sha256": self.ground_truth_sha256}


@dataclass(frozen=True)
class MethodSpecification:
    method_id: str
    implementation_version: str
    method_configuration: FrozenConfiguration
    ranking_eligible: bool
    ranking_definition: str | None

    def __post_init__(self) -> None:
        if not self.method_id.strip() or not self.implementation_version.strip():
            raise ValueError("Method ID and implementation version are required")
        if self.ranking_eligible and (not self.ranking_definition or not self.ranking_definition.strip()):
            raise ValueError("Rank-eligible methods require a ranking definition")

    def to_dict(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "implementation_version": self.implementation_version,
            "method_configuration": self.method_configuration.to_dict(),
            "ranking_eligible": self.ranking_eligible,
            "ranking_definition": self.ranking_definition,
        }


@dataclass(frozen=True)
class ExperimentSpecificationV2:
    schema_version: str
    experiment_id: str
    experiment_version: int
    dataset_id: str
    dataset_version: str
    dataset_manifest_hash: str
    split_manifest_hash: str
    split_lock_hash: str
    selected_images: tuple[SelectedImageIdentity, ...]
    methods: tuple[MethodSpecification, ...]
    preprocessing_configuration: FrozenConfiguration
    proposal_configuration: FrozenConfiguration
    feature_scoring_configuration: FrozenConfiguration
    maximum_proposal_count: int
    random_seeds: tuple[tuple[str, int], ...]
    deterministic_mode: bool
    evaluation_policy_id: str
    evaluation_policy_version: int
    evaluation_policy_hash: str
    matching_thresholds: tuple[float, ...]
    metric_definitions_hash: str
    allowed_fitting_splits: tuple[str, ...]
    forbidden_test_access: bool
    git_commit: str
    git_tree_state: str
    uncommitted_diff_hash: str | None
    python_version: str
    dependency_snapshot: FrozenConfiguration
    dependency_lock_hash: str
    operating_system_metadata: FrozenConfiguration
    hardware_metadata: FrozenConfiguration
    opencv_version: str
    opencv_backend: str
    creation_timestamp: str
    expected_executed_configuration_hashes: tuple[tuple[str, str], ...] = ()
    specification_hash: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != SPECIFICATION_SCHEMA_VERSION:
            raise ValueError("Unsupported experiment specification schema")
        required = (self.experiment_id, self.dataset_id, self.dataset_version, self.evaluation_policy_id, self.python_version, self.opencv_version, self.opencv_backend)
        if any(not value.strip() for value in required) or self.experiment_version <= 0:
            raise ValueError("Required scientific identity fields are missing")
        hashes = (self.dataset_manifest_hash, self.split_manifest_hash, self.split_lock_hash, self.evaluation_policy_hash, self.metric_definitions_hash, self.dependency_lock_hash)
        if any(not is_sha256(value) for value in hashes):
            raise ValueError("Manifest, policy, metric, and dependency hashes must be SHA-256")
        if not self.selected_images or not self.methods:
            raise ValueError("At least one selected image and method are required")
        image_ids = [image.image_id for image in self.selected_images]
        method_ids = [method.method_id for method in self.methods]
        if len(image_ids) != len(set(image_ids)) or len(method_ids) != len(set(method_ids)):
            raise ValueError("Selected image IDs and method IDs must be unique")
        if self.maximum_proposal_count <= 0:
            raise ValueError("Maximum proposal count must be positive")
        if not self.random_seeds or len({name for name, _ in self.random_seeds}) != len(self.random_seeds):
            raise ValueError("Named random seeds are required and must be unique")
        if not self.matching_thresholds or any(value < 0 or value > 1 for value in self.matching_thresholds):
            raise ValueError("Matching thresholds must be explicit and within [0, 1]")
        if not self.allowed_fitting_splits or "test" in self.allowed_fitting_splits or not self.forbidden_test_access:
            raise ValueError("Fitting splits must be declared and test access must be forbidden")
        if self.git_tree_state not in {"clean", "dirty"}:
            raise ValueError("Git tree state must be clean or dirty")
        if self.git_tree_state == "clean" and self.uncommitted_diff_hash is not None:
            raise ValueError("Clean Git state cannot have an uncommitted diff hash")
        if self.git_tree_state == "dirty" and (self.uncommitted_diff_hash is None or not is_sha256(self.uncommitted_diff_hash)):
            raise ValueError("Dirty Git state requires an uncommitted diff hash")
        try:
            timestamp = datetime.fromisoformat(self.creation_timestamp)
        except ValueError as error:
            raise ValueError("Creation timestamp must be ISO-8601") from error
        if timestamp.tzinfo is None:
            raise ValueError("Creation timestamp must include a timezone")
        expected = tuple((method.method_id, sha256_json(self.expected_executable_configuration(method.method_id))) for method in self.methods)
        if self.expected_executed_configuration_hashes and self.expected_executed_configuration_hashes != expected:
            raise ValueError("Specified executed-configuration hashes do not match the frozen specification")
        if not self.expected_executed_configuration_hashes:
            object.__setattr__(self, "expected_executed_configuration_hashes", expected)
        calculated = sha256_json(self.to_dict(include_specification_hash=False))
        if self.specification_hash and self.specification_hash != calculated:
            raise ValueError("Experiment specification hash mismatch")
        if not self.specification_hash:
            object.__setattr__(self, "specification_hash", calculated)

    @property
    def expected_pair_count(self) -> int:
        return len(self.selected_images) * len(self.methods)

    def method(self, method_id: str) -> MethodSpecification:
        try:
            return next(method for method in self.methods if method.method_id == method_id)
        except StopIteration as error:
            raise KeyError(method_id) from error

    def expected_executable_configuration(self, method_id: str) -> dict[str, object]:
        method = self.method(method_id)
        return {
            "method": method.to_dict(),
            "preprocessing": self.preprocessing_configuration.to_dict(),
            "proposal": self.proposal_configuration.to_dict(),
            "feature_and_scoring": self.feature_scoring_configuration.to_dict(),
            "maximum_proposal_count": self.maximum_proposal_count,
            "random_seeds": list(self.random_seeds),
            "deterministic_mode": self.deterministic_mode,
            "evaluation_policy": {
                "id": self.evaluation_policy_id,
                "version": self.evaluation_policy_version,
                "hash": self.evaluation_policy_hash,
                "matching_thresholds": list(self.matching_thresholds),
                "metric_definitions_hash": self.metric_definitions_hash,
            },
        }

    def verify_executed_configuration(self, method_id: str, executed: Mapping[str, object]) -> str:
        actual = sha256_json(executed)
        expected = dict(self.expected_executed_configuration_hashes)[method_id]
        if actual != expected:
            raise ValueError("Executed configuration differs from the immutable specification")
        return actual

    def to_dict(self, *, include_specification_hash: bool = True) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "experiment_version": self.experiment_version,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "split_manifest_hash": self.split_manifest_hash,
            "split_lock_hash": self.split_lock_hash,
            "selected_images": [image.to_dict() for image in self.selected_images],
            "methods": [method.to_dict() for method in self.methods],
            "preprocessing_configuration": self.preprocessing_configuration.to_dict(),
            "proposal_configuration": self.proposal_configuration.to_dict(),
            "feature_scoring_configuration": self.feature_scoring_configuration.to_dict(),
            "maximum_proposal_count": self.maximum_proposal_count,
            "random_seeds": list(self.random_seeds),
            "deterministic_mode": self.deterministic_mode,
            "evaluation_policy_id": self.evaluation_policy_id,
            "evaluation_policy_version": self.evaluation_policy_version,
            "evaluation_policy_hash": self.evaluation_policy_hash,
            "matching_thresholds": list(self.matching_thresholds),
            "metric_definitions_hash": self.metric_definitions_hash,
            "allowed_fitting_splits": list(self.allowed_fitting_splits),
            "forbidden_test_access": self.forbidden_test_access,
            "git_commit": self.git_commit,
            "git_tree_state": self.git_tree_state,
            "uncommitted_diff_hash": self.uncommitted_diff_hash,
            "python_version": self.python_version,
            "dependency_snapshot": self.dependency_snapshot.to_dict(),
            "dependency_lock_hash": self.dependency_lock_hash,
            "operating_system_metadata": self.operating_system_metadata.to_dict(),
            "hardware_metadata": self.hardware_metadata.to_dict(),
            "opencv_version": self.opencv_version,
            "opencv_backend": self.opencv_backend,
            "creation_timestamp": self.creation_timestamp,
            "expected_executed_configuration_hashes": list(self.expected_executed_configuration_hashes),
        }
        if include_specification_hash:
            payload["specification_hash"] = self.specification_hash
        return payload

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, value: str) -> "ExperimentSpecificationV2":
        payload = json.loads(value)
        if canonical_json(payload) != value:
            raise ValueError("Serialized specification must use canonical JSON")
        methods = tuple(
            MethodSpecification(
                str(item["method_id"]), str(item["implementation_version"]),
                FrozenConfiguration.from_dict(item["method_configuration"]),
                bool(item["ranking_eligible"]), item.get("ranking_definition"),
            )
            for item in payload["methods"]
        )
        images = tuple(
            SelectedImageIdentity(str(item["image_id"]), str(item["image_sha256"]), str(item["ground_truth_sha256"]))
            for item in payload["selected_images"]
        )
        return cls(
            str(payload["schema_version"]), str(payload["experiment_id"]),
            int(payload["experiment_version"]), str(payload["dataset_id"]),
            str(payload["dataset_version"]), str(payload["dataset_manifest_hash"]),
            str(payload["split_manifest_hash"]), str(payload["split_lock_hash"]),
            images, methods,
            FrozenConfiguration.from_dict(payload["preprocessing_configuration"]),
            FrozenConfiguration.from_dict(payload["proposal_configuration"]),
            FrozenConfiguration.from_dict(payload["feature_scoring_configuration"]),
            int(payload["maximum_proposal_count"]),
            tuple((str(name), int(seed)) for name, seed in payload["random_seeds"]),
            bool(payload["deterministic_mode"]), str(payload["evaluation_policy_id"]),
            int(payload["evaluation_policy_version"]), str(payload["evaluation_policy_hash"]),
            tuple(float(item) for item in payload["matching_thresholds"]),
            str(payload["metric_definitions_hash"]), tuple(str(item) for item in payload["allowed_fitting_splits"]),
            bool(payload["forbidden_test_access"]), str(payload["git_commit"]),
            str(payload["git_tree_state"]), payload.get("uncommitted_diff_hash"),
            str(payload["python_version"]), FrozenConfiguration.from_dict(payload["dependency_snapshot"]),
            str(payload["dependency_lock_hash"]), FrozenConfiguration.from_dict(payload["operating_system_metadata"]),
            FrozenConfiguration.from_dict(payload["hardware_metadata"]), str(payload["opencv_version"]),
            str(payload["opencv_backend"]), str(payload["creation_timestamp"]),
            tuple((str(method_id), str(configuration_hash)) for method_id, configuration_hash in payload["expected_executed_configuration_hashes"]),
            str(payload["specification_hash"]),
        )
