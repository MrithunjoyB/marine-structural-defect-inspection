"""Immutable learned-baseline inputs and outputs without false score equivalence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping

import numpy as np

from structvision.types import frozen_mapping


def readonly_array(value: np.ndarray, *, ndim: int, dtype: np.dtype, name: str) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype).copy()
    if array.ndim != ndim or array.size == 0:
        raise ValueError(f"{name} must be a non-empty {ndim}-dimensional array")
    array.setflags(write=False)
    return array


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    prefix = f"{array.dtype.str}\0{','.join(str(item) for item in array.shape)}\0C\0".encode("ascii")
    return hashlib.sha256(prefix + array.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class NormalFitSample:
    image: object
    image_id: str
    image_sha256: str
    ground_truth_sha256: str
    role: str = "normal_fit"
    image_outcome: str = "no_anomaly"
    colour_space: str | None = None
    alpha_handling: str | None = None
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not self.image_id or self.role != "normal_fit" or self.image_outcome != "no_anomaly":
            raise ValueError("Fitting accepts only explicit no_anomaly normal_fit samples")
        for value in (self.image_sha256, self.ground_truth_sha256):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError("Normal-fit identities require SHA-256 hashes")


@dataclass(frozen=True)
class LearnedProposal:
    proposal_id: str
    rank: int
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    component_anomaly_score: float
    area: int
    centroid_xy: tuple[float, float]
    threshold: float
    operating_point_id: str
    mask_hash: str
    extraction_policy_hash: str

    def __post_init__(self) -> None:
        mask = readonly_array(self.mask, ndim=2, dtype=np.uint8, name="proposal mask")
        if not np.all((mask == 0) | (mask == 255)) or not np.any(mask):
            raise ValueError("Proposal mask must use non-empty 0/255 binary encoding")
        object.__setattr__(self, "mask", mask)
        if self.rank <= 0 or not self.proposal_id:
            raise ValueError("Proposal ID and one-based rank are required")
        ys, xs = np.where(mask > 0)
        expected_box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        if self.bbox != expected_box or self.area != int(np.count_nonzero(mask)):
            raise ValueError("Proposal geometry must be recomputed from its final mask")
        if not math.isfinite(self.component_anomaly_score) or not math.isfinite(self.threshold):
            raise ValueError("Learned anomaly distances and thresholds must be finite")
        if self.mask_hash != array_hash(mask):
            raise ValueError("Proposal mask hash mismatch")

    @property
    def final_mask(self) -> np.ndarray:
        """High-level compatibility alias for scientific-contract adaptation."""
        return self.mask

    @property
    def priority_score(self) -> float:
        """Rankable anomaly distance; it is not classical priority or probability."""
        return self.component_anomaly_score

    def to_dict(self, *, include_mask: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "proposal_id": self.proposal_id,
            "rank": self.rank,
            "bbox": list(self.bbox),
            "bbox_convention": "half-open:x_min,y_min,x_max,y_max",
            "component_anomaly_score": self.component_anomaly_score,
            "score_semantics": "raw_patchcore_distance_not_probability",
            "area": self.area,
            "centroid_xy": list(self.centroid_xy),
            "threshold": self.threshold,
            "operating_point_id": self.operating_point_id,
            "mask_hash": self.mask_hash,
            "extraction_policy_hash": self.extraction_policy_hash,
        }
        if include_mask:
            payload["mask"] = self.mask.tolist()
        return payload


@dataclass(frozen=True)
class NormalFeatureAnalysisResult:
    image_id: str
    input_hash: str
    image_shape: tuple[int, int, int]
    image_anomaly_score: float
    anomaly_map: np.ndarray
    anomaly_map_hash: str
    anomaly_map_coordinate_system: str
    proposals: tuple[LearnedProposal, ...]
    model_artifact_hash: str
    calibration_artifact_hash: str
    configuration_hash: str
    implementation_id: str
    implementation_version: str
    preprocessing_metadata: tuple[tuple[str, object], ...]
    deterministic_mode: bool
    device: str
    inference_seconds: float
    warnings: tuple[str, ...]
    provenance: tuple[tuple[str, object], ...]

    def __post_init__(self) -> None:
        anomaly_map = readonly_array(self.anomaly_map, ndim=2, dtype=np.float32, name="anomaly map")
        if anomaly_map.shape != self.image_shape[:2] or not np.isfinite(anomaly_map).all():
            raise ValueError("Anomaly map must be finite and use full analysed-image coordinates")
        object.__setattr__(self, "anomaly_map", anomaly_map)
        if self.anomaly_map_hash != array_hash(anomaly_map):
            raise ValueError("Anomaly-map hash mismatch")
        if not math.isfinite(self.image_anomaly_score) or self.image_anomaly_score < 0:
            raise ValueError("Image anomaly score must be a finite non-negative distance")
        ranks = tuple(item.rank for item in self.proposals)
        if ranks != tuple(range(1, len(ranks) + 1)):
            raise ValueError("Learned proposal ranks must be contiguous and ordered")
        if any(item.mask.shape != self.image_shape[:2] for item in self.proposals):
            raise ValueError("Learned proposals must use full-resolution coordinates")
        if self.device != "cpu" or not self.deterministic_mode:
            raise ValueError("Scientific learned results must use deterministic CPU")

    def to_dict(self, *, include_map: bool = False, include_masks: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "image_id": self.image_id,
            "input_hash": self.input_hash,
            "image_shape": list(self.image_shape),
            "image_anomaly_score": self.image_anomaly_score,
            "score_semantics": "raw_patchcore_distance_not_probability",
            "anomaly_map": None,
            "anomaly_map_hash": self.anomaly_map_hash,
            "anomaly_map_shape": list(self.anomaly_map.shape),
            "anomaly_map_dtype": str(self.anomaly_map.dtype),
            "anomaly_map_coordinate_system": self.anomaly_map_coordinate_system,
            "proposals": [item.to_dict(include_mask=include_masks) for item in self.proposals],
            "model_artifact_hash": self.model_artifact_hash,
            "calibration_artifact_hash": self.calibration_artifact_hash,
            "configuration_hash": self.configuration_hash,
            "implementation_id": self.implementation_id,
            "implementation_version": self.implementation_version,
            "preprocessing_metadata": dict(self.preprocessing_metadata),
            "deterministic_mode": self.deterministic_mode,
            "device": self.device,
            "inference_seconds": self.inference_seconds,
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
        }
        if include_map:
            payload["anomaly_map"] = self.anomaly_map.tolist()
        return payload


@dataclass(frozen=True)
class NormalFeatureScoreResult:
    """Unthresholded model output used only inside calibration-validation."""

    image_id: str
    input_hash: str
    image_shape: tuple[int, int, int]
    image_anomaly_score: float
    anomaly_map: np.ndarray
    anomaly_map_hash: str
    model_artifact_hash: str
    configuration_hash: str
    preprocessing_metadata: tuple[tuple[str, object], ...]
    deterministic_mode: bool
    device: str
    inference_seconds: float

    def __post_init__(self) -> None:
        anomaly_map = readonly_array(self.anomaly_map, ndim=2, dtype=np.float32, name="anomaly map")
        if anomaly_map.shape != self.image_shape[:2] or not np.isfinite(anomaly_map).all():
            raise ValueError("Score map must be finite and use analysed-image coordinates")
        object.__setattr__(self, "anomaly_map", anomaly_map)
        if self.anomaly_map_hash != array_hash(anomaly_map):
            raise ValueError("Score-map hash mismatch")
        if not math.isfinite(self.image_anomaly_score):
            raise ValueError("Image anomaly score must be finite")
        if self.device != "cpu" or not self.deterministic_mode:
            raise ValueError("Calibration scores require deterministic CPU")


def immutable_metadata(value: Mapping[str, object] | None) -> tuple[tuple[str, object], ...]:
    return tuple(frozen_mapping(value))
