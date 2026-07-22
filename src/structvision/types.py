"""Immutable domain records returned by the reusable API."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, Union

import numpy as np

from .provenance import ProvenanceRecord


JSONScalar = Union[str, int, float, bool, None]
FrozenValue = Union[JSONScalar, tuple["FrozenValue", ...], tuple[tuple[str, "FrozenValue"], ...]]


class FrozenObject(tuple):
    """Tagged immutable JSON object representation (including the empty object)."""


class FrozenArray(tuple):
    """Tagged immutable JSON array representation (including the empty array)."""


def freeze_value(value: Any) -> FrozenValue:
    """Recursively remove mutable containers while preserving deterministic order."""
    if isinstance(value, Mapping):
        return FrozenObject((str(key), freeze_value(item)) for key, item in sorted(value.items(), key=lambda pair: str(pair[0])))
    if isinstance(value, (list, tuple)):
        return FrozenArray(freeze_value(item) for item in value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Domain metadata cannot contain NaN or infinity")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Unsupported immutable metadata value: {type(value).__name__}")


def thaw_value(value: FrozenValue) -> Any:
    if isinstance(value, FrozenObject):
        return {item[0]: thaw_value(item[1]) for item in value}
    if isinstance(value, (FrozenArray, tuple)):
        return [thaw_value(item) for item in value]
    return value


def frozen_mapping(value: Mapping[str, Any] | None) -> tuple[tuple[str, FrozenValue], ...]:
    frozen = freeze_value(value or {})
    if not isinstance(frozen, FrozenObject):
        raise TypeError("Expected a mapping")
    return frozen


def _readonly_array(value: np.ndarray, *, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    copied = np.ascontiguousarray(array).copy()
    copied.setflags(write=False)
    return copied


def _array_payload(array: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "order": "C",
        "encoding": "base64-raw-bytes",
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
        "data": base64.b64encode(contiguous.tobytes(order="C")).decode("ascii"),
    }


@dataclass(frozen=True)
class Proposal:
    """One ranked proposal; ``bbox`` is half-open ``(x_min, y_min, x_max, y_max)``."""

    proposal_id: str
    rank: int
    bbox: tuple[int, int, int, int]
    final_mask: np.ndarray
    raw_mask: np.ndarray | None
    proposal_score: float
    evidence_score: float
    heuristic_reliability: float
    priority_score: float
    component_scores: tuple[tuple[str, float], ...]
    area: int
    centroid: tuple[float, float]
    context_diagnostics: tuple[tuple[str, FrozenValue], ...]
    warnings: tuple[str, ...]
    rejection_information: tuple[str, ...]
    implementation_id: str
    implementation_version: str

    def __post_init__(self) -> None:
        if not self.proposal_id.strip() or self.rank <= 0:
            raise ValueError("Proposal ID and one-based rank are required")
        mask = _readonly_array(self.final_mask, ndim=2, name="final_mask")
        if mask.dtype != np.uint8 or not np.all((mask == 0) | (mask == 255)) or not np.any(mask):
            raise ValueError("final_mask must be a non-empty uint8 binary mask using 0 and 255")
        object.__setattr__(self, "final_mask", mask)
        if self.raw_mask is not None:
            raw = _readonly_array(self.raw_mask, ndim=2, name="raw_mask")
            if raw.shape != mask.shape or raw.dtype != np.uint8 or not np.all((raw == 0) | (raw == 255)):
                raise ValueError("raw_mask must match final_mask shape and uint8 binary encoding")
            object.__setattr__(self, "raw_mask", raw)
        if len(self.bbox) != 4 or any(type(item) is not int for item in self.bbox):
            raise ValueError("bbox must contain four integers")
        x1, y1, x2, y2 = self.bbox
        height, width = mask.shape
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("bbox is outside its mask or is not half-open")
        ys, xs = np.where(mask > 0)
        expected = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
        if expected != self.bbox:
            raise ValueError("bbox must be recomputed from the final mask")
        if self.area != int(np.count_nonzero(mask)):
            raise ValueError("area must equal the final-mask foreground count")
        for name in ("proposal_score", "evidence_score", "heuristic_reliability", "priority_score"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if any(not math.isfinite(float(value)) for _, value in self.component_scores):
            raise ValueError("component scores must be finite")
        if len({name for name, _ in self.component_scores}) != len(self.component_scores):
            raise ValueError("component score names must be unique")

    @property
    def mask_reliability(self) -> float:
        """Alias with the methodology term; this is not calibrated confidence."""
        return self.heuristic_reliability

    def to_dict(self) -> dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "rank": self.rank,
            "bbox": list(self.bbox),
            "bbox_convention": "half-open:x_min,y_min,x_max,y_max",
            "final_mask": _array_payload(self.final_mask),
            "raw_mask": _array_payload(self.raw_mask) if self.raw_mask is not None else None,
            "mask_coordinate_space": "analysed_image_pixels",
            "proposal_score": self.proposal_score,
            "evidence_score": self.evidence_score,
            "heuristic_reliability": self.heuristic_reliability,
            "priority_score": self.priority_score,
            "component_scores": {name: value for name, value in self.component_scores},
            "area": self.area,
            "centroid_xy": list(self.centroid),
            "context_diagnostics": {name: thaw_value(value) for name, value in self.context_diagnostics},
            "warnings": list(self.warnings),
            "rejection_information": list(self.rejection_information),
            "implementation_id": self.implementation_id,
            "implementation_version": self.implementation_version,
        }


@dataclass(frozen=True)
class AnalysisResult:
    image_id: str
    input_hash: str
    image_shape: tuple[int, int, int]
    normalised_colour_space: str
    proposals: tuple[Proposal, ...]
    anomaly_heatmap: np.ndarray | None
    preprocessing_metadata: tuple[tuple[str, FrozenValue], ...]
    configuration_hash: str
    implementation_id: str
    implementation_version: str
    deterministic_mode: bool
    timing_breakdown_seconds: tuple[tuple[str, float], ...]
    warnings: tuple[str, ...]
    provenance: ProvenanceRecord
    image_metadata: tuple[tuple[str, FrozenValue], ...] = ()
    diagnostics: tuple[tuple[str, FrozenValue], ...] = ()

    def __post_init__(self) -> None:
        if not self.image_id.strip():
            raise ValueError("image_id is required")
        for name, value in (("input_hash", self.input_hash), ("configuration_hash", self.configuration_hash)):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if len(self.image_shape) != 3 or self.image_shape[2] != 3 or min(self.image_shape) <= 0:
            raise ValueError("image_shape must describe a non-empty three-channel analysed image")
        if self.normalised_colour_space != "BGR":
            raise ValueError("The frozen implementation coordinate space is normalised BGR")
        ranks = [item.rank for item in self.proposals]
        identifiers = [item.proposal_id for item in self.proposals]
        if ranks != list(range(1, len(ranks) + 1)) or len(identifiers) != len(set(identifiers)):
            raise ValueError("Proposal ranks must be ordered, unique, contiguous, and one-based")
        if any(item.final_mask.shape != self.image_shape[:2] for item in self.proposals):
            raise ValueError("Every proposal mask must use analysed-image coordinates")
        if self.anomaly_heatmap is not None:
            heatmap = _readonly_array(self.anomaly_heatmap, ndim=3, name="anomaly_heatmap")
            if heatmap.shape != self.image_shape or heatmap.dtype != np.uint8:
                raise ValueError("anomaly_heatmap must be uint8 and match analysed-image shape")
            object.__setattr__(self, "anomaly_heatmap", heatmap)
        if len({name for name, _ in self.timing_breakdown_seconds}) != len(self.timing_breakdown_seconds):
            raise ValueError("Timing keys must be unique")
        if any(not math.isfinite(float(value)) or float(value) < 0 for _, value in self.timing_breakdown_seconds):
            raise ValueError("Timing values must be finite and non-negative")

    @property
    def identity(self) -> str:
        payload = f"{self.implementation_id}\0{self.configuration_hash}\0{self.image_id}\0{self.input_hash}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self, *, include_heatmap: bool = True) -> dict[str, object]:
        return {
            "image_id": self.image_id,
            "input_hash": self.input_hash,
            "image_shape": list(self.image_shape),
            "normalised_colour_space": self.normalised_colour_space,
            "proposals": [item.to_dict() for item in self.proposals],
            "anomaly_heatmap": _array_payload(self.anomaly_heatmap) if include_heatmap and self.anomaly_heatmap is not None else None,
            "preprocessing_metadata": {name: thaw_value(value) for name, value in self.preprocessing_metadata},
            "configuration_hash": self.configuration_hash,
            "implementation_id": self.implementation_id,
            "implementation_version": self.implementation_version,
            "deterministic_mode": self.deterministic_mode,
            "timing_breakdown_seconds": dict(self.timing_breakdown_seconds),
            "warnings": list(self.warnings),
            "provenance": self.provenance.to_dict(),
            "image_metadata": {name: thaw_value(value) for name, value in self.image_metadata},
            "diagnostics": {name: thaw_value(value) for name, value in self.diagnostics},
            "result_identity": self.identity,
        }

    def to_json(self, *, include_heatmap: bool = True) -> str:
        return json.dumps(
            self.to_dict(include_heatmap=include_heatmap), sort_keys=True,
            separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        )


@dataclass(frozen=True)
class AnalysisSample:
    image: object
    image_id: str
    colour_space: str | None = None
    alpha_handling: str | None = None
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not self.image_id.strip():
            raise ValueError("AnalysisSample.image_id is required")


@dataclass(frozen=True)
class BatchFailure:
    index: int
    image_id: str
    error_type: str
    message: str


@dataclass(frozen=True)
class BatchAnalysisResult(Sequence[AnalysisResult]):
    input_count: int
    results: tuple[AnalysisResult, ...]
    failures: tuple[BatchFailure, ...]
    fail_fast: bool
    worker_count: int

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, index: int | slice) -> AnalysisResult | tuple[AnalysisResult, ...]:
        return self.results[index]

    def __iter__(self) -> Iterator[AnalysisResult]:
        return iter(self.results)

    @property
    def completed_count(self) -> int:
        return len(self.results)

    @property
    def failed_count(self) -> int:
        return len(self.failures)
