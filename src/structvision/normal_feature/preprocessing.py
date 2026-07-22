"""Deterministic aspect-preserving preprocessing for the PatchCore adapter."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from scientific_contract.hashing import sha256_json
from structvision.inputs import NormalisedInput, normalise_input

from .configuration import NormalFeatureConfig


@dataclass(frozen=True)
class LetterboxGeometry:
    original_height: int
    original_width: int
    resized_height: int
    resized_width: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int

    def to_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class PreparedNormalFeatureInput:
    normalised_input: NormalisedInput
    tensor_chw: np.ndarray
    geometry: LetterboxGeometry
    preprocessing_hash: str

    def __post_init__(self) -> None:
        array = np.ascontiguousarray(self.tensor_chw, dtype=np.float32).copy()
        if array.ndim != 3 or array.shape[0] != 3 or not np.isfinite(array).all():
            raise ValueError("Prepared PatchCore input must be finite float32 CHW")
        array.setflags(write=False)
        object.__setattr__(self, "tensor_chw", array)


def preprocessing_contract(config: NormalFeatureConfig) -> dict[str, object]:
    return config.specification_sections()["preprocessing"]


def prepare_input(
    image: object,
    config: NormalFeatureConfig,
    *,
    colour_space: str | None = None,
    alpha_handling: str | None = None,
) -> PreparedNormalFeatureInput:
    source = normalise_input(image, colour_space=colour_space, alpha_handling=alpha_handling)
    bgr = source.image_bgr
    height, width = bgr.shape[:2]
    scale = min(config.input_width / width, config.input_height / height)
    resized_width = max(1, min(config.input_width, int(round(width * scale))))
    resized_height = max(1, min(config.input_height, int(round(height * scale))))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(bgr, (resized_width, resized_height), interpolation=interpolation)
    pad_width = config.input_width - resized_width
    pad_height = config.input_height - resized_height
    left = pad_width // 2
    right = pad_width - left
    top = pad_height // 2
    bottom = pad_height - top
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=tuple(reversed(config.padding_value_rgb)),
    )
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / np.float32(255.0)
    mean = np.asarray(config.input_normalisation_mean, dtype=np.float32)
    std = np.asarray(config.input_normalisation_std, dtype=np.float32)
    tensor = np.transpose((rgb - mean) / std, (2, 0, 1))
    geometry = LetterboxGeometry(height, width, resized_height, resized_width, top, bottom, left, right)
    contract = {"configuration": preprocessing_contract(config), "geometry": geometry.to_dict()}
    return PreparedNormalFeatureInput(source, tensor, geometry, sha256_json(contract))


def restore_anomaly_map(anomaly_map: np.ndarray, geometry: LetterboxGeometry) -> np.ndarray:
    array = np.asarray(anomaly_map, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Anomaly map must be a finite two-dimensional array")
    y2 = array.shape[0] - geometry.pad_bottom
    x2 = array.shape[1] - geometry.pad_right
    cropped = array[geometry.pad_top:y2, geometry.pad_left:x2]
    if cropped.shape != (geometry.resized_height, geometry.resized_width):
        raise ValueError("Anomaly map does not match recorded letterbox geometry")
    restored = cv2.resize(
        cropped, (geometry.original_width, geometry.original_height), interpolation=cv2.INTER_LINEAR,
    )
    return np.ascontiguousarray(restored, dtype=np.float32)
