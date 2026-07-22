"""Explicit, deterministic image normalisation for the public API."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from os import PathLike
from pathlib import Path

import cv2
import numpy as np

from .errors import (
    AmbiguousColourSpaceError,
    CorruptImageError,
    InvalidInputTypeError,
    UnsupportedChannelLayoutError,
)


@dataclass(frozen=True)
class NormalisedInput:
    image_bgr: np.ndarray
    input_hash: str
    source_hash: str
    source_type: str


def _array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    prefix = json.dumps(
        {"shape": list(contiguous.shape), "dtype": str(contiguous.dtype), "order": "C"},
        sort_keys=True, separators=(",", ":"),
    ).encode("ascii") + b"\0"
    return hashlib.sha256(prefix + contiguous.tobytes(order="C")).hexdigest()


def _validate_array(array: np.ndarray) -> None:
    if array.dtype != np.uint8:
        raise InvalidInputTypeError("NumPy image arrays must use uint8 values")
    if array.size == 0 or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise UnsupportedChannelLayoutError("Image arrays must be non-empty")


def _alpha_composite(colour_bgr: np.ndarray, alpha: np.ndarray, handling: str) -> np.ndarray:
    if handling == "drop":
        return colour_bgr.copy()
    backgrounds = {"composite_black": 0, "composite_white": 255}
    if handling not in backgrounds:
        raise AmbiguousColourSpaceError(
            "RGBA/BGRA input requires alpha_handling='drop', 'composite_black', or 'composite_white'"
        )
    foreground = colour_bgr.astype(np.uint16)
    opacity = alpha.astype(np.uint16)[..., None]
    background = backgrounds[handling]
    return ((foreground * opacity + background * (255 - opacity) + 127) // 255).astype(np.uint8)


def _normalise_array(
    array: np.ndarray, colour_space: str | None, alpha_handling: str | None,
) -> tuple[np.ndarray, str]:
    _validate_array(array)
    declared = colour_space.upper() if isinstance(colour_space, str) else None
    source_hash = _array_hash(array)
    if array.ndim == 2 or (array.ndim == 3 and array.shape[2] == 1):
        if declared not in {None, "GRAY", "GREY", "GRAYSCALE"}:
            raise AmbiguousColourSpaceError("Single-channel input must be declared GRAY")
        gray = array if array.ndim == 2 else array[..., 0]
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), source_hash
    if array.ndim != 3:
        raise UnsupportedChannelLayoutError("Images must be HxW, HxWx1, HxWx3, or HxWx4")
    channels = array.shape[2]
    if channels == 3:
        if declared not in {"RGB", "BGR"}:
            raise AmbiguousColourSpaceError("Three-channel NumPy arrays require colour_space='RGB' or 'BGR'")
        return (cv2.cvtColor(array, cv2.COLOR_RGB2BGR) if declared == "RGB" else array.copy()), source_hash
    if channels == 4:
        if declared not in {"RGBA", "BGRA"}:
            raise AmbiguousColourSpaceError("Four-channel NumPy arrays require colour_space='RGBA' or 'BGRA'")
        if alpha_handling is None:
            raise AmbiguousColourSpaceError("Four-channel input requires explicit alpha_handling")
        colour = array[..., :3]
        bgr = cv2.cvtColor(colour, cv2.COLOR_RGB2BGR) if declared == "RGBA" else colour.copy()
        return _alpha_composite(bgr, array[..., 3], alpha_handling), source_hash
    raise UnsupportedChannelLayoutError(f"Unsupported channel count: {channels}")


def normalise_input(
    image: object, *, colour_space: str | None = None, alpha_handling: str | None = None,
) -> NormalisedInput:
    """Return contiguous uint8 BGR pixels without guessing array colour order."""
    if isinstance(image, (str, Path, PathLike)):
        path = Path(image)
        if not path.is_file():
            raise CorruptImageError(f"Image path does not exist or is not a file: {path}")
        encoded = path.read_bytes()
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise CorruptImageError(f"Could not decode image: {path.name}")
        declared = colour_space.upper() if isinstance(colour_space, str) else None
        if decoded.ndim == 2:
            if declared not in {None, "GRAY", "GREY", "GRAYSCALE"}:
                raise AmbiguousColourSpaceError("Decoded single-channel path input is GRAY")
            bgr = cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR)
        elif decoded.ndim == 3 and decoded.shape[2] == 3:
            if declared not in {None, "BGR"}:
                raise AmbiguousColourSpaceError("Filesystem images are decoded by OpenCV as BGR")
            bgr = decoded
        elif decoded.ndim == 3 and decoded.shape[2] == 4:
            if declared not in {None, "BGRA"}:
                raise AmbiguousColourSpaceError("Four-channel filesystem images are decoded as BGRA")
            if alpha_handling is None:
                raise AmbiguousColourSpaceError("Four-channel path input requires explicit alpha_handling")
            bgr = _alpha_composite(decoded[..., :3], decoded[..., 3], alpha_handling)
        else:
            raise UnsupportedChannelLayoutError(f"Unsupported decoded image shape: {decoded.shape}")
        _validate_array(bgr)
        contiguous = np.ascontiguousarray(bgr)
        return NormalisedInput(
            contiguous, _array_hash(contiguous), hashlib.sha256(encoded).hexdigest(), "filesystem_path"
        )
    if isinstance(image, np.ndarray):
        bgr, source_hash = _normalise_array(image, colour_space, alpha_handling)
        contiguous = np.ascontiguousarray(bgr)
        return NormalisedInput(contiguous, _array_hash(contiguous), source_hash, "numpy_array")
    raise InvalidInputTypeError(
        f"Unsupported image input type: {type(image).__name__}; expected a filesystem path or NumPy array"
    )


def content_hash(value: object) -> str:
    """Hash a file's encoded bytes or an array's declared in-memory representation."""
    if isinstance(value, (str, Path, PathLike)):
        path = Path(value)
        if not path.is_file():
            raise CorruptImageError(f"Content path does not exist or is not a file: {path}")
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if isinstance(value, np.ndarray):
        _validate_array(value)
        return _array_hash(value)
    raise InvalidInputTypeError(f"Cannot hash unsupported content type: {type(value).__name__}")
