"""Write-free professor-demonstration facade around public detector APIs.

This module validates in-memory inputs, invokes one existing detector, and
adapts typed results for faithful visualisation and explicit export. It does
not contain detector mathematics, evaluation logic, or persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO, StringIO
import csv
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import warnings
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image, ImageFile, PngImagePlugin

from .api import StructuralAnomalyDetector
from .configuration import DetectorConfig
from .normal_feature.configuration import (
    IMPLEMENTATION_ID as PATCHCORE_IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION as PATCHCORE_IMPLEMENTATION_VERSION,
    WEIGHT_SHA256,
)
from .hybrid.artifact import (
    HYBRID_IMPLEMENTATION_ID,
    HYBRID_IMPLEMENTATION_VERSION,
)
from .types import thaw_value


CLASSICAL_METHOD = "structvision-classical-baseline-v1-frozen"
PATCHCORE_METHOD = PATCHCORE_IMPLEMENTATION_ID
HYBRID_METHOD = HYBRID_IMPLEMENTATION_ID
DEFAULT_METHOD = CLASSICAL_METHOD

EXPECTED_ENVIRONMENT_LOCK_HASH = (
    "be3a00936219aedbcc397f0b3e8c0af6d901489a06550f3b148c72e22cea87b8"
)
EXPECTED_PATCHCORE_MODEL_ID = (
    "4542d063a64eb22d795f7a7faabb3cad592f69bd1fe753abdda0e5428f4961e7"
)
EXPECTED_PATCHCORE_CALIBRATION_ID = (
    "a5a434281d7e16ffb5c0a9af65f5b27d100cd447f1d024b7cbc5199805a21a6f"
)
EXPECTED_HYBRID_MODEL_ID = (
    "ef275b0a853231a239eebcccab6c920667616695296450d2d44453d922c341e7"
)
EXPECTED_HYBRID_FUSION_ID = (
    "a21b5880c5d8f16d3869227455279ddbf18815d92ae7862e262cc2560de3d8d1"
)

MAX_ENCODED_BYTES = 32 * 1024 * 1024
MAX_DECODED_PIXELS = 40_000_000
VALIDATED_RANGE_WARNING_PIXELS = 12_000_000
VALIDATED_RANGE_WARNING_EDGE = 4096
ALLOWED_FORMATS = {"PNG", "JPEG", "TIFF"}
ALLOWED_SUFFIXES = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}
ALPHA_HANDLING_OPTIONS = ("drop", "composite_black", "composite_white")

SCORE_SEMANTICS = {
    CLASSICAL_METHOD: (
        "Classical proposal/evidence/priority scores and mask reliability are "
        "heuristics, not probabilities or calibrated confidence."
    ),
    PATCHCORE_METHOD: "PatchCore scores are raw nearest-normal distances, not probabilities.",
    HYBRID_METHOD: (
        "The hybrid score is a fixed explainable linear rank score, not a probability "
        "or calibrated confidence."
    ),
}

EVIDENCE_ROWS = (
    {
        "method": CLASSICAL_METHOD,
        "status": "stable frozen baseline",
        "micro_sensitivity_iou_0_25": 0.770833,
        "image_sensitivity": 0.894737,
        "proposal_precision": 0.168950,
        "clean_false_proposals_per_image": 4.411765,
        "assigned_pair_mean_iou": 0.621954,
        "interpretation": "Strongest current sensitivity evidence; high clean proposal burden.",
    },
    {
        "method": PATCHCORE_METHOD,
        "status": "protected development baseline",
        "micro_sensitivity_iou_0_25": 0.687500,
        "image_sensitivity": 0.868421,
        "proposal_precision": 0.673469,
        "clean_false_proposals_per_image": 0.176471,
        "assigned_pair_mean_iou": 0.542742,
        "interpretation": "Lower clean burden; poor thin-crack and pitting behaviour.",
    },
    {
        "method": HYBRID_METHOD,
        "status": "rejected development candidate",
        "micro_sensitivity_iou_0_25": 0.750000,
        "image_sensitivity": 0.868421,
        "proposal_precision": 0.720000,
        "clean_false_proposals_per_image": 0.323529,
        "assigned_pair_mean_iou": 0.631250,
        "interpretation": (
            "Lower burden and better precision/localisation, but failed the predeclared "
            "overall and image-level sensitivity-preservation rules."
        ),
    },
)


class DemonstrationInputError(ValueError):
    """The demonstration input cannot be processed safely."""


class LearnedEnvironmentUnavailableError(RuntimeError):
    """The selected learned method cannot run in the current exact environment."""


class DemonstrationArtifactError(RuntimeError):
    """A required immutable learned artifact is missing or changed."""


@dataclass(frozen=True)
class MethodStatus:
    method_id: str
    version: str
    status: str
    operational_role: str
    requirements: tuple[str, ...]
    evidence_limit: str
    recommended_default: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "version": self.version,
            "status": self.status,
            "operational_role": self.operational_role,
            "requirements": list(self.requirements),
            "evidence_limit": self.evidence_limit,
            "recommended_default": self.recommended_default,
        }


METHOD_STATUSES = (
    MethodStatus(
        CLASSICAL_METHOD,
        "1.0.0",
        "stable frozen baseline",
        "Current recommended demonstration method; local and write-free.",
        ("Base StructVision installation",),
        "Synthetic development evidence only; not real-world validation.",
        True,
    ),
    MethodStatus(
        PATCHCORE_METHOD,
        PATCHCORE_IMPLEMENTATION_VERSION,
        "protected development baseline",
        "Optional research comparison; not the recommended operational default.",
        (
            "Exact Python 3.12 learned environment",
            "Verified official backbone weight",
            "Immutable PatchCore model and calibration artifacts",
        ),
        "Development-only, non-confirmatory evidence.",
        False,
    ),
    MethodStatus(
        HYBRID_METHOD,
        HYBRID_IMPLEMENTATION_VERSION,
        "rejected development candidate",
        "Optional research comparison only; rejected under the predeclared protocol.",
        (
            "Exact Python 3.12 learned environment",
            "Verified official backbone weight",
            "Immutable hybrid normal-memory and fusion artifacts",
        ),
        "Development holdout, non-confirmatory; failed sensitivity preservation.",
        False,
    ),
)


@dataclass(frozen=True)
class LearnedRuntimePaths:
    """Caller-selected local artifacts; paths never enter exported provenance."""

    environment_lock: Path | None = None
    weight: Path | None = None
    patchcore_model: Path | None = None
    patchcore_calibration: Path | None = None
    hybrid_model: Path | None = None
    hybrid_fusion: Path | None = None

    @classmethod
    def from_environment(cls) -> "LearnedRuntimePaths":
        def selected(name: str) -> Path | None:
            value = os.environ.get(name)
            return Path(value).expanduser() if value else None

        return cls(
            environment_lock=selected("STRUCTVISION_ENVIRONMENT_LOCK"),
            weight=selected("STRUCTVISION_PATCHCORE_WEIGHT"),
            patchcore_model=selected("STRUCTVISION_PATCHCORE_MODEL_ARTIFACT"),
            patchcore_calibration=selected("STRUCTVISION_PATCHCORE_CALIBRATION_ARTIFACT"),
            hybrid_model=selected("STRUCTVISION_HYBRID_MODEL_ARTIFACT"),
            hybrid_fusion=selected("STRUCTVISION_HYBRID_FUSION_ARTIFACT"),
        )


@dataclass(frozen=True)
class MethodAvailability:
    method_id: str
    available: bool
    missing_requirements: tuple[str, ...]
    environment_issues: tuple[str, ...] = ()
    artifact_issues: tuple[str, ...] = ()

    @property
    def message(self) -> str:
        if self.available:
            return "Available in the current local environment."
        return "Unavailable: " + "; ".join(self.missing_requirements)


@dataclass(frozen=True)
class DecodedDemonstrationImage:
    """One validated in-memory image with no filesystem provenance."""

    image_bgr: np.ndarray
    encoded_sha256: str
    source_format: str
    source_mode: str
    colour_handling: str
    width: int
    height: int
    warnings: tuple[str, ...]
    fixture_label: str | None = None

    def __post_init__(self) -> None:
        array = np.ascontiguousarray(self.image_bgr, dtype=np.uint8).copy()
        if array.ndim != 3 or array.shape[2] != 3:
            raise DemonstrationInputError("Decoded image must be an H×W×3 uint8 BGR array")
        if (array.shape[1], array.shape[0]) != (self.width, self.height):
            raise DemonstrationInputError("Decoded image dimensions do not match metadata")
        array.setflags(write=False)
        object.__setattr__(self, "image_bgr", array)

    @property
    def image_id(self) -> str:
        prefix = "fixture" if self.fixture_label else "upload"
        return f"{prefix}-{self.encoded_sha256[:16]}"


@dataclass(frozen=True)
class DemonstrationAnalysis:
    """Generic presentation wrapper retaining the unchanged typed core result."""

    input_image: DecodedDemonstrationImage
    method: MethodStatus
    result: object
    analysed_image_bgr: np.ndarray
    coordinate_mapping: tuple[tuple[str, object], ...]
    created_timestamp_utc: str
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        image = np.ascontiguousarray(self.analysed_image_bgr, dtype=np.uint8).copy()
        shape = tuple(int(item) for item in getattr(self.result, "image_shape"))
        if image.shape != shape:
            raise ValueError("Presentation image and detector result coordinates differ")
        image.setflags(write=False)
        object.__setattr__(self, "analysed_image_bgr", image)

    @property
    def method_id(self) -> str:
        return self.method.method_id

    @property
    def configuration_hash(self) -> str:
        if self.method_id == HYBRID_METHOD:
            return str(getattr(self.result, "classical_configuration_hash"))
        return str(getattr(self.result, "configuration_hash"))

    @property
    def input_hash(self) -> str:
        return str(getattr(self.result, "input_hash"))

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return tuple(int(item) for item in getattr(self.result, "image_shape"))

    @property
    def score_semantics(self) -> str:
        return SCORE_SEMANTICS[self.method_id]


def method_status(method_id: str) -> MethodStatus:
    try:
        return next(item for item in METHOD_STATUSES if item.method_id == method_id)
    except StopIteration as error:
        raise ValueError(f"Unknown method identity: {method_id}") from error


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_package_issues() -> tuple[str, ...]:
    if platform.python_version_tuple()[:2] != ("3", "12"):
        return ("exact Python 3.12 learned environment is not active",)
    required = {
        "anomalib": ("anomalib", "2.5.1"),
        "torch": ("torch", "2.9.1"),
        "torchvision": ("torchvision", "0.24.1"),
        "timm": ("timm", "1.0.28"),
        "safetensors": ("safetensors", "0.8.0"),
        "scikit-learn": ("sklearn", "1.7.2"),
    }
    issues = []
    for package, (module, version) in required.items():
        if importlib.util.find_spec(module) is None:
            issues.append(f"{package}=={version} is not installed")
            continue
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            issues.append(f"{package}=={version} is not installed")
        else:
            if actual != version:
                issues.append(f"{package}=={version} required; found {actual}")
    return tuple(issues)


def _path_issue(
    path: Path | None,
    label: str,
    *,
    expected_file_hash: str | None = None,
    expected_stem: str | None = None,
    paired_npz: bool = False,
) -> tuple[str, ...]:
    if path is None:
        return (f"{label} path is not configured",)
    if not path.is_file():
        return (f"{label} is missing",)
    issues = []
    if expected_file_hash is not None and _sha256_file(path) != expected_file_hash:
        issues.append(f"{label} SHA-256 does not match the protected identity")
    if expected_stem is not None and path.stem != expected_stem:
        issues.append(f"{label} filename does not match artifact identity {expected_stem}")
    if paired_npz and not path.with_suffix(".npz").is_file():
        issues.append(f"{label} memory-bank .npz is missing")
    return tuple(issues)


def method_availability(
    method_id: str,
    runtime: LearnedRuntimePaths | None = None,
) -> MethodAvailability:
    method_status(method_id)
    if method_id == CLASSICAL_METHOD:
        return MethodAvailability(method_id, True, (), (), ())
    selected = runtime or LearnedRuntimePaths.from_environment()
    environment_issues = list(_required_package_issues())
    artifact_issues = list(_path_issue(
        selected.environment_lock,
        "environment lock",
        expected_file_hash=EXPECTED_ENVIRONMENT_LOCK_HASH,
    ))
    artifact_issues.extend(_path_issue(
        selected.weight,
        "official backbone weight",
        expected_file_hash=WEIGHT_SHA256,
    ))
    if method_id == PATCHCORE_METHOD:
        artifact_issues.extend(_path_issue(
            selected.patchcore_model,
            "PatchCore model artifact",
            expected_stem=EXPECTED_PATCHCORE_MODEL_ID,
            paired_npz=True,
        ))
        artifact_issues.extend(_path_issue(
            selected.patchcore_calibration,
            "PatchCore calibration artifact",
            expected_stem=EXPECTED_PATCHCORE_CALIBRATION_ID,
        ))
    else:
        artifact_issues.extend(_path_issue(
            selected.hybrid_model,
            "hybrid normal-memory artifact",
            expected_stem=EXPECTED_HYBRID_MODEL_ID,
            paired_npz=True,
        ))
        artifact_issues.extend(_path_issue(
            selected.hybrid_fusion,
            "hybrid fusion artifact",
            expected_stem=EXPECTED_HYBRID_FUSION_ID,
        ))
    environment_unique = tuple(dict.fromkeys(environment_issues))
    artifact_unique = tuple(dict.fromkeys(artifact_issues))
    combined = environment_unique + artifact_unique
    return MethodAvailability(
        method_id,
        not combined,
        combined,
        environment_unique,
        artifact_unique,
    )


def _alpha_composite(rgba: np.ndarray, handling: str) -> np.ndarray:
    if handling not in ALPHA_HANDLING_OPTIONS:
        raise DemonstrationInputError(
            "Alpha-bearing input requires --alpha-handling: drop, composite_black, or composite_white"
        )
    rgb = rgba[..., :3]
    if handling == "drop":
        return rgb.copy()
    alpha = rgba[..., 3:4].astype(np.uint16)
    background = 0 if handling == "composite_black" else 255
    return ((rgb.astype(np.uint16) * alpha + background * (255 - alpha) + 127) // 255).astype(np.uint8)


def decode_image_bytes(
    encoded: bytes,
    *,
    filename: str,
    alpha_handling: str | None = None,
    max_encoded_bytes: int = MAX_ENCODED_BYTES,
    max_decoded_pixels: int = MAX_DECODED_PIXELS,
) -> DecodedDemonstrationImage:
    """Validate and decode one PNG/JPEG/TIFF entirely in memory."""
    if not isinstance(encoded, bytes) or not encoded:
        raise DemonstrationInputError("Image upload is empty")
    if len(encoded) > max_encoded_bytes:
        raise DemonstrationInputError(
            f"Encoded image exceeds the {max_encoded_bytes}-byte demonstration limit"
        )
    suffix = Path(filename).suffix.lower()
    expected_format = ALLOWED_SUFFIXES.get(suffix)
    if expected_format is None:
        raise DemonstrationInputError("Supported file types are PNG, JPEG, and TIFF")
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(BytesIO(encoded))
            actual_format = str(image.format or "").upper()
            if actual_format not in ALLOWED_FORMATS or actual_format != expected_format:
                raise DemonstrationInputError(
                    f"Encoded format {actual_format or 'unknown'} does not match {suffix}"
                )
            width, height = image.size
            if width <= 0 or height <= 0:
                raise DemonstrationInputError("Image dimensions must be positive")
            if width * height > max_decoded_pixels:
                raise DemonstrationInputError(
                    f"Decoded image exceeds the {max_decoded_pixels}-pixel demonstration limit"
                )
            source_mode = image.mode
            frames = int(getattr(image, "n_frames", 1))
            image.seek(0)
            has_alpha = source_mode in {"RGBA", "LA"} or (
                source_mode == "P" and "transparency" in image.info
            )
            if has_alpha:
                if alpha_handling is None:
                    raise DemonstrationInputError(
                        "Alpha-bearing input requires explicit alpha handling"
                    )
                rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
                rgb = _alpha_composite(rgba, alpha_handling)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                colour_handling = f"{source_mode} → RGB ({alpha_handling}) → BGR"
            elif source_mode in {"1", "L"}:
                gray = np.asarray(image.convert("L"), dtype=np.uint8)
                bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                colour_handling = f"{source_mode} grayscale → uint8 BGR"
            elif source_mode in {"I", "I;16", "I;16B", "I;16L", "F"}:
                raise DemonstrationInputError(
                    f"{source_mode} high-bit-depth input is not supported by the current uint8 detector contract"
                )
            else:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                colour_handling = f"{source_mode} → RGB → BGR"
            image.load()
    except DemonstrationInputError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise DemonstrationInputError("Image was rejected as a possible decompression bomb") from error
    except Exception as error:
        raise DemonstrationInputError("Image is malformed or could not be decoded") from error
    input_warnings = []
    if frames > 1:
        input_warnings.append("Only the first TIFF frame is analysed.")
    if width * height > VALIDATED_RANGE_WARNING_PIXELS or max(width, height) > VALIDATED_RANGE_WARNING_EDGE:
        input_warnings.append(
            "Resolution exceeds the current demonstration operating-range warning threshold; "
            "memory use and small-region behaviour require separate validation."
        )
    return DecodedDemonstrationImage(
        bgr,
        sha256(encoded).hexdigest(),
        actual_format,
        source_mode,
        colour_handling,
        width,
        height,
        tuple(input_warnings),
    )


FIXTURE_LABELS = (
    "clean textured surface",
    "thin structural indication",
    "clustered pitting-like indication",
    "illumination nuisance",
    "reflective nuisance",
)


def demonstration_fixture(label: str) -> DecodedDemonstrationImage:
    """Return a deterministic synthetic UI fixture excluded from research cohorts."""
    if label not in FIXTURE_LABELS:
        raise ValueError(f"Unknown demonstration fixture: {label}")
    height, width = 384, 640
    yy, xx = np.indices((height, width))
    base = 118 + 13 * np.sin(xx / 23.0) + 9 * np.cos(yy / 17.0)
    image = np.clip(base, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if label == "thin structural indication":
        cv2.line(bgr, (90, 300), (545, 92), (35, 35, 35), 3, cv2.LINE_8)
    elif label == "clustered pitting-like indication":
        for index in range(11):
            x = 250 + (index * 37) % 170
            y = 135 + (index * 29) % 120
            cv2.circle(bgr, (x, y), 5 + index % 4, (45, 45, 45), -1, cv2.LINE_8)
    elif label == "illumination nuisance":
        gradient = np.linspace(0, 75, width, dtype=np.uint8)[None, :, None]
        bgr = np.clip(bgr.astype(np.uint16) + gradient.astype(np.uint16), 0, 255).astype(np.uint8)
    elif label == "reflective nuisance":
        cv2.ellipse(bgr, (355, 185), (120, 34), -18, 0, 360, (242, 242, 242), -1, cv2.LINE_8)
    digest_payload = f"{label}\0{height}\0{width}".encode("utf-8") + bgr.tobytes(order="C")
    return DecodedDemonstrationImage(
        bgr,
        sha256(digest_payload).hexdigest(),
        "GENERATED_FIXTURE",
        "BGR",
        "deterministic synthetic BGR fixture",
        width,
        height,
        (
            "Synthetic demonstration fixture only; excluded from every research evaluation cohort.",
            "This fixture is not real inspection evidence.",
        ),
        label,
    )


def _analysed_display_image(
    decoded: DecodedDemonstrationImage,
    analysed_shape: tuple[int, int, int],
) -> tuple[np.ndarray, tuple[tuple[str, object], ...]]:
    analysed_height, analysed_width = analysed_shape[:2]
    if decoded.image_bgr.shape[:2] == (analysed_height, analysed_width):
        display = decoded.image_bgr.copy()
        policy = "identity"
    else:
        interpolation = cv2.INTER_AREA if analysed_width < decoded.width else cv2.INTER_LINEAR
        display = cv2.resize(
            decoded.image_bgr,
            (analysed_width, analysed_height),
            interpolation=interpolation,
        )
        policy = "display-only resize to returned analysed-image coordinates"
    mapping = (
        ("original_width", decoded.width),
        ("original_height", decoded.height),
        ("analysed_width", analysed_width),
        ("analysed_height", analysed_height),
        ("x_scale_original_to_analysed", analysed_width / decoded.width),
        ("y_scale_original_to_analysed", analysed_height / decoded.height),
        ("rendering_policy", policy),
        ("result_identity_preserved_at_original_returned_resolution", True),
    )
    return display, mapping


def _run_patchcore(
    decoded: DecodedDemonstrationImage,
    runtime: LearnedRuntimePaths,
) -> object:
    from .normal_feature import (
        NormalFeatureAnomalyDetector,
        NormalFeatureConfig,
        load_calibration_artifact,
        load_model_artifact,
    )

    if not all((
        runtime.environment_lock,
        runtime.weight,
        runtime.patchcore_model,
        runtime.patchcore_calibration,
    )):
        raise DemonstrationArtifactError("Complete PatchCore runtime paths are required")
    try:
        model = load_model_artifact(runtime.patchcore_model)
        calibration = load_calibration_artifact(runtime.patchcore_calibration)
    except Exception as error:
        raise DemonstrationArtifactError(
            "PatchCore model or calibration artifact failed immutable verification"
        ) from error
    detector = NormalFeatureAnomalyDetector(
        NormalFeatureConfig(),
        weight_file=runtime.weight,
        environment_lock_hash=_sha256_file(runtime.environment_lock),
    )
    return detector.analyse(
        decoded.image_bgr,
        image_id=decoded.image_id,
        colour_space="BGR",
        model_artifact=model,
        calibration_artifact=calibration,
        operating_point_id="fp-budget-0.50",
    )


def _run_hybrid(
    decoded: DecodedDemonstrationImage,
    runtime: LearnedRuntimePaths,
) -> object:
    from .hybrid import ProposalGuidedHybridDetector, load_hybrid_fusion_artifact
    from .normal_feature import (
        NormalFeatureAnomalyDetector,
        NormalFeatureConfig,
        load_model_artifact,
    )

    if not all((
        runtime.environment_lock,
        runtime.weight,
        runtime.hybrid_model,
        runtime.hybrid_fusion,
    )):
        raise DemonstrationArtifactError("Complete hybrid runtime paths are required")
    try:
        model = load_model_artifact(runtime.hybrid_model)
        fusion = load_hybrid_fusion_artifact(runtime.hybrid_fusion)
    except Exception as error:
        raise DemonstrationArtifactError(
            "Hybrid model or fusion artifact failed immutable verification"
        ) from error
    normal = NormalFeatureAnomalyDetector(
        NormalFeatureConfig(),
        weight_file=runtime.weight,
        environment_lock_hash=_sha256_file(runtime.environment_lock),
    )
    detector = ProposalGuidedHybridDetector(
        classical_detector=StructuralAnomalyDetector(DetectorConfig()),
        normal_feature_detector=normal,
        normal_feature_model_artifact=model,
        fusion_artifact=fusion,
    )
    return detector.analyse(
        decoded.image_bgr,
        image_id=decoded.image_id,
        colour_space="BGR",
    )


def analyse_demonstration_image(
    decoded: DecodedDemonstrationImage,
    *,
    method_id: str = DEFAULT_METHOD,
    runtime: LearnedRuntimePaths | None = None,
) -> DemonstrationAnalysis:
    """Run one existing method without a sink, database, registry, or remote call."""
    status = method_status(method_id)
    availability = method_availability(method_id, runtime)
    if not availability.available:
        if availability.environment_issues:
            raise LearnedEnvironmentUnavailableError(availability.message)
        raise DemonstrationArtifactError(availability.message)
    selected = runtime or LearnedRuntimePaths.from_environment()
    if method_id == CLASSICAL_METHOD:
        result = StructuralAnomalyDetector(DetectorConfig()).analyse(
            decoded.image_bgr,
            image_id=decoded.image_id,
            colour_space="BGR",
            metadata={
                "source": "professor_demonstration_in_memory",
                "source_format": decoded.source_format,
                "source_mode": decoded.source_mode,
                "retained": False,
            },
        )
    elif method_id == PATCHCORE_METHOD:
        result = _run_patchcore(decoded, selected)
    elif method_id == HYBRID_METHOD:
        result = _run_hybrid(decoded, selected)
    else:  # protected by method_status, retained as a fail-closed guard
        raise ValueError(f"Unknown method identity: {method_id}")
    analysed_shape = tuple(int(item) for item in getattr(result, "image_shape"))
    display, mapping = _analysed_display_image(decoded, analysed_shape)
    combined_warnings = tuple(dict.fromkeys(
        decoded.warnings
        + tuple(getattr(result, "warnings", ()))
        + (status.evidence_limit, SCORE_SEMANTICS[method_id])
    ))
    return DemonstrationAnalysis(
        decoded,
        status,
        result,
        display,
        mapping,
        datetime.now(timezone.utc).isoformat(),
        combined_warnings,
    )


def _classical_rows(analysis: DemonstrationAnalysis) -> list[dict[str, object]]:
    rows = []
    for proposal in getattr(analysis.result, "proposals"):
        context = {name: thaw_value(value) for name, value in proposal.context_diagnostics}
        rows.append({
            "rank": proposal.rank,
            "proposal_id": proposal.proposal_id,
            "x_min": proposal.bbox[0],
            "y_min": proposal.bbox[1],
            "x_max": proposal.bbox[2],
            "y_max": proposal.bbox[3],
            "area": proposal.area,
            "proposal_score": proposal.proposal_score,
            "classical_evidence_score": proposal.evidence_score,
            "heuristic_mask_reliability": proposal.heuristic_reliability,
            "priority_score": proposal.priority_score,
            "normality_distance": None,
            "proposal_interior_mean": None,
            "proposal_high_quantile": None,
            "local_context_contrast": None,
            "spatial_agreement": None,
            "hybrid_score": None,
            "selected": True,
            "selection_reason": "returned by the frozen classical ranking and top-K policy",
            "context_diagnostics": context,
            "mask": proposal.final_mask,
        })
    return rows


def _patchcore_rows(analysis: DemonstrationAnalysis) -> list[dict[str, object]]:
    rows = []
    for proposal in getattr(analysis.result, "proposals"):
        rows.append({
            "rank": proposal.rank,
            "proposal_id": proposal.proposal_id,
            "x_min": proposal.bbox[0],
            "y_min": proposal.bbox[1],
            "x_max": proposal.bbox[2],
            "y_max": proposal.bbox[3],
            "area": proposal.area,
            "proposal_score": proposal.component_anomaly_score,
            "classical_evidence_score": None,
            "heuristic_mask_reliability": None,
            "priority_score": None,
            "normality_distance": proposal.component_anomaly_score,
            "proposal_interior_mean": None,
            "proposal_high_quantile": None,
            "local_context_contrast": None,
            "spatial_agreement": None,
            "hybrid_score": None,
            "selected": True,
            "selection_reason": (
                f"connected component above frozen operating point {proposal.operating_point_id}"
            ),
            "context_diagnostics": {
                "threshold": proposal.threshold,
                "operating_point_id": proposal.operating_point_id,
                "extraction_policy_hash": proposal.extraction_policy_hash,
            },
            "mask": proposal.mask,
        })
    return rows


def _hybrid_rows(analysis: DemonstrationAnalysis) -> list[dict[str, object]]:
    rows = []
    for diagnostic in getattr(analysis.result, "complete_candidate_diagnostics"):
        patchcore = dict(diagnostic.patchcore_candidate_evidence)
        rows.append({
            "rank": diagnostic.final_rank,
            "proposal_id": diagnostic.classical_proposal_id,
            "x_min": diagnostic.bbox[0],
            "y_min": diagnostic.bbox[1],
            "x_max": diagnostic.bbox[2],
            "y_max": diagnostic.bbox[3],
            "area": int(np.count_nonzero(diagnostic.mask)),
            "proposal_score": diagnostic.hybrid_score,
            "classical_evidence_score": diagnostic.classical_evidence_score,
            "heuristic_mask_reliability": diagnostic.heuristic_mask_reliability,
            "priority_score": diagnostic.classical_priority_score,
            "normality_distance": diagnostic.normalised_patchcore_evidence,
            "proposal_interior_mean": patchcore.get("patchcore_inside_mean"),
            "proposal_high_quantile": patchcore.get("patchcore_inside_q95"),
            "local_context_contrast": patchcore.get("patchcore_context_contrast"),
            "spatial_agreement": patchcore.get("patchcore_local_spatial_agreement"),
            "hybrid_score": diagnostic.hybrid_score,
            "selected": diagnostic.selected,
            "selection_reason": "; ".join(diagnostic.explanation),
            "context_diagnostics": {
                "classical_rank": diagnostic.classical_rank,
                "operating_threshold": diagnostic.operating_threshold,
                "generic_preservation_applied": diagnostic.generic_preservation_applied,
                "normalised_features": dict(diagnostic.normalised_features),
            },
            "mask": diagnostic.mask,
        })
    return sorted(
        rows,
        key=lambda item: (
            not bool(item["selected"]),
            int(item["rank"]) if item["rank"] is not None else 10**9,
            int(item["context_diagnostics"]["classical_rank"]),
            str(item["proposal_id"]),
        ),
    )


def candidate_rows(
    analysis: DemonstrationAnalysis,
    *,
    include_masks: bool = False,
) -> tuple[dict[str, object], ...]:
    """Return deterministic candidate records; missing metrics remain ``None``."""
    if analysis.method_id == CLASSICAL_METHOD:
        rows = _classical_rows(analysis)
    elif analysis.method_id == PATCHCORE_METHOD:
        rows = _patchcore_rows(analysis)
    else:
        rows = _hybrid_rows(analysis)
    if include_masks:
        return tuple(rows)
    return tuple({key: value for key, value in row.items() if key != "mask"} for row in rows)


def _candidate_rows_with_masks(analysis: DemonstrationAnalysis) -> tuple[dict[str, object], ...]:
    return candidate_rows(analysis, include_masks=True)


def _render_candidates(
    analysis: DemonstrationAnalysis,
    candidate_id: str | None,
) -> tuple[dict[str, object], ...]:
    rows = _candidate_rows_with_masks(analysis)
    if candidate_id is not None:
        selected = tuple(item for item in rows if item["proposal_id"] == candidate_id)
        if not selected:
            raise KeyError(candidate_id)
        return selected
    return tuple(item for item in rows if bool(item["selected"]))


def render_overlay(
    analysis: DemonstrationAnalysis,
    *,
    candidate_id: str | None = None,
) -> np.ndarray:
    """Render direct masks and half-open boxes in deterministic rank order."""
    canvas = analysis.analysed_image_bgr.copy()
    rows = _render_candidates(analysis, candidate_id)
    palette = (
        (46, 204, 113),
        (52, 152, 219),
        (155, 89, 182),
        (0, 165, 255),
        (241, 196, 15),
        (26, 188, 156),
        (203, 192, 255),
        (128, 128, 255),
    )
    ordered = sorted(
        rows,
        key=lambda item: (
            int(item["rank"]) if item["rank"] is not None else 10**9,
            str(item["proposal_id"]),
        ),
        reverse=True,
    )
    for index, row in enumerate(ordered):
        mask = np.asarray(row["mask"]) > 0
        colour = palette[index % len(palette)]
        colour_plane = np.empty_like(canvas)
        colour_plane[:] = colour
        canvas[mask] = cv2.addWeighted(canvas[mask], 0.55, colour_plane[mask], 0.45, 0.0)
    for index, row in enumerate(reversed(ordered)):
        colour = palette[(len(ordered) - index - 1) % len(palette)]
        x1, y1, x2, y2 = (int(row[key]) for key in ("x_min", "y_min", "x_max", "y_max"))
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), colour, 2, cv2.LINE_8)
        rank = row["rank"] if row["rank"] is not None else "rejected"
        label = f"{rank}: {row['proposal_id']}"
        cv2.putText(
            canvas,
            label,
            (x1, max(14, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            colour,
            1,
            cv2.LINE_AA,
        )
    return canvas


def candidate_mask(analysis: DemonstrationAnalysis, candidate_id: str) -> np.ndarray:
    rows = _render_candidates(analysis, candidate_id)
    return np.ascontiguousarray(rows[0]["mask"], dtype=np.uint8).copy()


def candidate_crop(analysis: DemonstrationAnalysis, candidate_id: str) -> np.ndarray:
    row = _render_candidates(analysis, candidate_id)[0]
    x1, y1, x2, y2 = (int(row[key]) for key in ("x_min", "y_min", "x_max", "y_max"))
    return analysis.analysed_image_bgr[y1:y2, x1:x2].copy()


def render_anomaly_overlay(analysis: DemonstrationAnalysis) -> np.ndarray | None:
    """Visualise an exposed anomaly map without changing proposal boundaries."""
    raw = getattr(analysis.result, "anomaly_heatmap", None)
    if raw is not None:
        heatmap = np.asarray(raw, dtype=np.uint8)
        if heatmap.ndim == 2:
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)
    else:
        raw = getattr(analysis.result, "anomaly_map", None)
        if raw is None:
            return None
        values = np.asarray(raw, dtype=np.float32)
        minimum, maximum = float(np.min(values)), float(np.max(values))
        scaled = np.zeros(values.shape, dtype=np.uint8)
        if maximum > minimum:
            scaled = np.clip((values - minimum) * (255.0 / (maximum - minimum)), 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    if heatmap.shape[:2] != analysis.analysed_image_bgr.shape[:2]:
        raise ValueError("Exposed anomaly evidence and returned coordinate system are misaligned")
    return cv2.addWeighted(analysis.analysed_image_bgr, 0.52, heatmap, 0.48, 0.0)


def pipeline_stages(method_id: str) -> tuple[dict[str, str], ...]:
    method_status(method_id)
    if method_id == CLASSICAL_METHOD:
        return (
            {"stage": "Input normalisation", "evidence": "Exposed: explicit BGR normalisation and input hash."},
            {"stage": "Preprocessing", "evidence": "Exposed: immutable preprocessing configuration and analysed shape."},
            {"stage": "Feature-evidence families", "evidence": "Exposed: anomaly heatmap and candidate contribution diagnostics."},
            {"stage": "Candidate-mask generation", "evidence": "Not exposed by the current frozen API."},
            {"stage": "Contextual scoring", "evidence": "Exposed: component scores and contextual diagnostics per returned candidate."},
            {"stage": "Ranking", "evidence": "Exposed: ordered IDs, ranks, and priority scores."},
            {"stage": "Mask refinement", "evidence": "Only final and raw masks are exposed; internal refinement stages are not."},
            {"stage": "Final proposals", "evidence": "Exposed: direct binary masks and half-open mask-derived boxes."},
        )
    if method_id == PATCHCORE_METHOD:
        return (
            {"stage": "Normal-feature memory", "evidence": "Immutable normal-only memory artifact; no refitting in demonstration."},
            {"stage": "Patch embedding", "evidence": "Official frozen Wide-ResNet-50-2 layer2/layer3 adapter."},
            {"stage": "Nearest-normal distance", "evidence": "Raw exact distance evidence; not a probability."},
            {"stage": "Dense anomaly map", "evidence": "Full-resolution inverse-projected map with content hash."},
            {"stage": "Calibrated operating point", "evidence": "Immutable development calibration artifact at fp-budget-0.50."},
            {"stage": "Component extraction", "evidence": "Threshold, 8-connectivity, no morphology, half-open boxes."},
        )
    return (
        {"stage": "Classical candidate masks", "evidence": "Unchanged frozen masks and candidate ordering."},
        {"stage": "PatchCore candidate evidence", "evidence": "Interior mean/q95, support, context contrast, spatial agreement."},
        {"stage": "Fixed fusion", "evidence": "0.60 classical + 0.40 normality; immutable normalisation."},
        {"stage": "Frozen threshold", "evidence": "Primary threshold 0.4704560134385654."},
        {"stage": "Reranking and selection", "evidence": "Complete pre-threshold diagnostics retained."},
        {"stage": "Research decision", "evidence": "Rejected development candidate under the predeclared protocol."},
    )


def _artifact_identities(analysis: DemonstrationAnalysis) -> dict[str, str | None]:
    if analysis.method_id == CLASSICAL_METHOD:
        return {
            "model_artifact_hash": None,
            "calibration_artifact_hash": None,
            "fusion_artifact_hash": None,
        }
    if analysis.method_id == PATCHCORE_METHOD:
        return {
            "model_artifact_hash": str(getattr(analysis.result, "model_artifact_hash")),
            "calibration_artifact_hash": str(getattr(analysis.result, "calibration_artifact_hash")),
            "fusion_artifact_hash": None,
        }
    return {
        "model_artifact_hash": str(getattr(analysis.result, "normal_feature_model_artifact_hash")),
        "calibration_artifact_hash": None,
        "fusion_artifact_hash": str(getattr(analysis.result, "fusion_artifact_hash")),
    }


def _serialisable_candidate(row: Mapping[str, object], shape: tuple[int, int]) -> dict[str, object]:
    payload = dict(row)
    payload["bbox"] = [
        payload.pop("x_min"),
        payload.pop("y_min"),
        payload.pop("x_max"),
        payload.pop("y_max"),
    ]
    payload["bbox_convention"] = "half-open:x_min,y_min,x_max,y_max"
    payload["mask_dimensions"] = list(shape)
    if "context_diagnostics" in payload:
        payload["context_diagnostics"] = dict(payload["context_diagnostics"])
    return payload


def export_payload(analysis: DemonstrationAnalysis) -> dict[str, object]:
    height, width = analysis.image_shape[:2]
    result = analysis.result
    image_score = getattr(result, "image_anomaly_score", None)
    return {
        "schema_version": "structvision-professor-analysis-v1",
        "export_timestamp_utc": analysis.created_timestamp_utc,
        "method": analysis.method.to_dict(),
        "processing_status": "completed",
        "input": {
            "input_image_hash": analysis.input_image.encoded_sha256,
            "normalised_input_hash": analysis.input_hash,
            "source_format": analysis.input_image.source_format,
            "source_mode": analysis.input_image.source_mode,
            "colour_handling": analysis.input_image.colour_handling,
            "original_dimensions": [analysis.input_image.height, analysis.input_image.width],
            "persisted": False,
            "absolute_path_recorded": False,
        },
        "analysis": {
            "analysed_dimensions": [height, width],
            "normalised_colour_space": "BGR",
            "configuration_hash": analysis.configuration_hash,
            "implementation_identity": analysis.method_id,
            "implementation_version": analysis.method.version,
            "development_status": analysis.method.status,
            "image_anomaly_score": image_score,
            "score_semantics": analysis.score_semantics,
            "bounding_box_convention": "half-open:x_min,y_min,x_max,y_max",
            "coordinate_mapping": dict(analysis.coordinate_mapping),
            "artifact_identities": _artifact_identities(analysis),
        },
        "candidates": [
            _serialisable_candidate(row, (height, width))
            for row in candidate_rows(analysis)
        ],
        "warnings": list(analysis.warnings),
        "privacy": {
            "external_service_used": False,
            "telemetry_used": False,
            "dataset_registry_mutated": False,
            "experiment_store_written": False,
            "upload_retained": False,
        },
    }


def analysis_json_bytes(analysis: DemonstrationAnalysis) -> bytes:
    return (
        json.dumps(
            export_payload(analysis),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def proposal_csv_bytes(analysis: DemonstrationAnalysis) -> bytes:
    rows = candidate_rows(analysis)
    candidate_fields = (
        "rank",
        "proposal_id",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "area",
        "proposal_score",
        "classical_evidence_score",
        "heuristic_mask_reliability",
        "priority_score",
        "normality_distance",
        "proposal_interior_mean",
        "proposal_high_quantile",
        "local_context_contrast",
        "spatial_agreement",
        "hybrid_score",
        "selected",
        "selection_reason",
    )
    metadata = (
        "method_identity",
        "method_version",
        "development_status",
        "configuration_hash",
        "input_image_hash",
        "export_timestamp_utc",
        "bbox_convention",
        "mask_height",
        "mask_width",
        "score_semantics",
        "warnings",
    )
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=metadata + candidate_fields)
    writer.writeheader()
    for row in rows:
        payload = {
            "method_identity": analysis.method_id,
            "method_version": analysis.method.version,
            "development_status": analysis.method.status,
            "configuration_hash": analysis.configuration_hash,
            "input_image_hash": analysis.input_image.encoded_sha256,
            "export_timestamp_utc": analysis.created_timestamp_utc,
            "bbox_convention": "half-open:x_min,y_min,x_max,y_max",
            "mask_height": analysis.image_shape[0],
            "mask_width": analysis.image_shape[1],
            "score_semantics": analysis.score_semantics,
            "warnings": " | ".join(analysis.warnings),
        }
        for field in candidate_fields:
            value = row[field]
            payload[field] = "N/A" if value is None else value
        writer.writerow(payload)
    return buffer.getvalue().encode("utf-8")


def technical_summary_bytes(analysis: DemonstrationAnalysis) -> bytes:
    selected = sum(bool(row["selected"]) for row in candidate_rows(analysis))
    lines = (
        "StructVision-AI technical result summary",
        f"Method identity: {analysis.method_id}",
        f"Implementation version: {analysis.method.version}",
        f"Development status: {analysis.method.status}",
        f"Configuration hash: {analysis.configuration_hash}",
        f"Input-image hash: {analysis.input_image.encoded_sha256}",
        f"Export timestamp (UTC): {analysis.created_timestamp_utc}",
        f"Analysed dimensions: {analysis.image_shape[1]} × {analysis.image_shape[0]}",
        "Bounding-box convention: half-open (x_min, y_min, x_max, y_max)",
        f"Selected proposals: {selected}",
        f"Score semantics: {analysis.score_semantics}",
        "Warnings: " + " | ".join(analysis.warnings),
        "No image was sent to an external service or persisted by the analysis path.",
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _png_metadata(analysis: DemonstrationAnalysis, *, mask_dimensions: tuple[int, int]) -> PngImagePlugin.PngInfo:
    metadata = PngImagePlugin.PngInfo()
    values = {
        "method_identity": analysis.method_id,
        "implementation_version": analysis.method.version,
        "development_status": analysis.method.status,
        "configuration_hash": analysis.configuration_hash,
        "input_image_hash": analysis.input_image.encoded_sha256,
        "export_timestamp_utc": analysis.created_timestamp_utc,
        "bounding_box_convention": "half-open:x_min,y_min,x_max,y_max",
        "mask_dimensions": f"{mask_dimensions[0]}x{mask_dimensions[1]}",
        "score_semantics": analysis.score_semantics,
        "warnings": " | ".join(analysis.warnings),
    }
    for key, value in values.items():
        metadata.add_text(f"structvision.{key}", value)
    return metadata


def _encode_png(
    array: np.ndarray,
    *,
    metadata: PngImagePlugin.PngInfo,
    bgr: bool,
) -> bytes:
    if bgr:
        image = Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))
    else:
        image = Image.fromarray(array)
    buffer = BytesIO()
    image.save(buffer, format="PNG", pnginfo=metadata, optimize=False, compress_level=6)
    return buffer.getvalue()


def annotated_png_bytes(
    analysis: DemonstrationAnalysis,
    *,
    candidate_id: str | None = None,
) -> bytes:
    rendered = render_overlay(analysis, candidate_id=candidate_id)
    return _encode_png(
        rendered,
        metadata=_png_metadata(analysis, mask_dimensions=analysis.image_shape[:2]),
        bgr=True,
    )


def binary_mask_png_bytes(
    analysis: DemonstrationAnalysis,
    candidate_id: str,
) -> bytes:
    mask = candidate_mask(analysis, candidate_id)
    return _encode_png(
        mask,
        metadata=_png_metadata(analysis, mask_dimensions=mask.shape),
        bgr=False,
    )
