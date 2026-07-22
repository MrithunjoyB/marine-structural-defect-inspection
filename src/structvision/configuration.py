"""Immutable, content-addressed configuration for the frozen classical adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from typing import Any, Mapping

from .errors import InvalidConfigurationError


IMPLEMENTATION_ID = "structvision-classical-baseline-v1-frozen"
IMPLEMENTATION_VERSION = "1.0.0"

# These values influence the protected implementation but are not arguments to
# its public functions. They remain immutable in v1; exposing them as tunable
# values would imply an algorithm change. Their digest is part of every config.
IMMUTABLE_IMPLEMENTATION_CONSTANTS = (
    ("preprocess.denoise", (8, 8, 7, 21)),
    ("preprocess.clahe", (2.2, (8, 8))),
    ("preprocess.sharpen_kernel", (0.0, -1.0, 0.0, -1.0, 5.2, -1.0, 0.0, -1.0, 0.0)),
    ("feature.gaussian_kernel", (5, 5)),
    ("feature.canny_multipliers", (0.55, 1.65, 10, 20)),
    ("feature.texture_window", (15, 15)),
    ("feature.colour_window", (21, 21)),
    ("feature.anomaly_weights", (0.18, 0.12, 0.16, 0.20, 0.12, 0.16, 0.06)),
    ("proposal.feature_percentile", 82),
    ("proposal.threshold_percentile", 65),
    ("proposal.fused_weights", (0.58, 32.0, 85.0)),
    ("proposal.multiscale_ratios", (0.008, 0.018, 0.04)),
    ("proposal.tile_ratios", (0.08, 0.16, 0.28)),
    ("proposal.merge_overlap", 0.24),
    ("proposal.merge_containment", 0.72),
    ("proposal.merge_coherence", 0.44),
    ("proposal.nms_iou", 0.28),
    ("proposal.nms_containment", 0.68),
    ("proposal.nms_mask_overlap", 0.18),
    ("proposal.refinement_open_kernel", (3, 3)),
    ("proposal.refinement_close_kernel", (5, 5)),
    ("proposal.priority_rounding_digits", 1),
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


IMMUTABLE_IMPLEMENTATION_CONSTANTS_HASH = hashlib.sha256(
    _canonical_json(IMMUTABLE_IMPLEMENTATION_CONSTANTS).encode("utf-8")
).hexdigest()

DEFAULT_EVIDENCE_WEIGHTS = (
    ("local_texture_contrast", 0.22),
    ("local_colour_contrast", 0.20),
    ("local_entropy_contrast", 0.13),
    ("edge_concentration", 0.14),
    ("gradient_contrast", 0.16),
    ("geometric_irregularity", 0.15),
)
DEFAULT_RELIABILITY_WEIGHTS = (
    ("perturbation_stability", 0.24),
    ("connectedness", 0.18),
    ("boundary_smoothness", 0.18),
    ("scale_agreement", 0.16),
    ("segmentation_coherence", 0.24),
)
DEFAULT_PRIORITY_WEIGHTS = (
    ("anomaly_evidence", 0.60),
    ("mask_reliability", 0.20),
    ("area_relevance", 0.10),
    ("novelty", 0.10),
)


def _require_bool(name: str, value: object) -> None:
    if type(value) is not bool:
        raise InvalidConfigurationError(f"{name} must be boolean")


def _require_int(name: str, value: object, minimum: int, maximum: int) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise InvalidConfigurationError(f"{name} must be an integer in [{minimum}, {maximum}]")


def _finite(name: str, value: object, minimum: float, maximum: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidConfigurationError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or not minimum <= numeric <= maximum:
        raise InvalidConfigurationError(f"{name} must be finite and in [{minimum}, {maximum}]")


def _exact_keys(payload: Mapping[str, object], record_type: type) -> None:
    expected = {item.name for item in fields(record_type)}
    observed = set(payload)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise InvalidConfigurationError(
            f"Invalid {record_type.__name__} keys; missing={missing}, unexpected={unexpected}"
        )


@dataclass(frozen=True)
class PreprocessingConfig:
    resize_width: int = 1024
    denoise: bool = True
    clahe: bool = True
    sharpen: bool = False
    brightness: int = 0
    contrast: int = 0

    def __post_init__(self) -> None:
        _require_int("resize_width", self.resize_width, 16, 32768)
        _require_int("brightness", self.brightness, -255, 255)
        _require_int("contrast", self.contrast, -99, 300)
        for name in ("denoise", "clahe", "sharpen"):
            _require_bool(name, getattr(self, name))


@dataclass(frozen=True)
class FeatureConfig:
    edge_sensitivity: int = 100
    texture_sensitivity: int = 35
    colour_sensitivity: int = 35
    threshold_level: int = 128

    def __post_init__(self) -> None:
        _require_int("edge_sensitivity", self.edge_sensitivity, 1, 255)
        _require_int("texture_sensitivity", self.texture_sensitivity, 0, 255)
        _require_int("colour_sensitivity", self.colour_sensitivity, 0, 255)
        _require_int("threshold_level", self.threshold_level, 0, 255)


@dataclass(frozen=True)
class ProposalConfig:
    minimum_area_pixels: int = 250
    minimum_relative_area: float = 0.0002
    maximum_relative_area: float = 0.85
    border_margin: float = 0.025
    maximum_proposal_count: int = 8
    edge_features: bool = True
    texture_features: bool = True
    colour_features: bool = True
    entropy_features: bool = True
    perturbation_stability: bool = True
    contextual_contrast: bool = True
    multi_scale_fusion: bool = True
    region_merging: bool = True
    mask_refinement: bool = True
    local_texture_context: bool = True
    local_colour_context: bool = True
    local_entropy_context: bool = True
    internal_boundary_edge: bool = True
    border_penalty: bool = True
    coherence_term: bool = True
    specular_suppression: bool = False
    specular_penalty_weight: float = 0.65
    specular_rejection_threshold: float = 0.50
    crack_safeguard_weight: float = 0.90
    pitting_safeguard_weight: float = 0.65

    def __post_init__(self) -> None:
        _require_int("minimum_area_pixels", self.minimum_area_pixels, 1, 2**31 - 1)
        _require_int("maximum_proposal_count", self.maximum_proposal_count, 1, 10000)
        _finite("minimum_relative_area", self.minimum_relative_area, 0.0, 1.0)
        _finite("maximum_relative_area", self.maximum_relative_area, 0.0, 1.0)
        _finite("border_margin", self.border_margin, 0.0, 0.5)
        if float(self.minimum_relative_area) >= float(self.maximum_relative_area):
            raise InvalidConfigurationError("minimum_relative_area must be below maximum_relative_area")
        boolean_fields = (
            "edge_features", "texture_features", "colour_features", "entropy_features",
            "perturbation_stability", "contextual_contrast", "multi_scale_fusion",
            "region_merging", "mask_refinement", "local_texture_context",
            "local_colour_context", "local_entropy_context", "internal_boundary_edge",
            "border_penalty", "coherence_term", "specular_suppression",
        )
        for name in boolean_fields:
            _require_bool(name, getattr(self, name))
        for name in (
            "specular_penalty_weight", "specular_rejection_threshold",
            "crack_safeguard_weight", "pitting_safeguard_weight",
        ):
            _finite(name, getattr(self, name), 0.0, 1.0)


@dataclass(frozen=True)
class ScoringConfig:
    evidence_weights: tuple[tuple[str, float], ...] = DEFAULT_EVIDENCE_WEIGHTS
    reliability_weights: tuple[tuple[str, float], ...] = DEFAULT_RELIABILITY_WEIGHTS
    priority_weights: tuple[tuple[str, float], ...] = DEFAULT_PRIORITY_WEIGHTS

    def __post_init__(self) -> None:
        for name, weights in (
            ("evidence_weights", self.evidence_weights),
            ("reliability_weights", self.reliability_weights),
            ("priority_weights", self.priority_weights),
        ):
            if not isinstance(weights, tuple) or not weights:
                raise InvalidConfigurationError(f"{name} must be a non-empty immutable tuple")
            keys = [key for key, _ in weights]
            if len(keys) != len(set(keys)):
                raise InvalidConfigurationError(f"{name} contains duplicate component names")
            for key, value in weights:
                if not isinstance(key, str) or not key:
                    raise InvalidConfigurationError(f"{name} contains an invalid component name")
                _finite(f"{name}.{key}", value, 0.0, 1_000_000.0)
            if sum(float(value) for _, value in weights) <= 0:
                raise InvalidConfigurationError(f"{name} must have positive total weight")


@dataclass(frozen=True)
class DetectorConfig:
    """Complete frozen configuration for one classical-baseline execution."""

    preprocessing: PreprocessingConfig = PreprocessingConfig()
    features: FeatureConfig = FeatureConfig()
    proposals: ProposalConfig = ProposalConfig()
    scoring: ScoringConfig = ScoringConfig()
    random_seed: int = 0
    deterministic_mode: bool = True
    implementation_id: str = IMPLEMENTATION_ID
    implementation_version: str = IMPLEMENTATION_VERSION
    implementation_constants_hash: str = IMMUTABLE_IMPLEMENTATION_CONSTANTS_HASH

    def __post_init__(self) -> None:
        if not isinstance(self.preprocessing, PreprocessingConfig):
            raise InvalidConfigurationError("preprocessing must be PreprocessingConfig")
        if not isinstance(self.features, FeatureConfig):
            raise InvalidConfigurationError("features must be FeatureConfig")
        if not isinstance(self.proposals, ProposalConfig):
            raise InvalidConfigurationError("proposals must be ProposalConfig")
        if not isinstance(self.scoring, ScoringConfig):
            raise InvalidConfigurationError("scoring must be ScoringConfig")
        _require_int("random_seed", self.random_seed, 0, 2**32 - 1)
        _require_bool("deterministic_mode", self.deterministic_mode)
        if self.implementation_id != IMPLEMENTATION_ID or self.implementation_version != IMPLEMENTATION_VERSION:
            raise InvalidConfigurationError("The frozen adapter implementation identity cannot be changed")
        if self.implementation_constants_hash != IMMUTABLE_IMPLEMENTATION_CONSTANTS_HASH:
            raise InvalidConfigurationError("Immutable implementation-constant identity mismatch")
        # The legacy proposal function does not expose weight injection. Reject
        # alternatives rather than silently hashing values that are not executed.
        if self.scoring != ScoringConfig():
            raise InvalidConfigurationError("Frozen v1 scoring weights cannot be changed by the adapter")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def configuration_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def specification_sections(self) -> dict[str, object]:
        """Return the exact v2 configuration partitions without adding defaults."""
        return {
            "preprocessing": asdict(self.preprocessing),
            "proposal": asdict(self.proposals),
            "feature_and_scoring": {
                "features": asdict(self.features),
                "scoring": asdict(self.scoring),
            },
            "method": self.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "DetectorConfig":
        if not isinstance(payload, Mapping):
            raise InvalidConfigurationError("Detector configuration must be a mapping")
        _exact_keys(payload, cls)
        preprocessing_payload = payload["preprocessing"]
        feature_payload = payload["features"]
        proposal_payload = payload["proposals"]
        scoring_payload = payload["scoring"]
        for value, record_type in (
            (preprocessing_payload, PreprocessingConfig),
            (feature_payload, FeatureConfig),
            (proposal_payload, ProposalConfig),
            (scoring_payload, ScoringConfig),
        ):
            if not isinstance(value, Mapping):
                raise InvalidConfigurationError(f"{record_type.__name__} must be a mapping")
            _exact_keys(value, record_type)
        scoring_values = dict(scoring_payload)
        for name in ("evidence_weights", "reliability_weights", "priority_weights"):
            scoring_values[name] = tuple((str(key), float(value)) for key, value in scoring_values[name])
        return cls(
            preprocessing=PreprocessingConfig(**dict(preprocessing_payload)),
            features=FeatureConfig(**dict(feature_payload)),
            proposals=ProposalConfig(**dict(proposal_payload)),
            scoring=ScoringConfig(**scoring_values),
            random_seed=payload["random_seed"],
            deterministic_mode=payload["deterministic_mode"],
            implementation_id=payload["implementation_id"],
            implementation_version=payload["implementation_version"],
            implementation_constants_hash=payload["implementation_constants_hash"],
        )

    @classmethod
    def from_json(cls, value: str) -> "DetectorConfig":
        try:
            payload = json.loads(value)
        except (TypeError, json.JSONDecodeError) as error:
            raise InvalidConfigurationError("Detector configuration is not valid JSON") from error
        if _canonical_json(payload) != value:
            raise InvalidConfigurationError("Detector configuration JSON must be canonical")
        return cls.from_dict(payload)
