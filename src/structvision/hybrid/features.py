"""Compact, deterministic candidate-level evidence for hybrid fusion."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import cv2
import numpy as np

from structvision.types import Proposal

from .errors import HybridFeatureError


FEATURE_ORDER = (
    "classical_priority",
    "classical_evidence",
    "classical_mask_reliability",
    "patchcore_inside_mean",
    "patchcore_inside_q95",
    "patchcore_high_support_fraction",
    "patchcore_context_contrast",
    "patchcore_local_spatial_agreement",
)
CLASSICAL_FEATURES = FEATURE_ORDER[:3]
NORMALITY_FEATURES = FEATURE_ORDER[3:]


@dataclass(frozen=True)
class CandidateFeatureDefinition:
    name: str
    definition: str
    expected_direction: str
    deterministic_implementation: str
    ablation_eligible: bool = True

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


FEATURE_DEFINITIONS = (
    CandidateFeatureDefinition(
        "classical_priority", "Frozen classical priority_score.", "higher supports anomaly",
        "Direct read from the byte-protected classical Proposal record.",
    ),
    CandidateFeatureDefinition(
        "classical_evidence", "Frozen classical evidence_score.", "higher supports anomaly",
        "Direct read from the byte-protected classical Proposal record.",
    ),
    CandidateFeatureDefinition(
        "classical_mask_reliability", "Frozen heuristic mask reliability.", "higher supports a stable candidate",
        "Direct read from the byte-protected classical Proposal record; not a calibrated probability.",
    ),
    CandidateFeatureDefinition(
        "patchcore_inside_mean", "Arithmetic mean PatchCore distance inside the classical mask.",
        "higher supports normality departure", "NumPy float64 mean over the full-resolution projected anomaly map.",
    ),
    CandidateFeatureDefinition(
        "patchcore_inside_q95", "Linear 0.95 quantile of PatchCore distance inside the classical mask.",
        "higher supports focal normality departure", "NumPy quantile(method='linear') over mask pixels.",
    ),
    CandidateFeatureDefinition(
        "patchcore_high_support_fraction", "Fraction of mask pixels above the clean fusion-fit map reference.",
        "higher supports normality departure", "Strict greater-than comparison to the frozen reference.",
    ),
    CandidateFeatureDefinition(
        "patchcore_context_contrast", "Inside mean minus mean over a deterministic local context ring.",
        "higher supports locally concentrated departure", "Binary dilation radius max(2,round(sqrt(area)/8)); ring excludes mask.",
    ),
    CandidateFeatureDefinition(
        "patchcore_local_spatial_agreement", "High-distance pixels inside mask divided by high-distance pixels in mask plus ring.",
        "higher supports spatial agreement", "Same deterministic ring and clean-derived reference; zero when local support is empty.",
    ),
)


@dataclass(frozen=True)
class CandidateEvidence:
    proposal_id: str
    feature_values: tuple[tuple[str, float], ...]
    context_ring_radius: int
    classical_mask_hash: str

    def __post_init__(self) -> None:
        if tuple(name for name, _ in self.feature_values) != FEATURE_ORDER:
            raise HybridFeatureError("Candidate feature order differs from the frozen identity")
        if any(not math.isfinite(float(value)) for _, value in self.feature_values):
            raise HybridFeatureError("Candidate features must be finite")
        if self.context_ring_radius < 2:
            raise HybridFeatureError("Context-ring radius is invalid")

    def as_dict(self) -> dict[str, float]:
        return dict(self.feature_values)


@dataclass(frozen=True)
class FeatureNormalisation:
    name: str
    lower_quantile_05: float
    upper_quantile_95: float
    method: str = "fusion-fit empirical q05/q95; affine scale then clamp to [0,1]"

    def __post_init__(self) -> None:
        if self.name not in FEATURE_ORDER:
            raise HybridFeatureError("Unknown normalisation feature")
        if not math.isfinite(self.lower_quantile_05) or not math.isfinite(self.upper_quantile_95):
            raise HybridFeatureError("Normalisation bounds must be finite")
        if self.lower_quantile_05 > self.upper_quantile_95:
            raise HybridFeatureError("Normalisation bounds are reversed")

    def apply(self, value: float) -> float:
        if not math.isfinite(float(value)):
            raise HybridFeatureError("Cannot normalise a non-finite feature")
        span = self.upper_quantile_95 - self.lower_quantile_05
        if span <= 1e-12:
            return 0.0
        return float(np.clip((float(value) - self.lower_quantile_05) / span, 0.0, 1.0))

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def _mask_hash(mask: np.ndarray) -> str:
    array = np.ascontiguousarray(mask)
    prefix = f"{array.dtype.str}\0{','.join(str(item) for item in array.shape)}\0C\0".encode("ascii")
    return hashlib.sha256(prefix + array.tobytes(order="C")).hexdigest()


def clean_map_reference(anomaly_maps: tuple[np.ndarray, ...]) -> float:
    """Median clean-image q95; fitting code must pass fusion-fit clean maps only."""
    if not anomaly_maps:
        raise HybridFeatureError("A clean fusion-fit map reference is required")
    quantiles = []
    for anomaly_map in anomaly_maps:
        array = np.asarray(anomaly_map, dtype=np.float32)
        if array.ndim != 2 or not np.isfinite(array).all():
            raise HybridFeatureError("Reference maps must be finite and two-dimensional")
        quantiles.append(float(np.quantile(array, 0.95, method="linear")))
    reference = float(np.median(np.asarray(quantiles, dtype=np.float64)))
    if not math.isfinite(reference):
        raise HybridFeatureError("Clean anomaly-map reference is not finite")
    return reference


def candidate_evidence(
    proposal: Proposal,
    anomaly_map: np.ndarray,
    *,
    high_anomaly_reference: float,
) -> CandidateEvidence:
    """Extract evidence without category, filename, or ground-truth inputs."""
    values = np.asarray(anomaly_map, dtype=np.float32)
    mask_bytes_before = proposal.final_mask.tobytes(order="C")
    if values.ndim != 2 or values.shape != proposal.final_mask.shape or not np.isfinite(values).all():
        raise HybridFeatureError("Classical mask and projected PatchCore map are not exactly aligned")
    if not math.isfinite(float(high_anomaly_reference)):
        raise HybridFeatureError("High-anomaly reference must be finite")
    mask = proposal.final_mask > 0
    inside = values[mask]
    if inside.size == 0:
        raise HybridFeatureError("Classical candidate mask is empty")
    radius = max(2, int(round(math.sqrt(proposal.area) / 8.0)))
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    ring = dilated & ~mask
    ring_values = values[ring]
    inside_mean = float(np.mean(inside, dtype=np.float64))
    ring_mean = float(np.mean(ring_values, dtype=np.float64)) if ring_values.size else inside_mean
    high = values > float(high_anomaly_reference)
    high_inside = int(np.count_nonzero(high & mask))
    high_local = int(np.count_nonzero(high & (mask | ring)))
    evidence = CandidateEvidence(
        proposal_id=proposal.proposal_id,
        feature_values=(
            ("classical_priority", float(proposal.priority_score)),
            ("classical_evidence", float(proposal.evidence_score)),
            ("classical_mask_reliability", float(proposal.heuristic_reliability)),
            ("patchcore_inside_mean", inside_mean),
            ("patchcore_inside_q95", float(np.quantile(inside, 0.95, method="linear"))),
            ("patchcore_high_support_fraction", high_inside / int(np.count_nonzero(mask))),
            ("patchcore_context_contrast", inside_mean - ring_mean),
            ("patchcore_local_spatial_agreement", high_inside / high_local if high_local else 0.0),
        ),
        context_ring_radius=radius,
        classical_mask_hash=_mask_hash(proposal.final_mask),
    )
    if proposal.final_mask.tobytes(order="C") != mask_bytes_before:
        raise HybridFeatureError("Classical candidate mask changed during feature extraction")
    return evidence


def fit_normalisation(evidence: tuple[CandidateEvidence, ...]) -> tuple[FeatureNormalisation, ...]:
    if not evidence:
        raise HybridFeatureError("Fusion-fit candidates are required for normalisation")
    return tuple(FeatureNormalisation(
        name,
        float(np.quantile([item.as_dict()[name] for item in evidence], 0.05, method="linear")),
        float(np.quantile([item.as_dict()[name] for item in evidence], 0.95, method="linear")),
    ) for name in FEATURE_ORDER)


def normalised_components(
    evidence: CandidateEvidence,
    normalisation: tuple[FeatureNormalisation, ...],
) -> tuple[float, float, tuple[tuple[str, float], ...]]:
    if tuple(item.name for item in normalisation) != FEATURE_ORDER:
        raise HybridFeatureError("Normalisation order differs from feature identity")
    raw = evidence.as_dict()
    scaled = tuple((item.name, item.apply(raw[item.name])) for item in normalisation)
    lookup = dict(scaled)
    classical = float(np.mean([lookup[name] for name in CLASSICAL_FEATURES], dtype=np.float64))
    normality = float(np.mean([lookup[name] for name in NORMALITY_FEATURES], dtype=np.float64))
    return classical, normality, scaled
