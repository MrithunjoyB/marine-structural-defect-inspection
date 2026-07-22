"""Compatibility adapter around the byte-protected legacy classical pipeline."""

from __future__ import annotations

from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import RLock
import time
from types import ModuleType
from typing import Iterator, Mapping

import cv2
import numpy as np

from .configuration import DetectorConfig
from .provenance import build_provenance, hash_module_sources
from .types import AnalysisResult, Proposal, freeze_value, frozen_mapping


_LEGACY_LOCK = RLock()


def _load_legacy_modules(temporary_root: Path) -> tuple[object, object, object, object]:
    """Load protected modules without importing the write-on-import legacy config."""
    preprocess = importlib.import_module("preprocess")
    sentinel = object()
    prior_config = sys.modules.get("config", sentinel)
    stub = ModuleType("config")
    stub.FEATURE_DIR = temporary_root / "feature_maps"
    stub.MASK_DIR = temporary_root / "masks"
    stub.OUTPUT_DIR = temporary_root / "outputs"
    try:
        sys.modules["config"] = stub
        features = importlib.import_module("feature_extraction")
        scoring = importlib.import_module("scoring")
        proposals = importlib.import_module("region_proposal")
    finally:
        if prior_config is sentinel:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = prior_config
    return preprocess, features, scoring, proposals


@contextmanager
def _redirect_legacy_artifacts(
    features: object, proposals: object, root: Path,
) -> Iterator[None]:
    prior = (
        getattr(features, "FEATURE_DIR"),
        getattr(proposals, "MASK_DIR"),
        getattr(proposals, "OUTPUT_DIR"),
    )
    features.FEATURE_DIR = root / "feature_maps"
    proposals.MASK_DIR = root / "masks"
    proposals.OUTPUT_DIR = root / "outputs"
    try:
        yield
    finally:
        features.FEATURE_DIR, proposals.MASK_DIR, proposals.OUTPUT_DIR = prior


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask > 0)
    return float(np.mean(xs)), float(np.mean(ys))


def _proposal_record(legacy: object, mask: np.ndarray, raw_mask: np.ndarray, config: DetectorConfig) -> Proposal:
    context = {
        "relative_area": legacy.relative_area,
        "aspect_ratio": legacy.aspect_ratio,
        "perimeter": legacy.perimeter,
        "edge_density": legacy.edge_density,
        "texture_score": legacy.texture_score,
        "colour_variation_score": legacy.color_variation_score,
        "gradient_strength": legacy.gradient_strength,
        "entropy": legacy.entropy,
        "contrast_difference": legacy.contrast_difference,
        "mask_stability": legacy.mask_stability,
        "local_texture_contrast": legacy.local_texture_contrast,
        "local_colour_contrast": legacy.local_colour_contrast,
        "local_entropy_contrast": legacy.local_entropy_contrast,
        "internal_vs_boundary_edge_ratio": legacy.internal_vs_boundary_edge_ratio,
        "coherence_score": legacy.coherence_score,
        "border_penalty": legacy.border_penalty,
        "area_reduction": legacy.area_reduction,
        "boundary_smoothness": legacy.boundary_smoothness,
        "dominant_features": legacy.dominant_features,
        "explanation": legacy.explanation,
        "priority_label": legacy.priority.label,
        "specular_diagnostics": legacy.specular_diagnostics,
    }
    rank = int(legacy.region_id[1:])
    priority = float(legacy.priority.score)
    return Proposal(
        proposal_id=legacy.region_id,
        rank=rank,
        bbox=tuple(int(value) for value in legacy.bbox),
        final_mask=mask,
        raw_mask=raw_mask,
        proposal_score=priority,
        evidence_score=float(legacy.anomaly_evidence_score),
        heuristic_reliability=float(legacy.mask_reliability_score),
        priority_score=priority,
        component_scores=tuple((str(name), float(value)) for name, value in legacy.feature_contributions.items()),
        area=int(legacy.pixel_area),
        centroid=_centroid(mask),
        context_diagnostics=frozen_mapping(context),
        warnings=(),
        rejection_information=(),
        implementation_id=config.implementation_id,
        implementation_version=config.implementation_version,
    )


def run_frozen_classical(
    image_bgr: np.ndarray, *, image_id: str, input_hash: str, source_hash: str,
    source_type: str, metadata: Mapping[str, object] | None, config: DetectorConfig,
) -> AnalysisResult:
    """Execute protected mathematics and convert its path outputs to memory records."""
    started = time.perf_counter()
    with _LEGACY_LOCK:
        with TemporaryDirectory(prefix="structvision-frozen-") as temporary:
            root = Path(temporary)
            preprocess, features, scoring, proposals_module = _load_legacy_modules(root)
            source_hashes = hash_module_sources((preprocess, features, proposals_module, scoring))
            with _redirect_legacy_artifacts(features, proposals_module, root):
                preprocess_started = time.perf_counter()
                processed = preprocess.apply_preprocessing(
                    image_bgr,
                    resize_width=config.preprocessing.resize_width,
                    denoise=config.preprocessing.denoise,
                    clahe=config.preprocessing.clahe,
                    sharpen=config.preprocessing.sharpen,
                    brightness=config.preprocessing.brightness,
                    contrast=config.preprocessing.contrast,
                )
                preprocess_elapsed = time.perf_counter() - preprocess_started
                feature_started = time.perf_counter()
                feature_maps = features.extract_feature_maps(
                    processed,
                    edge_sensitivity=config.features.edge_sensitivity,
                    texture_sensitivity=config.features.texture_sensitivity,
                    color_sensitivity=config.features.colour_sensitivity,
                    threshold_level=config.features.threshold_level,
                )
                feature_elapsed = time.perf_counter() - feature_started
                proposal_started = time.perf_counter()
                proposal_config = config.proposals
                ablation = proposals_module.AblationConfig(
                    edge_features=proposal_config.edge_features,
                    texture_features=proposal_config.texture_features,
                    colour_features=proposal_config.colour_features,
                    entropy_features=proposal_config.entropy_features,
                    stability=proposal_config.perturbation_stability,
                    contextual_contrast=proposal_config.contextual_contrast,
                    multi_scale_fusion=proposal_config.multi_scale_fusion,
                    region_merging=proposal_config.region_merging,
                    mask_refinement=proposal_config.mask_refinement,
                    local_texture_context=proposal_config.local_texture_context,
                    local_colour_context=proposal_config.local_colour_context,
                    local_entropy_context=proposal_config.local_entropy_context,
                    internal_boundary_edge=proposal_config.internal_boundary_edge,
                    border_penalty=proposal_config.border_penalty,
                    coherence_term=proposal_config.coherence_term,
                    specular_suppression=proposal_config.specular_suppression,
                    specular_penalty_weight=proposal_config.specular_penalty_weight,
                    specular_rejection_threshold=proposal_config.specular_rejection_threshold,
                    crack_safeguard_weight=proposal_config.crack_safeguard_weight,
                    pitting_safeguard_weight=proposal_config.pitting_safeguard_weight,
                )
                legacy_result = proposals_module.propose_regions(
                    processed,
                    feature_maps,
                    image_stem=f"api_{input_hash[:16]}",
                    min_area=proposal_config.minimum_area_pixels,
                    max_regions=proposal_config.maximum_proposal_count,
                    min_relative_area=proposal_config.minimum_relative_area,
                    max_relative_area=proposal_config.maximum_relative_area,
                    border_margin=proposal_config.border_margin,
                    ablation=ablation,
                )
                proposal_elapsed = time.perf_counter() - proposal_started
                conversion_started = time.perf_counter()
                converted = []
                for legacy_proposal in legacy_result.proposals:
                    final_mask = cv2.imread(str(legacy_proposal.mask_path), cv2.IMREAD_GRAYSCALE)
                    raw_mask = cv2.imread(str(legacy_proposal.raw_mask_path), cv2.IMREAD_GRAYSCALE)
                    if final_mask is None or raw_mask is None:
                        raise RuntimeError(f"Frozen adapter could not recover masks for {legacy_proposal.region_id}")
                    converted.append(_proposal_record(legacy_proposal, final_mask, raw_mask, config))
                artifact_count = sum(1 for path in root.rglob("*") if path.is_file())
                conversion_elapsed = time.perf_counter() - conversion_started
                diagnostics = legacy_result.diagnostics.to_dict()
                provenance = build_provenance(
                    source_type=source_type,
                    source_content_hash=source_hash,
                    actual_source_hashes=source_hashes,
                    temporary_artifact_count=artifact_count,
                )
                elapsed = time.perf_counter() - started
                return AnalysisResult(
                    image_id=image_id,
                    input_hash=input_hash,
                    image_shape=tuple(int(value) for value in processed.shape),
                    normalised_colour_space="BGR",
                    proposals=tuple(converted),
                    anomaly_heatmap=feature_maps.anomaly_heatmap,
                    preprocessing_metadata=frozen_mapping({
                        "input_shape": list(image_bgr.shape),
                        "analysed_shape": list(processed.shape),
                        "configuration": config.preprocessing.__dict__,
                    }),
                    configuration_hash=config.configuration_hash,
                    implementation_id=config.implementation_id,
                    implementation_version=config.implementation_version,
                    deterministic_mode=config.deterministic_mode,
                    timing_breakdown_seconds=(
                        ("preprocessing", preprocess_elapsed),
                        ("feature_extraction", feature_elapsed),
                        ("proposal_generation", proposal_elapsed),
                        ("adapter_conversion", conversion_elapsed),
                        ("core_total", elapsed),
                    ),
                    warnings=(),
                    provenance=provenance,
                    image_metadata=frozen_mapping(metadata),
                    diagnostics=frozen_mapping(diagnostics),
                )
