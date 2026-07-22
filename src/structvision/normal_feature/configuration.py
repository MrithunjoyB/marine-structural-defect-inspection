"""Immutable identity for the single predeclared PatchCore development baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from scientific_contract.hashing import sha256_json


IMPLEMENTATION_ID = "structvision-patchcore-baseline-v1-dev"
IMPLEMENTATION_VERSION = "1.0.0-dev1"
UPSTREAM_LIBRARY = "anomalib"
UPSTREAM_VERSION = "2.5.1"
BACKBONE = "wide_resnet50_2"
WEIGHT_MODEL_ID = "timm/wide_resnet50_2.racm_in1k"
WEIGHT_REVISION = "30f73aceaaa1911830a9795b83ab1908dba18719"
WEIGHT_FILENAME = "model.safetensors"
WEIGHT_SHA256 = "03b71d65fb2c73bb0de079a1781009f27a782ec481d2f64ab3bde9b1cdec3000"
WEIGHT_LICENCE = "Apache-2.0"
WEIGHT_SOURCE = "https://huggingface.co/timm/wide_resnet50_2.racm_in1k"


@dataclass(frozen=True)
class LearnedProposalConfig:
    """Predeclared map-to-proposal policy; threshold comes only from calibration."""

    connectivity: int = 8
    minimum_area_pixels: int = 16
    maximum_proposal_count: int = 8
    morphology: str = "none"
    component_score: str = "maximum_patchcore_distance"
    ranking: str = "component_score_desc_area_desc_bbox_lexical"
    bbox_convention: str = "half-open:x_min,y_min,x_max,y_max"

    def __post_init__(self) -> None:
        if self.connectivity not in {4, 8}:
            raise ValueError("Connectivity must be 4 or 8")
        if self.minimum_area_pixels <= 0 or self.maximum_proposal_count <= 0:
            raise ValueError("Proposal limits must be positive")
        if self.morphology != "none":
            raise ValueError("The v1 development baseline predeclares no morphology")

    @property
    def configuration_hash(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True)
class NormalFeatureConfig:
    """One resource-declared configuration; it is not a search space."""

    implementation_id: str = IMPLEMENTATION_ID
    implementation_version: str = IMPLEMENTATION_VERSION
    upstream_library: str = UPSTREAM_LIBRARY
    upstream_version: str = UPSTREAM_VERSION
    backbone: str = BACKBONE
    pretrained: bool = True
    pretrained_weight_model_id: str = WEIGHT_MODEL_ID
    pretrained_weight_revision: str = WEIGHT_REVISION
    pretrained_weight_filename: str = WEIGHT_FILENAME
    pretrained_weight_sha256: str = WEIGHT_SHA256
    pretrained_weight_source: str = WEIGHT_SOURCE
    pretrained_weight_licence: str = WEIGHT_LICENCE
    extracted_layers: tuple[str, ...] = ("layer2", "layer3")
    input_colour_space: str = "RGB"
    input_normalisation_mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    input_normalisation_std: tuple[float, ...] = (0.229, 0.224, 0.225)
    spatial_resolution_policy: str = "aspect_preserving_letterbox"
    input_height: int = 256
    input_width: int = 416
    resize_interpolation: str = "opencv_inter_area_down_linear_up"
    padding_value_rgb: tuple[int, ...] = (0, 0, 0)
    tiling_policy: str = "none"
    embedding_dimensions: int = 1536
    coreset_sampling_policy: str = "anomalib_k_center_greedy_sparse_projection"
    coreset_sampling_ratio: float = 0.001
    nearest_neighbour_count: int = 9
    nearest_neighbour_index: str = "anomalib_exact_chunked_brute_force"
    distance_metric: str = "euclidean"
    random_seed: int = 42
    device: str = "cpu"
    deterministic_mode: bool = True
    torch_num_threads: int = 1
    torch_num_interop_threads: int = 1
    anomaly_map_interpolation: str = "anomalib_bilinear_then_letterbox_inverse_linear"
    batch_size: int = 1
    proposal: LearnedProposalConfig = LearnedProposalConfig()

    def __post_init__(self) -> None:
        expected = (
            (self.implementation_id, IMPLEMENTATION_ID),
            (self.implementation_version, IMPLEMENTATION_VERSION),
            (self.upstream_library, UPSTREAM_LIBRARY),
            (self.upstream_version, UPSTREAM_VERSION),
            (self.backbone, BACKBONE),
            (self.pretrained_weight_model_id, WEIGHT_MODEL_ID),
            (self.pretrained_weight_revision, WEIGHT_REVISION),
            (self.pretrained_weight_filename, WEIGHT_FILENAME),
            (self.pretrained_weight_sha256, WEIGHT_SHA256),
            (self.pretrained_weight_source, WEIGHT_SOURCE),
            (self.pretrained_weight_licence, WEIGHT_LICENCE),
        )
        if any(actual != required for actual, required in expected):
            raise ValueError("The v1 development baseline identity is fixed")
        if not self.pretrained:
            raise ValueError("Random or untrained backbone weights are forbidden")
        if self.device != "cpu" or not self.deterministic_mode:
            raise ValueError("The scientific reference requires deterministic CPU execution")
        if self.extracted_layers != ("layer2", "layer3") or self.embedding_dimensions != 1536:
            raise ValueError("The predeclared PatchCore feature contract is fixed")
        if (self.input_height, self.input_width) != (256, 416):
            raise ValueError("The normal-only resource assessment froze a 256x416 input")
        if (
            self.input_colour_space != "RGB"
            or self.input_normalisation_mean != (0.485, 0.456, 0.406)
            or self.input_normalisation_std != (0.229, 0.224, 0.225)
            or self.spatial_resolution_policy != "aspect_preserving_letterbox"
            or self.tiling_policy != "none"
        ):
            raise ValueError("The predeclared input and spatial contract is fixed")
        if (
            self.coreset_sampling_policy != "anomalib_k_center_greedy_sparse_projection"
            or self.coreset_sampling_ratio != 0.001
            or self.nearest_neighbour_index != "anomalib_exact_chunked_brute_force"
            or self.distance_metric != "euclidean"
        ):
            raise ValueError("The predeclared memory and distance contract is fixed")
        if self.nearest_neighbour_count != 9 or self.batch_size != 1:
            raise ValueError("PatchCore neighbour count and deterministic batch size are fixed")
        if (
            self.random_seed != 42
            or self.torch_num_threads != 1
            or self.torch_num_interop_threads != 1
            or self.proposal != LearnedProposalConfig()
        ):
            raise ValueError("The predeclared deterministic reference configuration is fixed")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["proposal"]["configuration_hash"] = self.proposal.configuration_hash
        return payload

    @property
    def configuration_hash(self) -> str:
        return sha256_json(self.to_dict())

    def specification_sections(self) -> dict[str, dict[str, object]]:
        value = self.to_dict()
        return {
            "preprocessing": {
                key: value[key]
                for key in (
                    "input_colour_space", "input_normalisation_mean", "input_normalisation_std",
                    "spatial_resolution_policy", "input_height", "input_width",
                    "resize_interpolation", "padding_value_rgb", "tiling_policy",
                    "anomaly_map_interpolation",
                )
            },
            "proposal": value["proposal"],
            "feature_and_scoring": {
                key: value[key]
                for key in (
                    "upstream_library", "upstream_version", "backbone", "pretrained",
                    "pretrained_weight_model_id", "pretrained_weight_revision",
                    "pretrained_weight_sha256", "pretrained_weight_licence", "extracted_layers",
                    "embedding_dimensions", "coreset_sampling_policy", "coreset_sampling_ratio",
                    "nearest_neighbour_count", "nearest_neighbour_index", "distance_metric",
                    "random_seed", "device", "deterministic_mode", "torch_num_threads",
                    "torch_num_interop_threads", "batch_size",
                )
            },
        }
