"""Optional normal-feature API; importing it does not import Torch or Anomalib."""

from .calibration import (
    CalibrationArtifact,
    CalibrationSample,
    DirectoryCalibrationArtifactSink,
    calibrate,
    load_calibration_artifact,
)
from .configuration import LearnedProposalConfig, NormalFeatureConfig
from .model_artifact import (
    DirectoryModelArtifactSink,
    NormalFeatureModelArtifact,
    load_model_artifact,
)
from .patchcore import NormalFeatureAnomalyDetector
from .types import (
    LearnedProposal,
    NormalFeatureAnalysisResult,
    NormalFeatureScoreResult,
    NormalFitSample,
)

__all__ = [
    "CalibrationArtifact",
    "CalibrationSample",
    "DirectoryCalibrationArtifactSink",
    "DirectoryModelArtifactSink",
    "LearnedProposal",
    "LearnedProposalConfig",
    "NormalFeatureAnalysisResult",
    "NormalFeatureAnomalyDetector",
    "NormalFeatureConfig",
    "NormalFeatureModelArtifact",
    "NormalFeatureScoreResult",
    "NormalFitSample",
    "calibrate",
    "load_calibration_artifact",
    "load_model_artifact",
]
