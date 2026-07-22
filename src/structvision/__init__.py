"""Reusable local API for StructVision's frozen classical baseline."""

from .api import StructuralAnomalyDetector
from .configuration import (
    DetectorConfig,
    FeatureConfig,
    PreprocessingConfig,
    ProposalConfig,
    ScoringConfig,
)
from .executor import ExperimentExecutorV2, ExperimentSample, V2ExecutionReport
from .errors import *
from .sinks import (
    ArtifactSink,
    MemoryResultSink,
    NullArtifactSink,
    NullResultSink,
    ResultSink,
    V2SQLiteResultSink,
)
from .types import AnalysisResult, AnalysisSample, BatchAnalysisResult, BatchFailure, Proposal

__all__ = [
    "AnalysisResult",
    "AnalysisSample",
    "ArtifactSink",
    "BatchAnalysisResult",
    "BatchFailure",
    "DetectorConfig",
    "FeatureConfig",
    "ExperimentExecutorV2",
    "ExperimentSample",
    "MemoryResultSink",
    "NullArtifactSink",
    "NullResultSink",
    "PreprocessingConfig",
    "Proposal",
    "ProposalConfig",
    "ResultSink",
    "ScoringConfig",
    "StructuralAnomalyDetector",
    "V2SQLiteResultSink",
    "V2ExecutionReport",
]

__version__ = "1.0.0"
