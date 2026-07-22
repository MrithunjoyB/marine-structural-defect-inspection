"""Fail-closed errors for the optional normal-feature baseline."""


class NormalFeatureError(RuntimeError):
    """Base error for the learned development-only boundary."""


class OptionalDependencyError(NormalFeatureError):
    """The exact optional learned environment is unavailable or incompatible."""


class DevelopmentProtocolError(NormalFeatureError):
    """A protected-cohort rule was violated."""


class WeightProvenanceError(NormalFeatureError):
    """Official pretrained-weight provenance could not be established."""


class ModelArtifactError(NormalFeatureError):
    """A fitted-model artifact is invalid, mismatched, or has been altered."""


class CalibrationError(NormalFeatureError):
    """A calibration input or artifact violates the development protocol."""


class DeterminismError(NormalFeatureError):
    """The deterministic CPU reference contract could not be enforced."""
