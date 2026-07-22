"""Fail-closed errors for proposal-guided hybrid development."""


class HybridProtocolError(RuntimeError):
    """The protected hybrid-development data contract was violated."""


class HybridFeatureError(RuntimeError):
    """Candidate evidence was invalid or misaligned."""


class HybridFusionError(RuntimeError):
    """Fusion fitting, selection, or replay violated the frozen contract."""


class HybridExperimentError(RuntimeError):
    """The one-shot development experiment could not be executed safely."""
