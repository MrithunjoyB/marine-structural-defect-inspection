"""Explicit failure types for the reusable StructVision API."""

from __future__ import annotations


class StructVisionError(Exception):
    """Base class for reusable API failures."""


class InvalidInputTypeError(StructVisionError, TypeError):
    """The supplied image object is not a supported input type."""


class AmbiguousColourSpaceError(StructVisionError, ValueError):
    """A channel layout cannot be interpreted without an explicit declaration."""


class CorruptImageError(StructVisionError, ValueError):
    """An image path exists but cannot be decoded as an image."""


class UnsupportedChannelLayoutError(StructVisionError, ValueError):
    """An array has an unsupported number or arrangement of channels."""


class InvalidConfigurationError(StructVisionError, ValueError):
    """Detector configuration is incomplete, non-finite, or out of range."""


class SpecificationMismatchError(StructVisionError, ValueError):
    """A v2 specification or executed configuration does not match its hash."""


class ProvenanceMismatchError(StructVisionError, ValueError):
    """Input, ground-truth, implementation, or runtime provenance differs."""


class DuplicateImageIDError(StructVisionError, ValueError):
    """A batch or experiment contains a repeated image identity."""


class DuplicateResultIdentityError(StructVisionError, ValueError):
    """A sink was asked to append a result identity it already contains."""


class SinkError(StructVisionError, RuntimeError):
    """An explicitly injected sink failed while accepting a record."""


class DeterministicModeError(StructVisionError, RuntimeError):
    """The requested deterministic execution contract cannot be honoured."""


class ExperimentExecutionError(StructVisionError, RuntimeError):
    """A v2 image-method execution failed."""
