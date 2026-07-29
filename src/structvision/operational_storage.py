"""Operational storage-mode selection for supported StructVision entry points.

Loading this module or discovering the active mode is write-free.  Supported
entry points use this context instead of independently parsing storage files.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .storage import (
    ConfiguredPath,
    LogicalRoot,
    PathIntent,
    StorageConfig,
    StorageConfigurationError,
    load_storage_config,
)


class OperationalStorageMode(str, Enum):
    """The two supported operational states."""

    NO_CONFIGURATION = "no_configuration"
    EXTERNAL = "external"


@dataclass(frozen=True)
class OperationalStorageContext:
    """Typed storage state shared by every supported operational entry point."""

    mode: OperationalStorageMode
    configuration: StorageConfig | None = None

    def __post_init__(self) -> None:
        mode = (
            self.mode
            if isinstance(self.mode, OperationalStorageMode)
            else OperationalStorageMode(self.mode)
        )
        if mode is OperationalStorageMode.NO_CONFIGURATION:
            if self.configuration is not None:
                raise StorageConfigurationError(
                    "No-configuration mode must not contain a storage configuration"
                )
        else:
            if self.configuration is None:
                raise StorageConfigurationError(
                    "External mode requires a complete storage configuration"
                )
            self.configuration.require_external()
        object.__setattr__(self, "mode", mode)

    @classmethod
    def discover(
        cls, explicit_configuration: Path | None = None
    ) -> "OperationalStorageContext":
        """Discover an explicit override or the preferred local configuration."""

        configuration = load_storage_config(
            explicit_configuration,
            required=explicit_configuration is not None,
        )
        if configuration is None:
            return cls(OperationalStorageMode.NO_CONFIGURATION)
        configuration.require_external()
        return cls(OperationalStorageMode.EXTERNAL, configuration)

    @classmethod
    def no_configuration(cls) -> "OperationalStorageContext":
        return cls(OperationalStorageMode.NO_CONFIGURATION)

    @classmethod
    def external(cls, configuration: StorageConfig) -> "OperationalStorageContext":
        return cls(OperationalStorageMode.EXTERNAL, configuration)

    @property
    def is_external(self) -> bool:
        return self.mode is OperationalStorageMode.EXTERNAL

    def _external_configuration(self) -> StorageConfig:
        if self.configuration is None:
            raise StorageConfigurationError(
                "This operation requires an active external storage configuration"
            )
        self.configuration.require_external()
        return self.configuration

    def authorise(
        self,
        logical_root: LogicalRoot,
        path: Path,
        *,
        intent: PathIntent,
    ) -> ConfiguredPath:
        """Authorise one caller-selected path in external mode."""

        return self._external_configuration().authorise_path(
            logical_root,
            path,
            intent=intent,
        )

    def authorise_private_input(self, path: Path) -> ConfiguredPath:
        return self.authorise(
            LogicalRoot.PRIVATE_DATA,
            path,
            intent=PathIntent.READ,
        )

    def authorise_run_output(self, path: Path) -> ConfiguredPath:
        return self.authorise(
            LogicalRoot.RUNS,
            path,
            intent=PathIntent.WRITE,
        )

    def authorise_release_output(self, path: Path) -> ConfiguredPath:
        return self.authorise(
            LogicalRoot.RELEASE,
            path,
            intent=PathIntent.WRITE,
        )

    def authorise_learned_resource(self, path: Path) -> ConfiguredPath:
        return self.authorise(
            LogicalRoot.LEARNED_ARTIFACT,
            path,
            intent=PathIntent.READ,
        )

    def to_dict(self, *, public: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {"mode": self.mode.value}
        if self.configuration is not None:
            payload["configuration"] = self.configuration.to_dict(public=public)
        return payload
