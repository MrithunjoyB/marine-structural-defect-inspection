"""Deterministic, hash-verified discovery of protected operational resources."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import stat
from typing import Iterable

from .operational_storage import OperationalStorageContext
from .storage import (
    LogicalRoot,
    PathIntent,
    ResourceBinding,
    ResourceRole,
    StorageConfigurationError,
)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ProtectedResource:
    """A verified private resource without any redistribution entitlement."""

    role: ResourceRole
    path: Path
    sha256: str
    configuration_identity: str
    redistribution_allowed: bool = False

    def to_dict(self, *, public: bool = True) -> dict[str, object]:
        return {
            "role": self.role.value,
            "path": "[redacted]" if public else str(self.path),
            "sha256": self.sha256,
            "configuration_identity": self.configuration_identity,
            "redistribution_allowed": False,
        }


class ProtectedResourceCatalog:
    """Resolve only explicit typed bindings from one external configuration."""

    def __init__(
        self,
        context: OperationalStorageContext,
        *,
        required_roles: Iterable[ResourceRole] = (),
    ):
        if not context.is_external or context.configuration is None:
            raise StorageConfigurationError(
                "Protected resource discovery requires external storage mode"
            )
        self.context = context
        self.configuration = context.configuration
        required = tuple(ResourceRole(role) for role in required_roles)
        missing = [
            role.value
            for role in required
            if self.configuration.resource_binding(role) is None
        ]
        if missing:
            raise StorageConfigurationError(
                "Missing required protected-resource bindings: "
                + ", ".join(sorted(missing))
            )

    def binding(self, role: ResourceRole) -> ResourceBinding | None:
        return self.configuration.resource_binding(role)

    def resolve(self, role: ResourceRole) -> ProtectedResource:
        selected_role = ResourceRole(role)
        binding = self.binding(selected_role)
        if binding is None:
            raise StorageConfigurationError(
                f"Protected resource is not configured: {selected_role.value}"
            )
        configured = self.configuration.configured_path(
            binding.logical_root,
            binding.relative_path,
            intent=PathIntent.READ,
        )
        try:
            metadata = configured.path.lstat()
        except OSError as error:
            raise StorageConfigurationError(
                f"Protected resource is unavailable: {selected_role.value}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise StorageConfigurationError(
                f"Protected resource is not a physical regular file: {selected_role.value}"
            )
        observed = _sha256_file(configured.path)
        if observed != binding.expected_sha256:
            raise StorageConfigurationError(
                f"Protected resource hash mismatch: {selected_role.value}"
            )
        return ProtectedResource(
            role=selected_role,
            path=configured.path,
            sha256=observed,
            configuration_identity=self.configuration.identity,
        )

    def resolve_optional(self, role: ResourceRole) -> ProtectedResource | None:
        return None if self.binding(role) is None else self.resolve(role)

    def resolve_selected(
        self,
        role: ResourceRole,
        selected_path: Path,
    ) -> ProtectedResource:
        """Verify that a caller selection is the bound resource for its role."""

        selected_role = ResourceRole(role)
        authorised = self.configuration.authorise_path(
            LogicalRoot.LEARNED_ARTIFACT,
            selected_path,
            intent=PathIntent.READ,
        )
        resource = self.resolve(selected_role)
        if authorised.path != resource.path:
            raise StorageConfigurationError(
                f"Selected path does not match bound resource: {selected_role.value}"
            )
        return resource
