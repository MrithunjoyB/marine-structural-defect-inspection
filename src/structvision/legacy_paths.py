"""Read-only, role-scoped resolution of immutable stored path references."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path

from .storage import (
    ConfiguredPath,
    LegacyReferenceRole,
    LegacyTranslationRule,
    LogicalRoot,
    StorageConfig,
    StorageConfigurationError,
)


class ResolutionStatus(str, Enum):
    DIRECT = "direct"
    TRANSLATED = "translated"
    UNAVAILABLE = "unavailable"
    REFUSED = "refused"


@dataclass(frozen=True)
class PathResolution:
    """Resolution result; it does not attest validity of referenced content."""

    logical_root_name: LogicalRoot
    configuration_identity: str
    original_stored_path: str
    resolved_path: Path | None
    role: LegacyReferenceRole
    status: ResolutionStatus
    translation_rule_identity: str | None
    public_safe: bool
    redistribution_allowed: bool
    reason: str
    scientific_validity_claimed: bool = False

    @property
    def available(self) -> bool:
        return self.status in {ResolutionStatus.DIRECT, ResolutionStatus.TRANSLATED}

    def to_dict(self, *, public: bool = False) -> dict[str, object]:
        return {
            "logical_root_name": self.logical_root_name.value,
            "configuration_identity": self.configuration_identity,
            "original_stored_path": (
                "[redacted]" if public else self.original_stored_path
            ),
            "resolved_path": (
                "[redacted]"
                if public and self.resolved_path is not None
                else str(self.resolved_path)
                if self.resolved_path is not None
                else None
            ),
            "role": self.role.value,
            "status": self.status.value,
            "translation_rule_identity": self.translation_rule_identity,
            "public_safe": self.public_safe,
            "redistribution_allowed": self.redistribution_allowed,
            "reason": self.reason,
            "scientific_validity_claimed": False,
        }


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _unsafe_raw_path(value: str) -> str | None:
    if not value:
        return "stored path is empty"
    if "\x00" in value:
        return "stored path contains a NUL byte"
    if any(part == ".." for part in value.replace("\\", "/").split("/")):
        return "stored path contains '..' traversal"
    if not Path(value).is_absolute():
        return "stored path must be absolute"
    return None


def _symlink_component(path: Path) -> Path | None:
    cursor = Path(path.anchor)
    for part in path.parts[1:]:
        cursor = cursor / part
        try:
            if cursor.is_symlink():
                return cursor
        except OSError:
            return cursor
    return None


class LegacyPathResolver:
    """Resolve only configured historical-report and annotation references."""

    def __init__(self, configuration: StorageConfig):
        self.configuration = configuration

    def _result(
        self,
        *,
        stored: str,
        role: LegacyReferenceRole,
        status: ResolutionStatus,
        resolved: Path | None,
        rule: LegacyTranslationRule | None,
        reason: str,
    ) -> PathResolution:
        target = (
            rule.target_root if rule is not None else {
                LegacyReferenceRole.HISTORICAL_REPORT: LogicalRoot.HISTORICAL_REPORT,
                LegacyReferenceRole.REGISTRY_ANNOTATION: LogicalRoot.RESEARCH_DATA,
            }[role]
        )
        return PathResolution(
            logical_root_name=target,
            configuration_identity=self.configuration.identity,
            original_stored_path=stored,
            resolved_path=resolved,
            role=role,
            status=status,
            translation_rule_identity=rule.identity if rule is not None else None,
            public_safe=bool(rule.public_safe) if rule is not None else False,
            redistribution_allowed=(
                bool(rule.redistribution_allowed) if rule is not None else False
            ),
            reason=reason,
        )

    def resolve(
        self,
        stored_path: str | os.PathLike[str],
        role: LegacyReferenceRole | str,
    ) -> PathResolution:
        try:
            selected_role = (
                role if isinstance(role, LegacyReferenceRole) else LegacyReferenceRole(role)
            )
        except ValueError as error:
            raise StorageConfigurationError(
                f"Unknown legacy-reference role: {role!r}"
            ) from error
        stored = os.fspath(stored_path)
        if not isinstance(stored, str):
            stored = str(stored)
        rule = self.configuration.rule(selected_role)
        issue = _unsafe_raw_path(stored)
        if issue is not None:
            return self._result(
                stored=stored,
                role=selected_role,
                status=ResolutionStatus.REFUSED,
                resolved=None,
                rule=rule,
                reason=issue,
            )

        raw = Path(stored)
        target_root = self.configuration.root(
            {
                LegacyReferenceRole.HISTORICAL_REPORT: LogicalRoot.HISTORICAL_REPORT,
                LegacyReferenceRole.REGISTRY_ANNOTATION: LogicalRoot.RESEARCH_DATA,
            }[selected_role]
        )
        permitted_direct_roots = [target_root]
        if rule is not None:
            permitted_direct_roots.append(rule.stored_prefix)

        matching_other = next(
            (
                candidate
                for candidate in self.configuration.translation_rules
                if candidate.role is not selected_role
                and _is_relative_to(raw, candidate.stored_prefix)
            ),
            None,
        )
        if matching_other is not None:
            return self._result(
                stored=stored,
                role=selected_role,
                status=ResolutionStatus.REFUSED,
                resolved=None,
                rule=rule,
                reason=(
                    f"stored path belongs to {matching_other.role.value}, not "
                    f"{selected_role.value}"
                ),
            )

        direct_root = next(
            (prefix for prefix in permitted_direct_roots if _is_relative_to(raw, prefix)),
            None,
        )
        if direct_root is not None:
            link = _symlink_component(raw)
            if link is not None:
                return self._result(
                    stored=stored,
                    role=selected_role,
                    status=ResolutionStatus.REFUSED,
                    resolved=None,
                    rule=rule,
                    reason=f"path traverses a symlink: {link}",
                )
        if raw.exists():
            if direct_root is None:
                return self._result(
                    stored=stored,
                    role=selected_role,
                    status=ResolutionStatus.REFUSED,
                    resolved=None,
                    rule=rule,
                    reason="existing path is outside every approved prefix for this role",
                )
            resolved = raw.resolve(strict=False)
            if not _is_relative_to(resolved, direct_root):
                return self._result(
                    stored=stored,
                    role=selected_role,
                    status=ResolutionStatus.REFUSED,
                    resolved=None,
                    rule=rule,
                    reason="resolved path escapes its approved direct prefix",
                )
            if not raw.is_file():
                return self._result(
                    stored=stored,
                    role=selected_role,
                    status=ResolutionStatus.REFUSED,
                    resolved=None,
                    rule=rule,
                    reason="resolved reference is not a regular file",
                )
            return self._result(
                stored=stored,
                role=selected_role,
                status=ResolutionStatus.DIRECT,
                resolved=resolved,
                rule=rule if direct_root == getattr(rule, "stored_prefix", None) else None,
                reason="existing regular file is inside an approved role-specific prefix",
            )

        if _is_relative_to(raw, target_root):
            return self._result(
                stored=stored,
                role=selected_role,
                status=ResolutionStatus.UNAVAILABLE,
                resolved=raw,
                rule=None,
                reason="configured direct target is not available",
            )
        if rule is None or not _is_relative_to(raw, rule.stored_prefix):
            return self._result(
                stored=stored,
                role=selected_role,
                status=ResolutionStatus.REFUSED,
                resolved=None,
                rule=rule,
                reason="stored absolute prefix is not approved for this role",
            )

        suffix = raw.relative_to(rule.stored_prefix)
        try:
            configured: ConfiguredPath = self.configuration.configured_path(
                rule.target_root,
                suffix,
            )
        except StorageConfigurationError as error:
            return self._result(
                stored=stored,
                role=selected_role,
                status=ResolutionStatus.REFUSED,
                resolved=None,
                rule=rule,
                reason=f"translated path was refused: {error}",
            )
        translated = configured.path
        link = _symlink_component(translated)
        if link is not None:
            return self._result(
                stored=stored,
                role=selected_role,
                status=ResolutionStatus.REFUSED,
                resolved=None,
                rule=rule,
                reason=f"translated path traverses a symlink: {link}",
            )
        if not translated.exists():
            return self._result(
                stored=stored,
                role=selected_role,
                status=ResolutionStatus.UNAVAILABLE,
                resolved=translated,
                rule=rule,
                reason="approved translated target is not available",
            )
        if not translated.is_file():
            return self._result(
                stored=stored,
                role=selected_role,
                status=ResolutionStatus.REFUSED,
                resolved=None,
                rule=rule,
                reason="translated reference is not a regular file",
            )
        return self._result(
            stored=stored,
            role=selected_role,
            status=ResolutionStatus.TRANSLATED,
            resolved=translated,
            rule=rule,
            reason="approved role-specific prefix translated to an existing regular file",
        )

    def resolve_historical_report(
        self, stored_path: str | os.PathLike[str]
    ) -> PathResolution:
        return self.resolve(stored_path, LegacyReferenceRole.HISTORICAL_REPORT)

    def resolve_registry_annotation(
        self, stored_path: str | os.PathLike[str]
    ) -> PathResolution:
        return self.resolve(stored_path, LegacyReferenceRole.REGISTRY_ANNOTATION)
