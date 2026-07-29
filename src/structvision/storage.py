"""Validated, write-free storage configuration for portable StructVision paths.

Loading or importing this module never creates a directory.  Callers must opt in
to either an external configuration or the clearly labelled legacy repository
compatibility state before requesting a runtime path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

try:  # Python 3.11+
    import tomllib
except ImportError:  # pragma: no cover - exercised by the supported Python 3.9 runtime
    import tomli as tomllib


CONFIG_SCHEMA = "org.structvision.storage"
CONFIG_SCHEMA_VERSION = 1
DEFAULT_EXTERNAL_DIRECTORY = "StructVision"
CONFIG_ENVIRONMENT_VARIABLE = "STRUCTVISION_STORAGE_CONFIG"


class StorageConfigurationError(ValueError):
    """A local storage configuration is missing, malformed, or unsafe."""


class StorageConfigurationMissingError(StorageConfigurationError):
    """No explicitly requested storage configuration is available."""


class LogicalRoot(str, Enum):
    SOURCE = "source_root"
    RUNS = "runs_root"
    TRASH = "trash_root"
    PROTECTED = "protected_root"
    REGISTRY = "registry_root"
    RESEARCH_DATA = "research_data_root"
    EXPERIMENT_STORE = "experiment_store_root"
    LEARNED_ARTIFACT = "learned_artifact_root"
    HISTORICAL_REPORT = "historical_report_root"
    ARTIFACT_CACHE = "artifact_cache_root"
    RELEASE = "release_root"
    PRIVATE_DATA = "private_data_root"


class RootAccess(str, Enum):
    READ_ONLY = "read_only"
    WRITABLE = "writable"


class PathIntent(str, Enum):
    READ = "read"
    WRITE = "write"


class MigrationState(str, Enum):
    EXTERNAL = "external"
    LEGACY_REPOSITORY_COMPATIBILITY = "legacy_repository_compatibility"


class LegacyReferenceRole(str, Enum):
    HISTORICAL_REPORT = "historical_report"
    REGISTRY_ANNOTATION = "registry_annotation"


ROOT_ACCESS: Mapping[LogicalRoot, RootAccess] = {
    LogicalRoot.SOURCE: RootAccess.READ_ONLY,
    LogicalRoot.RUNS: RootAccess.WRITABLE,
    LogicalRoot.TRASH: RootAccess.WRITABLE,
    LogicalRoot.PROTECTED: RootAccess.READ_ONLY,
    LogicalRoot.REGISTRY: RootAccess.READ_ONLY,
    LogicalRoot.RESEARCH_DATA: RootAccess.READ_ONLY,
    LogicalRoot.EXPERIMENT_STORE: RootAccess.READ_ONLY,
    LogicalRoot.LEARNED_ARTIFACT: RootAccess.READ_ONLY,
    LogicalRoot.HISTORICAL_REPORT: RootAccess.READ_ONLY,
    LogicalRoot.ARTIFACT_CACHE: RootAccess.WRITABLE,
    LogicalRoot.RELEASE: RootAccess.WRITABLE,
    LogicalRoot.PRIVATE_DATA: RootAccess.WRITABLE,
}

ROLE_TARGET_ROOT: Mapping[LegacyReferenceRole, LogicalRoot] = {
    LegacyReferenceRole.HISTORICAL_REPORT: LogicalRoot.HISTORICAL_REPORT,
    LegacyReferenceRole.REGISTRY_ANNOTATION: LogicalRoot.RESEARCH_DATA,
}

_PROTECTED_CHILDREN = frozenset(
    {
        LogicalRoot.REGISTRY,
        LogicalRoot.RESEARCH_DATA,
        LogicalRoot.EXPERIMENT_STORE,
        LogicalRoot.LEARNED_ARTIFACT,
        LogicalRoot.HISTORICAL_REPORT,
    }
)
_RULE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def preferred_config_path() -> Path:
    """Return the preferred local macOS path without creating it."""

    return Path.home() / "Library" / "Application Support" / "StructVision" / "config.toml"


def default_external_base() -> Path:
    """Return the proposed external base without creating it."""

    return Path.home() / DEFAULT_EXTERNAL_DIRECTORY


def _contains_parent_reference(value: str) -> bool:
    normalised = value.replace("\\", "/")
    return any(part == ".." for part in normalised.split("/"))


def _absolute_path(value: str | os.PathLike[str], *, label: str) -> Path:
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw.strip():
        raise StorageConfigurationError(f"{label} must be a non-empty path")
    if "\x00" in raw:
        raise StorageConfigurationError(f"{label} contains a NUL byte")
    if _contains_parent_reference(raw):
        raise StorageConfigurationError(f"{label} must not contain '..' traversal")
    expanded = Path(raw).expanduser()
    if not expanded.is_absolute():
        raise StorageConfigurationError(f"{label} must be absolute: {raw!r}")
    return expanded


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _existing_symlink(path: Path) -> Path | None:
    """Return the first existing symlink in a raw absolute path."""

    cursor = Path(path.anchor)
    for part in path.parts[1:]:
        cursor = cursor / part
        try:
            if cursor.is_symlink():
                return cursor
        except OSError as error:
            raise StorageConfigurationError(
                f"Could not inspect path component {cursor}: {error}"
            ) from error
    return None


def _validated_absolute(
    value: str | os.PathLike[str],
    *,
    label: str,
    reject_mount_root: bool = False,
) -> Path:
    raw = _absolute_path(value, label=label)
    link = _existing_symlink(raw)
    if link is not None:
        raise StorageConfigurationError(
            f"{label} traverses the symlink {link}; select a physical path"
        )
    resolved = raw.resolve(strict=False)
    if resolved == Path(resolved.anchor):
        raise StorageConfigurationError(f"{label} must not be a filesystem root")
    if resolved == Path.home().resolve(strict=False):
        raise StorageConfigurationError(f"{label} must not be the home directory")
    if reject_mount_root and resolved.exists() and os.path.ismount(resolved):
        raise StorageConfigurationError(f"{label} must not be a mount or volume root")
    return resolved


def _root_name(value: LogicalRoot | str) -> LogicalRoot:
    try:
        return value if isinstance(value, LogicalRoot) else LogicalRoot(value)
    except ValueError as error:
        raise StorageConfigurationError(f"Unknown logical root: {value!r}") from error


def _role_name(value: LegacyReferenceRole | str) -> LegacyReferenceRole:
    try:
        return value if isinstance(value, LegacyReferenceRole) else LegacyReferenceRole(value)
    except ValueError as error:
        raise StorageConfigurationError(
            f"Unknown legacy-reference role: {value!r}"
        ) from error


@dataclass(frozen=True)
class LegacyTranslationRule:
    """One exact, role-scoped historical-prefix translation."""

    role: LegacyReferenceRole
    identity: str
    stored_prefix: Path
    target_root: LogicalRoot
    public_safe: bool = False
    redistribution_allowed: bool = False

    def __post_init__(self) -> None:
        role = _role_name(self.role)
        target = _root_name(self.target_root)
        if not _RULE_ID.fullmatch(self.identity):
            raise StorageConfigurationError(
                "Translation identity must use 1-128 letters, digits, dots, dashes, or underscores"
            )
        prefix = _validated_absolute(
            self.stored_prefix,
            label=f"translation {self.identity} stored_prefix",
        )
        if target is not ROLE_TARGET_ROOT[role]:
            raise StorageConfigurationError(
                f"Role {role.value} may translate only into "
                f"{ROLE_TARGET_ROOT[role].value}, not {target.value}"
            )
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "target_root", target)
        object.__setattr__(self, "stored_prefix", prefix)

    def local_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "stored_prefix": str(self.stored_prefix),
            "target_root": self.target_root.value,
            "public_safe": self.public_safe,
            "redistribution_allowed": self.redistribution_allowed,
        }

    def public_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "target_root": self.target_root.value,
            "stored_prefix": "[redacted]",
            "public_safe": self.public_safe,
            "redistribution_allowed": self.redistribution_allowed,
        }


@dataclass(frozen=True)
class ConfiguredPath:
    """A path accompanied by its logical, non-scientific provenance."""

    logical_root: LogicalRoot
    configuration_identity: str
    path: Path
    public_safe: bool
    redistribution_allowed: bool

    def to_dict(self, *, public: bool = False) -> dict[str, object]:
        return {
            "logical_root_name": self.logical_root.value,
            "configuration_identity": self.configuration_identity,
            "path": "[redacted]" if public else str(self.path),
            "public_safe": self.public_safe,
            "redistribution_allowed": self.redistribution_allowed,
        }


@dataclass(frozen=True)
class StorageConfig:
    """Complete named-root configuration with an explicit migration state."""

    roots: Mapping[LogicalRoot, Path]
    migration_state: MigrationState = MigrationState.EXTERNAL
    translation_rules: tuple[LegacyTranslationRule, ...] = ()
    schema: str = CONFIG_SCHEMA
    schema_version: int = CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != CONFIG_SCHEMA:
            raise StorageConfigurationError(
                f"Configuration identity must be {CONFIG_SCHEMA!r}"
            )
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise StorageConfigurationError(
                f"Unsupported storage schema version {self.schema_version}; "
                f"expected {CONFIG_SCHEMA_VERSION}"
            )
        try:
            state = (
                self.migration_state
                if isinstance(self.migration_state, MigrationState)
                else MigrationState(self.migration_state)
            )
        except ValueError as error:
            raise StorageConfigurationError(
                f"Unknown migration_state: {self.migration_state!r}"
            ) from error

        normalised: dict[LogicalRoot, Path] = {}
        for raw_name, raw_path in self.roots.items():
            name = _root_name(raw_name)
            if name in normalised:
                raise StorageConfigurationError(f"Duplicate logical root: {name.value}")
            normalised[name] = _validated_absolute(
                raw_path,
                label=name.value,
                reject_mount_root=name is not LogicalRoot.SOURCE,
            )
        missing = set(LogicalRoot) - set(normalised)
        extra = set(normalised) - set(LogicalRoot)
        if missing or extra:
            detail = ", ".join(sorted(item.value for item in missing | extra))
            raise StorageConfigurationError(f"Named-root set is incomplete: {detail}")

        rules: list[LegacyTranslationRule] = []
        seen_roles: set[LegacyReferenceRole] = set()
        for rule in self.translation_rules:
            if not isinstance(rule, LegacyTranslationRule):
                raise StorageConfigurationError("translation_rules must be typed rules")
            if rule.role in seen_roles:
                raise StorageConfigurationError(
                    f"Only one translation rule is allowed for {rule.role.value}"
                )
            seen_roles.add(rule.role)
            rules.append(rule)

        object.__setattr__(self, "migration_state", state)
        object.__setattr__(self, "roots", normalised)
        object.__setattr__(self, "translation_rules", tuple(rules))
        if state is MigrationState.EXTERNAL:
            self._validate_external_layout()
        else:
            self._validate_legacy_layout()

    def _validate_external_layout(self) -> None:
        source = self.roots[LogicalRoot.SOURCE]
        for name, path in self.roots.items():
            if name is LogicalRoot.SOURCE:
                continue
            if _is_relative_to(path, source) or _is_relative_to(source, path):
                raise StorageConfigurationError(
                    f"{name.value} must be separate from the Git source root and its ancestors"
                )

        names = list(LogicalRoot)
        for index, left_name in enumerate(names):
            if left_name is LogicalRoot.SOURCE:
                continue
            left = self.roots[left_name]
            for right_name in names[index + 1 :]:
                if right_name is LogicalRoot.SOURCE:
                    continue
                right = self.roots[right_name]
                related = _is_relative_to(left, right) or _is_relative_to(right, left)
                if not related:
                    continue
                if left == right:
                    raise StorageConfigurationError(
                        f"Conflicting overlapping roots: {left_name.value} and "
                        f"{right_name.value}"
                    )
                permitted = (
                    left_name is LogicalRoot.PROTECTED and right_name in _PROTECTED_CHILDREN
                ) or (
                    right_name is LogicalRoot.PROTECTED and left_name in _PROTECTED_CHILDREN
                )
                if not permitted:
                    raise StorageConfigurationError(
                        f"Conflicting overlapping roots: {left_name.value} and "
                        f"{right_name.value}"
                    )

    def _validate_legacy_layout(self) -> None:
        source = self.roots[LogicalRoot.SOURCE]
        for name, path in self.roots.items():
            if not _is_relative_to(path, source):
                raise StorageConfigurationError(
                    f"Legacy compatibility root {name.value} must remain under source_root"
                )

    @property
    def identity(self) -> str:
        encoded = json.dumps(
            self.to_dict(public=False),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    @property
    def source_root(self) -> Path:
        return self.roots[LogicalRoot.SOURCE]

    def root(self, name: LogicalRoot | str) -> Path:
        return self.roots[_root_name(name)]

    def rule(self, role: LegacyReferenceRole | str) -> LegacyTranslationRule | None:
        wanted = _role_name(role)
        return next((rule for rule in self.translation_rules if rule.role is wanted), None)

    def configured_path(
        self,
        name: LogicalRoot | str,
        relative: str | os.PathLike[str] = "",
        *,
        intent: PathIntent | str = PathIntent.READ,
        allow_legacy_write: bool = False,
    ) -> ConfiguredPath:
        logical = _root_name(name)
        try:
            operation = intent if isinstance(intent, PathIntent) else PathIntent(intent)
        except ValueError as error:
            raise StorageConfigurationError(f"Unknown path intent: {intent!r}") from error
        if operation is PathIntent.WRITE and ROOT_ACCESS[logical] is RootAccess.READ_ONLY:
            raise StorageConfigurationError(f"{logical.value} is read-only")
        if (
            operation is PathIntent.WRITE
            and self.migration_state is MigrationState.LEGACY_REPOSITORY_COMPATIBILITY
            and not allow_legacy_write
        ):
            raise StorageConfigurationError(
                "Repository-local writes require allow_legacy_write=True in the "
                "explicit legacy compatibility state"
            )

        raw_relative = os.fspath(relative)
        if "\x00" in raw_relative or _contains_parent_reference(raw_relative):
            raise StorageConfigurationError("Relative path contains unsafe traversal")
        relative_path = Path(raw_relative)
        if relative_path.is_absolute():
            raise StorageConfigurationError("Configured relative path must not be absolute")
        target = self.roots[logical] / relative_path
        link = _existing_symlink(target)
        if link is not None:
            raise StorageConfigurationError(
                f"Configured path traverses the symlink {link}"
            )
        resolved = target.resolve(strict=False)
        if not _is_relative_to(resolved, self.roots[logical]):
            raise StorageConfigurationError(
                f"Configured path escapes {logical.value}"
            )
        public_safe = logical is LogicalRoot.RELEASE
        return ConfiguredPath(
            logical_root=logical,
            configuration_identity=self.identity,
            path=resolved,
            public_safe=public_safe,
            redistribution_allowed=public_safe,
        )

    def authorise_path(
        self,
        name: LogicalRoot | str,
        path: str | os.PathLike[str],
        *,
        intent: PathIntent | str = PathIntent.READ,
        allow_legacy_write: bool = False,
    ) -> ConfiguredPath:
        """Validate one caller-selected absolute path against a named root."""

        logical = _root_name(name)
        raw = _absolute_path(path, label=f"path for {logical.value}")
        link = _existing_symlink(raw)
        if link is not None:
            raise StorageConfigurationError(
                f"Path for {logical.value} traverses the symlink {link}"
            )
        resolved = raw.resolve(strict=False)
        root = self.roots[logical]
        if not _is_relative_to(resolved, root):
            raise StorageConfigurationError(
                f"Path is outside configured {logical.value}: {resolved}"
            )
        return self.configured_path(
            logical,
            resolved.relative_to(root),
            intent=intent,
            allow_legacy_write=allow_legacy_write,
        )

    def require_external(self) -> None:
        if self.migration_state is not MigrationState.EXTERNAL:
            raise StorageConfigurationError(
                "This operation requires external migration_state; "
                "legacy repository compatibility is not accepted"
            )

    def future_run_path(self, run_name: str) -> ConfiguredPath:
        """Return an external run location without creating it."""

        self.require_external()
        if not _RULE_ID.fullmatch(run_name):
            raise StorageConfigurationError(
                "Run name must use 1-128 letters, digits, dots, dashes, or underscores"
            )
        return self.configured_path(LogicalRoot.RUNS, run_name, intent=PathIntent.WRITE)

    def to_dict(self, *, public: bool = False) -> dict[str, object]:
        roots: dict[str, object] = {}
        for name in LogicalRoot:
            if public and name is LogicalRoot.PRIVATE_DATA:
                continue
            roots[name.value] = (
                {
                    "access": ROOT_ACCESS[name].value,
                    "path": "[redacted]",
                }
                if public
                else str(self.roots[name])
            )
        translations = {
            rule.role.value: (
                rule.public_dict() if public else rule.local_dict()
            )
            for rule in sorted(self.translation_rules, key=lambda item: item.role.value)
        }
        result: dict[str, object] = {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "migration_state": self.migration_state.value,
            "roots": roots,
            "translations": translations,
        }
        if public:
            result["configuration_identity"] = self.identity
            result["redacted_roots"] = [LogicalRoot.PRIVATE_DATA.value]
        return result

    def to_json(self, *, public: bool = True) -> str:
        return json.dumps(
            self.to_dict(public=public),
            indent=2,
            sort_keys=True,
        ) + "\n"

    def to_toml(self) -> str:
        """Serialise a local-only configuration deterministically."""

        lines = [
            f"schema = {json.dumps(self.schema)}",
            f"schema_version = {self.schema_version}",
            f"migration_state = {json.dumps(self.migration_state.value)}",
            "",
            "[roots]",
        ]
        for name in LogicalRoot:
            lines.append(f"{name.value} = {json.dumps(str(self.roots[name]))}")
        for rule in sorted(self.translation_rules, key=lambda item: item.role.value):
            lines.extend(
                [
                    "",
                    f"[translations.{rule.role.value}]",
                    f"identity = {json.dumps(rule.identity)}",
                    f"stored_prefix = {json.dumps(str(rule.stored_prefix))}",
                    f"target_root = {json.dumps(rule.target_root.value)}",
                    f"public_safe = {'true' if rule.public_safe else 'false'}",
                    (
                        "redistribution_allowed = "
                        f"{'true' if rule.redistribution_allowed else 'false'}"
                    ),
                ]
            )
        return "\n".join(lines) + "\n"

    @classmethod
    def proposed_external(
        cls,
        *,
        source_root: Path,
        external_base: Path | None = None,
        translation_rules: tuple[LegacyTranslationRule, ...] = (),
    ) -> "StorageConfig":
        source = _validated_absolute(source_root, label=LogicalRoot.SOURCE.value)
        base = _validated_absolute(
            external_base or default_external_base(),
            label="external_base",
            reject_mount_root=True,
        )
        protected = base / "Protected"
        return cls(
            roots={
                LogicalRoot.SOURCE: source,
                LogicalRoot.RUNS: base / "Runs",
                LogicalRoot.TRASH: base / "Trash",
                LogicalRoot.PROTECTED: protected,
                LogicalRoot.REGISTRY: protected / "Registry",
                LogicalRoot.RESEARCH_DATA: protected / "ResearchData",
                LogicalRoot.EXPERIMENT_STORE: protected / "ExperimentStores",
                LogicalRoot.LEARNED_ARTIFACT: protected / "LearnedArtifacts",
                LogicalRoot.HISTORICAL_REPORT: protected / "HistoricalReports",
                LogicalRoot.ARTIFACT_CACHE: base / "Cache",
                LogicalRoot.RELEASE: base / "Releases",
                LogicalRoot.PRIVATE_DATA: base / "PrivateData",
            },
            migration_state=MigrationState.EXTERNAL,
            translation_rules=translation_rules,
        )

    @classmethod
    def legacy_repository_compatibility(
        cls,
        source_root: Path,
        *,
        translation_rules: tuple[LegacyTranslationRule, ...] = (),
    ) -> "StorageConfig":
        """Construct the old layout explicitly; writes remain opt-in per call."""

        source = _validated_absolute(source_root, label=LogicalRoot.SOURCE.value)
        return cls(
            roots={
                LogicalRoot.SOURCE: source,
                LogicalRoot.RUNS: source / "outputs",
                LogicalRoot.TRASH: source / "research_data" / ".trash",
                LogicalRoot.PROTECTED: source,
                LogicalRoot.REGISTRY: source / "research_data" / "registry",
                LogicalRoot.RESEARCH_DATA: source / "research_data",
                LogicalRoot.EXPERIMENT_STORE: source / "outputs",
                LogicalRoot.LEARNED_ARTIFACT: source / "outputs",
                LogicalRoot.HISTORICAL_REPORT: source / "outputs",
                LogicalRoot.ARTIFACT_CACHE: source / ".cache" / "structvision",
                LogicalRoot.RELEASE: source / "dist",
                LogicalRoot.PRIVATE_DATA: source / "private_data",
            },
            migration_state=MigrationState.LEGACY_REPOSITORY_COMPATIBILITY,
            translation_rules=translation_rules,
        )


def discover_source_root(explicit: Path | None = None) -> Path:
    """Discover the checkout/package source without assuming a user directory."""

    if explicit is not None:
        candidate = _validated_absolute(explicit, label=LogicalRoot.SOURCE.value)
        if not candidate.is_dir():
            raise StorageConfigurationError(f"source_root is not a directory: {candidate}")
        return candidate
    module = Path(__file__).absolute()
    for candidate in module.parents:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "structvision"
        ).is_dir():
            return _validated_absolute(candidate, label=LogicalRoot.SOURCE.value)
    package_parent = module.parent.parent
    return _validated_absolute(package_parent, label=LogicalRoot.SOURCE.value)


def _parse_rule(
    role: LegacyReferenceRole,
    raw: object,
) -> LegacyTranslationRule:
    if not isinstance(raw, Mapping):
        raise StorageConfigurationError(
            f"translations.{role.value} must be a TOML table"
        )
    expected = {
        "identity",
        "stored_prefix",
        "target_root",
        "public_safe",
        "redistribution_allowed",
    }
    unexpected = set(raw) - expected
    missing = {"identity", "stored_prefix", "target_root"} - set(raw)
    if unexpected or missing:
        detail = ", ".join(sorted(unexpected | missing))
        raise StorageConfigurationError(
            f"Invalid translations.{role.value} fields: {detail}"
        )
    for key in ("identity", "stored_prefix", "target_root"):
        if not isinstance(raw[key], str):
            raise StorageConfigurationError(
                f"translations.{role.value}.{key} must be a string"
            )
    for key in ("public_safe", "redistribution_allowed"):
        if key in raw and type(raw[key]) is not bool:
            raise StorageConfigurationError(
                f"translations.{role.value}.{key} must be boolean"
            )
    try:
        return LegacyTranslationRule(
            role=role,
            identity=str(raw["identity"]),
            stored_prefix=Path(str(raw["stored_prefix"])),
            target_root=_root_name(str(raw["target_root"])),
            public_safe=bool(raw.get("public_safe", False)),
            redistribution_allowed=bool(raw.get("redistribution_allowed", False)),
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, StorageConfigurationError):
            raise
        raise StorageConfigurationError(
            f"Invalid translations.{role.value}: {error}"
        ) from error


def storage_config_from_mapping(raw: Mapping[str, Any]) -> StorageConfig:
    expected = {"schema", "schema_version", "migration_state", "roots", "translations"}
    unexpected = set(raw) - expected
    missing = {"schema", "schema_version", "migration_state", "roots"} - set(raw)
    if unexpected or missing:
        detail = ", ".join(sorted(unexpected | missing))
        raise StorageConfigurationError(f"Invalid top-level configuration fields: {detail}")
    roots_raw = raw["roots"]
    if not isinstance(roots_raw, Mapping):
        raise StorageConfigurationError("roots must be a TOML table")
    expected_roots = {item.value for item in LogicalRoot}
    if set(roots_raw) != expected_roots:
        detail = ", ".join(sorted(set(roots_raw) ^ expected_roots))
        raise StorageConfigurationError(f"Named-root set is incomplete or unknown: {detail}")
    if not isinstance(raw["schema"], str):
        raise StorageConfigurationError("schema must be a string")
    if type(raw["schema_version"]) is not int:
        raise StorageConfigurationError("schema_version must be an integer")
    if not isinstance(raw["migration_state"], str):
        raise StorageConfigurationError("migration_state must be a string")
    non_string_roots = sorted(
        name for name, value in roots_raw.items() if not isinstance(value, str)
    )
    if non_string_roots:
        raise StorageConfigurationError(
            "Root values must be strings: " + ", ".join(non_string_roots)
        )
    translations_raw = raw.get("translations", {})
    if not isinstance(translations_raw, Mapping):
        raise StorageConfigurationError("translations must be a TOML table")
    unknown_roles = set(translations_raw) - {item.value for item in LegacyReferenceRole}
    if unknown_roles:
        raise StorageConfigurationError(
            "Unknown translation roles: " + ", ".join(sorted(unknown_roles))
        )
    rules = tuple(
        _parse_rule(role, translations_raw[role.value])
        for role in LegacyReferenceRole
        if role.value in translations_raw
    )
    try:
        state = MigrationState(str(raw["migration_state"]))
    except ValueError as error:
        raise StorageConfigurationError(
            f"Unknown migration_state: {raw['migration_state']!r}"
        ) from error
    return StorageConfig(
        roots={
            LogicalRoot(name): Path(str(value))
            for name, value in roots_raw.items()
        },
        migration_state=state,
        translation_rules=rules,
        schema=str(raw["schema"]),
        schema_version=int(raw["schema_version"]),
    )


def load_storage_config(
    path: Path | None = None,
    *,
    required: bool = False,
) -> StorageConfig | None:
    """Load a local TOML file without creating or modifying any path."""

    selected = path
    if selected is None:
        configured = os.environ.get(CONFIG_ENVIRONMENT_VARIABLE)
        selected = Path(configured).expanduser() if configured else preferred_config_path()
    selected = _absolute_path(selected, label="configuration path")
    link = _existing_symlink(selected)
    if link is not None:
        raise StorageConfigurationError(
            f"Configuration path traverses the symlink {link}"
        )
    if not selected.is_file():
        if required or path is not None or os.environ.get(CONFIG_ENVIRONMENT_VARIABLE):
            raise StorageConfigurationMissingError(
                f"Storage configuration does not exist: {selected}"
            )
        return None
    try:
        payload = tomllib.loads(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise StorageConfigurationError(
            f"Could not load storage configuration {selected}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise StorageConfigurationError("Storage configuration must be a TOML table")
    return storage_config_from_mapping(payload)


def load_external_storage_config(path: Path | None = None) -> StorageConfig:
    config = load_storage_config(path, required=True)
    assert config is not None
    config.require_external()
    return config
