"""Side-effect-free readers for externally stored protected evidence."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
import sqlite3
import stat
from typing import Iterable

from .legacy_paths import LegacyPathResolver, PathResolution
from .operational_storage import OperationalStorageContext
from .resources import ProtectedResourceCatalog
from .storage import LogicalRoot, ResourceRole, StorageConfigurationError


class ProtectedWriteRefusedError(PermissionError):
    """A caller attempted to mutate immutable protected evidence."""


def _physical_directory(path: Path, *, label: str) -> Path:
    selected = Path(path)
    if not selected.is_absolute():
        raise StorageConfigurationError(f"{label} must be absolute")
    cursor = Path(selected.anchor)
    for part in selected.parts[1:]:
        cursor = cursor / part
        try:
            metadata = cursor.lstat()
        except OSError as error:
            raise StorageConfigurationError(f"{label} is unavailable") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise StorageConfigurationError(f"{label} traverses a symlink")
    if not selected.is_dir():
        raise StorageConfigurationError(f"{label} is not a directory")
    return selected.resolve(strict=True)


def _physical_file(path: Path, *, label: str) -> Path:
    selected = Path(path)
    try:
        metadata = selected.lstat()
    except OSError as error:
        raise StorageConfigurationError(f"{label} is unavailable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise StorageConfigurationError(f"{label} is not a physical regular file")
    return selected.resolve(strict=True)


def _connect_read_only(path: Path) -> sqlite3.Connection:
    selected = _physical_file(path, label="protected SQLite database")
    connection = sqlite3.connect(
        selected.as_uri() + "?mode=ro&immutable=1",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    return connection


def _rows(connection: sqlite3.Connection, query: str, values: Iterable[object] = ()):
    return tuple(dict(row) for row in connection.execute(query, tuple(values)).fetchall())


class ReadOnlyRegistry:
    """Read registry metadata while keeping registry and data roots separate."""

    def __init__(
        self,
        registry_root: Path,
        research_data_root: Path,
        *,
        path_resolver: LegacyPathResolver | None = None,
    ):
        self.registry_root = _physical_directory(
            registry_root,
            label="registry_root",
        )
        self.research_data_root = _physical_directory(
            research_data_root,
            label="research_data_root",
        )
        self.database_path = _physical_file(
            self.registry_root / "datasets.sqlite",
            label="registry database",
        )
        self.manifest_path = _physical_file(
            self.registry_root / "dataset_manifest.json",
            label="registry manifest",
        )
        self.path_resolver = path_resolver

    @classmethod
    def from_operational_storage(
        cls,
        context: OperationalStorageContext,
        *,
        path_resolver: LegacyPathResolver | None = None,
    ) -> "ReadOnlyRegistry":
        if not context.is_external or context.configuration is None:
            raise StorageConfigurationError(
                "Protected registry access requires external storage mode"
            )
        catalog = ProtectedResourceCatalog(
            context,
            required_roles=(
                ResourceRole.REGISTRY_DATABASE,
                ResourceRole.REGISTRY_MANIFEST,
            ),
        )
        database = catalog.resolve(ResourceRole.REGISTRY_DATABASE)
        manifest = catalog.resolve(ResourceRole.REGISTRY_MANIFEST)
        registry_root = context.configuration.root(LogicalRoot.REGISTRY)
        instance = cls.__new__(cls)
        instance.registry_root = _physical_directory(
            registry_root,
            label="registry_root",
        )
        instance.research_data_root = _physical_directory(
            context.configuration.root(LogicalRoot.RESEARCH_DATA),
            label="research_data_root",
        )
        instance.database_path = database.path
        instance.manifest_path = manifest.path
        instance.path_resolver = path_resolver
        return instance

    def _query(self, query: str, values: Iterable[object] = ()):
        with closing(_connect_read_only(self.database_path)) as connection:
            return _rows(connection, query, values)

    def datasets(self, dataset_id: str | None = None) -> tuple[dict[str, object], ...]:
        if dataset_id is None:
            return self._query(
                "SELECT * FROM datasets ORDER BY registered_timestamp, dataset_id"
            )
        return self._query(
            "SELECT * FROM datasets WHERE dataset_id=? "
            "ORDER BY registered_timestamp, dataset_version",
            (dataset_id,),
        )

    def images(self, dataset_id: str | None = None) -> tuple[dict[str, object], ...]:
        if dataset_id is None:
            return self._query("SELECT * FROM images ORDER BY image_id")
        return self._query(
            "SELECT * FROM images WHERE dataset_id=? ORDER BY image_id",
            (dataset_id,),
        )

    def plans(self, plan_id: str | None = None) -> tuple[dict[str, object], ...]:
        if plan_id is None:
            return self._query("SELECT * FROM experiment_plans ORDER BY plan_id")
        return self._query(
            "SELECT * FROM experiment_plans WHERE plan_id=? ORDER BY plan_id",
            (plan_id,),
        )

    def metadata(self, dataset_id: str, version: str | None = None) -> dict[str, object]:
        rows = (
            self._query(
                "SELECT * FROM datasets WHERE dataset_id=? AND dataset_version=? "
                "ORDER BY registered_timestamp DESC LIMIT 1",
                (dataset_id, version),
            )
            if version is not None
            else self._query(
                "SELECT * FROM datasets WHERE dataset_id=? "
                "ORDER BY registered_timestamp DESC LIMIT 1",
                (dataset_id,),
            )
        )
        if not rows:
            raise KeyError(dataset_id)
        return rows[0]

    def registered_image_path(self, image: dict[str, object]) -> Path:
        dataset_id = str(image["dataset_id"])
        stored_filename = str(image["stored_filename"])
        if not dataset_id or not stored_filename:
            raise StorageConfigurationError("Registered image metadata is incomplete")
        relative = Path("raw") / dataset_id / stored_filename
        if relative.is_absolute() or ".." in relative.parts:
            raise StorageConfigurationError("Registered image path contains traversal")
        target = self.research_data_root / relative
        resolved = _physical_file(target, label="registered image")
        try:
            resolved.relative_to(self.research_data_root)
        except ValueError as error:
            raise StorageConfigurationError(
                "Registered image escapes research_data_root"
            ) from error
        return resolved

    def annotation_resolution(
        self, image: dict[str, object]
    ) -> PathResolution | None:
        stored = str(image.get("annotation_path") or "")
        if not stored:
            return None
        if self.path_resolver is None:
            raise StorageConfigurationError(
                "Registry annotation resolution requires a typed LegacyPathResolver"
            )
        return self.path_resolver.resolve_registry_annotation(stored)

    @staticmethod
    def reject_write(*_args, **_kwargs) -> None:
        raise ProtectedWriteRefusedError("Protected registry access is read-only")


_STORE_TABLE = {
    ResourceRole.HISTORICAL_STORE: "automatic_results",
    ResourceRole.RESEARCH_EVALUATION_STORE: "experiment_records",
    ResourceRole.PATCHCORE_STORE: "result_rows",
    ResourceRole.HYBRID_STORE: "result_rows",
}


class ProtectedExperimentStoreReader:
    """Hash-verified, immutable readers for the four protected SQLite stores."""

    def __init__(self, resources: ProtectedResourceCatalog):
        missing = [
            role
            for role in _STORE_TABLE
            if resources.binding(role) is None
        ]
        if missing:
            raise StorageConfigurationError(
                "Missing required protected-store bindings: "
                + ", ".join(sorted(role.value for role in missing))
            )
        self.resources = resources

    def _database(self, role: ResourceRole) -> Path:
        if role not in _STORE_TABLE:
            raise StorageConfigurationError(
                f"Resource is not an experiment store: {role.value}"
            )
        return self.resources.resolve(role).path

    def row_count(self, role: ResourceRole) -> int:
        table = _STORE_TABLE[role]
        with closing(_connect_read_only(self._database(role))) as connection:
            return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    def results(
        self,
        role: ResourceRole,
        *,
        limit: int | None = None,
    ) -> tuple[dict[str, object], ...]:
        table = _STORE_TABLE[role]
        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError("limit must be a non-negative integer or None")
        query = f"SELECT * FROM {table}"
        values: tuple[object, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            values = (limit,)
        with closing(_connect_read_only(self._database(role))) as connection:
            return _rows(connection, query, values)

    def schema(self, role: ResourceRole) -> tuple[dict[str, object], ...]:
        with closing(_connect_read_only(self._database(role))) as connection:
            return _rows(
                connection,
                "SELECT type,name,tbl_name,sql FROM sqlite_master "
                "ORDER BY type,name",
            )

    @staticmethod
    def reject_write(*_args, **_kwargs) -> None:
        raise ProtectedWriteRefusedError(
            "Protected experiment-store access is read-only"
        )
