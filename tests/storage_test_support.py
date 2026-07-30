from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile
from typing import Iterator, Mapping, Sequence
from unittest.mock import patch

from structvision.storage import (
    CONFIG_ENVIRONMENT_VARIABLE,
    LogicalRoot,
    ResourceBinding,
    ResourceRole,
    StorageConfig,
)


STORAGE_RELATED_ENVIRONMENT_VARIABLES = frozenset(
    {
        CONFIG_ENVIRONMENT_VARIABLE,
        "STRUCTVISION_ENVIRONMENT_LOCK",
        "STRUCTVISION_HYBRID_FUSION_ARTIFACT",
        "STRUCTVISION_HYBRID_MODEL_ARTIFACT",
        "STRUCTVISION_PATCHCORE_CALIBRATION_ARTIFACT",
        "STRUCTVISION_PATCHCORE_MODEL_ARTIFACT",
        "STRUCTVISION_PATCHCORE_WEIGHT",
        "STRUCTVISION_PROTECTED_TEST_ROOT",
    }
)
HOME_RELATED_ENVIRONMENT_VARIABLES = frozenset(
    {
        "APPDATA",
        "HOME",
        "LOCALAPPDATA",
        "TEMP",
        "TMP",
        "TMPDIR",
        "USERPROFILE",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
    }
)


@dataclass(frozen=True)
class StorageTestEnvironment:
    root: Path
    home: Path
    environment: Mapping[str, str]
    configuration: StorageConfig | None = None
    configuration_path: Path | None = None
    invalid_configuration_path: Path | None = None

    @property
    def preferred_configuration_path(self) -> Path:
        return (
            self.home
            / "Library"
            / "Application Support"
            / "StructVision"
            / "config.toml"
        )

    @property
    def private_data_root(self) -> Path:
        if self.configuration is None:
            raise AssertionError("no synthetic external configuration is active")
        return self.configuration.root(LogicalRoot.PRIVATE_DATA)

    @property
    def runs_root(self) -> Path:
        if self.configuration is None:
            raise AssertionError("no synthetic external configuration is active")
        return self.configuration.root(LogicalRoot.RUNS)

    def subprocess_environment(
        self,
        updates: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        controlled = dict(self.environment)
        controlled["PYTHONDONTWRITEBYTECODE"] = "1"
        if updates:
            controlled.update(updates)
        return controlled


def _controlled_environment(root: Path, home: Path) -> dict[str, str]:
    environment = dict(os.environ)
    for name in (
        STORAGE_RELATED_ENVIRONMENT_VARIABLES
        | HOME_RELATED_ENVIRONMENT_VARIABLES
    ):
        environment.pop(name, None)
    temporary = root / "tmp"
    xdg = home / ".xdg"
    temporary.mkdir()
    for name in ("cache", "config", "data"):
        (xdg / name).mkdir(parents=True)
    environment.update(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "XDG_CACHE_HOME": str(xdg / "cache"),
            "XDG_CONFIG_HOME": str(xdg / "config"),
            "XDG_DATA_HOME": str(xdg / "data"),
            "TMPDIR": str(temporary),
            "TMP": str(temporary),
            "TEMP": str(temporary),
        }
    )
    return environment


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_store(path: Path, table: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(f"CREATE TABLE {table} (result_id TEXT PRIMARY KEY)")
        connection.execute(
            f"INSERT INTO {table} VALUES ('synthetic-metadata-only-row')"
        )


def _write_registry(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE datasets (dataset_id TEXT PRIMARY KEY, "
            "dataset_version TEXT, metadata_json TEXT, registered_timestamp TEXT)"
        )
        connection.execute(
            "INSERT INTO datasets VALUES "
            "('synthetic-metadata-only','1','{}','2026-01-01T00:00:00Z')"
        )
        connection.execute(
            "CREATE TABLE images (image_id TEXT PRIMARY KEY, dataset_id TEXT, "
            "stored_filename TEXT, annotation_path TEXT)"
        )
        connection.execute(
            "CREATE TABLE experiment_plans (plan_id TEXT PRIMARY KEY, "
            "dataset_id TEXT)"
        )


def _synthetic_configuration(root: Path) -> StorageConfig:
    source = root / "source"
    external = root / "external"
    source.mkdir()
    initial = StorageConfig.proposed_external(
        source_root=source,
        external_base=external,
    )
    for configured_root in sorted(
        set(initial.roots.values()),
        key=lambda path: (len(path.parts), str(path)),
    ):
        configured_root.mkdir(parents=True, exist_ok=True)

    registry_root = initial.root(LogicalRoot.REGISTRY)
    registry_database = registry_root / "datasets.sqlite"
    _write_registry(registry_database)
    registry_manifest = registry_root / "dataset_manifest.json"
    registry_manifest.write_text(
        json.dumps(
            {
                "fixture": "synthetic metadata only",
                "redistribution_allowed": False,
                "schema_version": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    experiment_root = initial.root(LogicalRoot.EXPERIMENT_STORE)
    store_definitions = {
        ResourceRole.HISTORICAL_STORE: (
            Path("historical.sqlite3"),
            "automatic_results",
        ),
        ResourceRole.RESEARCH_EVALUATION_STORE: (
            Path("evaluation.sqlite3"),
            "experiment_records",
        ),
        ResourceRole.PATCHCORE_STORE: (
            Path("patchcore.sqlite3"),
            "result_rows",
        ),
        ResourceRole.HYBRID_STORE: (
            Path("hybrid.sqlite3"),
            "result_rows",
        ),
    }
    for relative_path, table in store_definitions.values():
        _write_store(experiment_root / relative_path, table)

    learned_root = initial.root(LogicalRoot.LEARNED_ARTIFACT)
    learned_definitions = {
        ResourceRole.LEARNED_ENVIRONMENT_LOCK: Path(
            "environment-lock.metadata.json"
        ),
        ResourceRole.OFFICIAL_WEIGHT: Path("official-weight.metadata.json"),
        ResourceRole.PATCHCORE_MODEL: Path("patchcore-model.metadata.json"),
        ResourceRole.PATCHCORE_CALIBRATION: Path(
            "patchcore-calibration.metadata.json"
        ),
        ResourceRole.HYBRID_MODEL: Path("hybrid-model.metadata.json"),
        ResourceRole.HYBRID_FUSION: Path("hybrid-fusion.metadata.json"),
    }
    for role, relative_path in learned_definitions.items():
        selected = learned_root / relative_path
        selected.write_text(
            json.dumps(
                {
                    "fixture": "synthetic metadata only",
                    "redistribution_allowed": False,
                    "role": role.value,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )

    bindings = [
        ResourceBinding(
            ResourceRole.REGISTRY_DATABASE,
            LogicalRoot.REGISTRY,
            registry_database.relative_to(registry_root),
            _digest(registry_database),
            redistribution_allowed=False,
        ),
        ResourceBinding(
            ResourceRole.REGISTRY_MANIFEST,
            LogicalRoot.REGISTRY,
            registry_manifest.relative_to(registry_root),
            _digest(registry_manifest),
            redistribution_allowed=False,
        ),
    ]
    bindings.extend(
        ResourceBinding(
            role,
            LogicalRoot.EXPERIMENT_STORE,
            relative_path,
            _digest(experiment_root / relative_path),
            redistribution_allowed=False,
        )
        for role, (relative_path, _table) in store_definitions.items()
    )
    bindings.extend(
        ResourceBinding(
            role,
            LogicalRoot.LEARNED_ARTIFACT,
            relative_path,
            _digest(learned_root / relative_path),
            redistribution_allowed=False,
        )
        for role, relative_path in learned_definitions.items()
    )
    return StorageConfig.proposed_external(
        source_root=source,
        external_base=external,
        resource_bindings=tuple(bindings),
    )


@contextmanager
def isolated_no_configuration() -> Iterator[StorageTestEnvironment]:
    with tempfile.TemporaryDirectory(
        prefix="structvision-no-config-",
        dir=Path(tempfile.gettempdir()).resolve(),
    ) as temporary:
        root = Path(temporary).resolve()
        home = root / "home"
        home.mkdir()
        environment = _controlled_environment(root, home)
        state = StorageTestEnvironment(
            root=root,
            home=home,
            environment=environment,
        )
        if state.preferred_configuration_path.exists():
            raise AssertionError("isolated HOME unexpectedly contains a preferred config")
        with patch.dict(os.environ, environment, clear=True):
            yield state


@contextmanager
def synthetic_external_configuration() -> Iterator[StorageTestEnvironment]:
    with tempfile.TemporaryDirectory(
        prefix="structvision-synthetic-external-",
        dir=Path(tempfile.gettempdir()).resolve(),
    ) as temporary:
        root = Path(temporary).resolve()
        home = root / "home"
        home.mkdir()
        environment = _controlled_environment(root, home)
        configuration = _synthetic_configuration(root)
        preferred = (
            home
            / "Library"
            / "Application Support"
            / "StructVision"
            / "config.toml"
        )
        preferred.parent.mkdir(parents=True)
        preferred.write_text(configuration.to_toml(), encoding="utf-8")
        state = StorageTestEnvironment(
            root=root,
            home=home,
            environment=environment,
            configuration=configuration,
            configuration_path=preferred,
        )
        with patch.dict(os.environ, environment, clear=True):
            yield state


@contextmanager
def explicit_invalid_configuration() -> Iterator[StorageTestEnvironment]:
    with isolated_no_configuration() as isolated:
        malformed = isolated.root / "explicit-malformed-config.toml"
        malformed.write_text("[roots\n", encoding="utf-8")
        yield StorageTestEnvironment(
            root=isolated.root,
            home=isolated.home,
            environment=dict(isolated.environment),
            invalid_configuration_path=malformed,
        )


def run_in_storage_environment(
    mode: str,
    command: Sequence[str],
) -> subprocess.CompletedProcess:
    selected = {
        "no-config": isolated_no_configuration,
        "synthetic-external": synthetic_external_configuration,
    }[mode]
    with selected() as state:
        return subprocess.run(
            list(command),
            env=state.subprocess_environment(),
            check=False,
        )


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a command in a hermetic StructVision storage-test state."
    )
    parser.add_argument(
        "mode",
        choices=("no-config", "synthetic-external"),
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(arguments)
    command = list(parsed.command)
    if command[:1] == ["--"]:
        command.pop(0)
    if not command:
        parser.error("a command is required after --")
    return run_in_storage_environment(parsed.mode, command).returncode


if __name__ == "__main__":
    raise SystemExit(main())
