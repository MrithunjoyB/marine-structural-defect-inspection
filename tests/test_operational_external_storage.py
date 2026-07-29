from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from structvision import (
    LegacyPathResolver,
    LegacyReferenceRole,
    LegacyTranslationRule,
    LearnedRuntimePaths,
    LogicalRoot,
    OperationalStorageContext,
    OperationalStorageMode,
    ProtectedExperimentStoreReader,
    ProtectedResourceCatalog,
    ProtectedWriteRefusedError,
    ReadOnlyRegistry,
    ResourceBinding,
    ResourceRole,
    StorageConfig,
    StorageConfigurationError,
)
from structvision.legacy_paths import ResolutionStatus
from structvision.storage import CONFIG_ENVIRONMENT_VARIABLE


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_image(path: Path) -> None:
    yy, xx = np.indices((96, 128))
    image = np.clip(
        115 + 18 * np.sin(xx / 8.0) + 11 * np.cos(yy / 9.0),
        0,
        255,
    ).astype(np.uint8)
    Image.fromarray(image).save(path)


class ExternalFixture:
    def __init__(
        self,
        root: Path,
        *,
        translations: tuple[LegacyTranslationRule, ...] = (),
        resources: tuple[ResourceBinding, ...] = (),
    ):
        self.root = root.resolve()
        self.source = self.root / "source"
        self.source.mkdir()
        self.external = self.root / "external"
        self.config = StorageConfig.proposed_external(
            source_root=self.source,
            external_base=self.external,
            translation_rules=translations,
            resource_bindings=resources,
        )

    def config_file(self, name: str = "config.toml") -> Path:
        selected = self.root / name
        selected.write_text(self.config.to_toml(), encoding="utf-8")
        return selected


class OperationalStorageContextTests(unittest.TestCase):
    def test_preferred_discovery_explicit_override_and_no_configuration(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            fixture = ExternalFixture(root)
            preferred = fixture.config_file("preferred.toml")
            with patch.dict(os.environ, {}, clear=True), patch(
                "structvision.storage.preferred_config_path",
                return_value=preferred,
            ):
                discovered = OperationalStorageContext.discover()
            self.assertEqual(discovered.mode, OperationalStorageMode.EXTERNAL)
            self.assertEqual(discovered.configuration.identity, fixture.config.identity)

            alternate = fixture.config_file("alternate.toml")
            with patch(
                "structvision.storage.preferred_config_path",
                return_value=root / "missing.toml",
            ):
                explicit = OperationalStorageContext.discover(alternate)
            self.assertEqual(explicit.configuration.identity, fixture.config.identity)

            with patch.dict(os.environ, {}, clear=True), patch(
                "structvision.storage.preferred_config_path",
                return_value=root / "missing.toml",
            ):
                absent = OperationalStorageContext.discover()
            self.assertEqual(absent.mode, OperationalStorageMode.NO_CONFIGURATION)

    def test_malformed_and_legacy_configuration_fail_closed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            malformed = root / "malformed.toml"
            malformed.write_text("[roots\n", encoding="utf-8")
            with self.assertRaises(StorageConfigurationError):
                OperationalStorageContext.discover(malformed)

            source = root / "source"
            source.mkdir()
            legacy = StorageConfig.legacy_repository_compatibility(source)
            legacy_file = root / "legacy.toml"
            legacy_file.write_text(legacy.to_toml(), encoding="utf-8")
            with self.assertRaisesRegex(StorageConfigurationError, "external"):
                OperationalStorageContext.discover(legacy_file)

    def test_public_context_serialisation_contains_no_absolute_path(self):
        with TemporaryDirectory() as temporary:
            fixture = ExternalFixture(Path(temporary))
            public = json.dumps(
                OperationalStorageContext.external(fixture.config).to_dict(public=True)
            )
            self.assertNotIn(str(fixture.root), public)


class OfficialEntryPointStorageTests(unittest.TestCase):
    def test_live_console_authorises_private_input_and_runs_output(self):
        from structvision import live_console

        with TemporaryDirectory() as temporary:
            fixture = ExternalFixture(Path(temporary))
            private_root = fixture.config.root(LogicalRoot.PRIVATE_DATA)
            runs_root = fixture.config.root(LogicalRoot.RUNS)
            private_root.mkdir(parents=True)
            runs_root.mkdir(parents=True)
            image = private_root / "fixture.png"
            write_image(image)
            configuration = fixture.config_file()

            accepted = runs_root / "accepted"
            self.assertEqual(
                live_console.main(
                    [
                        "--input",
                        str(image),
                        "--output-dir",
                        str(accepted),
                        "--storage-config",
                        str(configuration),
                    ]
                ),
                live_console.EXIT_SUCCESS,
            )
            self.assertTrue((accepted / "RUN_MANIFEST.json").is_file())

            outside_image = fixture.root / "outside.png"
            write_image(outside_image)
            refused_input = runs_root / "refused-input"
            self.assertEqual(
                live_console.main(
                    [
                        "--input",
                        str(outside_image),
                        "--output-dir",
                        str(refused_input),
                        "--storage-config",
                        str(configuration),
                    ]
                ),
                live_console.EXIT_STORAGE_CONFIGURATION,
            )
            self.assertFalse(refused_input.exists())

            refused_output = fixture.root / "outside-output"
            self.assertEqual(
                live_console.main(
                    [
                        "--input",
                        str(image),
                        "--output-dir",
                        str(refused_output),
                        "--storage-config",
                        str(configuration),
                    ]
                ),
                live_console.EXIT_STORAGE_CONFIGURATION,
            )
            self.assertFalse(refused_output.exists())

    def test_analysis_cli_stdout_only_is_write_free_in_external_mode(self):
        with TemporaryDirectory() as temporary:
            fixture = ExternalFixture(Path(temporary))
            private_root = fixture.config.root(LogicalRoot.PRIVATE_DATA)
            private_root.mkdir(parents=True)
            image = private_root / "fixture.png"
            write_image(image)
            configuration = fixture.config_file()
            before = sorted(path.relative_to(fixture.root) for path in fixture.root.rglob("*"))
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(ROOT / "src")
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "structvision.cli",
                    "--input",
                    str(image),
                    "--stdout-json",
                    "--storage-config",
                    str(configuration),
                ],
                cwd=fixture.root,
                env=environment,
                capture_output=True,
                check=False,
            )
            after = sorted(path.relative_to(fixture.root) for path in fixture.root.rglob("*"))
            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
            self.assertEqual(before, after)

    def test_modern_streamlit_starts_with_external_or_no_configuration(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("optional Streamlit demonstration dependency is unavailable")
        with TemporaryDirectory() as temporary:
            fixture = ExternalFixture(Path(temporary))
            configuration = fixture.config_file()
            with patch.dict(
                os.environ,
                {CONFIG_ENVIRONMENT_VARIABLE: str(configuration)},
                clear=False,
            ):
                configured = AppTest.from_file(
                    str(ROOT / "apps" / "structvision_demo.py"),
                    default_timeout=20,
                ).run()
            self.assertFalse(configured.exception)
            self.assertEqual([item.value for item in configured.title], ["StructVision-AI"])

            with patch.dict(
                os.environ,
                {CONFIG_ENVIRONMENT_VARIABLE: ""},
                clear=False,
            ), patch(
                "structvision.storage.preferred_config_path",
                return_value=fixture.root / "absent.toml",
            ):
                unconfigured = AppTest.from_file(
                    str(ROOT / "apps" / "structvision_demo.py"),
                    default_timeout=20,
                ).run()
            self.assertFalse(unconfigured.exception)

    def test_modern_streamlit_masks_malformed_configuration_details(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("optional Streamlit demonstration dependency is unavailable")
        with TemporaryDirectory() as temporary:
            malformed = Path(temporary) / "private-malformed.toml"
            malformed.write_text("[roots\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {CONFIG_ENVIRONMENT_VARIABLE: str(malformed)},
                clear=False,
            ):
                application = AppTest.from_file(
                    str(ROOT / "apps" / "structvision_demo.py"),
                    default_timeout=20,
                ).run()
            self.assertFalse(application.exception)
            self.assertTrue(application.error)
            message = application.error[0].value
            self.assertNotIn(str(malformed), message)


def create_registry(registry_root: Path, research_root: Path) -> None:
    registry_root.mkdir(parents=True)
    (research_root / "raw" / "set-a").mkdir(parents=True)
    database = registry_root / "datasets.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE datasets (dataset_id TEXT, dataset_version TEXT, "
            "metadata_json TEXT, registered_timestamp TEXT)"
        )
        connection.execute(
            "INSERT INTO datasets VALUES ('set-a','1','{}','2026-01-01')"
        )
        connection.execute(
            "CREATE TABLE images (image_id TEXT, dataset_id TEXT, "
            "stored_filename TEXT, annotation_path TEXT)"
        )
        connection.execute(
            "INSERT INTO images VALUES "
            "('image-a','set-a','image-a.png','research_data/annotations/set-a/mask-a.png')"
        )
        connection.execute(
            "CREATE TABLE experiment_plans (plan_id TEXT, dataset_id TEXT)"
        )
        connection.execute("INSERT INTO experiment_plans VALUES ('plan-a','set-a')")
    (registry_root / "dataset_manifest.json").write_text(
        '{"fixture":"metadata-only"}\n',
        encoding="utf-8",
    )
    (research_root / "raw" / "set-a" / "image-a.png").write_bytes(
        b"opaque registered image fixture"
    )


class ReadOnlyEvidenceTests(unittest.TestCase):
    def test_registry_reader_uses_separate_roots_and_never_writes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            fixture = ExternalFixture(root)
            registry = fixture.config.root(LogicalRoot.REGISTRY)
            research = fixture.config.root(LogicalRoot.RESEARCH_DATA)
            create_registry(registry, research)
            configured = StorageConfig.proposed_external(
                source_root=fixture.source,
                external_base=fixture.external,
                resource_bindings=(
                    ResourceBinding(
                        ResourceRole.REGISTRY_DATABASE,
                        LogicalRoot.REGISTRY,
                        Path("datasets.sqlite"),
                        digest(registry / "datasets.sqlite"),
                    ),
                    ResourceBinding(
                        ResourceRole.REGISTRY_MANIFEST,
                        LogicalRoot.REGISTRY,
                        Path("dataset_manifest.json"),
                        digest(registry / "dataset_manifest.json"),
                    ),
                ),
            )
            before = digest(registry / "datasets.sqlite")
            reader = ReadOnlyRegistry.from_operational_storage(
                OperationalStorageContext.external(configured)
            )
            self.assertEqual(reader.metadata("set-a")["dataset_version"], "1")
            image = reader.images("set-a")[0]
            self.assertEqual(reader.registered_image_path(image).name, "image-a.png")
            self.assertEqual(reader.plans("plan-a")[0]["dataset_id"], "set-a")
            with self.assertRaises(ProtectedWriteRefusedError):
                reader.reject_write("DELETE FROM images")
            self.assertEqual(before, digest(registry / "datasets.sqlite"))

    def test_protected_store_reader_is_hash_verified_and_read_only(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            fixture = ExternalFixture(root)
            experiment_root = fixture.config.root(LogicalRoot.EXPERIMENT_STORE)
            experiment_root.mkdir(parents=True)
            definitions = (
                (ResourceRole.HISTORICAL_STORE, "historical.sqlite3", "automatic_results"),
                (
                    ResourceRole.RESEARCH_EVALUATION_STORE,
                    "evaluation.sqlite3",
                    "experiment_records",
                ),
                (ResourceRole.PATCHCORE_STORE, "patchcore.sqlite3", "result_rows"),
                (ResourceRole.HYBRID_STORE, "hybrid.sqlite3", "result_rows"),
            )
            bindings = []
            original_hashes = {}
            for role, filename, table in definitions:
                database = experiment_root / filename
                with sqlite3.connect(database) as connection:
                    connection.execute(f"CREATE TABLE {table} (result_id TEXT)")
                    connection.execute(f"INSERT INTO {table} VALUES ('fixture')")
                original_hashes[role] = digest(database)
                bindings.append(
                    ResourceBinding(
                        role,
                        LogicalRoot.EXPERIMENT_STORE,
                        Path(filename),
                        original_hashes[role],
                    )
                )
            configured = StorageConfig.proposed_external(
                source_root=fixture.source,
                external_base=fixture.external,
                resource_bindings=tuple(bindings),
            )
            context = OperationalStorageContext.external(configured)
            reader = ProtectedExperimentStoreReader(
                ProtectedResourceCatalog(context)
            )
            for role, filename, _ in definitions:
                self.assertEqual(reader.row_count(role), 1)
                self.assertEqual(reader.results(role, limit=1)[0]["result_id"], "fixture")
                self.assertTrue(reader.schema(role))
                self.assertEqual(
                    original_hashes[role],
                    digest(experiment_root / filename),
                )
            with self.assertRaises(ProtectedWriteRefusedError):
                reader.reject_write("UPDATE result_rows SET result_id='changed'")

    def test_external_legacy_executor_is_refused_before_any_access(self):
        from registered_experiment import (
            ExternalRegisteredExperimentExecutionDisabledError,
            execute_plan,
        )

        class AccessTrap:
            def __getattr__(self, name):
                raise AssertionError(f"protected access attempted: {name}")

        with TemporaryDirectory() as temporary:
            context = OperationalStorageContext.external(
                ExternalFixture(Path(temporary)).config
            )
            with self.assertRaises(
                ExternalRegisteredExperimentExecutionDisabledError
            ):
                execute_plan(
                    AccessTrap(),
                    AccessTrap(),
                    "plan",
                    operational_storage=context,
                )
            with patch(
                "registered_experiment.OperationalStorageContext.discover",
                return_value=context,
            ):
                with self.assertRaises(
                    ExternalRegisteredExperimentExecutionDisabledError
                ):
                    execute_plan(AccessTrap(), AccessTrap(), "plan")


class ResolverV2Tests(unittest.TestCase):
    def _fixture(self, root: Path):
        source = root / "source"
        source.mkdir()
        old = root / "Documents" / "Legacy"
        rules = (
            LegacyTranslationRule(
                LegacyReferenceRole.HISTORICAL_REPORT,
                "history-absolute-v2",
                old / "research_data" / "reports",
                LogicalRoot.HISTORICAL_REPORT,
            ),
            LegacyTranslationRule(
                LegacyReferenceRole.HISTORICAL_REPORT,
                "history-relative-v2",
                Path("research_data/reports"),
                LogicalRoot.HISTORICAL_REPORT,
            ),
            LegacyTranslationRule(
                LegacyReferenceRole.REGISTRY_ANNOTATION,
                "annotation-absolute-v2",
                old / "research_data" / "annotations",
                LogicalRoot.RESEARCH_DATA,
                Path("annotations"),
            ),
            LegacyTranslationRule(
                LegacyReferenceRole.REGISTRY_ANNOTATION,
                "annotation-relative-v2",
                Path("research_data/annotations"),
                LogicalRoot.RESEARCH_DATA,
                Path("annotations"),
            ),
        )
        config = StorageConfig.proposed_external(
            source_root=source,
            external_base=root / "external",
            translation_rules=rules,
        )
        return config, LegacyPathResolver(config), old

    def test_absolute_relative_destination_subpath_and_forced_external(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            config, resolver, old = self._fixture(root)
            history = config.root(LogicalRoot.HISTORICAL_REPORT) / "set" / "r.png"
            annotation = (
                config.root(LogicalRoot.RESEARCH_DATA)
                / "annotations"
                / "set"
                / "a.png"
            )
            history.parent.mkdir(parents=True)
            annotation.parent.mkdir(parents=True)
            history.write_bytes(b"external history")
            annotation.write_bytes(b"external annotation")

            legacy_history = old / "research_data" / "reports" / "set" / "r.png"
            legacy_annotation = (
                old / "research_data" / "annotations" / "set" / "a.png"
            )
            legacy_history.parent.mkdir(parents=True)
            legacy_annotation.parent.mkdir(parents=True)
            legacy_history.write_bytes(b"legacy history must not be selected")
            legacy_annotation.write_bytes(b"legacy annotation must not be selected")

            resolutions = (
                resolver.resolve_historical_report(str(legacy_history)),
                resolver.resolve_historical_report("research_data/reports/set/r.png"),
                resolver.resolve_registry_annotation(str(legacy_annotation)),
                resolver.resolve_registry_annotation(
                    "research_data/annotations/set/a.png"
                ),
            )
            self.assertEqual(
                [item.status for item in resolutions],
                [ResolutionStatus.TRANSLATED] * 4,
            )
            self.assertEqual(resolutions[0].resolved_path, history)
            self.assertEqual(resolutions[1].resolved_path, history)
            self.assertEqual(resolutions[2].resolved_path, annotation)
            self.assertEqual(resolutions[3].resolved_path, annotation)
            self.assertNotEqual(resolutions[0].resolved_path, legacy_history)

            round_trip = StorageConfig.proposed_external(
                source_root=config.source_root,
                external_base=root / "external",
                translation_rules=config.translation_rules,
            )
            config_file = root / "config.toml"
            config_file.write_text(round_trip.to_toml(), encoding="utf-8")
            from structvision.storage import load_storage_config

            loaded = load_storage_config(config_file)
            self.assertEqual(len(loaded.rules(LegacyReferenceRole.HISTORICAL_REPORT)), 2)

    def test_ambiguity_wrong_role_traversal_symlink_and_public_redaction(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            config, resolver, old = self._fixture(root)
            with self.assertRaisesRegex(StorageConfigurationError, "overlapping"):
                StorageConfig.proposed_external(
                    source_root=config.source_root,
                    external_base=root / "ambiguous",
                    translation_rules=(
                        LegacyTranslationRule(
                            LegacyReferenceRole.HISTORICAL_REPORT,
                            "one",
                            Path("research_data"),
                            LogicalRoot.HISTORICAL_REPORT,
                        ),
                        LegacyTranslationRule(
                            LegacyReferenceRole.HISTORICAL_REPORT,
                            "two",
                            Path("research_data/reports"),
                            LogicalRoot.HISTORICAL_REPORT,
                        ),
                    ),
                )
            wrong = resolver.resolve_historical_report(
                "research_data/annotations/set/a.png"
            )
            self.assertEqual(wrong.status, ResolutionStatus.REFUSED)
            self.assertIn("registry_annotation", wrong.reason)
            traversal = resolver.resolve_historical_report(
                "research_data/reports/../secret.png"
            )
            self.assertEqual(traversal.status, ResolutionStatus.REFUSED)

            target_root = config.root(LogicalRoot.HISTORICAL_REPORT)
            outside = root / "outside"
            outside.mkdir()
            target_root.mkdir(parents=True)
            (target_root / "escape").symlink_to(outside, target_is_directory=True)
            linked = resolver.resolve_historical_report(
                "research_data/reports/escape/file.png"
            )
            self.assertEqual(linked.status, ResolutionStatus.REFUSED)
            self.assertIn("symlink", linked.reason)

            public = json.dumps(
                resolver.resolve_historical_report(
                    str(old / "research_data" / "reports" / "private.png")
                ).to_dict(public=True)
            )
            self.assertNotIn(str(root), public)

    def test_external_translation_does_not_require_legacy_checkout(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            config, resolver, old = self._fixture(root)
            self.assertFalse(old.exists())
            target = config.root(LogicalRoot.HISTORICAL_REPORT) / "set" / "report.png"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"external metadata target")
            resolution = resolver.resolve_historical_report(
                str(old / "research_data" / "reports" / "set" / "report.png")
            )
            self.assertEqual(resolution.status, ResolutionStatus.TRANSLATED)
            self.assertEqual(resolution.resolved_path, target)


class ResourceDiscoveryAndHandoffTests(unittest.TestCase):
    def test_learned_resources_are_binding_discovered_and_hash_checked(self):
        with TemporaryDirectory() as temporary:
            fixture = ExternalFixture(Path(temporary))
            learned_root = fixture.config.root(LogicalRoot.LEARNED_ARTIFACT)
            learned_root.mkdir(parents=True)
            mapping = {
                ResourceRole.LEARNED_ENVIRONMENT_LOCK: "environment.lock",
                ResourceRole.OFFICIAL_WEIGHT: "weight.bin",
                ResourceRole.PATCHCORE_MODEL: "patchcore.json",
                ResourceRole.PATCHCORE_CALIBRATION: "calibration.json",
                ResourceRole.HYBRID_MODEL: "hybrid.json",
                ResourceRole.HYBRID_FUSION: "fusion.json",
            }
            bindings = []
            for role, filename in mapping.items():
                selected = learned_root / filename
                selected.write_bytes(f"fixture:{role.value}".encode())
                bindings.append(
                    ResourceBinding(
                        role,
                        LogicalRoot.LEARNED_ARTIFACT,
                        Path(filename),
                        digest(selected),
                    )
                )
            config = StorageConfig.proposed_external(
                source_root=fixture.source,
                external_base=fixture.external,
                resource_bindings=tuple(bindings),
            )
            context = OperationalStorageContext.external(config)
            with patch.dict(os.environ, {}, clear=True):
                runtime = LearnedRuntimePaths.from_environment(context)
            self.assertEqual(runtime.environment_lock, learned_root / "environment.lock")
            self.assertEqual(runtime.weight, learned_root / "weight.bin")
            public = json.dumps(
                ProtectedResourceCatalog(context)
                .resolve(ResourceRole.OFFICIAL_WEIGHT)
                .to_dict(public=True)
            )
            self.assertNotIn(str(fixture.root), public)

            changed = learned_root / "weight.bin"
            changed.write_bytes(b"changed")
            with self.assertRaisesRegex(StorageConfigurationError, "hash mismatch"):
                ProtectedResourceCatalog(context).resolve(ResourceRole.OFFICIAL_WEIGHT)
            changed.write_bytes(
                f"fixture:{ResourceRole.OFFICIAL_WEIGHT.value}".encode()
            )

            substitute = learned_root / "substitute.bin"
            substitute.write_bytes(b"substitute")
            with patch.dict(
                os.environ,
                {"STRUCTVISION_PATCHCORE_WEIGHT": str(substitute)},
                clear=True,
            ):
                with self.assertRaisesRegex(
                    StorageConfigurationError,
                    "bound resource",
                ):
                    LearnedRuntimePaths.from_environment(context)

    def test_handoff_output_is_authorised_under_release_root(self):
        from scripts import build_technical_handoff as builder

        with TemporaryDirectory() as temporary:
            fixture = ExternalFixture(Path(temporary))
            context = OperationalStorageContext.external(fixture.config)
            accepted = fixture.config.root(LogicalRoot.RELEASE) / "handoff"
            outside = fixture.root / "outside"
            self.assertEqual(
                builder._operational_output(accepted, context),
                accepted,
            )
            with self.assertRaisesRegex(StorageConfigurationError, "release_root"):
                builder._operational_output(outside, context)


if __name__ == "__main__":
    unittest.main()
