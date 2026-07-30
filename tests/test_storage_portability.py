from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from structvision.legacy_paths import LegacyPathResolver, ResolutionStatus
from structvision.storage import (
    LegacyReferenceRole,
    LegacyTranslationRule,
    LogicalRoot,
    MigrationState,
    PathIntent,
    ResourceBinding,
    ResourceRole,
    StorageConfig,
    StorageConfigurationError,
    StorageConfigurationMissingError,
    load_storage_config,
)
from storage_test_support import (
    isolated_no_configuration,
    synthetic_external_configuration,
)


ROOT = Path(__file__).resolve().parents[1]


def physical(path: Path) -> Path:
    return path.resolve(strict=False)


class StorageFixture:
    def __init__(self, root: Path, *, translations=()):
        self.root = physical(root)
        self.source = self.root / "source"
        self.external = self.root / "external"
        self.source.mkdir()
        self.config = StorageConfig.proposed_external(
            source_root=self.source,
            external_base=self.external,
            translation_rules=tuple(translations),
        )

    def write_config(self, name: str = "config.toml") -> Path:
        path = self.root / name
        path.write_text(self.config.to_toml(), encoding="utf-8")
        return path


class StorageConfigurationTests(unittest.TestCase):
    def test_valid_external_named_roots_and_access_types(self):
        with TemporaryDirectory() as temporary:
            fixture = StorageFixture(Path(temporary))
            self.assertEqual(set(fixture.config.roots), set(LogicalRoot))
            self.assertTrue(
                all(path.is_absolute() for path in fixture.config.roots.values())
            )
            run = fixture.config.future_run_path("RUN-001")
            self.assertEqual(run.logical_root, LogicalRoot.RUNS)
            self.assertFalse(run.path.exists())
            with self.assertRaisesRegex(StorageConfigurationError, "read-only"):
                fixture.config.configured_path(
                    LogicalRoot.REGISTRY,
                    "datasets.sqlite",
                    intent=PathIntent.WRITE,
                )

    def test_module_import_creates_no_directory(self):
        with isolated_no_configuration() as isolated:
            root = isolated.root
            before = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    (
                        "import structvision.storage; "
                        "import structvision.legacy_paths; "
                        "print('imported')"
                    ),
                ],
                cwd=root,
                env=isolated.subprocess_environment(
                    {"PYTHONPATH": str(ROOT / "src")}
                ),
                capture_output=True,
                check=False,
            )
            after = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        self.assertEqual(before, after)

    def test_missing_and_malformed_configuration_are_explicit(self):
        with TemporaryDirectory() as temporary:
            root = physical(Path(temporary))
            missing = root / "missing.toml"
            with self.assertRaises(StorageConfigurationMissingError):
                load_storage_config(missing)
            malformed = root / "malformed.toml"
            malformed.write_text("[roots\n", encoding="utf-8")
            with self.assertRaisesRegex(
                StorageConfigurationError, "Could not load storage configuration"
            ):
                load_storage_config(malformed)

    def test_loading_configuration_does_not_create_named_roots(self):
        with TemporaryDirectory() as temporary:
            fixture = StorageFixture(Path(temporary))
            config_path = fixture.write_config()
            before = {
                name: path.exists() for name, path in fixture.config.roots.items()
            }
            loaded = load_storage_config(config_path)
            after = {
                name: path.exists() for name, path in fixture.config.roots.items()
            }
            self.assertIsNotNone(loaded)
            self.assertEqual(before, after)
            self.assertFalse(after[LogicalRoot.RUNS])

    def test_relative_and_traversal_roots_are_refused(self):
        with TemporaryDirectory() as temporary:
            fixture = StorageFixture(Path(temporary))
            roots = dict(fixture.config.roots)
            roots[LogicalRoot.RUNS] = Path("relative")
            with self.assertRaisesRegex(StorageConfigurationError, "must be absolute"):
                StorageConfig(roots=roots)
            roots = dict(fixture.config.roots)
            roots[LogicalRoot.RUNS] = Path(str(fixture.root) + "/external/../escape")
            with self.assertRaisesRegex(StorageConfigurationError, "traversal"):
                StorageConfig(roots=roots)

    def test_symlink_root_and_symlinked_ancestor_are_refused(self):
        with TemporaryDirectory() as temporary:
            fixture = StorageFixture(Path(temporary))
            real = fixture.root / "real"
            real.mkdir()
            direct_link = fixture.root / "direct-link"
            direct_link.symlink_to(real, target_is_directory=True)
            roots = dict(fixture.config.roots)
            roots[LogicalRoot.RUNS] = direct_link
            with self.assertRaisesRegex(StorageConfigurationError, "symlink"):
                StorageConfig(roots=roots)

            ancestor = fixture.root / "ancestor-link"
            ancestor.symlink_to(real, target_is_directory=True)
            roots = dict(fixture.config.roots)
            roots[LogicalRoot.RUNS] = ancestor / "runs"
            with self.assertRaisesRegex(StorageConfigurationError, "symlink"):
                StorageConfig(roots=roots)

    def test_filesystem_home_and_repository_roots_are_refused(self):
        with TemporaryDirectory() as temporary:
            fixture = StorageFixture(Path(temporary))
            roots = dict(fixture.config.roots)
            roots[LogicalRoot.RUNS] = Path("/")
            with self.assertRaisesRegex(StorageConfigurationError, "filesystem root"):
                StorageConfig(roots=roots)
            roots = dict(fixture.config.roots)
            roots[LogicalRoot.RUNS] = Path.home()
            with self.assertRaisesRegex(StorageConfigurationError, "home directory"):
                StorageConfig(roots=roots)
            roots = dict(fixture.config.roots)
            roots[LogicalRoot.RUNS] = fixture.source / "outputs"
            with self.assertRaisesRegex(StorageConfigurationError, "Git source root"):
                StorageConfig(roots=roots)

    def test_conflicting_protected_private_and_release_roots_are_refused(self):
        with TemporaryDirectory() as temporary:
            fixture = StorageFixture(Path(temporary))
            roots = dict(fixture.config.roots)
            roots[LogicalRoot.PRIVATE_DATA] = roots[LogicalRoot.PROTECTED]
            with self.assertRaisesRegex(StorageConfigurationError, "overlapping"):
                StorageConfig(roots=roots)
            roots = dict(fixture.config.roots)
            roots[LogicalRoot.PRIVATE_DATA] = roots[LogicalRoot.RELEASE] / "private"
            with self.assertRaisesRegex(StorageConfigurationError, "overlapping"):
                StorageConfig(roots=roots)

    def test_identity_and_serialisation_are_deterministic(self):
        with TemporaryDirectory() as temporary:
            fixture = StorageFixture(Path(temporary))
            config_path = fixture.write_config()
            first = load_storage_config(config_path)
            second = load_storage_config(config_path)
            assert first is not None and second is not None
            self.assertEqual(first.identity, second.identity)
            self.assertEqual(first.to_toml(), second.to_toml())
            self.assertEqual(first.to_json(), second.to_json())

    def test_public_serialisation_redacts_private_paths_and_user_identity(self):
        with TemporaryDirectory() as temporary:
            fixture = StorageFixture(Path(temporary))
            public = fixture.config.to_json(public=True)
            payload = json.loads(public)
            self.assertNotIn(str(fixture.root), public)
            self.assertNotIn(str(Path.home()), public)
            self.assertNotIn(LogicalRoot.PRIVATE_DATA.value, payload["roots"])
            self.assertEqual(
                payload["redacted_roots"], [LogicalRoot.PRIVATE_DATA.value]
            )

    def test_source_does_not_embed_current_user_absolute_path(self):
        current_home = str(Path.home())
        for path in (
            ROOT / "src" / "structvision" / "storage.py",
            ROOT / "src" / "structvision" / "legacy_paths.py",
        ):
            self.assertNotIn(current_home, path.read_text(encoding="utf-8"))

    def test_explicit_legacy_mode_cannot_silently_write(self):
        with TemporaryDirectory() as temporary:
            source = physical(Path(temporary)) / "source"
            source.mkdir()
            config = StorageConfig.legacy_repository_compatibility(source)
            self.assertEqual(
                config.migration_state,
                MigrationState.LEGACY_REPOSITORY_COMPATIBILITY,
            )
            with self.assertRaisesRegex(
                StorageConfigurationError, "allow_legacy_write=True"
            ):
                config.configured_path(
                    LogicalRoot.RUNS,
                    "new-run",
                    intent=PathIntent.WRITE,
                )
            with self.assertRaisesRegex(StorageConfigurationError, "external"):
                config.future_run_path("new-run")


class LegacyPathResolutionTests(unittest.TestCase):
    def _fixture(self, root: Path):
        physical_root = physical(root)
        source = physical_root / "source"
        external = physical_root / "external"
        old_reports = physical_root / "retired-source" / "reports"
        old_annotations = physical_root / "retired-source" / "annotations"
        source.mkdir()
        report_rule = LegacyTranslationRule(
            role=LegacyReferenceRole.HISTORICAL_REPORT,
            identity="historical-report-prefix-v1",
            stored_prefix=old_reports,
            target_root=LogicalRoot.HISTORICAL_REPORT,
        )
        annotation_rule = LegacyTranslationRule(
            role=LegacyReferenceRole.REGISTRY_ANNOTATION,
            identity="registry-annotation-prefix-v1",
            stored_prefix=old_annotations,
            target_root=LogicalRoot.RESEARCH_DATA,
        )
        config = StorageConfig.proposed_external(
            source_root=source,
            external_base=external,
            translation_rules=(report_rule, annotation_rule),
        )
        return (
            config,
            LegacyPathResolver(config),
            old_reports,
            old_annotations,
        )

    def test_direct_valid_path(self):
        with TemporaryDirectory() as temporary:
            config, resolver, _, _ = self._fixture(Path(temporary))
            direct = config.root(LogicalRoot.HISTORICAL_REPORT) / "direct.txt"
            direct.parent.mkdir(parents=True)
            direct.write_bytes(b"metadata-only fixture")
            result = resolver.resolve_historical_report(str(direct))
            self.assertEqual(result.status, ResolutionStatus.DIRECT)
            self.assertEqual(result.resolved_path, direct)

    def test_translated_historical_report_and_registry_annotation(self):
        with TemporaryDirectory() as temporary:
            config, resolver, old_reports, old_annotations = self._fixture(
                Path(temporary)
            )
            report = config.root(LogicalRoot.HISTORICAL_REPORT) / "set" / "r.txt"
            report.parent.mkdir(parents=True)
            report.write_bytes(b"report reference fixture")
            annotation = config.root(LogicalRoot.RESEARCH_DATA) / "set" / "a.txt"
            annotation.parent.mkdir(parents=True)
            annotation.write_bytes(b"annotation reference fixture")
            report_result = resolver.resolve_historical_report(
                str(old_reports / "set" / "r.txt")
            )
            annotation_result = resolver.resolve_registry_annotation(
                str(old_annotations / "set" / "a.txt")
            )
            self.assertEqual(report_result.status, ResolutionStatus.TRANSLATED)
            self.assertEqual(report_result.resolved_path, report)
            self.assertEqual(
                annotation_result.status, ResolutionStatus.TRANSLATED
            )
            self.assertEqual(annotation_result.resolved_path, annotation)

    def test_unavailable_translated_target(self):
        with TemporaryDirectory() as temporary:
            _, resolver, old_reports, _ = self._fixture(Path(temporary))
            stored = old_reports / "missing.txt"
            result = resolver.resolve_historical_report(str(stored))
            self.assertEqual(result.status, ResolutionStatus.UNAVAILABLE)
            self.assertEqual(result.original_stored_path, str(stored))

    def test_unexpected_prefix_and_traversal_are_refused(self):
        with TemporaryDirectory() as temporary:
            _, resolver, old_reports, _ = self._fixture(Path(temporary))
            unexpected = physical(Path(temporary)) / "external-other" / "x.txt"
            self.assertEqual(
                resolver.resolve_historical_report(str(unexpected)).status,
                ResolutionStatus.REFUSED,
            )
            traversal = str(old_reports) + "/../secret.txt"
            result = resolver.resolve_historical_report(traversal)
            self.assertEqual(result.status, ResolutionStatus.REFUSED)
            self.assertIn("traversal", result.reason)

    def test_symlink_escape_is_refused(self):
        with TemporaryDirectory() as temporary:
            config, resolver, old_reports, _ = self._fixture(Path(temporary))
            outside = physical(Path(temporary)) / "outside"
            outside.mkdir()
            (outside / "secret.txt").write_bytes(b"do not resolve")
            target_root = config.root(LogicalRoot.HISTORICAL_REPORT)
            target_root.mkdir(parents=True)
            (target_root / "escape").symlink_to(outside, target_is_directory=True)
            result = resolver.resolve_historical_report(
                str(old_reports / "escape" / "secret.txt")
            )
            self.assertEqual(result.status, ResolutionStatus.REFUSED)
            self.assertIn("symlink", result.reason)

    def test_wrong_logical_role_is_refused(self):
        with TemporaryDirectory() as temporary:
            _, resolver, _, old_annotations = self._fixture(Path(temporary))
            result = resolver.resolve_historical_report(
                str(old_annotations / "mask.txt")
            )
            self.assertEqual(result.status, ResolutionStatus.REFUSED)
            self.assertIn("registry_annotation", result.reason)

    def test_immutable_files_remain_byte_identical_and_payload_is_not_decoded(self):
        with TemporaryDirectory() as temporary:
            root = physical(Path(temporary))
            _, resolver, old_reports, _ = self._fixture(root)
            database = root / "historical.sqlite3"
            manifest = root / "dataset_manifest.json"
            database.write_bytes(b"opaque database fixture")
            manifest.write_bytes(b'{"opaque":"manifest fixture"}\n')
            before = {
                database: sha256(database.read_bytes()).hexdigest(),
                manifest: sha256(manifest.read_bytes()).hexdigest(),
            }
            resolver.resolve_historical_report(str(old_reports / "missing.txt"))
            after = {
                path: sha256(path.read_bytes()).hexdigest() for path in before
            }
            self.assertEqual(before, after)

    def test_original_path_is_preserved_and_public_export_is_redacted(self):
        with TemporaryDirectory() as temporary:
            _, resolver, old_reports, _ = self._fixture(Path(temporary))
            stored = str(old_reports / "private" / "report.txt")
            result = resolver.resolve_historical_report(stored)
            self.assertEqual(result.original_stored_path, stored)
            private = result.to_dict(public=False)
            public = result.to_dict(public=True)
            self.assertEqual(private["original_stored_path"], stored)
            self.assertEqual(public["original_stored_path"], "[redacted]")
            self.assertNotIn(str(physical(Path(temporary))), json.dumps(public))
            self.assertFalse(public["scientific_validity_claimed"])


class StorageIntegrationTests(unittest.TestCase):
    def test_future_run_path_is_outside_repository(self):
        with TemporaryDirectory() as temporary:
            fixture = StorageFixture(Path(temporary))
            run = fixture.config.future_run_path("future-run")
            self.assertFalse(str(run.path).startswith(str(fixture.source)))
            self.assertTrue(str(run.path).startswith(str(fixture.external)))

    def test_live_console_output_contract_remains_explicit(self):
        source = (ROOT / "src" / "structvision" / "live_console.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"--output-dir"', source)
        self.assertIn("required=True", source)
        self.assertIn('"--storage-config"', source)
        self.assertIn("OperationalStorageContext.discover", source)

    def test_handoff_policy_excludes_local_configuration(self):
        from scripts import build_technical_handoff as builder

        self.assertIn("config.toml", builder.PROHIBITED_NAMES)
        self.assertIsNotNone(
            builder._is_prohibited(builder.PurePosixPath("config.toml"))
        )

    def test_legacy_streamlit_is_disabled_before_mutable_imports(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        guard = source.index("raise LegacyInterfaceDisabledError")
        self.assertLess(guard, source.index("from config import"))
        self.assertLess(guard, source.index("from dataset_intake import"))
        self.assertIn("apps/structvision_demo.py", source[:guard])

    def test_cli_external_config_accepts_only_runs_root_outputs(self):
        with synthetic_external_configuration() as fixture:
            runs = fixture.runs_root
            config_path = fixture.configuration_path
            private_data = fixture.private_data_root
            image_path = private_data / "fixture.png"
            yy, xx = np.indices((96, 128))
            image = np.clip(
                120 + 20 * np.sin(xx / 8.0) + 12 * np.cos(yy / 9.0),
                0,
                255,
            ).astype(np.uint8)
            Image.fromarray(image).save(image_path)
            accepted = runs / "analysis.json"
            refused = fixture.root / "outside.json"
            environment = fixture.subprocess_environment(
                {"PYTHONPATH": str(ROOT / "src")}
            )
            base = [
                sys.executable,
                "-B",
                "-m",
                "structvision.cli",
                "--input",
                str(image_path),
                "--storage-config",
                str(config_path),
            ]
            success = subprocess.run(
                [*base, "--json-out", str(accepted)],
                cwd=fixture.root,
                env=environment,
                capture_output=True,
                check=False,
            )
            failure = subprocess.run(
                [*base, "--json-out", str(refused)],
                cwd=fixture.root,
                env=environment,
                capture_output=True,
                check=False,
            )
            self.assertEqual(success.returncode, 0, success.stderr.decode())
            self.assertTrue(accepted.is_file())
            self.assertEqual(failure.returncode, 7, failure.stderr.decode())
            self.assertFalse(refused.exists())

    def test_registry_reader_uses_read_only_annotation_resolver_when_supplied(self):
        from types import SimpleNamespace

        from registered_experiment import load_ground_truth

        with TemporaryDirectory() as temporary:
            root = physical(Path(temporary))
            source = root / "source"
            source.mkdir()
            old_annotations = root / "retired" / "annotations"
            rule = LegacyTranslationRule(
                role=LegacyReferenceRole.REGISTRY_ANNOTATION,
                identity="registry-reader-v1",
                stored_prefix=old_annotations,
                target_root=LogicalRoot.RESEARCH_DATA,
            )
            config = StorageConfig.proposed_external(
                source_root=source,
                external_base=root / "external",
                translation_rules=(rule,),
            )
            target = config.root(LogicalRoot.RESEARCH_DATA) / "mask.png"
            target.parent.mkdir(parents=True)
            Image.fromarray(
                np.pad(
                    np.full((8, 8), 255, dtype=np.uint8),
                    ((4, 4), (4, 4)),
                )
            ).save(target)
            before = sha256(target.read_bytes()).hexdigest()
            row = SimpleNamespace(
                annotation_path=str(old_annotations / "mask.png"),
                height=16,
                width=16,
            )
            mask = load_ground_truth(row, LegacyPathResolver(config))
            after = sha256(target.read_bytes()).hexdigest()
            self.assertEqual(before, after)
            self.assertEqual(mask.shape, (16, 16))
            self.assertEqual(set(np.unique(mask)), {0, 255})

    def test_technical_demo_learned_paths_honor_explicit_storage_config(self):
        from structvision.demonstration import LearnedRuntimePaths
        from structvision.storage import CONFIG_ENVIRONMENT_VARIABLE

        with TemporaryDirectory() as temporary:
            fixture = StorageFixture(Path(temporary))
            learned = (
                fixture.config.root(LogicalRoot.LEARNED_ARTIFACT) / "model.json"
            )
            learned.parent.mkdir(parents=True)
            learned.write_bytes(b"bound learned model metadata")
            configured = StorageConfig.proposed_external(
                source_root=fixture.source,
                external_base=fixture.external,
                resource_bindings=(
                    ResourceBinding(
                        ResourceRole.PATCHCORE_MODEL,
                        LogicalRoot.LEARNED_ARTIFACT,
                        learned.relative_to(
                            fixture.config.root(LogicalRoot.LEARNED_ARTIFACT)
                        ),
                        sha256(learned.read_bytes()).hexdigest(),
                    ),
                ),
            )
            config_path = fixture.root / "config.toml"
            config_path.write_text(configured.to_toml(), encoding="utf-8")
            outside = fixture.root / "outside-model.json"
            with patch.dict(
                os.environ,
                {
                    CONFIG_ENVIRONMENT_VARIABLE: str(config_path),
                    "STRUCTVISION_PATCHCORE_MODEL_ARTIFACT": str(learned),
                },
                clear=True,
            ):
                runtime = LearnedRuntimePaths.from_environment()
            self.assertEqual(runtime.patchcore_model, learned)
            with patch.dict(
                os.environ,
                {
                    CONFIG_ENVIRONMENT_VARIABLE: str(config_path),
                    "STRUCTVISION_PATCHCORE_MODEL_ARTIFACT": str(outside),
                },
                clear=True,
            ):
                with self.assertRaisesRegex(
                    StorageConfigurationError, "learned_artifact_root"
                ):
                    LearnedRuntimePaths.from_environment()


if __name__ == "__main__":
    unittest.main()
