from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, PngImagePlugin

from structvision import CLASSICAL_METHOD, demonstration_fixture
from structvision import live_console


def fixture_bytes() -> bytes:
    fixture = demonstration_fixture("thin structural indication")
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text(
        "structvision.fixture_status",
        "synthetic demonstration fixture; excluded from research cohorts; "
        "not real inspection evidence",
    )
    buffer = BytesIO()
    Image.fromarray(fixture.image_bgr[:, :, ::-1]).save(
        buffer,
        format="PNG",
        pnginfo=metadata,
        optimize=False,
        compress_level=6,
    )
    return buffer.getvalue()


def tree_snapshot(root: Path) -> tuple[tuple[str, ...], dict[str, bytes]]:
    directories = tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_dir()
        )
    )
    files = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    return directories, files


def plausible_marker(**changes: object) -> dict[str, object]:
    manifest_digest = "0" * 64
    marker: dict[str, object] = {
        "schema_version": live_console.OWNERSHIP_SCHEMA,
        "tool_identity": live_console.OWNERSHIP_TOOL,
        "tool_version": live_console.OWNERSHIP_VERSION,
        "method_identity": CLASSICAL_METHOD,
        "run_manifest_path": "RUN_MANIFEST.json",
        "run_manifest_schema": live_console.RUN_MANIFEST_SCHEMA,
        "run_manifest_sha256": manifest_digest,
        "ownership_digest": live_console._ownership_digest(
            manifest_digest
        ),
        "completed": True,
    }
    marker.update(changes)
    return marker


class LiveConsoleFilesystemSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            dir=Path(tempfile.gettempdir()).resolve()
        )
        self.root = Path(self.temporary.name)
        self.input = self.root / "input-zone" / "demonstration-fixture.png"
        self.input.parent.mkdir()
        self.input.write_bytes(fixture_bytes())

    def tearDown(self):
        self.temporary.cleanup()

    def run_console(self, output: Path, *extra: str):
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = live_console.main(
                [
                    "--input",
                    str(self.input),
                    "--output-dir",
                    str(output),
                    *extra,
                ]
            )
        return status, stdout.getvalue(), stderr.getvalue()

    def assert_path_refused_without_loss(
        self,
        target: Path,
        *,
        expected_message: str,
    ) -> None:
        target.mkdir(parents=True, exist_ok=True)
        sentinel = target / "must-survive.txt"
        sentinel.write_text("preserve", encoding="utf-8")
        with self.assertRaisesRegex(
            live_console.UnsafeOutputTargetError,
            expected_message,
        ):
            live_console._safe_output_target(
                target,
                input_path=self.input,
            )
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")

    def test_home_and_standard_user_folders_are_refused_without_loss(self):
        fake_home = self.root / "fake-home"
        with patch.object(
            live_console,
            "_home_directory",
            return_value=fake_home,
        ):
            self.assert_path_refused_without_loss(
                fake_home,
                expected_message="home directory",
            )
            for name in live_console.STANDARD_USER_FOLDERS:
                with self.subTest(folder=name):
                    self.assert_path_refused_without_loss(
                        fake_home / name,
                        expected_message="Standard user folders",
                    )

    def test_filesystem_root_and_mount_root_shapes_are_refused_without_loss(self):
        fake_root = self.root / "mock-filesystem-root"
        fake_volume_root = self.root / "Volumes" / "DRIVE"
        fake_root.mkdir()
        (fake_root / "must-survive.txt").write_text(
            "preserve",
            encoding="utf-8",
        )
        with patch.object(
            live_console,
            "_is_filesystem_root",
            side_effect=lambda path: path == fake_root,
        ):
            with self.assertRaisesRegex(
                live_console.UnsafeOutputTargetError,
                "Filesystem roots",
            ):
                live_console._safe_output_target(
                    fake_root,
                    input_path=self.input,
                )
        self.assertEqual(
            (fake_root / "must-survive.txt").read_text(encoding="utf-8"),
            "preserve",
        )

        with patch.object(
            live_console,
            "_is_mount_root",
            side_effect=lambda path: path == fake_volume_root,
        ):
            self.assert_path_refused_without_loss(
                fake_volume_root,
                expected_message="Mount or external-volume roots",
            )

    def test_repository_root_and_parent_are_refused_without_loss(self):
        repository_parent = self.root / "repository-zone"
        repository = repository_parent / "repository"
        repository.mkdir(parents=True)
        with patch.object(
            live_console,
            "_repository_roots",
            return_value=(repository,),
        ):
            self.assert_path_refused_without_loss(
                repository,
                expected_message="repository",
            )
            self.assert_path_refused_without_loss(
                repository_parent,
                expected_message="repository",
            )

    def test_input_parent_is_refused_without_loss(self):
        sentinel = self.input.parent / "must-survive.txt"
        sentinel.write_text("preserve", encoding="utf-8")
        with self.assertRaisesRegex(
            live_console.UnsafeOutputTargetError,
            "input parent",
        ):
            live_console._safe_output_target(
                self.input.parent,
                input_path=self.input,
            )
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")

    def test_direct_symlink_and_symlinked_ancestor_are_refused_without_loss(self):
        direct_destination = self.root / "direct-destination"
        direct_destination.mkdir()
        direct_sentinel = direct_destination / "must-survive.txt"
        direct_sentinel.write_text("direct", encoding="utf-8")
        direct_link = self.root / "direct-link"
        direct_link.symlink_to(direct_destination, target_is_directory=True)

        ancestor_destination = self.root / "ancestor-destination"
        nested = ancestor_destination / "nested-output"
        nested.mkdir(parents=True)
        ancestor_sentinel = nested / "must-survive.txt"
        ancestor_sentinel.write_text("ancestor", encoding="utf-8")
        ancestor_link = self.root / "ancestor-link"
        ancestor_link.symlink_to(
            ancestor_destination,
            target_is_directory=True,
        )

        for target in (direct_link, ancestor_link / "nested-output"):
            with self.subTest(target=target.name):
                with self.assertRaisesRegex(
                    live_console.UnsafeOutputTargetError,
                    "symlink",
                ):
                    live_console._safe_output_target(
                        target,
                        input_path=self.input,
                    )
        self.assertEqual(direct_sentinel.read_text(encoding="utf-8"), "direct")
        self.assertEqual(
            ancestor_sentinel.read_text(encoding="utf-8"),
            "ancestor",
        )

    def test_unmarked_existing_directory_is_never_replaced(self):
        output = self.root / "unmarked"
        output.mkdir()
        sentinel = output / "must-survive.txt"
        sentinel.write_text("preserve", encoding="utf-8")
        with patch.object(
            live_console,
            "analyse_demonstration_image",
        ) as detector, patch.object(
            live_console.shutil,
            "rmtree",
        ) as remove_tree:
            status, _stdout, stderr = self.run_console(
                output,
                "--overwrite",
            )
        self.assertEqual(status, live_console.EXIT_UNSAFE_TARGET)
        self.assertIn("Unsafe output target", stderr)
        detector.assert_not_called()
        remove_tree.assert_not_called()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")

    def test_retired_marker_run_is_preserved_and_refused(self):
        output = self.root / "retired-marker-run"
        output.mkdir()
        sentinel = output / "must-survive.txt"
        sentinel.write_text("preserve", encoding="utf-8")
        retired_marker = output / ".structvision-professor-console-owner.json"
        retired_marker.write_text(
            json.dumps(
                {
                    "schema_version": (
                        "structvision-professor-console-run-owner-v1"
                    ),
                    "tool_identity": "structvision-professor-demo",
                    "completed": True,
                }
            ),
            encoding="utf-8",
        )
        with patch.object(
            live_console,
            "analyse_demonstration_image",
        ) as detector, patch.object(
            live_console.shutil,
            "rmtree",
        ) as remove_tree:
            status, _stdout, stderr = self.run_console(
                output,
                "--overwrite",
            )
        self.assertEqual(status, live_console.EXIT_UNSAFE_TARGET)
        self.assertIn("Unsafe output target", stderr)
        detector.assert_not_called()
        remove_tree.assert_not_called()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")
        self.assertTrue(retired_marker.is_file())

    def test_forged_malformed_and_wrong_tool_markers_are_refused(self):
        cases = {
            "forged": json.dumps(
                plausible_marker(ownership_digest="f" * 64)
            ).encode("utf-8"),
            "malformed": b"{not-json",
            "wrong-tool": json.dumps(
                plausible_marker(tool_identity="another-tool")
            ).encode("utf-8"),
            "wrong-version": json.dumps(
                plausible_marker(tool_version=999)
            ).encode("utf-8"),
        }
        for name, marker_payload in cases.items():
            with self.subTest(marker=name):
                output = self.root / f"marker-{name}"
                output.mkdir()
                sentinel = output / "must-survive.txt"
                sentinel.write_text(name, encoding="utf-8")
                (output / live_console.OWNERSHIP_MARKER_NAME).write_bytes(
                    marker_payload
                )
                with patch.object(
                    live_console,
                    "analyse_demonstration_image",
                ) as detector:
                    status, _stdout, stderr = self.run_console(
                        output,
                        "--overwrite",
                    )
                self.assertEqual(
                    status,
                    live_console.EXIT_UNSAFE_TARGET,
                )
                self.assertIn("Unsafe output target", stderr)
                detector.assert_not_called()
                self.assertEqual(sentinel.read_text(encoding="utf-8"), name)

    def test_valid_owned_run_replaces_only_after_staging_is_complete(self):
        output = self.root / "owned-run"
        first, _stdout, first_stderr = self.run_console(output)
        self.assertEqual(first, live_console.EXIT_SUCCESS, first_stderr)
        prior = tree_snapshot(output)
        original_write = live_console._write_run

        def observe_prior_while_staging(**kwargs):
            self.assertEqual(tree_snapshot(output), prior)
            return original_write(**kwargs)

        with patch.object(
            live_console,
            "_write_run",
            side_effect=observe_prior_while_staging,
        ):
            second, _stdout, second_stderr = self.run_console(
                output,
                "--overwrite",
            )
        self.assertEqual(second, live_console.EXIT_SUCCESS, second_stderr)
        live_console._validate_owned_run(output)
        self.assertFalse(
            any(
                live_console.STAGING_NAME_TOKEN in path.name
                or live_console.BACKUP_NAME_TOKEN in path.name
                for path in output.parent.iterdir()
            )
        )

    def test_install_failure_restores_previous_complete_run(self):
        output = self.root / "rollback-run"
        first, _stdout, first_stderr = self.run_console(output)
        self.assertEqual(first, live_console.EXIT_SUCCESS, first_stderr)
        prior = tree_snapshot(output)
        real_rename = os.rename

        def fail_staging_install(source, destination, *args, **kwargs):
            source_path = Path(source)
            if (
                live_console.STAGING_NAME_TOKEN in source_path.name
                and Path(destination) == output
            ):
                raise OSError("simulated staging install failure")
            return real_rename(source, destination, *args, **kwargs)

        with patch.object(
            live_console.os,
            "rename",
            side_effect=fail_staging_install,
        ):
            status, _stdout, stderr = self.run_console(
                output,
                "--overwrite",
            )
        self.assertEqual(status, live_console.EXIT_OUTPUT)
        self.assertIn("simulated staging install failure", stderr)
        self.assertEqual(tree_snapshot(output), prior)
        live_console._validate_owned_run(output)

    def test_target_change_between_validation_and_commit_is_refused(self):
        output = self.root / "race-run"
        first, _stdout, first_stderr = self.run_console(output)
        self.assertEqual(first, live_console.EXIT_SUCCESS, first_stderr)
        prior_directories, prior_files = tree_snapshot(output)
        original_write = live_console._write_run

        def inject_change_after_staging(**kwargs):
            result = original_write(**kwargs)
            (output / "race-sentinel.txt").write_text(
                "appeared during run",
                encoding="utf-8",
            )
            return result

        with patch.object(
            live_console,
            "_write_run",
            side_effect=inject_change_after_staging,
        ):
            status, _stdout, stderr = self.run_console(
                output,
                "--overwrite",
            )
        self.assertEqual(status, live_console.EXIT_UNSAFE_TARGET)
        self.assertIn("Unsafe output target", stderr)
        self.assertEqual(
            (output / "race-sentinel.txt").read_text(encoding="utf-8"),
            "appeared during run",
        )
        current_directories, current_files = tree_snapshot(output)
        self.assertEqual(current_directories, prior_directories)
        for relative, payload in prior_files.items():
            self.assertEqual(current_files[relative], payload)


if __name__ == "__main__":
    unittest.main()
