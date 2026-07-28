from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile

from scripts import build_technical_handoff as builder


ROOT = Path(__file__).resolve().parents[1]


def create_minimal_bundle(
    root: Path,
    commit: str,
    *,
    extra: dict[str, bytes] | None = None,
) -> Path:
    bundle = root / "StructVision-AI-Technical-Handoff"
    bundle.mkdir()
    for relative in sorted(builder.REQUIRED_RELATIVE_FILES):
        if relative in {builder.VERSION_FILE, builder.CHECKSUM_FILE}:
            continue
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture for {relative}\n".encode("utf-8"))
    archive = (
        bundle
        / "SOURCE"
        / f"{builder.SOURCE_ARCHIVE_PREFIX}{commit}.zip"
    )
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("README.md", "committed source\n")
        handle.writestr("src/example.py", "VALUE = 1\n")
    manifest = (
        f"source_commit={commit}\n"
        f"archive_sha256={sha256(archive.read_bytes()).hexdigest()}\n"
    )
    (bundle / "SOURCE" / "source-manifest.txt").write_text(
        manifest,
        encoding="utf-8",
    )
    for relative, payload in (extra or {}).items():
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    builder._finalise_integrity(
        bundle,
        commit=commit,
        branch="main",
        clean=True,
        timestamp="2026-07-29T00:00:00+00:00",
    )
    return bundle


class TechnicalHandoffBuilderTests(unittest.TestCase):
    def test_builder_uses_git_archive_and_never_recursive_worktree_copy(self):
        source = (
            ROOT / "scripts" / "build_technical_handoff.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"archive"', source)
        self.assertIn('"HEAD"', source)
        self.assertIn("git archive HEAD", source)
        self.assertNotIn("copytree(root", source)
        self.assertNotIn('["pip"', source)
        self.assertNotIn('"-m", "pip"', source)
        self.assertNotIn("requests.", source)
        self.assertNotIn("urlopen(", source)

    def test_checksum_order_is_deterministic(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "z.txt").write_text("z", encoding="utf-8")
            (root / "a.txt").write_text("a", encoding="utf-8")
            first = builder._checksum_text(root)
            second = builder._checksum_text(root)
        self.assertEqual(first, second)
        paths = [line.split("  ", 1)[1] for line in first.splitlines()]
        self.assertEqual(paths, sorted(paths))

    def test_clean_minimal_bundle_verifies_and_source_zip_opens(self):
        commit = "a" * 40
        with TemporaryDirectory() as temporary:
            bundle = create_minimal_bundle(Path(temporary), commit)
            count, size = builder.verify_handoff(
                bundle,
                expected_commit=commit,
            )
            archive = (
                bundle
                / "SOURCE"
                / f"{builder.SOURCE_ARCHIVE_PREFIX}{commit}.zip"
            )
            self.assertTrue(zipfile.is_zipfile(archive))
            with zipfile.ZipFile(archive) as handle:
                self.assertIsNone(handle.testzip())
        self.assertGreater(count, 0)
        self.assertGreater(size, 0)

    def test_verifier_detects_tampering_missing_and_unexpected_files(self):
        commit = "b" * 40
        with TemporaryDirectory() as temporary:
            bundle = create_minimal_bundle(Path(temporary), commit)
            (bundle / "README_FIRST.md").write_text(
                "tampered",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                builder.HandoffError,
                "Checksum mismatch",
            ):
                builder.verify_handoff(bundle, expected_commit=commit)
        with TemporaryDirectory() as temporary:
            bundle = create_minimal_bundle(Path(temporary), commit)
            (bundle / "README_FIRST.md").unlink()
            with self.assertRaisesRegex(
                builder.HandoffError,
                "Missing required",
            ):
                builder.verify_handoff(bundle, expected_commit=commit)
        with TemporaryDirectory() as temporary:
            bundle = create_minimal_bundle(Path(temporary), commit)
            (bundle / "unexpected.txt").write_text(
                "unexpected",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                builder.HandoffError,
                "Unexpected files",
            ):
                builder.verify_handoff(bundle, expected_commit=commit)

    def test_verifier_detects_prohibited_file_and_absolute_home_path(self):
        commit = "c" * 40
        with TemporaryDirectory() as temporary:
            bundle = create_minimal_bundle(
                Path(temporary),
                commit,
                extra={"SOURCE/model.pt": b"weight"},
            )
            with self.assertRaisesRegex(
                builder.HandoffError,
                "Exclusion audit failed",
            ):
                builder.verify_handoff(bundle, expected_commit=commit)
        with TemporaryDirectory() as temporary:
            bundle = create_minimal_bundle(
                Path(temporary),
                commit,
                extra={
                    "DOCUMENTATION/local-path.txt": (
                        b"/" + b"Users/example/private/image.png\n"
                    )
                },
            )
            with self.assertRaisesRegex(
                builder.HandoffError,
                "absolute user-home path",
            ):
                builder.verify_handoff(bundle, expected_commit=commit)

    def test_verifier_detects_wrong_source_commit(self):
        bundle_commit = "d" * 40
        expected_commit = "e" * 40
        with TemporaryDirectory() as temporary:
            bundle = create_minimal_bundle(
                Path(temporary),
                bundle_commit,
            )
            with self.assertRaisesRegex(
                builder.HandoffError,
                "Wrong source commit",
            ):
                builder.verify_handoff(
                    bundle,
                    expected_commit=expected_commit,
                )

    def test_prohibited_policy_covers_runtime_and_learned_artifacts(self):
        prohibited = (
            ".git/config",
            "venv/bin/python",
            ".venv/pyvenv.cfg",
            "__pycache__/x.pyc",
            ".pytest_cache/state",
            "model.safetensors",
            "memory.npz",
            "results.sqlite3",
            ".DS_Store",
        )
        for relative in prohibited:
            self.assertIsNotNone(
                builder._is_prohibited(
                    builder.PurePosixPath(relative)
                ),
                relative,
            )


if __name__ == "__main__":
    unittest.main()
