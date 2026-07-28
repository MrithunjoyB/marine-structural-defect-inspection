from __future__ import annotations

from pathlib import Path
import unittest

from scripts import build_technical_handoff as builder
from structvision import live_console


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOTS = (
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "apps",
    ROOT / "docs",
)
PUBLIC_TEXT_SUFFIXES = {".html", ".md", ".py", ".sh", ".svg", ".toml"}


class GlobalProductIdentityTests(unittest.TestCase):
    def test_primary_identity_and_status_language_are_global(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(
            "A modular, evidence-driven visual anomaly proposal and inspection "
            "research\nplatform for marine and structural imagery.",
            readme,
        )
        self.assertIn("stable frozen classical baseline", readme.lower())
        self.assertIn("protected development baseline", readme.lower())
        self.assertIn("rejected development candidate", readme.lower())
        self.assertIn("real-domain validation remains pending", readme.lower())

    def test_new_entry_point_is_registered_and_retired_entry_point_is_absent(self):
        metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn(
            'structvision-live-demo = "structvision.live_console:main"',
            metadata,
        )
        self.assertNotIn("structvision-professor-demo", metadata)
        self.assertTrue((ROOT / "src" / "structvision" / "live_console.py").is_file())
        self.assertFalse(
            (ROOT / "src" / "structvision" / "professor_console.py").exists()
        )

    def test_public_paths_and_public_text_have_no_meeting_identity(self):
        public_files = [ROOT / "README.md", ROOT / "pyproject.toml"]
        for base in PUBLIC_ROOTS:
            public_files.extend(path for path in base.rglob("*") if path.is_file())
        for path in public_files:
            relative = path.relative_to(ROOT).as_posix().lower()
            self.assertNotIn("professor", relative, relative)
            if path.suffix.lower() not in PUBLIC_TEXT_SUFFIXES:
                continue
            text = path.read_text(encoding="utf-8").lower()
            for phrase in (
                "professor",
                "internship",
                "department project",
                "student demo",
                "college project",
            ):
                self.assertNotIn(phrase, text, f"{phrase!r} in {relative}")

    def test_public_app_scripts_and_bundle_use_global_names(self):
        expected = (
            ROOT / "apps" / "structvision_demo.py",
            ROOT / "scripts" / "run_live_demo.sh",
            ROOT / "scripts" / "build_technical_handoff.py",
            ROOT / "docs" / "live-console-block-diagram.md",
            ROOT / "docs" / "technical-handoff.html",
        )
        self.assertTrue(all(path.is_file() for path in expected))
        self.assertEqual(
            builder.BUNDLE_NAME,
            "StructVision-AI-Technical-Handoff",
        )

    def test_live_marker_and_schema_identities_are_exact(self):
        self.assertEqual(
            live_console.OWNERSHIP_MARKER_NAME,
            ".structvision-live-console-owner.json",
        )
        self.assertEqual(
            live_console.OWNERSHIP_SCHEMA,
            "structvision-live-console-run-owner-v1",
        )
        self.assertEqual(live_console.OWNERSHIP_TOOL, "structvision-live-demo")
        self.assertEqual(
            live_console.RUN_MANIFEST_SCHEMA,
            "structvision-live-run-manifest-v1",
        )

    def test_generated_bundle_public_templates_have_no_meeting_identity(self):
        commit = "a" * 40
        payloads = (
            builder._readme_markdown(commit),
            builder._readme_html(commit),
            builder._run_demo_script(),
            builder._input_note(),
            builder._installation_macos(),
            builder._installation_linux(),
            builder._requirements_note(),
            builder._version_text(
                commit=commit,
                branch="main",
                clean=True,
                timestamp="2026-07-29T00:00:00+00:00",
                file_count=1,
                bundle_size=1,
            ),
        )
        for payload in payloads:
            self.assertNotIn("professor", payload.lower())
        self.assertIn("StructVision-AI", "\n".join(payloads))


if __name__ == "__main__":
    unittest.main()
