from __future__ import annotations

import ast
from contextlib import redirect_stderr, redirect_stdout
import csv
from hashlib import sha256
from io import BytesIO, StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, PngImagePlugin

from structvision import (
    CLASSICAL_METHOD,
    candidate_rows,
    demonstration_fixture,
)
from structvision import professor_console


ROOT = Path(__file__).resolve().parents[1]


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


class ProfessorConsoleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            dir=Path(tempfile.gettempdir()).resolve()
        )
        self.root = Path(self.temporary.name)
        self.input = self.root / "demonstration-fixture.png"
        self.input.write_bytes(fixture_bytes())

    def tearDown(self):
        self.temporary.cleanup()

    def run_console(self, output: Path | None = None, *extra: str):
        selected = output or self.root / "demo-run"
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = professor_console.main(
                [
                    "--input",
                    str(self.input),
                    "--output-dir",
                    str(selected),
                    *extra,
                ]
            )
        return status, stdout.getvalue(), stderr.getvalue(), selected

    def test_wrapper_imports_only_public_structvision_interface(self):
        path = ROOT / "src" / "structvision" / "professor_console.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        structvision_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                structvision_imports.extend(
                    alias.name
                    for alias in node.names
                    if alias.name.startswith("structvision")
                )
            elif (
                isinstance(node, ast.ImportFrom)
                and (node.module or "").startswith("structvision")
            ):
                structvision_imports.append(node.module or "")
        self.assertEqual(structvision_imports, ["structvision"])

    def test_wrapper_contains_no_detector_math_network_install_or_download(self):
        source = (
            ROOT / "src" / "structvision" / "professor_console.py"
        ).read_text(encoding="utf-8")
        for forbidden in (
            "apply_preprocessing",
            "threshold_level",
            "requests",
            "urllib",
            "socket",
            "subprocess",
            "pip install",
            "api_key",
            "hf_hub_download",
        ):
            self.assertNotIn(forbidden, source)
        launcher = (ROOT / "scripts" / "run_professor_demo.sh").read_text(
            encoding="utf-8"
        )
        for forbidden_command in ("pip ", "curl ", "wget ", "git clone"):
            self.assertNotIn(forbidden_command, launcher)

    def test_output_directory_is_required_and_default_method_is_fixed(self):
        with self.assertRaises(SystemExit) as raised:
            professor_console.build_parser().parse_args(
                ["--input", str(self.input)]
            )
        self.assertEqual(raised.exception.code, professor_console.EXIT_USAGE)
        help_text = professor_console.build_parser().format_help()
        self.assertNotIn("--method", help_text)
        self.assertIn("stable frozen", help_text)

    def test_existing_directory_is_protected_before_detector_execution(self):
        output = self.root / "existing"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with patch.object(
            professor_console,
            "analyse_demonstration_image",
        ) as detector:
            status, _stdout, stderr, _ = self.run_console(output)
        self.assertEqual(status, professor_console.EXIT_OUTPUT_EXISTS)
        self.assertIn("Output protection", stderr)
        detector.assert_not_called()
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_complete_run_calls_frozen_detector_once_and_writes_only_below_output(self):
        before = {
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*")
            if path.is_file()
        }
        original = professor_console.analyse_demonstration_image
        with patch.object(
            professor_console,
            "analyse_demonstration_image",
            wraps=original,
        ) as detector:
            status, stdout, stderr, output = self.run_console()
        self.assertEqual(status, professor_console.EXIT_SUCCESS, stderr)
        detector.assert_called_once()
        self.assertEqual(
            detector.call_args.kwargs["method_id"],
            CLASSICAL_METHOD,
        )
        after = {
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*")
            if path.is_file()
        }
        new_files = after - before
        self.assertTrue(new_files)
        self.assertTrue(
            all(path.startswith("demo-run/") for path in new_files)
        )
        self.assertIn("Measured processing stages:", stdout)
        self.assertIn("Selected proposals:", stdout)
        self.assertIn(professor_console.SCORE_WARNING, stdout)
        self.assertNotIn(str(self.root), stdout)
        self.assertFalse((output / "PROCESSING" / ".runtime").exists())

    def test_run_structure_hashes_timings_proposals_and_coordinate_contract(self):
        status, stdout, stderr, output = self.run_console()
        self.assertEqual(status, 0, stderr)
        required = {
            "INPUT/original.png",
            "INPUT/input-metadata.json",
            "PROCESSING/pipeline-stages.json",
            "PROCESSING/stage-timings.csv",
            "PROCESSING/anomaly-evidence.png",
            "PROCESSING/README.txt",
            "OUTPUT/overlay.png",
            "OUTPUT/proposals.csv",
            "OUTPUT/result.json",
            "OUTPUT/technical-summary.txt",
            "RUN_MANIFEST.json",
            professor_console.OWNERSHIP_MARKER_NAME,
            "CONSOLE_LOG.txt",
        }
        actual = {
            path.relative_to(output).as_posix()
            for path in output.rglob("*")
            if path.is_file()
        }
        self.assertTrue(required <= actual)
        manifest = json.loads(
            (output / "RUN_MANIFEST.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["detector_execution_count"], 1)
        self.assertEqual(
            manifest["method"]["method_id"],
            CLASSICAL_METHOD,
        )
        self.assertEqual(
            manifest["method"]["status"],
            "stable frozen baseline",
        )
        self.assertTrue(
            manifest["implementation"]["protected_source_hashes_verified"]
        )
        for item in manifest["files"]:
            path = output / item["path"]
            self.assertTrue(path.is_file(), item["path"])
            self.assertEqual(path.stat().st_size, item["size_bytes"])
            self.assertEqual(
                sha256(path.read_bytes()).hexdigest(),
                item["sha256"],
            )
        stages = [item["stage"] for item in manifest["timings"]]
        self.assertEqual(
            stages,
            [
                "input_normalisation",
                "preprocessing",
                "feature_extraction",
                "proposal_generation",
                "result_conversion",
                "total",
            ],
        )
        self.assertTrue(
            all(
                item["measurement"] == "measured"
                and item["seconds"] >= 0
                for item in manifest["timings"]
            )
        )
        selected = manifest["selected_proposal_count"]
        mask_paths = sorted((output / "OUTPUT" / "masks").glob("*.png"))
        self.assertEqual(len(mask_paths), selected)
        self.assertIn(f"Selected proposals: {selected}", stdout)
        result = json.loads(
            (output / "OUTPUT" / "result.json").read_text(encoding="utf-8")
        )
        dimensions = tuple(result["analysis"]["analysed_dimensions"])
        for mask in mask_paths:
            with Image.open(mask) as image:
                self.assertEqual(
                    image.size,
                    (dimensions[1], dimensions[0]),
                )
        with (output / "OUTPUT" / "proposals.csv").open(
            encoding="utf-8",
            newline="",
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), selected)
        self.assertTrue(
            all(
                row["bbox_convention"]
                == "half-open:x_min,y_min,x_max,y_max"
                for row in rows
            )
        )

    def test_input_hashes_stage_honesty_and_no_absolute_paths(self):
        status, _stdout, stderr, output = self.run_console()
        self.assertEqual(status, 0, stderr)
        metadata = json.loads(
            (output / "INPUT" / "input-metadata.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            metadata["source_encoded_sha256"],
            sha256(self.input.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            metadata["normalised_copy_sha256"],
            sha256((output / "INPUT" / "original.png").read_bytes()).hexdigest(),
        )
        stages = json.loads(
            (output / "PROCESSING" / "pipeline-stages.json").read_text(
                encoding="utf-8"
            )
        )
        evidence = " ".join(item["evidence"] for item in stages["stages"])
        self.assertIn(
            "Not exposed by the current frozen API.",
            evidence,
        )
        processing = (output / "PROCESSING" / "README.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("not invented", processing)
        for path in output.rglob("*"):
            if path.suffix.lower() not in {".json", ".csv", ".txt"}:
                continue
            payload = path.read_text(encoding="utf-8")
            self.assertNotIn(str(self.root), payload, path)
            self.assertNotIn("/Users/", payload, path)
            self.assertNotIn("API_KEY", payload, path)
        summary = (output / "OUTPUT" / "technical-summary.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("not confirmed defects", summary)
        self.assertNotIn("confirmed defect.", summary.lower())

    def test_overwrite_requires_explicit_flag_and_has_deterministic_exit_codes(self):
        output = self.root / "demo-run"
        first, _stdout, _stderr, _ = self.run_console(output)
        second, _stdout, _stderr, _ = self.run_console(output)
        third, _stdout, stderr, _ = self.run_console(output, "--overwrite")
        self.assertEqual(first, professor_console.EXIT_SUCCESS)
        self.assertEqual(second, professor_console.EXIT_OUTPUT_EXISTS)
        self.assertEqual(third, professor_console.EXIT_SUCCESS, stderr)
        invalid = self.root / "bad.png"
        invalid.write_bytes(b"not an image")
        prior = self.input
        self.input = invalid
        try:
            status, _stdout, _stderr, _ = self.run_console(
                self.root / "invalid-run"
            )
        finally:
            self.input = prior
        self.assertEqual(status, professor_console.EXIT_INPUT)


if __name__ == "__main__":
    unittest.main()
