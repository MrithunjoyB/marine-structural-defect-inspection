from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"


def cli_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SOURCE)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def write_image(path: Path) -> None:
    yy, xx = np.indices((96, 128))
    image = np.clip(110 + 25 * np.sin(xx / 8.0) + 10 * np.cos(yy / 9.0), 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)


class LiveCliTests(unittest.TestCase):
    def run_cli(self, arguments: list[str], cwd: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "structvision.cli", *arguments],
            cwd=cwd,
            env=cli_environment(),
            capture_output=True,
            check=False,
        )

    def test_no_write_default_and_machine_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "inspection.png"
            write_image(image)
            before = {path.name for path in root.iterdir()}
            completed = self.run_cli(
                ["--input", str(image), "--method", "classical", "--stdout-json"],
                root,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
            self.assertEqual(before, {path.name for path in root.iterdir()})
            payload = json.loads(completed.stdout)
            self.assertEqual(
                payload["method"]["method_id"],
                "structvision-classical-baseline-v1-frozen",
            )

    def test_explicit_outputs_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "inspection.png"
            output_json = root / "result.json"
            output_overlay = root / "overlay.png"
            write_image(image)
            completed = self.run_cli(
                [
                    "--input",
                    str(image),
                    "--json-out",
                    str(output_json),
                    "--overlay-out",
                    str(output_overlay),
                ],
                root,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
            self.assertTrue(output_json.is_file())
            self.assertTrue(output_overlay.is_file())

    def test_corrupt_input_and_invalid_method_have_deterministic_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corrupt = root / "corrupt.png"
            corrupt.write_bytes(b"not-an-image")
            completed = self.run_cli(["--input", str(corrupt)], root)
            self.assertEqual(completed.returncode, 3)
            invalid = self.run_cli(
                ["--input", str(corrupt), "--method", "not-a-method"],
                root,
            )
            self.assertEqual(invalid.returncode, 2)

    def test_unavailable_learned_environment_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "inspection.png"
            write_image(image)
            completed = self.run_cli(
                ["--input", str(image), "--method", "patchcore"],
                root,
            )
            self.assertEqual(completed.returncode, 4)
            self.assertIn(b"Learned environment unavailable", completed.stderr)

    def test_missing_learned_artifact_has_distinct_exit_code(self):
        from structvision.cli import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "inspection.png"
            write_image(image)
            with patch(
                "structvision.demonstration._required_package_issues",
                return_value=(),
            ):
                self.assertEqual(
                    main(["--input", str(image), "--method", "patchcore"]),
                    5,
                )


if __name__ == "__main__":
    unittest.main()
