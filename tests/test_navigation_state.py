from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LegacyApplicationPolicyTests(unittest.TestCase):
    def test_legacy_application_fails_closed_without_creating_runtime_paths(self):
        with TemporaryDirectory() as temporary:
            sandbox = Path(temporary)
            before = tuple(sandbox.iterdir())
            environment = dict(os.environ)
            environment.update(
                {
                    "PYTHONPATH": str(ROOT),
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )
            completed = subprocess.run(
                [sys.executable, "-B", "-c", "import app"],
                cwd=sandbox,
                env=environment,
                capture_output=True,
                check=False,
            )
            after = tuple(sandbox.iterdir())
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            b"streamlit run apps/structvision_demo.py",
            completed.stderr,
        )
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
