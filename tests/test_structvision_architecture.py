import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class StructVisionArchitectureTests(unittest.TestCase):
    @staticmethod
    def _build_python():
        candidates = [Path(sys.executable)]
        candidates.extend(Path.home().glob(".cache/codex-runtimes/*/dependencies/python/bin/python3"))
        for candidate in candidates:
            completed = subprocess.run(
                [str(candidate), "-c", "import setuptools, wheel"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if completed.returncode == 0:
                return str(candidate)
        raise unittest.SkipTest("No local Python build environment provides setuptools and wheel")

    def test_package_has_no_streamlit_absolute_local_path_or_api_key_dependency(self):
        root = Path(__file__).parents[1]
        for path in sorted((root / "src" / "structvision").glob("*.py")):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("streamlit", source.lower(), path.name)
            self.assertNotIn("/" + "Users/", source, path.name)
            self.assertNotIn("OPENAI_API_KEY", source, path.name)
            self.assertNotIn("GOOGLE_API_KEY", source, path.name)
        core = "\n".join(
            (root / "src" / "structvision" / name).read_text(encoding="utf-8")
            for name in ("api.py", "classical.py", "configuration.py", "inputs.py", "types.py")
        )
        self.assertNotIn("sqlite3", core)
        self.assertNotIn("session_state", core)
        self.assertNotIn("sidebar", core)

    def test_standard_package_installs_and_runs_outside_repository(self):
        root = Path(__file__).parents[1]
        with TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            project = temporary_root / "project"
            project.mkdir()
            for filename in (
                "pyproject.toml", "README.md", "preprocess.py", "feature_extraction.py",
                "region_proposal.py", "scoring.py",
            ):
                shutil.copy2(root / filename, project / filename)
            shutil.copytree(root / "src", project / "src")
            shutil.copytree(root / "scientific_contract", project / "scientific_contract")
            site = temporary_root / "site"
            installed = subprocess.run(
                [
                    self._build_python(), "-m", "pip", "install", "--no-deps", "--no-build-isolation",
                    "--no-compile", "--target", str(site), str(project),
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(installed.returncode, 0, installed.stderr)
            caller = temporary_root / "caller"
            caller.mkdir()
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(site)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import numpy as np; "
                        "from structvision import DetectorConfig, StructuralAnomalyDetector; "
                        "image=np.full((32,48,3),150,np.uint8); "
                        "result=StructuralAnomalyDetector(DetectorConfig()).analyse("
                        "image,image_id='outside',colour_space='BGR'); "
                        "assert result.image_id=='outside'; print(result.configuration_hash)"
                    ),
                ],
                cwd=caller,
                env=environment,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(len(completed.stdout.strip()), 64)
            self.assertEqual(list(caller.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
