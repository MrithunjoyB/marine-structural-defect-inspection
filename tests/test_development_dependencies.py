from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
import re
import unittest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def _distribution_name(requirement: str) -> str:
    return re.split(r"[\s<>=!~;\[]", requirement, maxsplit=1)[0].lower()


def _declared_surfaces() -> dict[str, set[str]]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    core = {
        _distribution_name(requirement)
        for requirement in project["project"]["dependencies"]
    }
    optional = project["project"]["optional-dependencies"]
    return {
        "core": core,
        "dev": {
            _distribution_name(requirement)
            for requirement in optional["dev"]
        },
        "demo": {
            _distribution_name(requirement)
            for requirement in optional["demo"]
        },
    }


class DevelopmentDependencyTests(unittest.TestCase):
    def test_default_test_imports_have_declared_installation_providers(self):
        surfaces = _declared_surfaces()
        expected = {
            "numpy": ("numpy", "core"),
            "cv2": ("opencv-python-headless", "core"),
            "PIL": ("pillow", "core"),
            "pytest": ("pytest", "dev"),
            "matplotlib": ("matplotlib", "dev"),
            "pandas": ("pandas", "dev"),
            "reportlab": ("reportlab", "dev"),
            "streamlit": ("streamlit", "demo"),
        }
        for import_name, (distribution, surface) in expected.items():
            self.assertIn(distribution, surfaces[surface], import_name)
            self.assertIsNotNone(find_spec(import_name), import_name)

    def test_matplotlib_constraint_is_exactly_on_development_surface(self):
        project = tomllib.loads(
            (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        self.assertIn(
            "matplotlib>=3.8.0",
            project["project"]["optional-dependencies"]["dev"],
        )

    def test_default_surfaces_exclude_heavy_learned_runtimes(self):
        surfaces = _declared_surfaces()
        declared = set().union(
            surfaces["core"],
            surfaces["dev"],
            surfaces["demo"],
        )
        self.assertTrue(
            {
                "torch",
                "torchvision",
                "anomalib",
                "ultralytics",
                "timm",
                "safetensors",
            }.isdisjoint(declared)
        )
