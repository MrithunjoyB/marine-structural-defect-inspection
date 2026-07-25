from __future__ import annotations

import ast
from hashlib import sha256
from pathlib import Path
import sqlite3
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProfessorArchitectureTests(unittest.TestCase):
    def test_client_imports_only_public_structvision_interface(self):
        client = ROOT / "apps" / "professor_demo.py"
        tree = ast.parse(client.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        structvision_imports = [name for name in imports if name.startswith("structvision")]
        self.assertEqual(structvision_imports, ["structvision"])
        self.assertNotIn("app", structvision_imports)

    def test_dependency_direction_and_streamlit_isolation(self):
        algorithm_files = tuple((ROOT / "src" / "structvision").rglob("*.py"))
        for path in algorithm_files:
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("apps.professor_demo", source)
            self.assertNotIn("import professor_demo", source)
            self.assertNotIn("import streamlit", source)
        client_source = (ROOT / "apps" / "professor_demo.py").read_text(encoding="utf-8")
        self.assertNotIn("research_analysis_ui", client_source)
        self.assertNotIn("region_proposal", client_source)
        self.assertNotIn("normal_feature.patchcore", client_source)
        self.assertNotIn("hybrid.detector", client_source)

    def test_no_automatic_install_download_or_paid_service(self):
        sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                ROOT / "src" / "structvision" / "demonstration.py",
                ROOT / "src" / "structvision" / "cli.py",
                ROOT / "apps" / "professor_demo.py",
            )
        )
        self.assertNotIn("subprocess", sources)
        self.assertNotIn("pip install", sources)
        self.assertNotIn("hf_hub_download", sources)
        self.assertNotIn("requests.", sources)
        self.assertNotIn("api_key=", sources.lower())
        self.assertNotIn("research_data/raw", sources)
        self.assertNotIn("datasets/images", sources)
        self.assertNotIn("professor_data", sources.lower())

    def test_streamlit_client_smoke_is_write_free(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("optional Streamlit demonstration dependency is unavailable")
        protected = (
            ROOT / "outputs" / "registered_experiment_results.sqlite3",
            ROOT / "research_data" / "registry" / "datasets.sqlite",
        )
        before = {
            path: sha256(path.read_bytes()).hexdigest()
            for path in protected
            if path.is_file()
        }
        app = AppTest.from_file(str(ROOT / "apps" / "professor_demo.py"), default_timeout=20)
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual([item.value for item in app.title], ["StructVision-AI"])
        self.assertIn("Analyse an Image", [item.label for item in app.tabs])
        after = {
            path: sha256(path.read_bytes()).hexdigest()
            for path in before
        }
        self.assertEqual(before, after)

    def test_protected_classical_sources_match_frozen_contract(self):
        expected = {
            "preprocess.py": "fcd5da2b563e420b18f5baaf6a73c276457b4b6c65b33531cfeaf917ffefcf48",
            "feature_extraction.py": "1ae26484de02f4d5764d2ee90ee519babe307192c12fa8deecfc50d96ff1976c",
            "region_proposal.py": "65815b84dd8078b11776ccb70e81688e47f4e7afe1624534d6872bec1e46f80a",
            "scoring.py": "d284c8012464003a0ddc5a697c4d85303fbe73a356f8ee7f649c5d75ebcd3a79",
            "explain.py": "efbf6e259d6657172257c583aa02222a1ed9e78df061b5f9825d516a0182583f",
            "report.py": "ae143390a52b36276a3f87193f3c01e4ba64d0747e73c11b5df4d7fcc5dc0752",
            "severity.py": "fd3405039e36df2c98b5b30a421098cf55fedbe39347cbaa215fd03b996f027c",
        }
        for name, digest in expected.items():
            self.assertEqual(sha256((ROOT / name).read_bytes()).hexdigest(), digest)

    def test_local_protected_stores_are_unchanged_when_present(self):
        stores = (
            (
                ROOT / "outputs" / "registered_experiment_results.sqlite3",
                "1ebde1de1f065b5b220366798147beb67dd10a446b7cd8840f988c9aeda9ce92",
                "SELECT COUNT(*) FROM automatic_results",
                888,
            ),
            (
                ROOT
                / "outputs"
                / "normal-feature-development"
                / "SYN-NORMAL-FEATURE-DEV-001-v1"
                / "v2-development-results.sqlite3",
                "3a0200e75fde0633587f961d297d91259868df7120f176f5abfa2af9e73febf1",
                "SELECT COUNT(*) FROM result_rows",
                144,
            ),
            (
                ROOT
                / "outputs"
                / "proposal-guided-hybrid"
                / "SYN-PROPOSAL-HYBRID-DEV-001-v1"
                / "v2-hybrid-development-results.sqlite3",
                "288c415703f4a2f98a9cb998467607510ce44bab0a0b4520b7136912399d3f22",
                "SELECT COUNT(*) FROM result_rows",
                216,
            ),
        )
        for path, digest, query, count in stores:
            if not path.is_file():
                continue
            self.assertEqual(sha256(path.read_bytes()).hexdigest(), digest)
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                self.assertEqual(connection.execute(query).fetchone()[0], count)
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
