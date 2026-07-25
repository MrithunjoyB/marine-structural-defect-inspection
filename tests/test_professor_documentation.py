from __future__ import annotations

import re
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
DOCS = (
    ROOT / "docs" / "algorithm-specification.md",
    ROOT / "docs" / "algorithm-pseudocode.md",
    ROOT / "docs" / "code-structure-guide.md",
    ROOT / "docs" / "professor-data-adapter.md",
    ROOT / "docs" / "professor-handoff.md",
    ROOT / "docs" / "professor-demo-runbook.md",
    ROOT / "docs" / "research-evidence-summary.md",
)


class ProfessorDocumentationTests(unittest.TestCase):
    def test_all_relative_markdown_links_resolve(self):
        files = DOCS + (ROOT / "README.md",)
        pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
        for document in files:
            text = document.read_text(encoding="utf-8")
            for target in pattern.findall(text):
                if "://" in target or target.startswith("#"):
                    continue
                relative = target.split("#", 1)[0]
                if not relative:
                    continue
                resolved = (document.parent / relative).resolve()
                self.assertTrue(
                    resolved.exists(),
                    f"{document.relative_to(ROOT)} contains unresolved link {target}",
                )

    def test_mermaid_fences_are_balanced_and_use_known_graph_types(self):
        for document in DOCS:
            text = document.read_text(encoding="utf-8")
            self.assertEqual(text.count("```mermaid"), len(re.findall(r"```mermaid", text)))
            if "```mermaid" in text:
                for block in text.split("```mermaid")[1:]:
                    body, closing, _ = block.partition("```")
                    self.assertTrue(closing, f"Unclosed Mermaid block in {document.name}")
                    first = next(line.strip() for line in body.splitlines() if line.strip())
                    self.assertTrue(
                        first.startswith(("flowchart ", "graph ", "sequenceDiagram")),
                        f"Unknown Mermaid diagram type in {document.name}",
                    )

    def test_pseudocode_references_real_modules_and_functions(self):
        text = (ROOT / "docs" / "algorithm-pseudocode.md").read_text(encoding="utf-8")
        expected = (
            ("src/structvision/api.py", "class StructuralAnomalyDetector"),
            ("src/structvision/inputs.py", "def normalise_input"),
            ("src/structvision/classical.py", "def run_frozen_classical"),
            ("src/structvision/normal_feature/patchcore.py", "def fit_normal"),
            ("src/structvision/normal_feature/patchcore.py", "def analyse"),
            ("src/structvision/hybrid/detector.py", "def analyse"),
            ("src/structvision/executor.py", "def execute"),
        )
        for path, symbol in expected:
            self.assertIn(path, text)
            self.assertIn(symbol, (ROOT / path).read_text(encoding="utf-8"))

    def test_stored_metrics_and_hybrid_rejection_match_source_document(self):
        source = (
            ROOT / "docs" / "results" / "proposal-guided-hybrid-development.md"
        ).read_text(encoding="utf-8")
        evidence = (ROOT / "docs" / "research-evidence-summary.md").read_text(encoding="utf-8")
        values = (
            "0.770833",
            "0.750000",
            "0.894737",
            "0.868421",
            "0.168950",
            "0.720000",
            "4.411765",
            "0.323529",
            "0.621954",
            "0.631250",
        )
        for value in values:
            self.assertIn(value, source)
            self.assertIn(value, evidence)
        for document in DOCS + (ROOT / "README.md",):
            text = document.read_text(encoding="utf-8").lower()
            if "hybrid" in text:
                self.assertTrue(
                    "rejected development candidate" in text
                    or "rejected under the predeclared protocol" in text,
                    f"Hybrid rejection missing from {document.name}",
                )

    def test_no_unsupported_claim_or_personal_promotion(self):
        forbidden = (
            "globally best",
            "state-of-the-art performance",
            "real-world validated",
            "publication-ready",
            "deployment-ready",
            "curriculum vitae",
            "placement",
        )
        for document in DOCS:
            text = document.read_text(encoding="utf-8").lower()
            for phrase in forbidden:
                self.assertNotIn(phrase, text, f"{phrase!r} in {document.name}")

    def test_required_handoff_and_adapter_sections_exist(self):
        handoff = (ROOT / "docs" / "professor-handoff.md").read_text(encoding="utf-8")
        for heading in (
            "What the system currently does",
            "What it does not yet prove",
            "Install the base system",
            "Why the hybrid was rejected",
            "Information required from the professor",
            "Proposed pilot protocol",
        ):
            self.assertIn(heading, handoff)
        adapter = (ROOT / "docs" / "professor-data-adapter.md").read_text(encoding="utf-8")
        for field in (
            "sample_id",
            "image_content_sha256",
            "acquisition_group_id",
            "annotation_version",
            "confidentiality_classification",
            "split_lock_hash",
        ):
            self.assertIn(field, adapter)


if __name__ == "__main__":
    unittest.main()
