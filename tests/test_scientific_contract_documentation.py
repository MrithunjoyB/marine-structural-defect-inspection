from pathlib import Path
import re
import unittest


class ScientificContractDocumentationTests(unittest.TestCase):
    def test_local_markdown_links_resolve(self):
        root = Path(__file__).parents[1]
        documents = [root / "README.md", *sorted((root / "docs").rglob("*.md"))]
        failures = []
        pattern = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
        for document in documents:
            for target in pattern.findall(document.read_text(encoding="utf-8")):
                target = target.strip().strip("<>")
                if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                    continue
                relative = target.split("#", 1)[0]
                if not (document.parent / relative).resolve().exists():
                    failures.append(f"{document.relative_to(root)} -> {target}")
        self.assertEqual(failures, [])

    def test_required_status_language_is_present(self):
        root = Path(__file__).parents[1]
        contract = (root / "docs" / "scientific-contract-v2.md").read_text(encoding="utf-8")
        overlap = (root / "docs" / "audits" / "historical-dataset-overlap.md").read_text(encoding="utf-8")
        self.assertIn("structvision-eval-v1-historical", contract)
        self.assertIn("structvision-eval-v2", contract)
        self.assertIn("historical engineering comparison — not confirmatory", overlap)
        self.assertIn("80", overlap)
        self.assertIn("13", overlap)

    def test_scientific_modules_have_no_streamlit_or_absolute_local_paths(self):
        root = Path(__file__).parents[1]
        for path in (root / "scientific_contract").glob("*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("streamlit", source.lower(), path.name)
            self.assertNotIn("/" + "Users/", source, path.name)


if __name__ == "__main__":
    unittest.main()
