from __future__ import annotations

from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


class LiveConsoleBlockDiagramTests(unittest.TestCase):
    def test_mermaid_has_two_valid_flowcharts_and_required_pipeline(self):
        source = (DOCS / "live-console-block-diagram.md").read_text(
            encoding="utf-8"
        )
        blocks = re.findall(r"```mermaid\n(.*?)```", source, re.DOTALL)
        self.assertEqual(len(blocks), 2)
        self.assertTrue(
            all(block.strip().startswith("flowchart TD") for block in blocks)
        )
        required = (
            "INPUT IMAGE",
            "VALIDATION AND NORMALISATION",
            "PREPROCESSING",
            "VISUAL EVIDENCE EXTRACTION",
            "CANDIDATE REGION GENERATION / BINARY MASKS",
            "CONTEXTUAL SCORING AND RANKING",
            "TYPED RESULTS AND PROVENANCE",
            "EXPLICIT OUTPUT FILES",
        )
        for label in required:
            self.assertIn(label, blocks[0])
        self.assertEqual(
            sum(block.count("[") for block in blocks),
            sum(block.count("]") for block in blocks),
        )

    def test_module_names_exist_and_statuses_are_exact(self):
        source = (DOCS / "live-console-block-diagram.md").read_text(
            encoding="utf-8"
        )
        modules = (
            "src/structvision/inputs.py",
            "preprocess.py",
            "feature_extraction.py",
            "region_proposal.py",
            "scoring.py",
            "src/structvision/classical.py",
            "src/structvision/types.py",
            "src/structvision/cli.py",
            "src/structvision/live_console.py",
        )
        for module in modules:
            self.assertIn(module, source)
            self.assertTrue((ROOT / module).is_file(), module)
        self.assertIn("stable default", source)
        self.assertIn("protected development baseline", source)
        self.assertIn("rejected development candidate", source)

    def test_svg_is_self_contained_accessible_and_has_both_architectures(self):
        path = DOCS / "live-console-block-diagram.svg"
        root = ET.fromstring(path.read_bytes())
        self.assertTrue(root.tag.endswith("svg"))
        text = path.read_text(encoding="utf-8")
        self.assertIn("<title", text)
        self.assertIn("<desc", text)
        self.assertNotIn("<script", text.lower())
        self.assertNotRegex(text, r"(?:href|src)=[\"']https?://")
        self.assertIn("STABLE DEFAULT", text)
        self.assertIn("DEVELOPMENT BASELINE", text)
        self.assertIn("REJECTED DEVELOPMENT CANDIDATE", text)

    def test_offline_html_has_no_remote_assets_and_guide_has_all_sections(self):
        diagram = (
            DOCS / "live-console-block-diagram.html"
        ).read_text(encoding="utf-8")
        guide = (DOCS / "technical-handoff.html").read_text(
            encoding="utf-8"
        )
        for payload in (diagram, guide):
            self.assertNotRegex(payload, r"https?://")
            self.assertNotIn("<script", payload.lower())
            self.assertIn("<style>", payload)
        self.assertIn(
            'src="live-console-block-diagram.svg"',
            diagram,
        )
        sections = (
            "Project purpose",
            "Current method status",
            "Block diagram",
            "Input–process–output explanation",
            "Exact Terminal commands",
            "Run folder structure",
            "Output-file definitions",
            "Important source modules",
            "Demonstration script",
            "Current evidence and limitations",
            "Why the hybrid is rejected",
            "Future private-data integration",
            "Security and privacy",
            "Troubleshooting",
            "Questions a technical reviewer may ask",
        )
        for heading in sections:
            self.assertIn(heading, guide)
        self.assertIn("structvision-live-demo", guide)
        self.assertIn("--verify", guide)
        self.assertIn("not confirmed defects", guide)


if __name__ == "__main__":
    unittest.main()
