from pathlib import Path
import unittest

import pytest

from protected_test_support import require_protected_files
from scientific_contract.dataset_audit import (
    DatasetImageIdentity,
    audit_dataset_overlap,
    audit_registered_datasets,
)


def image(dataset, identifier, sha, phash, split="train", category="crack", source="", template="", acquisition="", status="unique"):
    return DatasetImageIdentity(dataset, "1.0", identifier, f"{identifier}.png", split, category, sha, phash, source, template, acquisition, status)


class DatasetOverlapAuditTests(unittest.TestCase):
    def test_exact_test_overlap_groups_and_unsupported_unique_status(self):
        left = [image("pilot", "p1", "a" * 64, "0000", source="s1", template="t1", acquisition="a1")]
        right = [image("final", "f1", "a" * 64, "0000", split="test", source="s1", template="t1", acquisition="a1")]
        report = audit_dataset_overlap(left, right, perceptual_distance_threshold=1)
        self.assertEqual(len(report.exact_duplicates), 1)
        self.assertEqual(dict(report.overlap_by_right_split), {"test": 1})
        self.assertEqual(len(report.source_group_crossings), 1)
        self.assertEqual(len(report.template_group_crossings), 1)
        self.assertEqual(len(report.acquisition_group_crossings), 1)
        self.assertEqual(len(report.unsupported_unique_statuses), 1)
        self.assertFalse(report.confirmatory_test_protected)
        self.assertEqual(report.evidence_classification, "historical engineering comparison — not confirmatory")

    def test_perceptual_hash_reports_candidates_not_uniqueness_proof(self):
        left = [image("left", "l", "a" * 64, "0000")]
        right = [image("right", "r", "b" * 64, "0001")]
        report = audit_dataset_overlap(left, right, perceptual_distance_threshold=1)
        self.assertEqual(len(report.perceptual_near_duplicate_candidates), 1)
        self.assertEqual(report.perceptual_near_duplicate_candidates[0].distance, 1)
        self.assertEqual(len(report.unsupported_unique_statuses), 1)
        self.assertFalse(report.perceptual_uniqueness_established)

    @pytest.mark.protected_integration
    def test_current_historical_registry_confirms_80_and_13(self):
        root = require_protected_files(
            Path(__file__).parents[1],
            "research_data/registry/datasets.sqlite",
        )
        database = root / "research_data" / "registry" / "datasets.sqlite"
        report = audit_registered_datasets(
            database, "synthetic-expanded-pilot", "1.0",
            "synthetic-expanded", "1.0",
        )
        self.assertEqual(report.left_image_count, 80)
        self.assertEqual(len(report.exact_duplicates), 80)
        self.assertEqual(dict(report.overlap_by_right_split)["test"], 13)
        self.assertGreaterEqual(len(report.unsupported_unique_statuses), 80)
        self.assertFalse(report.perceptual_uniqueness_established)


if __name__ == "__main__":
    unittest.main()
