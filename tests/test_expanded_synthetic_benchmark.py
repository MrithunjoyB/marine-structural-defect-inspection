from dataclasses import asdict
from pathlib import Path
import json
import tempfile
import unittest

import cv2
import numpy as np

from ablation_study import CONFIG_BY_ID
from expanded_synthetic_benchmark import (
    EXPANDED_CATEGORIES, POSITIVE_CATEGORIES, generate_expanded_cases,
    validate_generated_cases,
)
from region_proposal import AblationConfig
from registered_experiment import AutomaticResult, RegisteredExperimentStore, match_proposals, pairing_audit
from research_analysis import filtered_csv, filtered_json
from research_dataset import (
    DatasetRegistry, check_leakage, prepare_split,
    register_expanded_synthetic_benchmark,
)


class ExpandedSyntheticBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.registry = DatasetRegistry(Path(cls.temp.name) / "research_data")
        cls.records, _, cls.validation = register_expanded_synthetic_benchmark(
            cls.registry, seed=42, samples_per_category=50
        )
        cls.split, cls.leaks, _ = prepare_split(
            cls.registry, "synthetic-expanded", (.6, .2, .2), 42,
            mode="Expanded Synthetic Benchmark",
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_deterministic_multi_seed_generation(self):
        first, first_parameters = generate_expanded_cases(42, 8)
        second, second_parameters = generate_expanded_cases(42, 8)
        different, _ = generate_expanded_cases(43, 8)
        self.assertEqual(first_parameters, second_parameters)
        self.assertTrue(all(np.array_equal(first[name][0], second[name][0]) for name in first))
        self.assertTrue(any(not np.array_equal(first[name][0], different[name][0]) for name in first))

    def test_500_unique_hashes_and_category_balance(self):
        self.assertTrue(self.validation["passed"])
        self.assertEqual(self.validation["image_count"], 500)
        self.assertEqual(self.validation["unique_sha256_count"], 500)
        self.assertEqual(set(self.validation["category_counts"]), set(EXPANDED_CATEGORIES))
        self.assertEqual(set(self.validation["category_counts"].values()), {50})

    def test_mask_and_clean_semantics(self):
        frame = self.registry.images("synthetic-expanded")
        for _, row in frame.iterrows():
            if row.anomaly_type in POSITIVE_CATEGORIES:
                mask = cv2.imread(row.annotation_path, cv2.IMREAD_GRAYSCALE)
                self.assertIsNotNone(mask); self.assertTrue(np.any(mask))
            else:
                self.assertEqual(row.image_outcome, "no_anomaly")
                self.assertEqual(row.annotation_path, "")
        clean = match_proposals([], np.zeros((20, 20), np.uint8))
        self.assertIsNone(clean["proposal_recall"]); self.assertIsNone(clean["top_1_hit"])

    def test_group_aware_exact_split_and_leakage(self):
        self.assertEqual(self.split.groupby("split").size().to_dict(), {"test": 100, "train": 300, "validation": 100})
        composition = self.split.groupby(["anomaly_type", "split"]).size().unstack(fill_value=0)
        self.assertTrue((composition[["train", "validation", "test"]].values == [30, 10, 10]).all())
        self.assertFalse(any(self.leaks.values())); self.assertFalse(any(check_leakage(self.split).values()))
        for column in ("source_group_id", "template_group_id", "near_duplicate_group_id"):
            populated = self.split[self.split[column].astype(str) != ""]
            self.assertTrue((populated.groupby(column).split.nunique() == 1).all())

    def test_deterministic_split_reproduction(self):
        before = dict(zip(self.split.image_id, self.split.split))
        repeated, leaks, _ = prepare_split(self.registry, "synthetic-expanded", (.6, .2, .2), 42, mode="Expanded Synthetic Benchmark")
        self.assertEqual(before, dict(zip(repeated.image_id, repeated.split)))
        self.assertFalse(any(leaks.values()))

    def test_algorithm_configuration_snapshots_are_frozen(self):
        self.assertEqual(asdict(CONFIG_BY_ID["ABL-RERANK-ONLY"].config), asdict(AblationConfig(multi_scale_fusion=False)))
        expected = AblationConfig(multi_scale_fusion=False, specular_suppression=True)
        self.assertEqual(asdict(CONFIG_BY_ID["ABL-RERANK-SPECULAR-SUPPRESS"].config), asdict(expected))

    def test_result_pairing_count_and_exports(self):
        image_ids = self.split[self.split.split == "test"].image_id.tolist()
        methods = ["contour-only baseline", "fixed-threshold baseline", "multi-scale fused method", "refined contextual method", "ABL-RERANK-ONLY", "ABL-RERANK-SPECULAR-SUPPRESS"]
        pairs = {(image_id, method) for image_id in image_ids for method in methods}
        self.assertEqual(len(pairs), 600)
        rows = [{"image_id": image_id, "method": method} for image_id, method in sorted(pairs)]
        import pandas as pd
        frame = pd.DataFrame(rows)
        frame["run_status"] = "completed"
        audit = pairing_audit(frame, image_ids, methods)
        self.assertTrue(audit["complete"]); self.assertEqual(audit["expected_rows"], 600)
        self.assertEqual(len(filtered_csv(frame)), len(frame.to_csv(index=False).encode()))
        self.assertEqual(len(json.loads(filtered_json(frame))), 600)

    def test_result_store_does_not_modify_existing_rows(self):
        store = RegisteredExperimentStore(Path(self.temp.name) / "results.sqlite3")
        self.assertEqual(list(store.dataframe().columns), list(AutomaticResult.__annotations__))


if __name__ == "__main__":
    unittest.main()
