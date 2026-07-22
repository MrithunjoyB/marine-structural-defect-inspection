from pathlib import Path
from datetime import datetime
import hashlib
import json
import sqlite3
import unittest

from structvision.hybrid.artifact import load_hybrid_fusion_artifact
from structvision.normal_feature.model_artifact import load_model_artifact


class HybridDevelopmentReferenceRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).parents[1]
        cls.run_directory = cls.root / "outputs/proposal-guided-hybrid/SYN-PROPOSAL-HYBRID-DEV-001-v1"
        cls.summary_path = cls.run_directory / "development-summary.json"
        if not cls.summary_path.is_file():
            raise unittest.SkipTest("Ignored hybrid development reference artifacts are not present")
        cls.summary = json.loads(cls.summary_path.read_text())

    def test_frozen_artifact_selection_and_honest_rejected_decision(self):
        summary = self.summary
        self.assertEqual(summary["manifest_hash"], "a1e6f9a83e5e8d73275236e6dc4fafd985e6e1ef2c4aef21fd4156dc821829a4")
        self.assertEqual(summary["normal_model_artifact_hash"], "ef275b0a853231a239eebcccab6c920667616695296450d2d44453d922c341e7")
        self.assertEqual(summary["fusion_artifact_hash"], "a21b5880c5d8f16d3869227455279ddbf18815d92ae7862e262cc2560de3d8d1")
        self.assertEqual(summary["fusion_fit"]["selected_configuration_id"], "cw-0.60_nw-0.40_floor-none")
        self.assertEqual(summary["development_decision"], "development candidate rejected under the predeclared protocol")
        self.assertEqual(summary["holdout_preservation_failures"], [
            "overall_micro_sensitivity_decrease_exceeds_0.02",
            "image_level_sensitivity_decreased",
        ])
        self.assertTrue(summary["holdout_clean_burden_improved"])
        self.assertTrue(summary["holdout_primary_budget_met"])

    def test_artifacts_roundtrip_and_holdout_was_attempted_once(self):
        model_id = self.summary["normal_model_artifact_hash"]
        fusion_id = self.summary["fusion_artifact_hash"]
        model = load_model_artifact(self.run_directory / f"hybrid-model-artifacts/{model_id}.json")
        fusion = load_hybrid_fusion_artifact(self.run_directory / f"fusion-artifacts/{fusion_id}.json")
        self.assertEqual(model.artifact_hash, model_id)
        self.assertEqual(model.memory_bank_shape, (116, 1536))
        self.assertEqual(fusion.artifact_hash, fusion_id)
        self.assertEqual(fusion.selected_coefficients, (0.6, 0.4))
        self.assertEqual(len(fusion.evaluated_configurations), 15)
        events = [json.loads(line) for line in (self.run_directory / "holdout-attempts.jsonl").read_text().splitlines()]
        self.assertEqual([item["event"] for item in events], ["primary_holdout_started", "primary_holdout_finished"])
        self.assertEqual(events[-1]["status"], "completed")
        self.assertLess(datetime.fromisoformat(fusion.creation_timestamp), datetime.fromisoformat(events[0]["timestamp"]))

    def test_three_method_v2_store_is_complete_and_paired(self):
        database = self.run_directory / "v2-hybrid-development-results.sqlite3"
        connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            counts = dict(connection.execute(
                "SELECT method_implementation_id,count(*) FROM result_rows GROUP BY method_implementation_id"
            ))
            policies = dict(connection.execute(
                "SELECT evaluation_policy_id,count(*) FROM result_rows GROUP BY evaluation_policy_id"
            ))
            attempts = connection.execute("SELECT count(*) FROM execution_attempts").fetchone()[0]
            per_image = connection.execute(
                "SELECT image_id,count(*),count(DISTINCT method_implementation_id) "
                "FROM result_rows GROUP BY image_id"
            ).fetchall()
        finally:
            connection.close()
        self.assertEqual(counts, {
            "structvision-classical-baseline-v1-frozen": 72,
            "structvision-patchcore-baseline-v1-dev": 72,
            "structvision-proposal-guided-hybrid-v1-dev": 72,
        })
        self.assertEqual(policies, {"structvision-eval-v2": 216})
        self.assertEqual(attempts, 1)
        self.assertEqual(len(per_image), 72)
        self.assertTrue(all(total == 3 and methods == 3 for _, total, methods in per_image))
        self.assertEqual(self.summary["expected_result_rows"], 216)
        self.assertEqual(self.summary["actual_result_rows"], 216)
        self.assertTrue(self.summary["pairing_complete"])

    def test_protected_stores_implementations_and_prior_development_artifacts_are_unchanged(self):
        self.assertTrue(self.summary["protected_unchanged"])
        self.assertEqual(self.summary["protected_before"], self.summary["protected_after"])
        self.assertEqual(self.summary["protected_after"]["historical_automatic_rows"], 888)
        self.assertFalse(self.summary["historical_test_access"])
        self.assertFalse(self.summary["professor_data_access"])
        self.assertFalse(self.summary["api_key_required"])


if __name__ == "__main__":
    unittest.main()
