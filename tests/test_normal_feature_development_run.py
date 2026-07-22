from pathlib import Path
import hashlib
import json
import sqlite3
import unittest


class CompletedNormalFeatureDevelopmentRunTests(unittest.TestCase):
    """Read-only guards for the ignored reference artifacts when locally available."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).parents[1]
        cls.run_directory = cls.root / "outputs/normal-feature-development/SYN-NORMAL-FEATURE-DEV-001-v1"
        cls.summary_path = cls.run_directory / "development-summary.json"
        cls.database = cls.run_directory / "v2-development-results.sqlite3"
        if not cls.summary_path.is_file() or not cls.database.is_file():
            raise unittest.SkipTest("Ignored normal-feature development artifacts are not present")
        cls.summary = json.loads(cls.summary_path.read_text(encoding="utf-8"))

    def test_scope_artifacts_and_reference_policy(self):
        summary = self.summary
        self.assertEqual(summary["experiment_id"], "SYN-NORMAL-FEATURE-DEV-001")
        self.assertEqual(summary["experiment_version"], 1)
        self.assertEqual(summary["classification"], "development-only — non-confirmatory")
        self.assertEqual(summary["normal_fit_count"], 91)
        self.assertEqual(summary["calibration_validation_count"], 72)
        self.assertFalse(summary["historical_test_access"])
        self.assertFalse(summary["professor_data_access"])
        self.assertFalse(summary["hybrid_method_implemented"])
        self.assertFalse(summary["deprecated_balanced_score_used"])
        self.assertEqual(summary["scientific_device"], "cpu")
        self.assertFalse(summary["mps_scientific_reference"])

    def test_complete_two_method_v2_pairing_and_append_only_attempt(self):
        connection = sqlite3.connect(self.database.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            attempt = connection.execute(
                "SELECT expected_pairs,attempted_pairs,completed_pairs,failed_pairs,"
                "skipped_pairs,unique_stored_pairs FROM execution_attempts"
            ).fetchone()
            rows = connection.execute(
                "SELECT image_id,method_implementation_id,evaluation_policy_id,"
                "model_artifact_identity FROM ("
                "SELECT image_id,method_implementation_id,evaluation_policy_id,"
                "json_extract(proposal_output_details_json,'$.model_artifact_hash') "
                "AS model_artifact_identity FROM result_rows)"
            ).fetchall()
        finally:
            connection.close()
        self.assertEqual(attempt, (144, 144, 144, 0, 0, 144))
        self.assertEqual(len(rows), 144)
        by_image = {}
        for image_id, method, policy, artifact in rows:
            by_image.setdefault(image_id, set()).add(method)
            self.assertEqual(policy, "structvision-eval-v2")
            if method == "structvision-patchcore-baseline-v1-dev":
                self.assertEqual(artifact, self.summary["model_artifact_hash"])
        self.assertEqual(len(by_image), 72)
        self.assertTrue(all(methods == {
            "structvision-classical-baseline-v1-frozen",
            "structvision-patchcore-baseline-v1-dev",
        } for methods in by_image.values()))
        self.assertTrue(self.summary["pairing_complete"])
        self.assertEqual(self.summary["expected_result_rows"], 144)
        self.assertEqual(self.summary["actual_result_rows"], 144)

    def test_persisted_artifact_identities_and_store_hash(self):
        summary = self.summary
        model = self.run_directory / "model-artifacts" / f"{summary['model_artifact_hash']}.json"
        bank = self.run_directory / "model-artifacts" / f"{summary['model_artifact_hash']}.npz"
        calibration = self.run_directory / "calibration-artifacts" / f"{summary['calibration_artifact_hash']}.json"
        self.assertTrue(model.is_file() and bank.is_file() and calibration.is_file())
        self.assertEqual(summary["manifest_hash"], "2aa40b9db145a37522775b7ac605ae201b91e564cde881528fd6d41f449f3d58")
        self.assertEqual(summary["weight_hash"], "03b71d65fb2c73bb0de079a1781009f27a782ec481d2f64ab3bde9b1cdec3000")
        self.assertEqual(summary["dependency_lock_hash"], "be3a00936219aedbcc397f0b3e8c0af6d901489a06550f3b148c72e22cea87b8")
        self.assertEqual(summary["memory_bank_shape"], [151, 1536])
        self.assertEqual(
            hashlib.sha256(self.database.read_bytes()).hexdigest(),
            "3a0200e75fde0633587f961d297d91259868df7120f176f5abfa2af9e73febf1",
        )


if __name__ == "__main__":
    unittest.main()
