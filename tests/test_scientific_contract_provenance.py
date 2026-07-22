from dataclasses import replace
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import unittest

from scientific_contract.evaluation_policy import default_evaluation_policy
from scientific_contract.hashing import canonical_json, sha256_json
from scientific_contract.provenance import capture_git_state
from scientific_contract.result_store import (
    ExecutionAttemptSummary,
    RESULT_SCHEMA_VERSION,
    ResultRowV2,
    V2ResultStore,
)
from scientific_contract.specification import (
    SPECIFICATION_SCHEMA_VERSION,
    ExperimentSpecificationV2,
    FrozenConfiguration,
    MethodSpecification,
    SelectedImageIdentity,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def configuration(**values):
    return FrozenConfiguration.from_value(values)


def make_spec(method_parameter=1, tree_state="clean", diff_hash=None):
    policy = default_evaluation_policy()
    method = MethodSpecification(
        "method-a", "implementation-1",
        configuration(parameter=method_parameter, threshold=0.7),
        True, "score descending then proposal_id",
    )
    return ExperimentSpecificationV2(
        SPECIFICATION_SCHEMA_VERSION, "EXP-V2", 1, "dataset", "2.0",
        HASH_A, HASH_B, HASH_C,
        (SelectedImageIdentity("image-1", HASH_D, HASH_A),), (method,),
        configuration(clahe=True, clip_limit=2.0),
        configuration(minimum_area=30, morphology="fixed"),
        configuration(weights={"texture": 0.5}, score_threshold=0.7),
        8, (("python", 42), ("numpy", 42)), True,
        policy.policy_id, policy.policy_version, policy.configuration_hash,
        policy.threshold_analyses, policy.metric_definitions_hash,
        ("train", "validation"), True,
        "1" * 40, tree_state, diff_hash,
        "3.12.1", configuration(packages={"numpy": "2.0"}), HASH_B,
        configuration(system="test-os"), configuration(machine="test-cpu"),
        "4.10", "parallel:test", "2026-07-22T10:00:00+05:30",
    )


def make_result(spec, result_id="result-1", attempt_id="attempt-1"):
    return ResultRowV2(
        result_id, spec.specification_hash,
        dict(spec.expected_executed_configuration_hashes)["method-a"],
        "method-a", "implementation-1", spec.evaluation_policy_id,
        spec.evaluation_policy_version, spec.evaluation_policy_hash,
        "image-1", HASH_D, HASH_A, None,
        canonical_json({"proposals": []}),
        default_evaluation_policy().matching_policy_hash,
        RESULT_SCHEMA_VERSION, attempt_id, True,
        "2026-07-22T10:01:00+05:30", "completed",
        canonical_json({
            "proposals": [], "truths": [], "similarity_matrix": [],
            "proposal_decisions": [], "unmatched_truth_ids": [],
        }), canonical_json({"micro_component_sensitivity": None}),
    )


class SpecificationTests(unittest.TestCase):
    def test_specification_hash_is_deterministic_and_serialisable(self):
        first = make_spec()
        second = make_spec()
        self.assertEqual(first.specification_hash, second.specification_hash)
        loaded = ExperimentSpecificationV2.from_json(first.to_json())
        self.assertEqual(loaded, first)
        self.assertEqual(first.expected_pair_count, 1)

    def test_any_executed_parameter_changes_configuration_and_spec_hash(self):
        first = make_spec(1)
        second = make_spec(2)
        self.assertNotEqual(first.specification_hash, second.specification_hash)
        self.assertNotEqual(first.expected_executed_configuration_hashes, second.expected_executed_configuration_hashes)

    def test_empty_placeholder_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            FrozenConfiguration.from_value({})

    def test_execution_mismatch_fails_closed(self):
        spec = make_spec()
        expected = spec.expected_executable_configuration("method-a")
        self.assertEqual(spec.verify_executed_configuration("method-a", expected), dict(spec.expected_executed_configuration_hashes)["method-a"])
        changed = {**expected, "maximum_proposal_count": 9}
        with self.assertRaises(ValueError):
            spec.verify_executed_configuration("method-a", changed)

    def test_clean_and_dirty_git_state_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Scientific Test"], cwd=root, check=True)
            (root / "tracked.txt").write_text("baseline", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "baseline"], cwd=root, check=True)
            clean = capture_git_state(root)
            self.assertTrue(clean.clean)
            self.assertIsNone(clean.uncommitted_diff_hash)
            (root / "tracked.txt").write_text("changed", encoding="utf-8")
            dirty = capture_git_state(root)
            self.assertFalse(dirty.clean)
            self.assertEqual(len(dirty.uncommitted_diff_hash), 64)


class AppendOnlyStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "scientific-v2.sqlite3"
        self.store = V2ResultStore(self.path)
        self.spec = make_spec()
        self.store.register_specification(self.spec)

    def tearDown(self):
        self.temp.cleanup()

    def summary(self, attempt_id="attempt-1"):
        return ExecutionAttemptSummary(
            attempt_id, self.spec.specification_hash, "completed",
            1, 1, 1, 0, 0, 1,
            "2026-07-22T10:00:00+05:30", "2026-07-22T10:02:00+05:30",
        )

    def test_schema_history_foreign_keys_and_hash_persistence(self):
        row = make_result(self.spec)
        self.store.append_attempt(self.spec, self.summary(), [row])
        counts = self.store.counts()
        self.assertEqual(counts["schema_versions"], 1)
        self.assertEqual(counts["migration_history"], 1)
        self.assertEqual(counts["result_rows"], 1)
        with self.store.connect() as connection:
            self.assertEqual(connection.execute("PRAGMA foreign_keys").fetchone()[0], 1)
            stored = connection.execute("SELECT image_content_hash,ground_truth_content_hash,evaluation_policy_hash,executed_configuration_hash FROM result_rows").fetchone()
        self.assertEqual(tuple(stored), (HASH_D, HASH_A, self.spec.evaluation_policy_hash, dict(self.spec.expected_executed_configuration_hashes)["method-a"]))

    def test_result_identity_is_append_only(self):
        row = make_result(self.spec)
        self.store.append_attempt(self.spec, self.summary(), [row])
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.append_attempt(self.spec, self.summary(), [row])
        self.assertEqual(self.store.counts()["result_rows"], 1)

    def test_expected_and_actual_pair_invariants(self):
        with self.assertRaises(ValueError):
            ExecutionAttemptSummary("a", self.spec.specification_hash, "completed", 2, 1, 1, 0, 0, 1, "start", "end")
        wrong = replace(self.summary(), expected_pairs=2, status="partially_completed")
        with self.assertRaises(ValueError):
            self.store.append_attempt(self.spec, wrong, [make_result(self.spec)])

    def test_store_rejects_spec_or_configuration_mismatch(self):
        row = replace(make_result(self.spec), executed_configuration_hash=HASH_B)
        with self.assertRaises(ValueError):
            self.store.append_attempt(self.spec, self.summary(), [row])
        self.assertEqual(self.store.counts()["result_rows"], 0)

    def test_new_storage_implementation_contains_no_replace_statement(self):
        source = (Path(__file__).parents[1] / "scientific_contract" / "result_store.py").read_text(encoding="utf-8")
        forbidden = "INSERT" + " OR " + "REPLACE"
        self.assertNotIn(forbidden, source.upper())


if __name__ == "__main__":
    unittest.main()
