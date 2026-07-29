from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import unittest

import pytest

from protected_test_support import require_protected_files
from structvision.hybrid.errors import HybridExperimentError, HybridFusionError
from structvision.hybrid.artifact import (
    DECLARED_BUDGETS,
    EvaluatedFusionConfiguration,
    FusionOperatingPoint,
    FusionSearchConfiguration,
)
from structvision.hybrid.experiment import HoldoutAttemptLedger
from structvision.hybrid.selection import _select, coefficient_search_space


def evaluated(
    configuration_id,
    classical_weight,
    *,
    primary_fp=0.4,
    primary_micro=0.8,
    primary_iou=0.6,
    primary_failures=(),
    floor=None,
):
    search = FusionSearchConfiguration(
        configuration_id, classical_weight, 1.0 - classical_weight, floor,
    )
    points = []
    for budget in DECLARED_BUDGETS:
        fp = primary_fp if budget == 0.5 else min(0.2, budget)
        failures = primary_failures if budget == 0.5 else ()
        points.append(FusionOperatingPoint(
            budget, 0.5, fp, 0.1, primary_micro, 0.8, 0.8, 0.7,
            primary_iou, 0.5, (("thin_crack", 1.0),),
            fp <= budget, not failures, tuple(failures),
        ))
    return EvaluatedFusionConfiguration(search, tuple(points))


class HybridSelectionAndExperimentTests(unittest.TestCase):
    def test_coefficient_search_is_fixed_nonnegative_identifiable_and_deterministic(self):
        first = coefficient_search_space()
        second = coefficient_search_space()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 15)
        self.assertEqual(len({item.configuration_id for item in first}), 15)
        self.assertTrue(all(item.classical_weight > 0 and item.normality_weight >= 0 for item in first))
        self.assertTrue(all(abs(item.classical_weight + item.normality_weight - 1.0) < 1e-12 for item in first))
        self.assertEqual({item.preservation_floor for item in first}, {None, 0.8, 0.9})
        with self.assertRaises(HybridFusionError):
            FusionSearchConfiguration("zero-classical", 0.0, 1.0, None)
        with self.assertRaises(HybridFusionError):
            FusionSearchConfiguration("negative", 1.1, -0.1, None)

    def test_selection_enforces_budget_and_preservation_and_retains_failure(self):
        over_budget = evaluated("over-budget", 0.8, primary_fp=0.6)
        failed_preservation = evaluated(
            "failed-preservation", 0.8,
            primary_failures=("thin_crack_sensitivity_decreased",),
        )
        self.assertIsNone(_select((over_budget, failed_preservation)))

    def test_selection_tie_break_is_deterministic_and_prefers_simple_conservative_fusion(self):
        with_floor = evaluated("with-floor", 0.9, floor=0.9)
        no_floor_lower_classical = evaluated("no-floor-080", 0.8)
        no_floor_higher_classical = evaluated("no-floor-090", 0.9)
        candidates = (with_floor, no_floor_lower_classical, no_floor_higher_classical)
        self.assertEqual(_select(candidates), no_floor_higher_classical)
        self.assertEqual(_select(tuple(reversed(candidates))), no_floor_higher_classical)

    def test_holdout_attempt_ledger_permits_exactly_one_primary_attempt(self):
        with TemporaryDirectory() as temporary:
            ledger = HoldoutAttemptLedger(Path(temporary) / "attempts.jsonl")
            ledger.start(attempt_id="primary", fusion_artifact_hash="a" * 64, specification_hash="b" * 64)
            with self.assertRaises(HybridExperimentError):
                ledger.start(attempt_id="second", fusion_artifact_hash="a" * 64, specification_hash="b" * 64)
            ledger.finish(attempt_id="primary", status="completed")
            lines = ledger.path.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            with self.assertRaises(HybridExperimentError):
                ledger.finish(attempt_id="primary", status="completed")

    @pytest.mark.protected_integration
    def test_protected_historical_hashes_and_rows_remain_at_baseline(self):
        root = require_protected_files(
            Path(__file__).parents[1],
            "outputs/research_evaluation.sqlite3",
            "outputs/registered_experiment_results.sqlite3",
            "research_data/registry/datasets.sqlite",
        )
        expected = {
            "outputs/research_evaluation.sqlite3": "9a77d748dbf9780f5f0e104bea3412ddaadcad10b54a2c1fceed0e532acef640",
            "outputs/registered_experiment_results.sqlite3": "1ebde1de1f065b5b220366798147beb67dd10a446b7cd8840f988c9aeda9ce92",
            "research_data/registry/datasets.sqlite": "50513870f8b4b55b8616f505467858c917dc58a58c525950d729f2486df04632",
        }
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
