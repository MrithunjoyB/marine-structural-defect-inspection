import unittest

import numpy as np

from scientific_contract.evaluation_policy import default_evaluation_policy
from scientific_contract.matching import (
    EncodedMask,
    GroundTruthRecord,
    ProposalRecord,
    ProposalSet,
    TruthInstance,
    canonical_ground_truth,
    match_one_to_one,
    reconstruct_matching,
)


def rectangle(y1, y2, x1, x2, shape=(20, 20)):
    mask = np.zeros(shape, np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def proposal(identifier, mask, score=None, rank=None):
    return ProposalRecord(identifier, EncodedMask.from_array(mask), score, rank)


def truth(identifier, mask):
    return TruthInstance(identifier, EncodedMask.from_array(mask))


class OneToOneMatchingTests(unittest.TestCase):
    def setUp(self):
        self.policy = default_evaluation_policy()

    def test_assignment_is_deterministic_and_one_to_one(self):
        first_truth = truth("T1", rectangle(2, 8, 2, 8))
        second_truth = truth("T2", rectangle(2, 8, 10, 16))
        covering = rectangle(2, 8, 2, 16)
        proposals = ProposalSet("unordered", (proposal("P1", covering),), False)
        ground_truth = GroundTruthRecord("image", "anomaly_present", (first_truth, second_truth))
        first = match_one_to_one(proposals, ground_truth, 0.25, self.policy)
        second = match_one_to_one(proposals, ground_truth, 0.25, self.policy)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.matched_proposal_count, 1)
        self.assertEqual(first.matched_truth_count, 1)
        self.assertEqual(len(first.unmatched_truth_ids), 1)

    def test_one_truth_cannot_credit_multiple_proposals(self):
        target = truth("T1", rectangle(4, 12, 4, 12))
        proposals = ProposalSet("unordered", (
            proposal("P1", rectangle(4, 12, 4, 12)),
            proposal("P2", rectangle(4, 12, 4, 12)),
        ), False)
        result = match_one_to_one(proposals, GroundTruthRecord("image", "anomaly_present", (target,)), 0.25, self.policy)
        self.assertEqual(result.matched_proposal_count, 1)
        self.assertEqual(result.matched_truth_count, 1)
        self.assertEqual(sum(item.assigned_truth_id == "T1" for item in result.proposal_decisions), 1)

    def test_tie_breaking_is_stable(self):
        mask = rectangle(4, 12, 4, 12)
        proposals = ProposalSet("unordered", (proposal("P2", mask), proposal("P1", mask)), False)
        truths = GroundTruthRecord("image", "anomaly_present", (truth("T2", mask), truth("T1", mask)))
        assignments = []
        for _ in range(5):
            result = match_one_to_one(proposals, truths, 0.25, self.policy)
            assignments.append(tuple((item.proposal_id, item.assigned_truth_id) for item in result.proposal_decisions))
        self.assertEqual(len(set(assignments)), 1)

    def test_strict_named_iou_thresholds(self):
        target_mask = rectangle(0, 10, 0, 10)
        target = GroundTruthRecord("image", "anomaly_present", (truth("T", target_mask),))
        cases = {
            0.10: rectangle(0, 1, 0, 10),
            0.25: rectangle(0, 5, 0, 5),
            0.50: rectangle(0, 5, 0, 10),
        }
        for threshold, candidate in cases.items():
            result = match_one_to_one(ProposalSet("m", (proposal("P", candidate),), False), target, threshold, self.policy)
            self.assertEqual(result.matched_truth_count, 1, threshold)
        below = rectangle(0, 4, 0, 10)
        self.assertEqual(match_one_to_one(ProposalSet("m", (proposal("P", below),), False), target, 0.50, self.policy).matched_truth_count, 0)

    def test_centroid_is_diagnostic_only(self):
        thin = np.zeros((20, 20), np.uint8)
        thin[10, 2:18] = 1
        square = rectangle(7, 14, 7, 14)
        result = match_one_to_one(
            ProposalSet("m", (proposal("P", square),), False),
            GroundTruthRecord("image", "anomaly_present", (truth("T", thin),)),
            0.50, self.policy,
        )
        decision = result.proposal_decisions[0]
        self.assertTrue(decision.centroid_inside_truth_diagnostic)
        self.assertFalse(decision.matched)

    def test_unmatched_proposal_and_truth_accounting(self):
        proposals = ProposalSet("m", (proposal("P1", rectangle(0, 4, 0, 4)), proposal("P2", rectangle(15, 19, 15, 19))), False)
        truths = GroundTruthRecord("image", "anomaly_present", (truth("T1", rectangle(0, 4, 0, 4)), truth("T2", rectangle(8, 12, 8, 12))))
        result = match_one_to_one(proposals, truths, 0.25, self.policy)
        self.assertEqual(result.matched_proposal_count, 1)
        self.assertEqual(result.unmatched_truth_ids, ("T2",))
        self.assertEqual(sum(not item.matched for item in result.proposal_decisions), 1)

    def test_stored_details_reconstruct_without_inference(self):
        mask = rectangle(3, 12, 3, 12)
        proposals = ProposalSet("ranked", (proposal("P", mask, 0.9, 1),), True, "score descending then proposal_id")
        result = match_one_to_one(proposals, GroundTruthRecord("image", "anomaly_present", (truth("T", mask),)), 0.25, self.policy)
        rebuilt = reconstruct_matching(result.to_dict(), self.policy)
        self.assertEqual([item.assigned_truth_id for item in rebuilt.proposal_decisions], ["T"])
        self.assertEqual(rebuilt.similarity_matrix, result.similarity_matrix)


class RankingAndAnnotationTests(unittest.TestCase):
    def test_duplicate_missing_gap_and_score_order_fail(self):
        mask = rectangle(1, 4, 1, 4)
        with self.assertRaises(ValueError):
            ProposalSet("m", (proposal("P1", mask, 0.9, 1), proposal("P2", mask, 0.8, 1)), True, "score")
        with self.assertRaises(ValueError):
            ProposalSet("m", (proposal("P1", mask, 0.9, 1), proposal("P2", mask, 0.8, None)), True, "score")
        with self.assertRaises(ValueError):
            ProposalSet("m", (proposal("P1", mask, 0.9, 1), proposal("P2", mask, 0.8, 3)), True, "score")
        with self.assertRaises(ValueError):
            ProposalSet("m", (proposal("P1", mask, 0.9, 2), proposal("P2", mask, 0.8, 1)), True, "score")

    def test_score_ties_require_proposal_id_order(self):
        mask = rectangle(1, 4, 1, 4)
        ProposalSet("m", (proposal("P1", mask, 0.5, 1), proposal("P2", mask, 0.5, 2)), True, "score then ID")
        with self.assertRaises(ValueError):
            ProposalSet("m", (proposal("P1", mask, 0.5, 2), proposal("P2", mask, 0.5, 1)), True, "score then ID")

    def test_clean_semantics_and_legacy_empty_mask_warning(self):
        empty = np.zeros((10, 10), np.uint8)
        clean = canonical_ground_truth("clean", "no_anomaly", [("legacy", empty, "")], allow_legacy_empty_clean_mask=True)
        self.assertEqual(clean.truth_instances, ())
        self.assertIn("legacy empty clean mask", clean.legacy_warning)
        with self.assertRaises(ValueError):
            canonical_ground_truth("clean", "no_anomaly", [("bad", rectangle(1, 2, 1, 2, (10, 10)), "")])
        with self.assertRaises(ValueError):
            GroundTruthRecord("positive", "anomaly_present", ())


if __name__ == "__main__":
    unittest.main()
