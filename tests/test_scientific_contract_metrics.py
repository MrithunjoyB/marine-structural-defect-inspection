import unittest

import numpy as np

from scientific_contract.matching import EncodedMask, GroundTruthRecord, ProposalRecord, ProposalSet, TruthInstance
from scientific_contract.metrics import aggregate_metrics, evaluate_image


def mask(y1, y2, x1, x2):
    value = np.zeros((30, 30), np.uint8)
    value[y1:y2, x1:x2] = 1
    return value


def proposal(identifier, value, score=None, rank=None):
    return ProposalRecord(identifier, EncodedMask.from_array(value), score, rank)


def truth(identifier, value):
    return TruthInstance(identifier, EncodedMask.from_array(value))


class MetricContractTests(unittest.TestCase):
    def test_micro_macro_image_clean_precision_and_localisation(self):
        first_truth = GroundTruthRecord("i1", "anomaly_present", (truth("t1", mask(1, 5, 1, 5)),))
        first_output = ProposalSet("method", (proposal("p1", mask(1, 5, 1, 5)),), False)
        second_truth = GroundTruthRecord("i2", "anomaly_present", (
            truth("t1", mask(1, 5, 1, 5)), truth("t2", mask(10, 14, 10, 14)), truth("t3", mask(20, 24, 20, 24)),
        ))
        second_output = ProposalSet("method", (proposal("p1", mask(1, 5, 1, 5)), proposal("p2", mask(25, 29, 25, 29))), False)
        clean_truth = GroundTruthRecord("i3", "no_anomaly", ())
        clean_output = ProposalSet("method", (proposal("p1", mask(5, 9, 20, 24)),), False)
        evaluations = [
            evaluate_image(first_output, first_truth, category="crack", acquisition_group_id="a"),
            evaluate_image(second_output, second_truth, category="pitting", acquisition_group_id="b"),
            evaluate_image(clean_output, clean_truth, category="specular", acquisition_group_id="c"),
        ]
        result = aggregate_metrics(evaluations)
        self.assertEqual(result.micro_component_sensitivity, 0.5)
        self.assertAlmostEqual(result.macro_per_positive_image_recall, 2 / 3)
        self.assertEqual(result.image_level_detection_sensitivity, 1.0)
        self.assertEqual(result.proposal_precision, 0.5)
        self.assertEqual(result.clean_false_proposals_per_image, 1.0)
        self.assertEqual(result.clean_images_with_any_proposal, 1.0)
        self.assertEqual(result.assigned_pair_ious, (1.0, 1.0))
        self.assertEqual(result.assigned_pair_dice, (1.0, 1.0))
        self.assertTrue(all(value is None for _, value in result.top_k_component_sensitivity))
        self.assertEqual(dict(result.nuisance_false_proposals_per_image)["specular"], 1.0)
        self.assertIsNone(result.primary_endpoint_selector)
        self.assertNotIn("balanced_score", result.to_dict())

    def test_ranked_top_k_is_computed_only_for_valid_ordered_output(self):
        truths = GroundTruthRecord("i1", "anomaly_present", (truth("t", mask(10, 15, 10, 15)),))
        ranked = ProposalSet("ranked", (
            proposal("p1", mask(1, 5, 1, 5), 0.9, 1),
            proposal("p2", mask(10, 15, 10, 15), 0.8, 2),
        ), True, "score descending then proposal_id")
        metrics = aggregate_metrics([evaluate_image(ranked, truths)])
        values = dict(metrics.top_k_component_sensitivity)
        self.assertEqual(values[1], 0.0)
        self.assertEqual(values[3], 1.0)
        self.assertEqual(values[5], 1.0)
        self.assertEqual(values[8], 1.0)

    def test_undefined_values_remain_null(self):
        clean = GroundTruthRecord("clean", "no_anomaly", ())
        empty = ProposalSet("method", (), False)
        clean_metrics = aggregate_metrics([evaluate_image(empty, clean)])
        self.assertIsNone(clean_metrics.micro_component_sensitivity)
        self.assertIsNone(clean_metrics.macro_per_positive_image_recall)
        self.assertIsNone(clean_metrics.image_level_detection_sensitivity)
        self.assertIsNone(clean_metrics.proposal_precision)
        self.assertEqual(clean_metrics.clean_false_proposals_per_image, 0.0)
        positive = GroundTruthRecord("positive", "anomaly_present", (truth("t", mask(1, 3, 1, 3)),))
        positive_metrics = aggregate_metrics([evaluate_image(empty, positive)])
        self.assertEqual(positive_metrics.micro_component_sensitivity, 0.0)
        self.assertIsNone(positive_metrics.clean_false_proposals_per_image)
        self.assertIsNone(positive_metrics.clean_images_with_any_proposal)

    def test_threshold_localisation_sensitivity_is_explicit(self):
        target = GroundTruthRecord("i", "anomaly_present", (truth("t", mask(0, 10, 0, 10)),))
        candidate = ProposalSet("method", (proposal("p", mask(0, 5, 0, 10)),), False)
        result = aggregate_metrics([evaluate_image(candidate, target)])
        self.assertEqual(dict(result.sensitivity_by_iou_threshold), {0.1: 1.0, 0.25: 1.0, 0.5: 1.0})


if __name__ == "__main__":
    unittest.main()
