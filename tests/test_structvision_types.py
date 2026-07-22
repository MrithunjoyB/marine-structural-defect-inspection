from dataclasses import FrozenInstanceError
import json
import unittest

import numpy as np

from structvision.provenance import ProvenanceRecord
from structvision.types import AnalysisResult, Proposal, frozen_mapping


def proposal(rank=1, bbox=(2, 1, 6, 5), score=42.0):
    mask = np.zeros((8, 10), np.uint8)
    mask[1:5, 2:6] = 255
    return Proposal(
        f"R{rank:03d}", rank, bbox, mask, mask.copy(), score, 40.0, 30.0, score,
        (("texture", 1.0),), 16, (3.5, 2.5), frozen_mapping({"ring": "valid"}),
        (), (), "structvision-classical-baseline-v1-frozen", "1.0.0",
    )


def provenance():
    return ProvenanceRecord("adapter", "numpy_array", "a" * 64, (), True, "1", "1", 0, "none")


class DomainTypeTests(unittest.TestCase):
    def test_proposal_is_immutable_and_mask_is_read_only(self):
        item = proposal()
        with self.assertRaises(FrozenInstanceError):
            item.rank = 2
        with self.assertRaises(ValueError):
            item.final_mask[1, 2] = 0
        self.assertEqual(item.mask_reliability, item.heuristic_reliability)

    def test_mask_bbox_score_and_area_validation(self):
        with self.assertRaises(ValueError):
            proposal(bbox=(2, 1, 5, 5))
        with self.assertRaises(ValueError):
            proposal(score=float("nan"))
        item = proposal()
        self.assertEqual(item.bbox, (2, 1, 6, 5))
        self.assertEqual(item.area, 16)

    def test_result_requires_ordered_unique_ranks_and_matching_coordinates(self):
        heatmap = np.zeros((8, 10, 3), np.uint8)
        result = AnalysisResult(
            "image", "a" * 64, (8, 10, 3), "BGR", (proposal(),), heatmap,
            frozen_mapping({"resize": 1024}), "b" * 64,
            "structvision-classical-baseline-v1-frozen", "1.0.0", True,
            (("total", 1.0),), (), provenance(), (), frozen_mapping({"final": 1}),
        )
        encoded = result.to_json()
        self.assertEqual(encoded, result.to_json())
        decoded = json.loads(encoded)
        self.assertEqual(decoded["proposals"][0]["bbox_convention"], "half-open:x_min,y_min,x_max,y_max")
        self.assertEqual(decoded["proposals"][0]["final_mask"]["dtype"], "uint8")
        with self.assertRaises(ValueError):
            AnalysisResult(
                "image", "a" * 64, (8, 10, 3), "BGR", (proposal(rank=2),), heatmap,
                (), "b" * 64, "structvision-classical-baseline-v1-frozen", "1.0.0", True,
                (), (), provenance(), (), (),
            )


if __name__ == "__main__":
    unittest.main()
