from dataclasses import replace
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from structvision import (
    AnalysisSample,
    DetectorConfig,
    DuplicateImageIDError,
    PreprocessingConfig,
    ProposalConfig,
    StructuralAnomalyDetector,
)
from structvision.errors import (
    AmbiguousColourSpaceError,
    CorruptImageError,
    UnsupportedChannelLayoutError,
)
from structvision.inputs import normalise_input


def detector():
    config = DetectorConfig(
        preprocessing=PreprocessingConfig(256, False, False, False, 0, 0),
        proposals=ProposalConfig(minimum_area_pixels=20, maximum_proposal_count=4),
    )
    return StructuralAnomalyDetector(config)


def surface():
    image = np.full((96, 144, 3), 155, np.uint8)
    cv2.line(image, (15, 80), (130, 15), (30, 30, 30), 3, cv2.LINE_AA)
    return image


def parity_fields(result):
    return [
        (item.proposal_id, item.rank, item.bbox, item.priority_score, item.evidence_score,
         item.heuristic_reliability, item.final_mask.tobytes(), item.raw_mask.tobytes())
        for item in result.proposals
    ]


class PublicApiTests(unittest.TestCase):
    def test_public_import_and_one_numpy_image(self):
        result = detector().analyse(surface(), image_id="frame-001", colour_space="BGR")
        self.assertEqual(result.image_id, "frame-001")
        self.assertEqual(result.image_shape, surface().shape)
        self.assertEqual(result.normalised_colour_space, "BGR")
        if result.proposals:
            self.assertFalse(result.proposals[0].final_mask.flags.writeable)

    def test_rgb_bgr_grayscale_and_rgba_normalisation(self):
        bgr = surface()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.assertTrue(np.array_equal(
            normalise_input(bgr, colour_space="BGR").image_bgr,
            normalise_input(rgb, colour_space="RGB").image_bgr,
        ))
        left = detector().analyse(bgr, image_id="bgr", colour_space="BGR")
        right = detector().analyse(rgb, image_id="rgb", colour_space="RGB")
        self.assertEqual(parity_fields(left), parity_fields(right))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_result = detector().analyse(gray, image_id="gray")
        self.assertEqual(gray_result.image_shape, bgr.shape)
        rgba = np.dstack((rgb, np.full(rgb.shape[:2], 128, np.uint8)))
        rgba_result = detector().analyse(
            rgba, image_id="rgba", colour_space="RGBA", alpha_handling="composite_white"
        )
        self.assertEqual(rgba_result.image_shape, bgr.shape)
        with self.assertRaises(AmbiguousColourSpaceError):
            detector().analyse(rgba, image_id="ambiguous-alpha", colour_space="RGBA")

    def test_three_channel_arrays_never_guess_colour_space(self):
        with self.assertRaises(AmbiguousColourSpaceError):
            detector().analyse(surface(), image_id="ambiguous")

    def test_filesystem_path_and_corrupt_image(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.png"
            self.assertTrue(cv2.imwrite(str(path), surface()))
            result = detector().analyse(path, image_id="path")
            self.assertEqual(result.provenance.source_type, "filesystem_path")
            corrupt = Path(temporary) / "broken.png"
            corrupt.write_bytes(b"not-an-image")
            with self.assertRaises(CorruptImageError):
                detector().analyse(corrupt, image_id="broken")

    def test_invalid_channel_shape(self):
        invalid = np.zeros((12, 14, 2), np.uint8)
        with self.assertRaises(UnsupportedChannelLayoutError):
            detector().analyse(invalid, image_id="invalid", colour_space="BGR")

    def test_ordered_batch_duplicate_rejection_and_error_isolation(self):
        samples = [
            AnalysisSample(surface(), "first", "BGR"),
            AnalysisSample(surface(), "second", "BGR"),
        ]
        batch = detector().analyse_batch(samples)
        self.assertEqual([result.image_id for result in batch], ["first", "second"])
        self.assertEqual(batch.completed_count, 2)
        with self.assertRaises(DuplicateImageIDError):
            detector().analyse_batch([samples[0], samples[0]])
        isolated = detector().analyse_batch(
            [samples[0], AnalysisSample(np.zeros((4, 4, 2), np.uint8), "bad", "BGR")],
            fail_fast=False,
        )
        self.assertEqual(isolated.completed_count, 1)
        self.assertEqual(isolated.failed_count, 1)
        self.assertEqual(isolated.failures[0].image_id, "bad")

    def test_repeated_analysis_is_deterministic_except_declared_timing(self):
        first = detector().analyse(surface(), image_id="same", colour_space="BGR")
        second = detector().analyse(surface(), image_id="same", colour_space="BGR")
        self.assertEqual(first.input_hash, second.input_hash)
        self.assertEqual(first.configuration_hash, second.configuration_hash)
        self.assertEqual(first.diagnostics, second.diagnostics)
        self.assertEqual(parity_fields(first), parity_fields(second))

    def test_default_analysis_writes_nothing_to_caller_directory(self):
        original = Path.cwd()
        with TemporaryDirectory() as temporary:
            target = Path(temporary)
            os.chdir(target)
            try:
                before = tuple(target.rglob("*"))
                detector().analyse(surface(), image_id="no-write", colour_space="BGR")
                after = tuple(target.rglob("*"))
            finally:
                os.chdir(original)
        self.assertEqual(before, ())
        self.assertEqual(after, ())

    def test_only_single_worker_is_supported(self):
        with self.assertRaises(ValueError):
            detector().analyse_batch([AnalysisSample(surface(), "one", "BGR")], worker_count=2)


if __name__ == "__main__":
    unittest.main()
