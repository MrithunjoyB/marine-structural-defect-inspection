from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from structvision import DetectorConfig, StructuralAnomalyDetector
from structvision.classical import _load_legacy_modules, _redirect_legacy_artifacts
from structvision.inputs import normalise_input
from structvision.types import thaw_value


def generated_fixture_set():
    """Create deterministic arrays in memory; never load dataset payloads."""
    rng = np.random.default_rng(20260722)
    base = np.clip(150 + rng.normal(0, 5, (96, 144, 3)), 0, 255).astype(np.uint8)
    fixtures = {"clean_normal_texture": (base.copy(), "BGR", None)}
    crack = base.copy(); cv2.line(crack, (8, 85), (135, 10), (25, 25, 25), 3, cv2.LINE_AA)
    fixtures["thin_crack"] = (crack, "BGR", None)
    pits = base.copy()
    for point in ((45, 45), (58, 52), (72, 42), (65, 65)):
        cv2.circle(pits, point, 6, (45, 55, 65), -1, cv2.LINE_AA)
    fixtures["pitting_cluster"] = (pits, "BGR", None)
    weld = base.copy(); cv2.ellipse(weld, (72, 48), (54, 12), -10, 0, 360, (78, 95, 112), -1, cv2.LINE_AA)
    fixtures["weld_disturbance"] = (weld, "BGR", None)
    specular = base.copy()
    for point in ((35, 25), (100, 60), (80, 30)):
        cv2.circle(specular, point, 5, (255, 255, 255), -1, cv2.LINE_AA)
    fixtures["specular_highlight"] = (specular, "BGR", None)
    border = base.copy(); border[:, :10] = 0
    fixtures["border_artifact"] = (border, "BGR", None)
    gradient = np.tile(np.linspace(65, 225, 144, dtype=np.uint8), (96, 1))
    fixtures["illumination_gradient"] = (cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR), "BGR", None)
    fixtures["grayscale_input"] = (cv2.cvtColor(crack, cv2.COLOR_BGR2GRAY), None, None)
    rgba = np.dstack((cv2.cvtColor(pits, cv2.COLOR_BGR2RGB), np.full(pits.shape[:2], 190, np.uint8)))
    fixtures["rgba_input"] = (rgba, "RGBA", "composite_black")
    non_square = np.full((70, 190, 3), 160, np.uint8); cv2.rectangle(non_square, (105, 12), (175, 58), (70, 90, 110), -1)
    fixtures["non_square"] = (non_square, "BGR", None)
    small = np.full((24, 32, 3), 150, np.uint8); cv2.line(small, (3, 20), (28, 4), (20, 20, 20), 1)
    fixtures["small_valid"] = (small, "BGR", None)
    large_rng = np.random.default_rng(9)
    large = np.clip(145 + large_rng.normal(0, 4, (420, 700, 3)), 0, 255).astype(np.uint8)
    cv2.line(large, (40, 380), (655, 35), (35, 35, 35), 5, cv2.LINE_AA)
    fixtures["large_controlled"] = (large, "BGR", None)
    return fixtures


def direct_legacy(normalised, config, stem):
    with TemporaryDirectory(prefix="parity-direct-") as temporary:
        root = Path(temporary)
        preprocess, features, _, region = _load_legacy_modules(root)
        with _redirect_legacy_artifacts(features, region, root):
            processed = preprocess.apply_preprocessing(
                normalised,
                resize_width=config.preprocessing.resize_width,
                denoise=config.preprocessing.denoise,
                clahe=config.preprocessing.clahe,
                sharpen=config.preprocessing.sharpen,
                brightness=config.preprocessing.brightness,
                contrast=config.preprocessing.contrast,
            )
            maps = features.extract_feature_maps(
                processed,
                edge_sensitivity=config.features.edge_sensitivity,
                texture_sensitivity=config.features.texture_sensitivity,
                color_sensitivity=config.features.colour_sensitivity,
                threshold_level=config.features.threshold_level,
            )
            result = region.propose_regions(
                processed,
                maps,
                stem,
                min_area=config.proposals.minimum_area_pixels,
                max_regions=config.proposals.maximum_proposal_count,
                min_relative_area=config.proposals.minimum_relative_area,
                max_relative_area=config.proposals.maximum_relative_area,
                border_margin=config.proposals.border_margin,
            )
            records = []
            for item in result.proposals:
                final = cv2.imread(str(item.mask_path), cv2.IMREAD_GRAYSCALE)
                raw = cv2.imread(str(item.raw_mask_path), cv2.IMREAD_GRAYSCALE)
                records.append((item, final.copy(), raw.copy()))
            diagnostics = result.diagnostics.to_dict()
            heatmap = maps.anomaly_heatmap.copy()
            artifact_count = sum(path.is_file() for path in root.rglob("*"))
    return processed.copy(), records, diagnostics, heatmap, artifact_count


class FrozenBaselineParityTests(unittest.TestCase):
    """Exact parity over generated arrays and temporary artifacts only."""

    def test_generated_fixture_set_has_exact_direct_api_parity(self):
        config = DetectorConfig()
        detector = StructuralAnomalyDetector(config)
        for name, (source, colour_space, alpha_handling) in (
            generated_fixture_set().items()
        ):
            with self.subTest(fixture=name):
                if name == "large_controlled":
                    self.assertLess(source.nbytes, 1_000_000)
                normalised = normalise_input(
                    source, colour_space=colour_space, alpha_handling=alpha_handling
                )
                processed, legacy, diagnostics, heatmap, direct_artifacts = direct_legacy(
                    normalised.image_bgr, config, f"direct_{name}"
                )
                api = detector.analyse(
                    source,
                    image_id=name,
                    colour_space=colour_space,
                    alpha_handling=alpha_handling,
                )
                self.assertGreater(direct_artifacts, 0)
                self.assertGreater(api.provenance.temporary_artifact_count, 0)
                self.assertTrue(api.provenance.protected_hashes_verified)
                self.assertEqual(api.image_shape, processed.shape)
                self.assertTrue(np.array_equal(api.anomaly_heatmap, heatmap))
                self.assertEqual(len(api.proposals), len(legacy))
                self.assertEqual([item.proposal_id for item in api.proposals], [item.region_id for item, _, _ in legacy])
                self.assertEqual([item.rank for item in api.proposals], list(range(1, len(legacy) + 1)))
                for actual, (expected, final_mask, raw_mask) in zip(api.proposals, legacy):
                    self.assertEqual(actual.bbox, expected.bbox)
                    self.assertEqual(actual.area, expected.pixel_area)
                    self.assertEqual(actual.priority_score, expected.priority.score)
                    self.assertEqual(actual.proposal_score, expected.priority.score)
                    self.assertEqual(actual.evidence_score, expected.anomaly_evidence_score)
                    self.assertEqual(actual.heuristic_reliability, expected.mask_reliability_score)
                    self.assertEqual(dict(actual.component_scores), expected.feature_contributions)
                    self.assertEqual(actual.final_mask.tobytes(), final_mask.tobytes())
                    self.assertEqual(actual.raw_mask.tobytes(), raw_mask.tobytes())
                    self.assertEqual(actual.warnings, ())
                    self.assertEqual(actual.rejection_information, ())
                api_diagnostics = {key: thaw_value(value) for key, value in api.diagnostics}
                self.assertEqual(set(api_diagnostics), set(diagnostics))
                self.assertEqual(api_diagnostics, diagnostics)
                self.assertEqual(api.warnings, ())


if __name__ == "__main__":
    unittest.main()
