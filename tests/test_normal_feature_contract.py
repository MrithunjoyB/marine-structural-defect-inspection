from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import os
import subprocess
import sys
import unittest

import numpy as np

from structvision.normal_feature.calibration import (
    CalibrationSample,
    DirectoryCalibrationArtifactSink,
    calibrate,
    load_calibration_artifact,
)
from structvision.normal_feature.configuration import (
    WEIGHT_FILENAME,
    WEIGHT_LICENCE,
    WEIGHT_MODEL_ID,
    WEIGHT_REVISION,
    WEIGHT_SHA256,
    WEIGHT_SOURCE,
    LearnedProposalConfig,
    NormalFeatureConfig,
)
from structvision.normal_feature.errors import (
    CalibrationError,
    ModelArtifactError,
    WeightProvenanceError,
)
from structvision.normal_feature.model_artifact import (
    DirectoryModelArtifactSink,
    NormalFeatureModelArtifact,
    load_model_artifact,
)
from structvision.normal_feature.preprocessing import prepare_input, restore_anomaly_map
from structvision.normal_feature.patchcore import EXACT_RUNTIME_VERSIONS, NormalFeatureAnomalyDetector
from structvision.normal_feature.proposal_extraction import extract_proposals


DIGEST = "a" * 64


def model_artifact():
    config = NormalFeatureConfig()
    return NormalFeatureModelArtifact.create(
        config=config,
        selected_normal_fit_ids=("fit-1",),
        normal_fit_image_hashes=(("fit-1", "b" * 64),),
        normal_fit_manifest_hash="c" * 64,
        weight_provenance={
            "source": WEIGHT_SOURCE,
            "model_id": WEIGHT_MODEL_ID,
            "revision": WEIGHT_REVISION,
            "filename": WEIGHT_FILENAME,
            "sha256": WEIGHT_SHA256,
            "licence": WEIGHT_LICENCE,
        },
        memory_bank=np.arange(24, dtype=np.float32).reshape(3, 8),
        selected_coreset_indices=(0, 4, 9),
        preprocessing_hash="d" * 64,
        environment_lock_hash="e" * 64,
        hardware_runtime_metadata={"device": "cpu"},
        git_commit="f" * 40,
        git_dirty_state="clean",
        git_diff_hash=None,
        creation_timestamp="2026-07-22T00:00:00+00:00",
    )


class NormalFeatureContractTests(unittest.TestCase):
    def test_dependency_boundary_exact_lock_and_resolved_environment(self):
        root = Path(__file__).parents[1]
        project = (root / "pyproject.toml").read_text(encoding="utf-8")
        base_section, learned_section = project.split("[dependency-groups]", 1)
        for heavy in ("torch", "torchvision", "anomalib", "scikit-learn"):
            self.assertNotIn(f'"{heavy}', base_section)
        self.assertIn("opencv-python>=4.9.0.80; python_version < '3.12'", base_section)
        self.assertIn("opencv-python-headless>=4.9.0.80; python_version >= '3.12'", base_section)
        for distribution, version in EXACT_RUNTIME_VERSIONS.items():
            self.assertIn(f'"{distribution}=={version}"', learned_section)
        lock = root / "requirements/pylock.normal-feature-macos-arm64.toml"
        self.assertEqual(
            hashlib.sha256(lock.read_bytes()).hexdigest(),
            "be3a00936219aedbcc397f0b3e8c0af6d901489a06550f3b148c72e22cea87b8",
        )
        lock_text = lock.read_text(encoding="utf-8")
        for distribution, version in EXACT_RUNTIME_VERSIONS.items():
            canonical_name = distribution.lower().replace("_", "-")
            self.assertIn(f'name = "{canonical_name}"\nversion = "{version}"', lock_text)
        report = json.loads((root / "requirements/normal-feature-macos-arm64-environment.json").read_text())
        self.assertEqual(report["python_version"], "3.12.13")
        self.assertEqual(report["machine"], "arm64")
        self.assertFalse(report["api_key_required"])
        self.assertFalse(report["mps_scientific_reference"])
        self.assertGreaterEqual(report["package_count"], 70)

    def test_base_import_does_not_import_torch_anomalib_or_streamlit(self):
        command = [
            sys.executable, "-c",
            "import sys,structvision.normal_feature; print(int('torch' in sys.modules),int('anomalib' in sys.modules),int('streamlit' in sys.modules))",
        ]
        completed = subprocess.run(command, cwd="/private/tmp", check=True, capture_output=True, text=True)
        self.assertEqual(completed.stdout.strip(), "0 0 0")

    def test_predeclared_configuration_rejects_random_weights_mps_and_search_variants(self):
        for change in (
            {"pretrained": False},
            {"device": "mps"},
            {"deterministic_mode": False},
            {"coreset_sampling_ratio": 0.01},
            {"input_height": 224},
            {"pretrained_weight_source": "https://unofficial.invalid/model"},
            {"pretrained_weight_licence": "unknown"},
            {"proposal": LearnedProposalConfig(minimum_area_pixels=4)},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(NormalFeatureConfig(), **change)

    def test_official_weight_is_required_and_changed_or_missing_weight_fails_closed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            detector = NormalFeatureAnomalyDetector(
                weight_file=root / WEIGHT_FILENAME,
                environment_lock_hash="a" * 64,
                repository_root=Path(__file__).parents[1],
            )
            with self.assertRaises(WeightProvenanceError):
                detector._verify_weight()
            detector.weight_file.write_bytes(b"not-the-official-weight")
            with self.assertRaises(WeightProvenanceError):
                detector._verify_weight()

    def test_preprocessing_preserves_aspect_with_recorded_letterbox_and_inverse_map(self):
        image = np.full((300, 500, 3), 128, dtype=np.uint8)
        prepared = prepare_input(image, NormalFeatureConfig(), colour_space="BGR")
        self.assertEqual(prepared.tensor_chw.shape, (3, 256, 416))
        self.assertEqual((prepared.geometry.resized_height, prepared.geometry.resized_width), (250, 416))
        self.assertEqual((prepared.geometry.pad_top, prepared.geometry.pad_bottom), (3, 3))
        restored = restore_anomaly_map(np.ones((256, 416), np.float32), prepared.geometry)
        self.assertEqual(restored.shape, (300, 500))
        self.assertTrue(np.allclose(restored, 1.0))

    def test_explicit_rgb_and_bgr_inputs_are_equivalent_and_no_files_are_written(self):
        rgb = np.zeros((20, 30, 3), dtype=np.uint8)
        rgb[..., 0], rgb[..., 1], rgb[..., 2] = 11, 71, 193
        bgr = rgb[..., ::-1].copy()
        with TemporaryDirectory() as temporary:
            previous = Path.cwd()
            try:
                os.chdir(temporary)
                before = tuple(Path(temporary).iterdir())
                rgb_prepared = prepare_input(rgb, NormalFeatureConfig(), colour_space="RGB")
                bgr_prepared = prepare_input(bgr, NormalFeatureConfig(), colour_space="BGR")
                anomaly_map = np.zeros((20, 30), dtype=np.float32)
                extract_proposals(
                    anomaly_map,
                    threshold=1.0,
                    operating_point_id="fp-budget-0.50",
                    config=LearnedProposalConfig(),
                )
                self.assertEqual(before, tuple(Path(temporary).iterdir()))
            finally:
                os.chdir(previous)
        self.assertTrue(np.array_equal(rgb_prepared.tensor_chw, bgr_prepared.tensor_chw))
        with self.assertRaises(ValueError):
            prepare_input(rgb, NormalFeatureConfig())

    def test_proposal_extraction_is_ranked_not_scan_order_and_uses_half_open_boxes(self):
        anomaly_map = np.zeros((40, 60), np.float32)
        anomaly_map[3:8, 4:9] = 2.0
        anomaly_map[25:31, 40:47] = 8.0
        proposals = extract_proposals(
            anomaly_map, threshold=1.0, operating_point_id="fp-budget-0.50",
            config=LearnedProposalConfig(minimum_area_pixels=4),
        )
        self.assertEqual([item.rank for item in proposals], [1, 2])
        self.assertEqual(proposals[0].bbox, (40, 25, 47, 31))
        self.assertEqual(proposals[1].bbox, (4, 3, 9, 8))
        self.assertGreater(proposals[0].component_anomaly_score, proposals[1].component_anomaly_score)
        replay = extract_proposals(
            anomaly_map, threshold=1.0, operating_point_id="fp-budget-0.50",
            config=LearnedProposalConfig(minimum_area_pixels=4),
        )
        self.assertEqual([item.to_dict() for item in proposals], [item.to_dict() for item in replay])
        self.assertEqual(len({item.proposal_id for item in proposals}), len(proposals))
        self.assertTrue(all(item.operating_point_id == "fp-budget-0.50" for item in proposals))

    def test_model_artifact_round_trip_tamper_and_mismatch_detection(self):
        artifact = model_artifact()
        self.assertFalse(artifact.memory_bank.flags.writeable)
        self.assertEqual(artifact.normal_fit_image_hashes, (("fit-1", "b" * 64),))
        self.assertEqual(artifact.configuration_hash, NormalFeatureConfig().configuration_hash)
        self.assertEqual(artifact.environment_lock_hash, "e" * 64)
        self.assertEqual(artifact.backbone_weight_hash, WEIGHT_SHA256)
        self.assertEqual(artifact.weight_provenance["licence"], "Apache-2.0")
        with self.assertRaises(TypeError):
            artifact.weight_provenance["source"] = "changed"
        with self.assertRaises(TypeError):
            artifact.model_configuration["proposal"]["morphology"] = "changed"
        with TemporaryDirectory() as temporary:
            sink = DirectoryModelArtifactSink(Path(temporary))
            sink.write(artifact)
            path = Path(temporary) / f"{artifact.artifact_hash}.json"
            loaded = load_model_artifact(path)
            self.assertEqual(loaded.artifact_hash, artifact.artifact_hash)
            self.assertTrue(np.array_equal(loaded.memory_bank, artifact.memory_bank))
            with self.assertRaises(ModelArtifactError):
                sink.write(artifact)
            bank = Path(temporary) / f"{artifact.artifact_hash}.npz"
            np.savez_compressed(bank, memory_bank=np.zeros((3, 8), np.float32))
            with self.assertRaises(ModelArtifactError):
                load_model_artifact(path)
        with self.assertRaises(ModelArtifactError):
            replace(artifact, configuration_hash="0" * 64)

    def test_calibration_uses_validation_only_preserves_curve_and_budget_order(self):
        samples = []
        for index in range(4):
            anomaly_map = np.zeros((24, 32), np.float32)
            truth = np.zeros((24, 32), np.uint8)
            outcome = "no_anomaly" if index < 2 else "anomaly_present"
            category = "clean" if index < 2 else "thin_crack"
            if index == 1:
                anomaly_map[2:7, 2:7] = 0.3
            if index >= 2:
                anomaly_map[8:15, 10:18] = 1.0 + index
                truth[8:15, 10:18] = 255
            samples.append(CalibrationSample(
                f"cal-{index}", "calibration_validation", category, outcome, anomaly_map, truth,
            ))
        artifact = calibrate(
            samples,
            model_artifact_hash="a" * 64,
            calibration_manifest_hash="b" * 64,
            proposal_config=LearnedProposalConfig(minimum_area_pixels=4),
            quantile_count=11,
        )
        self.assertEqual(tuple(point.threshold for point in artifact.curve), artifact.candidate_thresholds)
        thresholds = [point.threshold for point in artifact.operating_points]
        self.assertTrue(all(left >= right for left, right in zip(thresholds, thresholds[1:])))
        self.assertEqual(artifact.false_proposal_budgets, (0.25, 0.5, 1.0))
        with TemporaryDirectory() as temporary:
            sink = DirectoryCalibrationArtifactSink(Path(temporary))
            sink.write(artifact)
            loaded = load_calibration_artifact(Path(temporary) / f"{artifact.artifact_hash}.json")
            self.assertEqual(loaded, artifact)
        with self.assertRaises(CalibrationError):
            replace(samples[0], role="test")
        with self.assertRaises(CalibrationError):
            calibrate(samples, model_artifact_hash="a" * 64, calibration_manifest_hash="b" * 64,
                      proposal_config=LearnedProposalConfig(), false_proposal_budgets=(0.5, 0.25))


if __name__ == "__main__":
    unittest.main()
