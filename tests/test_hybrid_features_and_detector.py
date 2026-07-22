from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import inspect
import os
import subprocess
import sys
import unittest

import numpy as np

from scientific_contract.evaluation_policy import default_evaluation_policy
from structvision.configuration import DetectorConfig
from structvision.hybrid.artifact import (
    DECLARED_BUDGETS,
    HYBRID_ARTIFACT_SCHEMA_VERSION,
    HYBRID_IMPLEMENTATION_ID,
    HYBRID_IMPLEMENTATION_VERSION,
    DirectoryHybridFusionArtifactSink,
    EvaluatedFusionConfiguration,
    FusionOperatingPoint,
    FusionSearchConfiguration,
    HybridFusionArtifact,
    load_hybrid_fusion_artifact,
)
from structvision.hybrid.detector import ProposalGuidedHybridDetector
from structvision.hybrid.errors import HybridFeatureError, HybridFusionError
from structvision.hybrid.features import (
    FEATURE_DEFINITIONS,
    FEATURE_ORDER,
    FeatureNormalisation,
    candidate_evidence,
    fit_normalisation,
)
from structvision.inputs import normalise_input
from structvision.normal_feature.configuration import (
    WEIGHT_FILENAME,
    WEIGHT_LICENCE,
    WEIGHT_MODEL_ID,
    WEIGHT_REVISION,
    WEIGHT_SHA256,
    WEIGHT_SOURCE,
    NormalFeatureConfig,
)
from structvision.normal_feature.model_artifact import NormalFeatureModelArtifact
from structvision.normal_feature.types import NormalFeatureScoreResult, array_hash
from structvision.provenance import ProvenanceRecord
from structvision.types import AnalysisResult, Proposal, frozen_mapping


def proposal():
    mask = np.zeros((40, 60), np.uint8)
    mask[10:24, 16:38] = 255
    return Proposal(
        "R001", 1, (16, 10, 38, 24), mask, mask.copy(), 61.0, 55.0, 72.0, 61.0,
        (("texture", 0.7),), int(np.count_nonzero(mask)), (26.5, 16.5),
        frozen_mapping({"ring": "valid"}), (), (),
        "structvision-classical-baseline-v1-frozen", "1.0.0",
    )


def operating_point(budget):
    return FusionOperatingPoint(
        budget, 0.4, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.2,
        (("pitting_cluster", 1.0), ("thin_crack", 1.0), ("weld_disturbance", 1.0)),
        True, True, (),
    )


def model_artifact(protocol_hash="a" * 64, lock_hash="e" * 64):
    config = NormalFeatureConfig()
    return NormalFeatureModelArtifact.create(
        config=config,
        selected_normal_fit_ids=("fit-1",),
        normal_fit_image_hashes=(("fit-1", "b" * 64),),
        normal_fit_manifest_hash=protocol_hash,
        weight_provenance={
            "source": WEIGHT_SOURCE, "model_id": WEIGHT_MODEL_ID, "revision": WEIGHT_REVISION,
            "filename": WEIGHT_FILENAME, "sha256": WEIGHT_SHA256, "licence": WEIGHT_LICENCE,
        },
        memory_bank=np.ones((2, 8), np.float32),
        selected_coreset_indices=(0, 1), preprocessing_hash="c" * 64,
        environment_lock_hash=lock_hash, hardware_runtime_metadata={"device": "cpu"},
        git_commit="d" * 40, git_dirty_state="clean", git_diff_hash=None,
        creation_timestamp="2026-07-22T00:00:00+00:00",
    )


def fusion_artifact(model):
    search = FusionSearchConfiguration("one", 0.8, 0.2, None)
    points = tuple(operating_point(budget) for budget in DECLARED_BUDGETS)
    evaluated = EvaluatedFusionConfiguration(search, points)
    bounds = tuple(FeatureNormalisation(name, 0.0, 100.0) for name in FEATURE_ORDER)
    return HybridFusionArtifact.create(
        schema_version=HYBRID_ARTIFACT_SCHEMA_VERSION,
        implementation_identity=HYBRID_IMPLEMENTATION_ID,
        implementation_version=HYBRID_IMPLEMENTATION_VERSION,
        hybrid_protocol_hash="a" * 64,
        normal_feature_model_artifact_hash=model.artifact_hash,
        frozen_classical_configuration_hash=DetectorConfig().configuration_hash,
        candidate_feature_definitions=FEATURE_DEFINITIONS,
        feature_order_identity=FEATURE_ORDER,
        high_anomaly_reference=0.5,
        normalisation_statistics=bounds,
        coefficient_search_space=(search,),
        evaluated_configurations=(evaluated,),
        preservation_constraints=(("overall_micro_max_decrease", 0.02),),
        selection_status="selected",
        selected_configuration_id="one",
        selected_coefficients=(0.8, 0.2),
        selected_preservation_floor=None,
        selected_operating_points=points,
        selected_operating_threshold=0.4,
        false_proposal_budget=0.5,
        fusion_fit_image_hashes=(("fit-image", "1" * 64),),
        fusion_fit_truth_hashes=(("fit-image", "2" * 64),),
        evaluation_policy_hash=default_evaluation_policy().configuration_hash,
        environment_lock_hash="e" * 64,
        code_commit="f" * 40, git_dirty_state="clean", git_diff_hash=None,
        deterministic_seed=73021, creation_timestamp="2026-07-22T00:00:00+00:00",
    )


class FakeClassical:
    config = DetectorConfig()

    def analyse(self, image, *, image_id, colour_space=None, alpha_handling=None, metadata=None):
        normalised = normalise_input(image, colour_space=colour_space, alpha_handling=alpha_handling)
        item = proposal()
        provenance = ProvenanceRecord("adapter", "numpy_array", normalised.source_hash, (), True, "1", "1", 0, "none")
        return AnalysisResult(
            image_id, normalised.input_hash, (40, 60, 3), "BGR", (item,), None,
            frozen_mapping({"resize": 1024}), self.config.configuration_hash,
            self.config.implementation_id, self.config.implementation_version, True,
            (("core_total", 0.01),), (), provenance, (), (),
        )


class FakeNormal:
    config = NormalFeatureConfig()
    environment_lock_hash = "e" * 64

    def score(self, image, *, model_artifact, image_id, colour_space=None, alpha_handling=None):
        normalised = normalise_input(image, colour_space=colour_space, alpha_handling=alpha_handling)
        anomaly_map = np.linspace(0.0, 1.0, 40 * 60, dtype=np.float32).reshape(40, 60)
        return NormalFeatureScoreResult(
            image_id, normalised.input_hash, (40, 60, 3), 1.0, anomaly_map,
            array_hash(anomaly_map), model_artifact.artifact_hash,
            self.config.configuration_hash, (), True, "cpu", 0.01,
        )


class HybridFeatureAndDetectorTests(unittest.TestCase):
    def test_feature_alignment_determinism_finiteness_and_mask_parity(self):
        item = proposal()
        anomaly_map = np.linspace(0.0, 1.0, 40 * 60, dtype=np.float32).reshape(40, 60)
        before = item.final_mask.tobytes()
        first = candidate_evidence(item, anomaly_map, high_anomaly_reference=0.5)
        second = candidate_evidence(item, anomaly_map, high_anomaly_reference=0.5)
        self.assertEqual(first, second)
        self.assertEqual(first.context_ring_radius, max(2, round(np.sqrt(item.area) / 8)))
        self.assertEqual(tuple(name for name, _ in first.feature_values), FEATURE_ORDER)
        self.assertTrue(all(np.isfinite(value) for _, value in first.feature_values))
        self.assertEqual(before, item.final_mask.tobytes())
        self.assertEqual(tuple(item.name for item in fit_normalisation((first, second))), FEATURE_ORDER)
        with self.assertRaises(HybridFeatureError):
            candidate_evidence(item, np.zeros((39, 60), np.float32), high_anomaly_reference=0.5)

    def test_inference_feature_api_has_no_category_filename_or_truth_input(self):
        parameters = set(inspect.signature(candidate_evidence).parameters)
        self.assertEqual(parameters, {"proposal", "anomaly_map", "high_anomaly_reference"})
        self.assertFalse({"category", "filename", "ground_truth"} & parameters)

    def test_artifact_roundtrip_and_tamper_rejection(self):
        artifact = fusion_artifact(model_artifact())
        with TemporaryDirectory() as temporary:
            sink = DirectoryHybridFusionArtifactSink(Path(temporary))
            sink.write(artifact)
            path = Path(temporary) / f"{artifact.artifact_hash}.json"
            self.assertEqual(load_hybrid_fusion_artifact(path), artifact)
            with self.assertRaises(HybridFusionError):
                sink.write(artifact)
            path.write_text(path.read_text().replace('"classical_weight":0.8', '"classical_weight":0.7'), encoding="utf-8")
            with self.assertRaises(HybridFusionError):
                load_hybrid_fusion_artifact(path)

    def test_public_detector_retains_all_diagnostics_and_byte_identical_masks(self):
        model = model_artifact()
        artifact = fusion_artifact(model)
        detector = ProposalGuidedHybridDetector(
            classical_detector=FakeClassical(), normal_feature_detector=FakeNormal(),
            normal_feature_model_artifact=model, fusion_artifact=artifact,
        )
        image = np.zeros((40, 60, 3), np.uint8)
        result = detector.analyse(image, image_id="frame-1", colour_space="BGR")
        replayed = detector.analyse(image, image_id="frame-1", colour_space="BGR")
        self.assertEqual(result.complete_original_classical_candidate_count, 1)
        self.assertEqual(len(result.complete_candidate_diagnostics), 1)
        self.assertEqual(result.complete_candidate_diagnostics[0].mask.tobytes(), proposal().final_mask.tobytes())
        self.assertEqual(
            [item.to_dict() for item in result.complete_candidate_diagnostics],
            [item.to_dict() for item in replayed.complete_candidate_diagnostics],
        )
        self.assertEqual(
            [item.to_dict() for item in result.proposals],
            [item.to_dict() for item in replayed.proposals],
        )
        self.assertEqual(result.fusion_artifact_hash, artifact.artifact_hash)
        self.assertEqual(result.normal_feature_model_artifact_hash, model.artifact_hash)
        self.assertNotIn("probability", result.complete_candidate_diagnostics[0].to_dict()["score_semantics"].replace("not_probability", ""))
        for budget in DECLARED_BUDGETS:
            replay = detector.reselect(result, false_proposal_budget=budget)
            self.assertEqual([item.rank for item in replay.proposals], list(range(1, len(replay.proposals) + 1)))

    def test_hybrid_import_has_no_ui_or_api_key_dependency(self):
        root = Path(__file__).parents[1]
        environment = {
            name: value for name, value in os.environ.items()
            if "API_KEY" not in name.upper()
        }
        environment["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root)))
        script = """
import sys
import structvision.hybrid
blocked = {'streamlit', 'gradio', 'flask', 'fastapi', 'django'} & set(sys.modules)
assert not blocked, blocked
"""
        subprocess.run([sys.executable, "-c", script], cwd=root, env=environment, check=True)


if __name__ == "__main__":
    unittest.main()
