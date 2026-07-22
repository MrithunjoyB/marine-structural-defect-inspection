import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from structvision.inputs import content_hash
from structvision.normal_feature.model_artifact import DirectoryModelArtifactSink, load_model_artifact
from structvision.normal_feature.patchcore import NormalFeatureAnomalyDetector
from structvision.normal_feature.types import NormalFitSample


@unittest.skipUnless(os.environ.get("STRUCTVISION_PATCHCORE_WEIGHT"), "exact learned runtime/official weight not selected")
class OfficialPatchCoreRuntimeTests(unittest.TestCase):
    def test_official_fit_offline_replay_and_deterministic_hashes(self):
        weight = Path(os.environ["STRUCTVISION_PATCHCORE_WEIGHT"])
        detector = NormalFeatureAnomalyDetector(
            weight_file=weight, environment_lock_hash="a" * 64,
            repository_root=Path(__file__).parents[1],
        )
        rng = np.random.default_rng(42)
        samples = []
        for index in range(2):
            image = rng.integers(0, 256, size=(300, 500, 3), dtype=np.uint8)
            truth = np.zeros((300, 500), np.uint8)
            samples.append(NormalFitSample(
                image, f"normal-{index}", content_hash(image), content_hash(truth), colour_space="BGR",
            ))
        first = detector.fit_normal(samples, normal_fit_manifest_hash="b" * 64)
        self.assertTrue(any(
            type(model).__module__.startswith("anomalib.")
            for model in detector._runtime_models.values()
        ))
        second_detector = NormalFeatureAnomalyDetector(
            weight_file=weight, environment_lock_hash="a" * 64,
            repository_root=Path(__file__).parents[1],
        )
        second = second_detector.fit_normal(samples, normal_fit_manifest_hash="b" * 64)
        self.assertEqual(first.memory_bank_hash, second.memory_bank_hash)
        self.assertEqual(first.selected_coreset_indices, second.selected_coreset_indices)
        first_score = detector.score(samples[0].image, model_artifact=first, image_id="normal-0", colour_space="BGR")
        second_score = second_detector.score(samples[0].image, model_artifact=second, image_id="normal-0", colour_space="BGR")
        self.assertEqual(first_score.anomaly_map_hash, second_score.anomaly_map_hash)
        self.assertTrue(np.isfinite(first_score.anomaly_map).all())
        self.assertEqual(first_score.anomaly_map.shape, samples[0].image.shape[:2])
        with TemporaryDirectory() as temporary:
            sink = DirectoryModelArtifactSink(Path(temporary))
            sink.write(first)
            loaded = load_model_artifact(Path(temporary) / f"{first.artifact_hash}.json")
            replay = NormalFeatureAnomalyDetector(
                weight_file=weight, environment_lock_hash="a" * 64,
                repository_root=Path(__file__).parents[1],
            ).score(samples[0].image, model_artifact=loaded, image_id="normal-0", colour_space="BGR")
            self.assertEqual(first_score.anomaly_map_hash, replay.anomaly_map_hash)


if __name__ == "__main__":
    unittest.main()
