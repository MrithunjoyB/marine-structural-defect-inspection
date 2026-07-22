from dataclasses import FrozenInstanceError, replace
import math
import unittest

from structvision.configuration import (
    DEFAULT_EVIDENCE_WEIGHTS,
    DEFAULT_PRIORITY_WEIGHTS,
    DEFAULT_RELIABILITY_WEIGHTS,
    DetectorConfig,
    FeatureConfig,
    PreprocessingConfig,
    ProposalConfig,
    ScoringConfig,
)
from structvision.errors import InvalidConfigurationError


class DetectorConfigurationTests(unittest.TestCase):
    def test_configuration_is_frozen_canonical_and_round_trips(self):
        config = DetectorConfig()
        with self.assertRaises(FrozenInstanceError):
            config.random_seed = 4
        encoded = config.to_json()
        self.assertEqual(DetectorConfig.from_json(encoded), config)
        self.assertEqual(DetectorConfig.from_dict(config.to_dict()), config)
        self.assertEqual(config.configuration_hash, DetectorConfig().configuration_hash)
        self.assertNotIn("NaN", encoded)

    def test_baseline_defaults_match_frozen_legacy_arguments(self):
        config = DetectorConfig()
        self.assertEqual(config.preprocessing, PreprocessingConfig(1024, True, True, False, 0, 0))
        self.assertEqual(config.features, FeatureConfig(100, 35, 35, 128))
        self.assertEqual(
            config.proposals,
            ProposalConfig(
                minimum_area_pixels=250,
                minimum_relative_area=0.0002,
                maximum_relative_area=0.85,
                border_margin=0.025,
                maximum_proposal_count=8,
            ),
        )
        self.assertEqual(config.scoring.evidence_weights, DEFAULT_EVIDENCE_WEIGHTS)
        self.assertEqual(config.scoring.reliability_weights, DEFAULT_RELIABILITY_WEIGHTS)
        self.assertEqual(config.scoring.priority_weights, DEFAULT_PRIORITY_WEIGHTS)
        self.assertFalse(config.proposals.specular_suppression)

    def test_hash_changes_for_every_executable_public_field(self):
        baseline = DetectorConfig()
        variants = []
        for field, value in baseline.preprocessing.__dict__.items():
            changed = (not value) if isinstance(value, bool) else value + 1
            variants.append(replace(baseline, preprocessing=replace(baseline.preprocessing, **{field: changed})))
        for field, value in baseline.features.__dict__.items():
            changed = value + 1 if value < 255 else value - 1
            variants.append(replace(baseline, features=replace(baseline.features, **{field: changed})))
        for field, value in baseline.proposals.__dict__.items():
            if isinstance(value, bool):
                changed = not value
            elif field == "minimum_area_pixels" or field == "maximum_proposal_count":
                changed = value + 1
            else:
                changed = value + 0.00001 if value < 0.9 else value - 0.00001
            variants.append(replace(baseline, proposals=replace(baseline.proposals, **{field: changed})))
        variants.append(replace(baseline, random_seed=1))
        variants.append(replace(baseline, deterministic_mode=False))
        self.assertTrue(variants)
        self.assertEqual(len({item.configuration_hash for item in variants}), len(variants))
        self.assertNotIn(baseline.configuration_hash, {item.configuration_hash for item in variants})

    def test_invalid_ranges_nan_infinity_and_mutable_weight_shapes_fail(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(InvalidConfigurationError):
                ProposalConfig(border_margin=value)
        with self.assertRaises(InvalidConfigurationError):
            ProposalConfig(minimum_relative_area=0.9, maximum_relative_area=0.8)
        with self.assertRaises(InvalidConfigurationError):
            PreprocessingConfig(contrast=-100)
        with self.assertRaises(InvalidConfigurationError):
            ScoringConfig(evidence_weights=[("x", 1.0)])
        with self.assertRaises(InvalidConfigurationError):
            DetectorConfig(scoring=ScoringConfig(evidence_weights=(("x", 1.0),)))

    def test_missing_unknown_and_noncanonical_serialisation_fail(self):
        payload = DetectorConfig().to_dict()
        payload.pop("random_seed")
        with self.assertRaises(InvalidConfigurationError):
            DetectorConfig.from_dict(payload)
        with self.assertRaises(InvalidConfigurationError):
            DetectorConfig.from_json('{"random_seed":0}')


if __name__ == "__main__":
    unittest.main()
