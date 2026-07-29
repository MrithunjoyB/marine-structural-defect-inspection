from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pytest

from protected_test_support import require_protected_files
from scientific_contract.dataset_audit import hamming_distance, read_registry_dataset
from structvision.hybrid.errors import HybridProtocolError
from structvision.hybrid.protocol import (
    create_hybrid_development_manifest,
    fusion_fit_view,
    load_hybrid_manifest,
    write_hybrid_manifest,
)


class HybridDevelopmentProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).parents[1]
        cls.manifest = load_hybrid_manifest(
            cls.root / "development_data/hybrid-development-manifest-v1.json"
        )

    def test_counts_categories_and_deterministic_manifest_identity(self):
        self.assertEqual(self.manifest.manifest_hash, "a1e6f9a83e5e8d73275236e6dc4fafd985e6e1ef2c4aef21fd4156dc821829a4")
        self.assertEqual((len(self.manifest.normal_fit), len(self.manifest.fusion_fit), len(self.manifest.development_holdout)), (70, 126, 72))
        self.assertEqual({item.image_outcome for item in self.manifest.normal_fit}, {"no_anomaly"})
        self.assertEqual({item.image_outcome for item in self.manifest.fusion_fit}, {"no_anomaly", "anomaly_present"})
        self.assertEqual({item.image_outcome for item in self.manifest.development_holdout}, {"no_anomaly", "anomaly_present"})
        for role in (self.manifest.fusion_fit, self.manifest.development_holdout):
            self.assertTrue({"thin_crack", "pitting_cluster", "weld_disturbance"}.issubset({item.category for item in role}))

    def test_no_id_hash_perceptual_or_declared_group_crossing(self):
        roles = (self.manifest.normal_fit, self.manifest.fusion_fit, self.manifest.development_holdout)
        self.assertEqual({item.split_role for item in self.manifest.selected_images}, {"train", "validation"})
        for index, left in enumerate(roles):
            for right in roles[index + 1:]:
                self.assertFalse({item.image_id for item in left} & {item.image_id for item in right})
                self.assertFalse({item.image_sha256 for item in left} & {item.image_sha256 for item in right})
                for field in ("source_group_id", "template_group_id", "acquisition_group_id"):
                    self.assertFalse(
                        {getattr(item, field) for item in left if getattr(item, field)}
                        & {getattr(item, field) for item in right if getattr(item, field)}
                    )
                self.assertTrue(all(
                    not left_item.perceptual_hash or not right_item.perceptual_hash
                    or hamming_distance(left_item.perceptual_hash, right_item.perceptual_hash) > 3
                    for left_item in left for right_item in right
                ))

    @pytest.mark.protected_integration
    def test_selected_roles_have_no_pilot_or_historical_test_overlap(self):
        root = require_protected_files(
            self.root,
            "research_data/registry/datasets.sqlite",
        )
        registry = root / "research_data/registry/datasets.sqlite"
        pilot = read_registry_dataset(registry, "synthetic-expanded-pilot", "1.0")
        expanded = read_registry_dataset(registry, "synthetic-expanded", "1.0")
        historical_test = tuple(item for item in expanded if item.split == "test")
        selected = self.manifest.selected_images
        for protected in (pilot, historical_test):
            self.assertFalse(
                {item.image_sha256 for item in selected}
                & {item.image_sha256 for item in protected}
            )
            self.assertTrue(all(
                not left.perceptual_hash or not right.perceptual_hash
                or hamming_distance(left.perceptual_hash, right.perceptual_hash) > 3
                for left in selected for right in protected
            ))
        reasons = {reason for item in self.manifest.exclusions for reason in item.reasons}
        self.assertTrue(any(reason.startswith("pilot:") for reason in reasons))
        self.assertTrue(any(reason.startswith("historical_test:") for reason in reasons))

    def test_fusion_fit_view_cannot_expose_holdout(self):
        view = fusion_fit_view(self.manifest)
        self.assertEqual({item.role for item in view.identities}, {"hybrid_fusion_fit"})
        self.assertFalse({item.image_id for item in view.identities} & {item.image_id for item in self.manifest.development_holdout})

    def test_committed_manifest_roundtrip_immutability_and_tamper_detection(self):
        committed = load_hybrid_manifest(self.root / "development_data/hybrid-development-manifest-v1.json")
        self.assertEqual(committed, self.manifest)
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            write_hybrid_manifest(self.manifest, path)
            self.assertEqual(load_hybrid_manifest(path), self.manifest)
            with self.assertRaises(HybridProtocolError):
                write_hybrid_manifest(self.manifest, path)
            path.write_text(path.read_text().replace("hybrid_normal_fit", "test", 1), encoding="utf-8")
            with self.assertRaises((HybridProtocolError, TypeError, ValueError)):
                load_hybrid_manifest(path)

    def test_forbidden_role_and_anomaly_in_normal_fit_fail_closed(self):
        with self.assertRaises(HybridProtocolError):
            replace(self.manifest.normal_fit[0], split_role="test")
        with self.assertRaises(HybridProtocolError):
            replace(self.manifest.normal_fit[0], image_outcome="anomaly_present")

    @pytest.mark.protected_integration
    def test_optional_protected_stores_reproduce_committed_manifest(self):
        root = require_protected_files(
            self.root,
            "research_data/registry/datasets.sqlite",
            "outputs/registered_experiment_results.sqlite3",
        )
        generated = create_hybrid_development_manifest(
            repository_root=root,
            registry_database=root / "research_data/registry/datasets.sqlite",
            historical_result_database=(
                root / "outputs/registered_experiment_results.sqlite3"
            ),
        )
        self.assertEqual(generated, self.manifest)


if __name__ == "__main__":
    unittest.main()
