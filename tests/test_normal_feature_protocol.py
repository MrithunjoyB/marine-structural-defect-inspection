from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import unittest

from structvision.development_protocol import (
    create_protected_development_manifest,
    load_development_manifest,
    write_development_manifest,
)
from structvision.normal_feature.errors import DevelopmentProtocolError


class ProtectedDevelopmentProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).parents[1]
        cls.manifest = create_protected_development_manifest(
            repository_root=cls.root,
            registry_database=cls.root / "research_data/registry/datasets.sqlite",
            historical_result_database=cls.root / "outputs/registered_experiment_results.sqlite3",
        )

    def test_protected_counts_roles_categories_and_hash_are_deterministic(self):
        second = create_protected_development_manifest(
            repository_root=self.root,
            registry_database=self.root / "research_data/registry/datasets.sqlite",
            historical_result_database=self.root / "outputs/registered_experiment_results.sqlite3",
        )
        self.assertEqual(len(self.manifest.normal_fit), 91)
        self.assertEqual(len(self.manifest.calibration_validation), 72)
        self.assertEqual(self.manifest.manifest_hash, second.manifest_hash)
        self.assertEqual(self.manifest.manifest_hash, "2aa40b9db145a37522775b7ac605ae201b91e564cde881528fd6d41f449f3d58")
        self.assertEqual({item.image_outcome for item in self.manifest.normal_fit}, {"no_anomaly"})
        self.assertEqual(
            {item.category for item in self.manifest.calibration_validation},
            set(self.manifest.expected_validation_categories),
        )

    def test_no_test_or_pilot_exact_identity_and_no_group_crossing(self):
        self.assertNotIn("test", {item.split_role for item in self.manifest.selected_images})
        fit = self.manifest.normal_fit
        validation = self.manifest.calibration_validation
        self.assertFalse({item.image_sha256 for item in fit} & {item.image_sha256 for item in validation})
        self.assertFalse({item.source_group_id for item in fit} & {item.source_group_id for item in validation})
        self.assertFalse({item.template_group_id for item in fit} & {item.template_group_id for item in validation})
        selected = {item.image_id for item in self.manifest.selected_images}
        protected_exact = {
            item.image_id for item in self.manifest.exclusions
            if any("exact_sha256" in reason for reason in item.reasons)
        }
        self.assertFalse(selected & protected_exact)

    def test_manifest_round_trip_is_immutable_and_tamper_detected(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            write_development_manifest(self.manifest, path)
            self.assertEqual(load_development_manifest(path), self.manifest)
            with self.assertRaises(DevelopmentProtocolError):
                write_development_manifest(self.manifest, path)
            path.write_text(path.read_text().replace("normal_fit", "normal_fitz", 1), encoding="utf-8")
            with self.assertRaises((DevelopmentProtocolError, TypeError, ValueError)):
                load_development_manifest(path)

    def test_missing_category_and_forbidden_role_fail_closed(self):
        with self.assertRaises(DevelopmentProtocolError):
            replace(self.manifest, missing_categories=("thin_crack",))
        with self.assertRaises(DevelopmentProtocolError):
            replace(self.manifest.selected_images[0], split_role="test")

    def test_registry_and_historical_store_hashes_are_recorded(self):
        registry = self.root / "research_data/registry/datasets.sqlite"
        historical = self.root / "outputs/registered_experiment_results.sqlite3"
        self.assertEqual(hashlib.sha256(registry.read_bytes()).hexdigest(), self.manifest.source_registry_sha256)
        self.assertEqual(hashlib.sha256(historical.read_bytes()).hexdigest(), self.manifest.historical_result_store_sha256)


if __name__ == "__main__":
    unittest.main()
