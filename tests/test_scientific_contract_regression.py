from pathlib import Path
import hashlib
import json
import sqlite3
import unittest


REGISTERED_DATA_REFERENCE = {
    "schema_version": "structvision-registered-data-reference-v1",
    "payload_access_policy": "forbidden-in-default-tests",
    "entries": {
        "registered_images": {
            "file_count": 613,
            "aggregate_sha256": (
                "1a091e24311e3d7e56398bfbf3d561c461941bee087eea729456e75d86f274f7"
            ),
        },
        "registered_annotations": {
            "file_count": 323,
            "aggregate_sha256": (
                "299906f6e2fac5e00f3f67e2cb2789d4d61326c4741d0b08b0f36f2f96165382"
            ),
        },
        "registered_splits": {
            "file_count": 2,
            "aggregate_sha256": (
                "2a526a3a0326d42c96587b53bc7bfb462fdc156929b8ee60ea63f8bc35aef876"
            ),
        },
    },
}
REGISTERED_DATA_REFERENCE_IDENTITY = (
    "ac7823dea49d4b763fb2aecb04122af89b05c9ccf03f15741cdedd68ad1e6fc3"
)


class HistoricalEvidenceRegressionTests(unittest.TestCase):
    """Read-only guard for the local reviewed evidence stores when available."""

    def test_historical_888_rows_and_database_hash_are_unchanged(self):
        root = Path(__file__).parents[1]
        database = root / "outputs" / "registered_experiment_results.sqlite3"
        if not database.exists():
            self.skipTest("Ignored historical runtime result store is not present")
        before = hashlib.sha256(database.read_bytes()).hexdigest()
        uri = database.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        try:
            count = connection.execute("SELECT count(*) FROM automatic_results").fetchone()[0]
        finally:
            connection.close()
        after = hashlib.sha256(database.read_bytes()).hexdigest()
        self.assertEqual(count, 888)
        self.assertEqual(before, "1ebde1de1f065b5b220366798147beb67dd10a446b7cd8840f988c9aeda9ce92")
        self.assertEqual(after, before)

    def test_all_historical_store_hashes_and_counts_are_unchanged(self):
        root = Path(__file__).parents[1]
        expected = {
            "outputs/registered_experiment_results.sqlite3": ("1ebde1de1f065b5b220366798147beb67dd10a446b7cd8840f988c9aeda9ce92", "automatic_results", 888),
            "outputs/research_evaluation.sqlite3": ("9a77d748dbf9780f5f0e104bea3412ddaadcad10b54a2c1fceed0e532acef640", "experiment_records", 0),
        }
        for relative, (expected_hash, table, expected_rows) in expected.items():
            database = root / relative
            if not database.exists():
                continue
            self.assertEqual(hashlib.sha256(database.read_bytes()).hexdigest(), expected_hash)
            connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
            try:
                self.assertEqual(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0], expected_rows)
            finally:
                connection.close()

    def test_protected_algorithm_sources_are_byte_unchanged(self):
        root = Path(__file__).parents[1]
        expected = {
            "preprocess.py": "fcd5da2b563e420b18f5baaf6a73c276457b4b6c65b33531cfeaf917ffefcf48",
            "feature_extraction.py": "1ae26484de02f4d5764d2ee90ee519babe307192c12fa8deecfc50d96ff1976c",
            "region_proposal.py": "65815b84dd8078b11776ccb70e81688e47f4e7afe1624534d6872bec1e46f80a",
            "scoring.py": "d284c8012464003a0ddc5a697c4d85303fbe73a356f8ee7f649c5d75ebcd3a79",
            "config.py": "21c41875fdaaa947eaf0c71e3e6c695325e07b0a2cfdfdb0b27822b263a74385",
            "expanded_synthetic_benchmark.py": "1f33297f25e2cb1dd208ce11cada3eb073b316e6a24bf819180fb40f945fcf0c",
            "synthetic_benchmark.py": "c22750f063cf0b120e7957f934eb5227d40ea9a82ea03d30657c6d302e185278",
        }
        for relative, expected_hash in expected.items():
            self.assertEqual(hashlib.sha256((root / relative).read_bytes()).hexdigest(), expected_hash, relative)

    def test_registered_data_reference_metadata_is_unchanged(self):
        """Validate committed identities without opening registered payload files."""
        canonical = json.dumps(
            REGISTERED_DATA_REFERENCE,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(
            hashlib.sha256(canonical).hexdigest(),
            REGISTERED_DATA_REFERENCE_IDENTITY,
        )
        self.assertEqual(
            REGISTERED_DATA_REFERENCE["payload_access_policy"],
            "forbidden-in-default-tests",
        )
        for metadata in REGISTERED_DATA_REFERENCE["entries"].values():
            self.assertGreater(metadata["file_count"], 0)
            self.assertRegex(metadata["aggregate_sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
