from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from scientific_contract.evaluation_policy import default_evaluation_policy
from scientific_contract.specification import (
    ExperimentSpecificationV2,
    FrozenConfiguration,
    MethodSpecification,
    SPECIFICATION_SCHEMA_VERSION,
    SelectedImageIdentity,
)
from structvision import (
    DetectorConfig,
    ExperimentExecutorV2,
    ExperimentSample,
    MemoryResultSink,
    NullResultSink,
    PreprocessingConfig,
    ProposalConfig,
    V2SQLiteResultSink,
)
from structvision.errors import ProvenanceMismatchError, SinkError, SpecificationMismatchError
from structvision.inputs import content_hash


DIGEST = "a" * 64


def fast_config():
    return DetectorConfig(
        preprocessing=PreprocessingConfig(256, False, False, False, 0, 0),
        proposals=ProposalConfig(minimum_area_pixels=20, maximum_proposal_count=4),
    )


def write_case(root: Path, name: str, *, empty_truth: bool = False):
    image = np.full((96, 144, 3), 155, np.uint8)
    cv2.line(image, (15, 80), (130, 15), (30, 30, 30), 4, cv2.LINE_AA)
    truth = np.zeros(image.shape[:2], np.uint8)
    if not empty_truth:
        cv2.line(truth, (15, 80), (130, 15), 255, 8, cv2.LINE_AA)
    image_path = root / f"{name}.png"
    truth_path = root / f"{name}-truth.png"
    self_image = cv2.imencode(".png", image)[1].tobytes()
    self_truth = cv2.imencode(".png", truth)[1].tobytes()
    image_path.write_bytes(self_image)
    truth_path.write_bytes(self_truth)
    return ExperimentSample(name, image_path, truth_path, "anomaly_present", category="thin_crack")


def specification(samples, config=None, *, proposal_payload=None):
    config = config or fast_config()
    sections = config.specification_sections()
    policy = default_evaluation_policy()
    method = MethodSpecification(
        config.implementation_id,
        config.implementation_version,
        FrozenConfiguration.from_value(config.to_dict()),
        True,
        "priority_score descending; frozen proposal order; proposal_id tie-break",
    )
    return ExperimentSpecificationV2(
        schema_version=SPECIFICATION_SCHEMA_VERSION,
        experiment_id="TEMP-V2-API",
        experiment_version=1,
        dataset_id="temporary-fixtures",
        dataset_version="1",
        dataset_manifest_hash=DIGEST,
        split_manifest_hash="b" * 64,
        split_lock_hash="c" * 64,
        selected_images=tuple(
            SelectedImageIdentity(item.image_id, content_hash(item.image), content_hash(item.ground_truth))
            for item in samples
        ),
        methods=(method,),
        preprocessing_configuration=FrozenConfiguration.from_value(sections["preprocessing"]),
        proposal_configuration=FrozenConfiguration.from_value(proposal_payload or sections["proposal"]),
        feature_scoring_configuration=FrozenConfiguration.from_value(sections["feature_and_scoring"]),
        maximum_proposal_count=config.proposals.maximum_proposal_count,
        random_seeds=(("structvision", config.random_seed),),
        deterministic_mode=config.deterministic_mode,
        evaluation_policy_id=policy.policy_id,
        evaluation_policy_version=policy.policy_version,
        evaluation_policy_hash=policy.configuration_hash,
        matching_thresholds=policy.threshold_analyses,
        metric_definitions_hash=policy.metric_definitions_hash,
        allowed_fitting_splits=("train", "validation"),
        forbidden_test_access=True,
        git_commit="dca2ac4ed42ac9da225eaf3886931bcaedc92eeb",
        git_tree_state="clean",
        uncommitted_diff_hash=None,
        python_version="3.9-test",
        dependency_snapshot=FrozenConfiguration.from_value({"runtime": "temporary"}),
        dependency_lock_hash="d" * 64,
        operating_system_metadata=FrozenConfiguration.from_value({"system": "test"}),
        hardware_metadata=FrozenConfiguration.from_value({"hardware": "test"}),
        opencv_version=cv2.__version__,
        opencv_backend="test",
        creation_timestamp=datetime.now(timezone.utc).isoformat(),
    )


def proposal_signature(report):
    return [
        (item.proposal_id, item.rank, item.bbox, item.priority_score, item.final_mask.tobytes())
        for item in report.analyses[0].result.proposals
    ]


class V2ExecutorTests(unittest.TestCase):
    def test_valid_temporary_lifecycle_counters_matching_and_memory_sink(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = write_case(root, "image-1")
            spec = specification((sample,))
            sink = MemoryResultSink()
            report = ExperimentExecutorV2().execute(
                spec, (sample,), sink=sink, execution_attempt_id="attempt-valid"
            )
            self.assertEqual(report.expected_count, 1)
            self.assertEqual(report.attempted_count, 1)
            self.assertEqual(report.completed_count, 1)
            self.assertEqual(report.failed_count, 0)
            self.assertEqual(report.skipped_count, 0)
            self.assertEqual(len(sink.records), 1)
            self.assertEqual(report.rows[0].completion_status, "completed")
            self.assertTrue(report.rows[0].matching_details_json)
            self.assertTrue(report.rows[0].metrics_json)

    def test_specification_hash_and_executed_configuration_mismatches_fail_closed(self):
        with TemporaryDirectory() as temporary:
            sample = write_case(Path(temporary), "image-1")
            spec = specification((sample,))
            object.__setattr__(spec, "specification_hash", "0" * 64)
            with self.assertRaises(SpecificationMismatchError):
                ExperimentExecutorV2().execute(spec, (sample,))
            config = fast_config()
            altered = dict(config.specification_sections()["proposal"])
            altered["minimum_area_pixels"] = altered["minimum_area_pixels"] + 1
            inconsistent = specification((sample,), config, proposal_payload=altered)
            with self.assertRaises(SpecificationMismatchError):
                ExperimentExecutorV2().execute(inconsistent, (sample,))

    def test_image_and_ground_truth_hash_mismatches_fail_before_analysis(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = write_case(root, "image-1")
            spec = specification((sample,))
            Path(sample.image).write_bytes(Path(sample.image).read_bytes() + b"changed")
            with self.assertRaises(ProvenanceMismatchError):
                ExperimentExecutorV2().execute(spec, (sample,))
            sample = write_case(root, "image-2")
            spec = specification((sample,))
            Path(sample.ground_truth).write_bytes(Path(sample.ground_truth).read_bytes() + b"changed")
            with self.assertRaises(ProvenanceMismatchError):
                ExperimentExecutorV2().execute(spec, (sample,))

    def test_null_or_absent_sink_creates_no_result_store(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = write_case(root, "image-1")
            spec = specification((sample,))
            before = sorted(path.name for path in root.iterdir())
            ExperimentExecutorV2().execute(spec, (sample,), execution_attempt_id="none")
            ExperimentExecutorV2().execute(
                spec, (sample,), sink=NullResultSink(), execution_attempt_id="null"
            )
            after = sorted(path.name for path in root.iterdir())
            self.assertEqual(after, before)

    def test_explicit_sqlite_sink_is_append_only(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = write_case(root, "image-1")
            spec = specification((sample,))
            sink = V2SQLiteResultSink(root / "temporary-v2.sqlite3")
            report = ExperimentExecutorV2().execute(
                spec, (sample,), sink=sink, execution_attempt_id="append-once"
            )
            counts = sink.store.counts()
            self.assertEqual(counts["result_rows"], 1)
            self.assertEqual(counts["execution_attempts"], 1)
            with self.assertRaises(SinkError):
                sink.write(report)
            self.assertEqual(sink.store.counts()["result_rows"], 1)

    def test_deterministic_replay_preserves_algorithm_outputs(self):
        with TemporaryDirectory() as temporary:
            sample = write_case(Path(temporary), "image-1")
            spec = specification((sample,))
            first = ExperimentExecutorV2().execute(spec, (sample,), execution_attempt_id="replay-a")
            second = ExperimentExecutorV2().execute(spec, (sample,), execution_attempt_id="replay-b")
            self.assertEqual(proposal_signature(first), proposal_signature(second))
            self.assertEqual(first.analyses[0].result.diagnostics, second.analyses[0].result.diagnostics)

    def test_per_image_failure_isolation_has_explicit_counts(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            good = write_case(root, "good")
            bad = write_case(root, "bad", empty_truth=True)
            spec = specification((good, bad))
            report = ExperimentExecutorV2().execute(
                spec, (good, bad), fail_fast=False, execution_attempt_id="isolated"
            )
            self.assertEqual(report.expected_count, 2)
            self.assertEqual(report.attempted_count, 2)
            self.assertEqual(report.completed_count, 1)
            self.assertEqual(report.failed_count, 1)
            self.assertEqual([row.completion_status for row in report.rows], ["completed", "failed"])

    def test_sample_order_must_match_the_specification(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = write_case(root, "first")
            second = write_case(root, "second")
            spec = specification((first, second))
            with self.assertRaises(ProvenanceMismatchError):
                ExperimentExecutorV2().execute(spec, (second, first))


if __name__ == "__main__":
    unittest.main()
