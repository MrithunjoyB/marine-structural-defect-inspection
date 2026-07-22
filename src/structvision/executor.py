"""Prospective v2 executor using the same public detector API as direct callers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping
from uuid import uuid4

import cv2
import numpy as np

from scientific_contract.evaluation_policy import default_evaluation_policy
from scientific_contract.hashing import canonical_json, sha256_json
from scientific_contract.matching import (
    EncodedMask,
    GroundTruthRecord,
    ProposalRecord,
    ProposalSet,
    TruthInstance,
)
from scientific_contract.metrics import ImageEvaluationV2, evaluate_image
from scientific_contract.result_store import (
    ExecutionAttemptSummary,
    RESULT_SCHEMA_VERSION,
    ResultRowV2,
)
from scientific_contract.specification import ExperimentSpecificationV2

from .api import StructuralAnomalyDetector
from .configuration import DetectorConfig
from .errors import (
    DuplicateImageIDError,
    ExperimentExecutionError,
    ProvenanceMismatchError,
    SinkError,
    SpecificationMismatchError,
)
from .inputs import content_hash
from .sinks import ResultSink
from .types import AnalysisResult


@dataclass(frozen=True)
class ExperimentSample:
    """Caller-supplied locations/content for one immutable selected-image identity."""

    image_id: str
    image: object
    ground_truth: object
    ground_truth_status: str
    colour_space: str | None = None
    alpha_handling: str | None = None
    category: str = ""
    acquisition_group_id: str = ""
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not self.image_id.strip():
            raise ValueError("ExperimentSample.image_id is required")
        if self.ground_truth_status not in {"anomaly_present", "no_anomaly"}:
            raise ValueError("ground_truth_status must be anomaly_present or no_anomaly")


@dataclass(frozen=True)
class ExecutionAnalysis:
    image_id: str
    method_id: str
    result: AnalysisResult
    evaluation: ImageEvaluationV2


@dataclass(frozen=True)
class V2ExecutionReport:
    specification: ExperimentSpecificationV2
    summary: ExecutionAttemptSummary
    rows: tuple[ResultRowV2, ...]
    analyses: tuple[ExecutionAnalysis, ...]

    @property
    def execution_attempt_id(self) -> str:
        return self.summary.execution_attempt_id

    @property
    def identity(self) -> str:
        return self.execution_attempt_id

    @property
    def expected_count(self) -> int:
        return self.summary.expected_pairs

    @property
    def attempted_count(self) -> int:
        return self.summary.attempted_pairs

    @property
    def completed_count(self) -> int:
        return self.summary.completed_pairs

    @property
    def failed_count(self) -> int:
        return self.summary.failed_pairs

    @property
    def skipped_count(self) -> int:
        return self.summary.skipped_pairs


def _verify_specification_hash(specification: ExperimentSpecificationV2) -> None:
    actual = sha256_json(specification.to_dict(include_specification_hash=False))
    if actual != specification.specification_hash:
        raise SpecificationMismatchError("Experiment specification hash mismatch")


def _detector_config(specification: ExperimentSpecificationV2, method_id: str) -> tuple[DetectorConfig, str]:
    method = specification.method(method_id)
    try:
        config = DetectorConfig.from_dict(method.method_configuration.value)
    except Exception as error:
        raise SpecificationMismatchError(
            f"Method {method_id} does not contain a complete DetectorConfig"
        ) from error
    sections = config.specification_sections()
    comparisons = (
        ("preprocessing", specification.preprocessing_configuration.value, sections["preprocessing"]),
        ("proposal", specification.proposal_configuration.value, sections["proposal"]),
        ("feature_and_scoring", specification.feature_scoring_configuration.value, sections["feature_and_scoring"]),
    )
    for name, specified, executed in comparisons:
        if canonical_json(specified) != canonical_json(executed):
            raise SpecificationMismatchError(f"Executed {name} configuration differs from the immutable specification")
    if method.method_id != config.implementation_id or method.implementation_version != config.implementation_version:
        raise SpecificationMismatchError("Method identity/version differs from DetectorConfig")
    if method.method_configuration.configuration_hash != config.configuration_hash:
        raise SpecificationMismatchError("Method configuration hash differs from DetectorConfig")
    if specification.maximum_proposal_count != config.proposals.maximum_proposal_count:
        raise SpecificationMismatchError("Maximum proposal count differs from DetectorConfig")
    if specification.deterministic_mode != config.deterministic_mode:
        raise SpecificationMismatchError("Deterministic-mode state differs from DetectorConfig")
    seeds = dict(specification.random_seeds)
    if seeds.get("structvision") != config.random_seed:
        raise SpecificationMismatchError("Named structvision random seed differs from DetectorConfig")
    executed = specification.expected_executable_configuration(method_id)
    try:
        executed_hash = specification.verify_executed_configuration(method_id, executed)
    except Exception as error:
        raise SpecificationMismatchError("Executed configuration hash mismatch") from error
    return config, executed_hash


def _read_truth(value: object) -> np.ndarray:
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_file():
            raise ProvenanceMismatchError(f"Ground-truth path is missing: {path}")
        decoded = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ProvenanceMismatchError(f"Could not decode ground truth: {path.name}")
        array = decoded
    elif isinstance(value, np.ndarray):
        array = np.asarray(value)
    else:
        raise ProvenanceMismatchError(f"Unsupported ground-truth type: {type(value).__name__}")
    if array.dtype != np.uint8 or array.size == 0:
        raise ProvenanceMismatchError("Ground truth must be a non-empty uint8 image")
    if array.ndim == 3 and array.shape[2] in {3, 4}:
        array = cv2.cvtColor(array[..., :3], cv2.COLOR_BGR2GRAY)
    elif array.ndim == 3 and array.shape[2] == 1:
        array = array[..., 0]
    elif array.ndim != 2:
        raise ProvenanceMismatchError(f"Unsupported ground-truth shape: {array.shape}")
    return np.ascontiguousarray((array > 0).astype(np.uint8) * 255)


def _ground_truth_record(sample: ExperimentSample, shape: tuple[int, int]) -> GroundTruthRecord:
    mask = _read_truth(sample.ground_truth)
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8) * 255
    if sample.ground_truth_status == "no_anomaly":
        if np.any(mask):
            raise ProvenanceMismatchError("A no_anomaly sample cannot contain foreground truth pixels")
        return GroundTruthRecord(sample.image_id, "no_anomaly", ())
    count, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    if count <= 1:
        raise ProvenanceMismatchError("An anomaly_present sample requires non-empty ground truth")
    truths = tuple(
        TruthInstance(
            f"T{index:03d}",
            EncodedMask.from_array((labels == index).astype(np.uint8)),
            sample.category,
        )
        for index in range(1, count)
    )
    return GroundTruthRecord(sample.image_id, "anomaly_present", truths)


def _proposal_set(result: AnalysisResult, method_id: str, ranking_definition: str | None) -> ProposalSet:
    records = tuple(
        ProposalRecord(
            proposal.proposal_id,
            EncodedMask.from_array(proposal.final_mask),
            proposal.priority_score,
            proposal.rank,
        )
        for proposal in result.proposals
    )
    return ProposalSet(
        method_id,
        records,
        True,
        ranking_definition or "priority_score descending; frozen proposal order; proposal_id tie-break",
    )


def _success_row(
    *, specification: ExperimentSpecificationV2, sample: ExperimentSample,
    selected: object, method: object, executed_hash: str, attempt_id: str,
    result: AnalysisResult, evaluation: ImageEvaluationV2, recorded: str,
) -> ResultRowV2:
    policy = default_evaluation_policy()
    matching = evaluation.result_at(policy.primary_match_threshold)
    proposal_details = result.to_dict(include_heatmap=False)
    metrics = asdict(evaluation)
    return ResultRowV2(
        result_id=sha256_json({"attempt": attempt_id, "image": sample.image_id, "method": method.method_id}),
        experiment_specification_hash=specification.specification_hash,
        executed_configuration_hash=executed_hash,
        method_implementation_id=method.method_id,
        method_implementation_version=method.implementation_version,
        evaluation_policy_id=specification.evaluation_policy_id,
        evaluation_policy_version=specification.evaluation_policy_version,
        evaluation_policy_hash=specification.evaluation_policy_hash,
        image_id=sample.image_id,
        image_content_hash=selected.image_sha256,
        ground_truth_content_hash=selected.ground_truth_sha256,
        proposal_output_artifact_hash=None,
        proposal_output_details_json=canonical_json(proposal_details),
        matching_policy_hash=policy.matching_policy_hash,
        result_schema_version=RESULT_SCHEMA_VERSION,
        execution_attempt_id=attempt_id,
        deterministic_mode=specification.deterministic_mode,
        recorded_timestamp=recorded,
        completion_status="completed",
        matching_details_json=canonical_json(matching.to_dict()),
        metrics_json=canonical_json(metrics),
    )


def _failed_row(
    *, specification: ExperimentSpecificationV2, sample: ExperimentSample,
    selected: object, method: object, executed_hash: str, attempt_id: str,
    error: Exception, recorded: str,
) -> ResultRowV2:
    policy = default_evaluation_policy()
    return ResultRowV2(
        result_id=sha256_json({"attempt": attempt_id, "image": sample.image_id, "method": method.method_id}),
        experiment_specification_hash=specification.specification_hash,
        executed_configuration_hash=executed_hash,
        method_implementation_id=method.method_id,
        method_implementation_version=method.implementation_version,
        evaluation_policy_id=specification.evaluation_policy_id,
        evaluation_policy_version=specification.evaluation_policy_version,
        evaluation_policy_hash=specification.evaluation_policy_hash,
        image_id=sample.image_id,
        image_content_hash=selected.image_sha256,
        ground_truth_content_hash=selected.ground_truth_sha256,
        proposal_output_artifact_hash=None,
        proposal_output_details_json=canonical_json({
            "proposals": [], "failure_type": type(error).__name__, "failure_message": str(error),
        }),
        matching_policy_hash=policy.matching_policy_hash,
        result_schema_version=RESULT_SCHEMA_VERSION,
        execution_attempt_id=attempt_id,
        deterministic_mode=specification.deterministic_mode,
        recorded_timestamp=recorded,
        completion_status="failed",
        matching_details_json="{}",
        metrics_json="{}",
    )


class ExperimentExecutorV2:
    """Fail-closed prospective executor; it never reads historical stores or UI state."""

    def __init__(self, *, worker_count: int = 1) -> None:
        if worker_count != 1:
            raise ValueError("V2 deterministic execution currently requires worker_count=1")
        self.worker_count = worker_count

    def execute(
        self,
        specification: ExperimentSpecificationV2,
        samples: Iterable[ExperimentSample],
        *,
        sink: ResultSink | None = None,
        fail_fast: bool = True,
        execution_attempt_id: str | None = None,
    ) -> V2ExecutionReport:
        if not isinstance(specification, ExperimentSpecificationV2):
            raise TypeError("specification must be ExperimentSpecificationV2")
        if type(fail_fast) is not bool:
            raise TypeError("fail_fast must be boolean")
        _verify_specification_hash(specification)
        policy = default_evaluation_policy()
        if (
            specification.evaluation_policy_id != policy.policy_id
            or specification.evaluation_policy_version != policy.policy_version
            or specification.evaluation_policy_hash != policy.configuration_hash
            or specification.metric_definitions_hash != policy.metric_definitions_hash
            or tuple(specification.matching_thresholds) != policy.threshold_analyses
        ):
            raise SpecificationMismatchError("Specification does not select the implemented v2 evaluation policy")
        prepared = tuple(samples)
        identifiers = [sample.image_id for sample in prepared]
        if len(identifiers) != len(set(identifiers)):
            raise DuplicateImageIDError("V2 samples contain duplicate image IDs")
        expected_ids = [item.image_id for item in specification.selected_images]
        if identifiers != expected_ids:
            raise ProvenanceMismatchError("Samples must exactly match selected-image identity and order")
        selected_by_id = {item.image_id: item for item in specification.selected_images}
        for sample in prepared:
            selected = selected_by_id[sample.image_id]
            actual_image_hash = content_hash(sample.image)
            actual_truth_hash = content_hash(sample.ground_truth)
            if actual_image_hash != selected.image_sha256:
                raise ProvenanceMismatchError(f"Image hash mismatch for {sample.image_id}")
            if actual_truth_hash != selected.ground_truth_sha256:
                raise ProvenanceMismatchError(f"Ground-truth hash mismatch for {sample.image_id}")
        method_runtime = {
            method.method_id: _detector_config(specification, method.method_id)
            for method in specification.methods
        }
        attempt_id = execution_attempt_id or str(uuid4())
        if not attempt_id.strip():
            raise ValueError("execution_attempt_id cannot be empty")
        started = datetime.now(timezone.utc).isoformat()
        rows: list[ResultRowV2] = []
        analyses: list[ExecutionAnalysis] = []
        completed = 0
        failed = 0
        for sample in prepared:
            selected = selected_by_id[sample.image_id]
            for method in specification.methods:
                config, executed_hash = method_runtime[method.method_id]
                pair_started = time.perf_counter()
                try:
                    result = StructuralAnomalyDetector(config).analyse(
                        sample.image,
                        image_id=sample.image_id,
                        colour_space=sample.colour_space,
                        alpha_handling=sample.alpha_handling,
                        metadata=sample.metadata,
                    )
                    if not result.provenance.protected_hashes_verified:
                        raise ProvenanceMismatchError("Protected classical source hashes differ from the frozen contract")
                    ground_truth = _ground_truth_record(sample, result.image_shape[:2])
                    proposal_set = _proposal_set(result, method.method_id, method.ranking_definition)
                    evaluation = evaluate_image(
                        proposal_set,
                        ground_truth,
                        category=sample.category,
                        acquisition_group_id=sample.acquisition_group_id,
                        processing_time_seconds=time.perf_counter() - pair_started,
                        hardware_context=specification.hardware_metadata.canonical_payload,
                        cache_state="uncontrolled_process_cache",
                        policy=policy,
                    )
                    recorded = datetime.now(timezone.utc).isoformat()
                    rows.append(_success_row(
                        specification=specification, sample=sample, selected=selected,
                        method=method, executed_hash=executed_hash, attempt_id=attempt_id,
                        result=result, evaluation=evaluation, recorded=recorded,
                    ))
                    analyses.append(ExecutionAnalysis(sample.image_id, method.method_id, result, evaluation))
                    completed += 1
                except Exception as error:
                    if fail_fast:
                        raise ExperimentExecutionError(
                            f"V2 execution failed for image={sample.image_id}, method={method.method_id}"
                        ) from error
                    rows.append(_failed_row(
                        specification=specification, sample=sample, selected=selected,
                        method=method, executed_hash=executed_hash, attempt_id=attempt_id,
                        error=error, recorded=datetime.now(timezone.utc).isoformat(),
                    ))
                    failed += 1
        completed_timestamp = datetime.now(timezone.utc).isoformat()
        attempted = len(rows)
        expected = specification.expected_pair_count
        status = "completed" if completed == expected and failed == 0 else ("failed" if completed == 0 else "partially_completed")
        summary = ExecutionAttemptSummary(
            attempt_id,
            specification.specification_hash,
            status,
            expected,
            attempted,
            completed,
            failed,
            0,
            attempted,
            started,
            completed_timestamp,
        )
        report = V2ExecutionReport(specification, summary, tuple(rows), tuple(analyses))
        if sink is not None:
            try:
                sink.write(report)
            except SinkError:
                raise
            except Exception as error:
                raise SinkError(f"Result sink failed for attempt {attempt_id}") from error
        return report
