"""Mixed classical/PatchCore development execution under scientific-contract v2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import platform
import time
from typing import Iterable
from uuid import uuid4

import cv2
import numpy as np

from scientific_contract.evaluation_policy import default_evaluation_policy
from scientific_contract.hashing import canonical_json, sha256_json
from scientific_contract.matching import EncodedMask, ProposalRecord, ProposalSet
from scientific_contract.metrics import ImageEvaluationV2, evaluate_image
from scientific_contract.result_store import ExecutionAttemptSummary, RESULT_SCHEMA_VERSION, ResultRowV2
from scientific_contract.specification import (
    ExperimentSpecificationV2,
    FrozenConfiguration,
    MethodSpecification,
    SPECIFICATION_SCHEMA_VERSION,
    SelectedImageIdentity,
)
from structvision.api import StructuralAnomalyDetector
from structvision.configuration import DetectorConfig
from structvision.development_protocol import ProtectedDevelopmentManifest
from structvision.errors import ProvenanceMismatchError, SpecificationMismatchError
from structvision.executor import ExperimentSample, _ground_truth_record, _proposal_set
from structvision.inputs import content_hash
from structvision.normal_feature.calibration import CalibrationArtifact
from structvision.normal_feature.configuration import NormalFeatureConfig
from structvision.normal_feature.model_artifact import NormalFeatureModelArtifact
from structvision.normal_feature.patchcore import EXACT_RUNTIME_VERSIONS, NormalFeatureAnomalyDetector, _git_metadata
from structvision.normal_feature.proposal_extraction import extract_proposals
from structvision.normal_feature.types import NormalFeatureAnalysisResult
from structvision.sinks import ResultSink


DEVELOPMENT_EXPERIMENT_ID = "SYN-NORMAL-FEATURE-DEV-001"
DEVELOPMENT_EXPERIMENT_VERSION = 1
PRIMARY_OPERATING_POINT_ID = "fp-budget-0.50"


@dataclass(frozen=True)
class DevelopmentExecutionAnalysis:
    image_id: str
    method_id: str
    result: object
    evaluation: ImageEvaluationV2


@dataclass(frozen=True)
class DevelopmentExecutionReport:
    specification: ExperimentSpecificationV2
    summary: ExecutionAttemptSummary
    rows: tuple[ResultRowV2, ...]
    analyses: tuple[DevelopmentExecutionAnalysis, ...]
    learned_budget_evaluations: tuple[tuple[str, tuple[ImageEvaluationV2, ...]], ...]

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
    def completed_count(self) -> int:
        return self.summary.completed_pairs


def development_experiment_samples(
    manifest: ProtectedDevelopmentManifest,
    repository_root: Path,
) -> tuple[ExperimentSample, ...]:
    root = Path(repository_root)
    samples = []
    for item in manifest.calibration_validation:
        if item.ground_truth_kind == "implicit_verified_zero_mask":
            image = root / item.image_path
            decoded = cv2.imdecode(np.frombuffer(image.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise ProvenanceMismatchError(f"Could not decode selected development image: {item.image_id}")
            truth: object = np.zeros(decoded.shape[:2], dtype=np.uint8)
        else:
            truth = root / str(item.ground_truth_path)
        samples.append(ExperimentSample(
            image_id=item.image_id,
            image=root / item.image_path,
            ground_truth=truth,
            ground_truth_status=item.image_outcome,
            category=item.category,
            acquisition_group_id=item.acquisition_group_id,
            metadata={
                "development_role": item.role,
                "source_group_id": item.source_group_id,
                "template_group_id": item.template_group_id,
            },
        ))
    return tuple(samples)


def create_development_experiment_specification(
    *,
    manifest: ProtectedDevelopmentManifest,
    model_artifact: NormalFeatureModelArtifact,
    calibration_artifact: CalibrationArtifact,
    dependency_lock_hash: str,
    repository_root: Path,
    classical_config: DetectorConfig | None = None,
    learned_config: NormalFeatureConfig | None = None,
) -> ExperimentSpecificationV2:
    classical = classical_config or DetectorConfig()
    learned = learned_config or NormalFeatureConfig()
    if model_artifact.configuration_hash != learned.configuration_hash:
        raise SpecificationMismatchError("Learned config and model artifact differ")
    if calibration_artifact.model_artifact_hash != model_artifact.artifact_hash:
        raise SpecificationMismatchError("Calibration and model artifacts differ")
    policy = default_evaluation_policy()
    methods = (
        MethodSpecification(
            classical.implementation_id,
            classical.implementation_version,
            FrozenConfiguration.from_value(classical.to_dict()),
            True,
            "priority_score descending; frozen proposal order; proposal_id tie-break",
        ),
        MethodSpecification(
            learned.implementation_id,
            learned.implementation_version,
            FrozenConfiguration.from_value({
                "detector_config": learned.to_dict(),
                "model_artifact_hash": model_artifact.artifact_hash,
                "calibration_artifact_hash": calibration_artifact.artifact_hash,
                "primary_operating_point_id": PRIMARY_OPERATING_POINT_ID,
            }),
            True,
            "component anomaly score descending; area descending; bbox lexical tie-break",
        ),
    )
    classical_sections = classical.specification_sections()
    learned_sections = learned.specification_sections()
    commit, tree_state, diff_hash = _git_metadata(Path(repository_root))
    return ExperimentSpecificationV2(
        schema_version=SPECIFICATION_SCHEMA_VERSION,
        experiment_id=DEVELOPMENT_EXPERIMENT_ID,
        experiment_version=DEVELOPMENT_EXPERIMENT_VERSION,
        dataset_id=manifest.source_dataset_id,
        dataset_version=manifest.source_dataset_version,
        dataset_manifest_hash=manifest.source_registry_sha256,
        split_manifest_hash=manifest.manifest_hash,
        split_lock_hash=manifest.manifest_hash,
        selected_images=tuple(
            SelectedImageIdentity(item.image_id, item.image_sha256, item.ground_truth_sha256)
            for item in manifest.calibration_validation
        ),
        methods=methods,
        preprocessing_configuration=FrozenConfiguration.from_value({
            classical.implementation_id: classical_sections["preprocessing"],
            learned.implementation_id: learned_sections["preprocessing"],
        }),
        proposal_configuration=FrozenConfiguration.from_value({
            classical.implementation_id: classical_sections["proposal"],
            learned.implementation_id: learned_sections["proposal"],
        }),
        feature_scoring_configuration=FrozenConfiguration.from_value({
            classical.implementation_id: classical_sections["feature_and_scoring"],
            learned.implementation_id: learned_sections["feature_and_scoring"],
        }),
        maximum_proposal_count=8,
        random_seeds=(("structvision", classical.random_seed), ("patchcore", learned.random_seed)),
        deterministic_mode=True,
        evaluation_policy_id=policy.policy_id,
        evaluation_policy_version=policy.policy_version,
        evaluation_policy_hash=policy.configuration_hash,
        matching_thresholds=policy.threshold_analyses,
        metric_definitions_hash=policy.metric_definitions_hash,
        allowed_fitting_splits=("train", "validation"),
        forbidden_test_access=True,
        git_commit=commit,
        git_tree_state=tree_state,
        uncommitted_diff_hash=diff_hash,
        python_version=platform.python_version(),
        dependency_snapshot=FrozenConfiguration.from_value({
            "normal_feature_exact": EXACT_RUNTIME_VERSIONS,
            "classical_package_version": "1.0.0",
        }),
        dependency_lock_hash=dependency_lock_hash,
        operating_system_metadata=FrozenConfiguration.from_value({
            "platform": platform.platform(), "machine": platform.machine(),
        }),
        hardware_metadata=FrozenConfiguration.from_value({
            "reference_device": "cpu", "thread_count": learned.torch_num_threads,
            "mps_scientific_reference": False,
        }),
        opencv_version=cv2.__version__,
        opencv_backend="opencv-python-headless macOS arm64",
        creation_timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _learned_proposal_set(result: NormalFeatureAnalysisResult, ranking_definition: str) -> ProposalSet:
    return ProposalSet(
        result.implementation_id,
        tuple(ProposalRecord(
            item.proposal_id, EncodedMask.from_array(item.mask), item.component_anomaly_score, item.rank,
        ) for item in result.proposals),
        True,
        ranking_definition,
    )


def _row(
    *,
    specification: ExperimentSpecificationV2,
    sample: ExperimentSample,
    method: MethodSpecification,
    executed_hash: str,
    attempt_id: str,
    result_details: dict[str, object],
    proposal_set: ProposalSet,
    evaluation: ImageEvaluationV2,
) -> ResultRowV2:
    policy = default_evaluation_policy()
    selected = next(item for item in specification.selected_images if item.image_id == sample.image_id)
    matching = evaluation.result_at(policy.primary_match_threshold)
    proposal_payload = canonical_json(result_details)
    return ResultRowV2(
        result_id=sha256_json({"attempt": attempt_id, "image": sample.image_id, "method": method.method_id}),
        experiment_specification_hash=specification.specification_hash,
        executed_configuration_hash=executed_hash,
        method_implementation_id=method.method_id,
        method_implementation_version=method.implementation_version,
        evaluation_policy_id=policy.policy_id,
        evaluation_policy_version=policy.policy_version,
        evaluation_policy_hash=policy.configuration_hash,
        image_id=sample.image_id,
        image_content_hash=selected.image_sha256,
        ground_truth_content_hash=selected.ground_truth_sha256,
        proposal_output_artifact_hash=sha256_json({
            "image_id": sample.image_id,
            "method_id": method.method_id,
            "proposal_set": proposal_set.to_dict(),
        }),
        proposal_output_details_json=proposal_payload,
        matching_policy_hash=policy.matching_policy_hash,
        result_schema_version=RESULT_SCHEMA_VERSION,
        execution_attempt_id=attempt_id,
        deterministic_mode=True,
        recorded_timestamp=datetime.now(timezone.utc).isoformat(),
        completion_status="completed",
        matching_details_json=canonical_json(matching.to_dict()),
        metrics_json=canonical_json(asdict(evaluation)),
    )


class DevelopmentExperimentExecutorV2:
    """Execute exactly the protected two-method development matrix."""

    def __init__(self, learned_detector: NormalFeatureAnomalyDetector):
        self.learned_detector = learned_detector

    def execute(
        self,
        specification: ExperimentSpecificationV2,
        samples: Iterable[ExperimentSample],
        *,
        model_artifact: NormalFeatureModelArtifact,
        calibration_artifact: CalibrationArtifact,
        sink: ResultSink | None = None,
        execution_attempt_id: str | None = None,
    ) -> DevelopmentExecutionReport:
        if specification.experiment_id != DEVELOPMENT_EXPERIMENT_ID or specification.experiment_version != 1:
            raise SpecificationMismatchError("Unexpected development experiment identity")
        if not specification.forbidden_test_access or specification.allowed_fitting_splits != ("train", "validation"):
            raise SpecificationMismatchError("Protected fitting/test-access policy differs")
        expected_methods = (
            DetectorConfig().implementation_id,
            self.learned_detector.config.implementation_id,
        )
        if tuple(item.method_id for item in specification.methods) != expected_methods:
            raise SpecificationMismatchError("Development method matrix differs from the protected two-method plan")
        prepared = tuple(samples)
        if tuple(item.image_id for item in prepared) != tuple(item.image_id for item in specification.selected_images):
            raise ProvenanceMismatchError("Development samples differ from the protected manifest order")
        selected = {item.image_id: item for item in specification.selected_images}
        for sample in prepared:
            if sample.metadata is None or sample.metadata.get("development_role") != "calibration_validation":
                raise ProvenanceMismatchError("Only calibration_validation samples may be executed")
            if content_hash(sample.image) != selected[sample.image_id].image_sha256:
                raise ProvenanceMismatchError(f"Image hash mismatch: {sample.image_id}")
            if content_hash(sample.ground_truth) != selected[sample.image_id].ground_truth_sha256:
                raise ProvenanceMismatchError(f"Ground-truth hash mismatch: {sample.image_id}")
        classical_method, learned_method = specification.methods
        classical_config = DetectorConfig.from_dict(classical_method.method_configuration.value)
        learned_payload = learned_method.method_configuration.value
        if (
            sha256_json(learned_payload["detector_config"]) != self.learned_detector.config.configuration_hash
            or learned_payload["model_artifact_hash"] != model_artifact.artifact_hash
            or learned_payload["calibration_artifact_hash"] != calibration_artifact.artifact_hash
            or learned_payload["primary_operating_point_id"] != PRIMARY_OPERATING_POINT_ID
        ):
            raise SpecificationMismatchError("Executed learned artifacts/configuration differ from the specification")
        executed_hashes = {
            method.method_id: specification.verify_executed_configuration(
                method.method_id, specification.expected_executable_configuration(method.method_id),
            )
            for method in specification.methods
        }
        attempt_id = execution_attempt_id or str(uuid4())
        started = datetime.now(timezone.utc).isoformat()
        rows: list[ResultRowV2] = []
        analyses: list[DevelopmentExecutionAnalysis] = []
        auxiliary: dict[str, list[ImageEvaluationV2]] = {
            item.operating_point_id: [] for item in calibration_artifact.operating_points
        }
        policy = default_evaluation_policy()
        for sample in prepared:
            truth = None
            pair_started = time.perf_counter()
            classical_result = StructuralAnomalyDetector(classical_config).analyse(
                sample.image, image_id=sample.image_id, metadata=sample.metadata,
            )
            if not classical_result.provenance.protected_hashes_verified:
                raise ProvenanceMismatchError("Frozen classical source hashes differ")
            truth = _ground_truth_record(sample, classical_result.image_shape[:2])
            classical_set = _proposal_set(classical_result, classical_method.method_id, classical_method.ranking_definition)
            classical_evaluation = evaluate_image(
                classical_set, truth, category=sample.category,
                acquisition_group_id=sample.acquisition_group_id,
                processing_time_seconds=time.perf_counter() - pair_started,
                hardware_context=specification.hardware_metadata.canonical_payload,
                cache_state="controlled_single_process_order",
                policy=policy,
            )
            rows.append(_row(
                specification=specification, sample=sample, method=classical_method,
                executed_hash=executed_hashes[classical_method.method_id], attempt_id=attempt_id,
                result_details=classical_result.to_dict(include_heatmap=False),
                proposal_set=classical_set, evaluation=classical_evaluation,
            ))
            analyses.append(DevelopmentExecutionAnalysis(
                sample.image_id, classical_method.method_id, classical_result, classical_evaluation,
            ))
            learned_result = self.learned_detector.analyse(
                sample.image,
                model_artifact=model_artifact,
                calibration_artifact=calibration_artifact,
                operating_point_id=PRIMARY_OPERATING_POINT_ID,
                image_id=sample.image_id,
            )
            learned_set = _learned_proposal_set(learned_result, str(learned_method.ranking_definition))
            learned_evaluation = evaluate_image(
                learned_set, truth, category=sample.category,
                acquisition_group_id=sample.acquisition_group_id,
                processing_time_seconds=learned_result.inference_seconds,
                hardware_context=specification.hardware_metadata.canonical_payload,
                cache_state="controlled_single_process_order",
                policy=policy,
            )
            rows.append(_row(
                specification=specification, sample=sample, method=learned_method,
                executed_hash=executed_hashes[learned_method.method_id], attempt_id=attempt_id,
                result_details=learned_result.to_dict(include_map=False, include_masks=False),
                proposal_set=learned_set, evaluation=learned_evaluation,
            ))
            analyses.append(DevelopmentExecutionAnalysis(
                sample.image_id, learned_method.method_id, learned_result, learned_evaluation,
            ))
            for operating_point in calibration_artifact.operating_points:
                proposals = extract_proposals(
                    learned_result.anomaly_map,
                    threshold=operating_point.threshold,
                    operating_point_id=operating_point.operating_point_id,
                    config=self.learned_detector.config.proposal,
                )
                proposal_set = ProposalSet(
                    learned_method.method_id,
                    tuple(ProposalRecord(
                        item.proposal_id, EncodedMask.from_array(item.mask),
                        item.component_anomaly_score, item.rank,
                    ) for item in proposals),
                    True,
                    str(learned_method.ranking_definition),
                )
                auxiliary[operating_point.operating_point_id].append(evaluate_image(
                    proposal_set, truth, category=sample.category,
                    acquisition_group_id=sample.acquisition_group_id, policy=policy,
                ))
        summary = ExecutionAttemptSummary(
            attempt_id,
            specification.specification_hash,
            "completed",
            specification.expected_pair_count,
            len(rows),
            len(rows),
            0,
            0,
            len(rows),
            started,
            datetime.now(timezone.utc).isoformat(),
        )
        report = DevelopmentExecutionReport(
            specification, summary, tuple(rows), tuple(analyses),
            tuple((identity, tuple(values)) for identity, values in auxiliary.items()),
        )
        if sink is not None:
            sink.write(report)
        return report
