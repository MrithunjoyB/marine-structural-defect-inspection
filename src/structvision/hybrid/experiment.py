"""One-shot three-method hybrid development holdout experiment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
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
from scientific_contract.result_store import ExecutionAttemptSummary, ResultRowV2
from scientific_contract.specification import (
    SPECIFICATION_SCHEMA_VERSION,
    ExperimentSpecificationV2,
    FrozenConfiguration,
    MethodSpecification,
    SelectedImageIdentity,
)
from structvision.api import StructuralAnomalyDetector
from structvision.configuration import DetectorConfig
from structvision.errors import ProvenanceMismatchError, SpecificationMismatchError
from structvision.executor import ExperimentSample, _ground_truth_record, _proposal_set
from structvision.inputs import content_hash
from structvision.learned_executor import _learned_proposal_set, _row
from structvision.normal_feature.calibration import CalibrationArtifact
from structvision.normal_feature.configuration import NormalFeatureConfig
from structvision.normal_feature.model_artifact import NormalFeatureModelArtifact
from structvision.normal_feature.patchcore import EXACT_RUNTIME_VERSIONS, NormalFeatureAnomalyDetector, _git_metadata
from structvision.sinks import ResultSink

from .artifact import DECLARED_BUDGETS, HYBRID_IMPLEMENTATION_ID, PRIMARY_BUDGET, HybridFusionArtifact
from .detector import HybridAnalysisResult, ProposalGuidedHybridDetector
from .errors import HybridExperimentError
from .protocol import DETERMINISTIC_SEED, HybridDevelopmentManifest


HYBRID_EXPERIMENT_ID = "SYN-PROPOSAL-HYBRID-DEV-001"
HYBRID_EXPERIMENT_VERSION = 1
BASELINE_PATCHCORE_OPERATING_POINT_ID = "fp-budget-0.50"


@dataclass(frozen=True)
class HybridExecutionAnalysis:
    image_id: str
    method_id: str
    result: object
    evaluation: ImageEvaluationV2


@dataclass(frozen=True)
class HybridExecutionReport:
    specification: ExperimentSpecificationV2
    summary: ExecutionAttemptSummary
    rows: tuple[ResultRowV2, ...]
    analyses: tuple[HybridExecutionAnalysis, ...]
    hybrid_budget_evaluations: tuple[tuple[float, tuple[ImageEvaluationV2, ...]], ...]


class HoldoutAttemptLedger:
    """Append-only primary-attempt gate outside every historical result store."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _events(self) -> tuple[dict[str, object], ...]:
        if not self.path.exists():
            return ()
        events = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            events.append(json.loads(line))
        return tuple(events)

    def start(self, *, attempt_id: str, fusion_artifact_hash: str, specification_hash: str) -> None:
        if self._events():
            raise HybridExperimentError("The primary hybrid holdout has already been attempted")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "event": "primary_holdout_started",
            "attempt_id": attempt_id,
            "fusion_artifact_hash": fusion_artifact_hash,
            "specification_hash": specification_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self.path.open("x", encoding="utf-8") as stream:
            stream.write(canonical_json(event) + "\n")

    def finish(self, *, attempt_id: str, status: str, detail: str = "") -> None:
        events = self._events()
        if len(events) != 1 or events[0].get("attempt_id") != attempt_id:
            raise HybridExperimentError("Holdout-attempt ledger identity differs")
        if status not in {"completed", "failed_software"}:
            raise HybridExperimentError("Unknown holdout completion status")
        event = {
            "event": "primary_holdout_finished",
            "attempt_id": attempt_id,
            "status": status,
            "detail": detail,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(event) + "\n")


def holdout_samples(
    manifest: HybridDevelopmentManifest,
    repository_root: Path,
) -> tuple[ExperimentSample, ...]:
    root = Path(repository_root).resolve()
    samples = []
    for identity in manifest.development_holdout:
        if identity.role != "hybrid_development_holdout" or identity.split_role != "validation":
            raise HybridExperimentError("Holdout loader received a non-holdout identity")
        image = root / identity.image_path
        if content_hash(image) != identity.image_sha256:
            raise ProvenanceMismatchError(f"Holdout image identity mismatch: {identity.image_id}")
        if identity.ground_truth_kind == "implicit_verified_zero_mask":
            decoded = cv2.imdecode(np.frombuffer(image.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise ProvenanceMismatchError(f"Could not decode selected holdout image: {identity.image_id}")
            truth: object = np.zeros(decoded.shape[:2], dtype=np.uint8)
        else:
            truth = root / str(identity.ground_truth_path)
        if content_hash(truth) != identity.ground_truth_sha256:
            raise ProvenanceMismatchError(f"Holdout truth identity mismatch: {identity.image_id}")
        samples.append(ExperimentSample(
            image_id=identity.image_id,
            image=image,
            ground_truth=truth,
            ground_truth_status=identity.image_outcome,
            category=identity.category,
            acquisition_group_id=identity.acquisition_group_id,
            metadata={
                "hybrid_role": identity.role,
                "source_group_id": identity.source_group_id,
                "template_group_id": identity.template_group_id,
            },
        ))
    return tuple(samples)


def create_hybrid_experiment_specification(
    *,
    manifest: HybridDevelopmentManifest,
    baseline_model_artifact: NormalFeatureModelArtifact,
    baseline_calibration_artifact: CalibrationArtifact,
    hybrid_model_artifact: NormalFeatureModelArtifact,
    fusion_artifact: HybridFusionArtifact,
    dependency_lock_hash: str,
    repository_root: Path,
    classical_config: DetectorConfig | None = None,
    learned_config: NormalFeatureConfig | None = None,
) -> ExperimentSpecificationV2:
    classical = classical_config or DetectorConfig()
    learned = learned_config or NormalFeatureConfig()
    if fusion_artifact.hybrid_protocol_hash != manifest.manifest_hash:
        raise SpecificationMismatchError("Fusion artifact and holdout manifest differ")
    if fusion_artifact.normal_feature_model_artifact_hash != hybrid_model_artifact.artifact_hash:
        raise SpecificationMismatchError("Hybrid model and fusion artifact differ")
    if baseline_calibration_artifact.model_artifact_hash != baseline_model_artifact.artifact_hash:
        raise SpecificationMismatchError("Historical PatchCore model/calibration artifacts differ")
    policy = default_evaluation_policy()
    methods = (
        MethodSpecification(
            classical.implementation_id, classical.implementation_version,
            FrozenConfiguration.from_value(classical.to_dict()), True,
            "priority_score descending; frozen proposal order; proposal_id tie-break",
        ),
        MethodSpecification(
            learned.implementation_id, learned.implementation_version,
            FrozenConfiguration.from_value({
                "detector_config": learned.to_dict(),
                "model_artifact_hash": baseline_model_artifact.artifact_hash,
                "calibration_artifact_hash": baseline_calibration_artifact.artifact_hash,
                "primary_operating_point_id": BASELINE_PATCHCORE_OPERATING_POINT_ID,
            }), True,
            "component anomaly distance descending; area descending; bbox lexical tie-break",
        ),
        MethodSpecification(
            fusion_artifact.implementation_identity, fusion_artifact.implementation_version,
            FrozenConfiguration.from_value({
                "classical_configuration_hash": classical.configuration_hash,
                "normal_feature_configuration_hash": learned.configuration_hash,
                "normal_feature_model_artifact_hash": hybrid_model_artifact.artifact_hash,
                "fusion_artifact_hash": fusion_artifact.artifact_hash,
                "primary_false_proposal_budget": PRIMARY_BUDGET,
            }), True,
            "hybrid linear rank score descending; classical proposal_id lexical tie-break",
        ),
    )
    classical_sections = classical.specification_sections()
    learned_sections = learned.specification_sections()
    commit, tree_state, diff_hash = _git_metadata(Path(repository_root))
    return ExperimentSpecificationV2(
        schema_version=SPECIFICATION_SCHEMA_VERSION,
        experiment_id=HYBRID_EXPERIMENT_ID,
        experiment_version=HYBRID_EXPERIMENT_VERSION,
        dataset_id=manifest.source_dataset_id,
        dataset_version=manifest.source_dataset_version,
        dataset_manifest_hash=manifest.source_registry_sha256,
        split_manifest_hash=manifest.manifest_hash,
        split_lock_hash=manifest.manifest_hash,
        selected_images=tuple(SelectedImageIdentity(
            item.image_id, item.image_sha256, item.ground_truth_sha256,
        ) for item in manifest.development_holdout),
        methods=methods,
        preprocessing_configuration=FrozenConfiguration.from_value({
            classical.implementation_id: classical_sections["preprocessing"],
            learned.implementation_id: learned_sections["preprocessing"],
            HYBRID_IMPLEMENTATION_ID: {
                "classical": classical_sections["preprocessing"],
                "normal_feature": learned_sections["preprocessing"],
            },
        }),
        proposal_configuration=FrozenConfiguration.from_value({
            classical.implementation_id: classical_sections["proposal"],
            learned.implementation_id: learned_sections["proposal"],
            HYBRID_IMPLEMENTATION_ID: {
                "source": "complete frozen classical proposal set",
                "selection_artifact": fusion_artifact.artifact_hash,
            },
        }),
        feature_scoring_configuration=FrozenConfiguration.from_value({
            classical.implementation_id: classical_sections["feature_and_scoring"],
            learned.implementation_id: learned_sections["feature_and_scoring"],
            HYBRID_IMPLEMENTATION_ID: {
                "fusion_artifact": fusion_artifact.to_dict(),
                "score_semantics": "explainable_linear_rank_score_not_probability",
            },
        }),
        maximum_proposal_count=8,
        random_seeds=(("structvision", classical.random_seed), ("patchcore", learned.random_seed), ("hybrid", DETERMINISTIC_SEED)),
        deterministic_mode=True,
        evaluation_policy_id=policy.policy_id,
        evaluation_policy_version=policy.policy_version,
        evaluation_policy_hash=policy.configuration_hash,
        matching_thresholds=policy.threshold_analyses,
        metric_definitions_hash=policy.metric_definitions_hash,
        allowed_fitting_splits=("train",),
        forbidden_test_access=True,
        git_commit=commit,
        git_tree_state=tree_state,
        uncommitted_diff_hash=diff_hash,
        python_version=platform.python_version(),
        dependency_snapshot=FrozenConfiguration.from_value({
            "normal_feature_exact": EXACT_RUNTIME_VERSIONS,
            "classical_package_version": "1.0.0",
            "hybrid_implementation_version": fusion_artifact.implementation_version,
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


def _hybrid_proposal_set(result: HybridAnalysisResult, ranking_definition: str) -> ProposalSet:
    return ProposalSet(
        result.implementation_id,
        tuple(ProposalRecord(
            item.proposal_id, EncodedMask.from_array(item.final_mask), item.hybrid_score, item.rank,
        ) for item in result.proposals),
        True,
        ranking_definition,
    )


class HybridDevelopmentExperimentExecutorV2:
    def __init__(
        self,
        *,
        classical_detector: StructuralAnomalyDetector,
        baseline_normal_detector: NormalFeatureAnomalyDetector,
        hybrid_detector: ProposalGuidedHybridDetector,
    ) -> None:
        self.classical_detector = classical_detector
        self.baseline_normal_detector = baseline_normal_detector
        self.hybrid_detector = hybrid_detector

    def execute(
        self,
        specification: ExperimentSpecificationV2,
        samples: Iterable[ExperimentSample],
        *,
        baseline_model_artifact: NormalFeatureModelArtifact,
        baseline_calibration_artifact: CalibrationArtifact,
        sink: ResultSink,
        execution_attempt_id: str,
    ) -> HybridExecutionReport:
        if sink is None:
            raise HybridExperimentError("Hybrid results require a new explicit append-only sink")
        if (specification.experiment_id, specification.experiment_version) != (HYBRID_EXPERIMENT_ID, 1):
            raise SpecificationMismatchError("Unexpected hybrid experiment identity")
        if specification.allowed_fitting_splits != ("train",) or not specification.forbidden_test_access:
            raise SpecificationMismatchError("Hybrid test-access policy differs")
        expected_methods = (
            DetectorConfig().implementation_id,
            NormalFeatureConfig().implementation_id,
            HYBRID_IMPLEMENTATION_ID,
        )
        if tuple(item.method_id for item in specification.methods) != expected_methods:
            raise SpecificationMismatchError("Hybrid three-method matrix differs")
        prepared = tuple(samples)
        if tuple(item.image_id for item in prepared) != tuple(item.image_id for item in specification.selected_images):
            raise ProvenanceMismatchError("Holdout samples differ from the frozen manifest order")
        selected = {item.image_id: item for item in specification.selected_images}
        for sample in prepared:
            if sample.metadata is None or sample.metadata.get("hybrid_role") != "hybrid_development_holdout":
                raise ProvenanceMismatchError("Only protected holdout samples may enter the executor")
            if content_hash(sample.image) != selected[sample.image_id].image_sha256:
                raise ProvenanceMismatchError(f"Image hash mismatch: {sample.image_id}")
            if content_hash(sample.ground_truth) != selected[sample.image_id].ground_truth_sha256:
                raise ProvenanceMismatchError(f"Truth hash mismatch: {sample.image_id}")
        classical_method, patchcore_method, hybrid_method = specification.methods
        executed_hashes = {
            method.method_id: specification.verify_executed_configuration(
                method.method_id, specification.expected_executable_configuration(method.method_id),
            ) for method in specification.methods
        }
        rows: list[ResultRowV2] = []
        analyses: list[HybridExecutionAnalysis] = []
        auxiliary: dict[float, list[ImageEvaluationV2]] = {budget: [] for budget in DECLARED_BUDGETS}
        policy = default_evaluation_policy()
        started = datetime.now(timezone.utc).isoformat()
        for sample in prepared:
            classical_started = time.perf_counter()
            classical = self.classical_detector.analyse(
                sample.image, image_id=sample.image_id, metadata=sample.metadata,
            )
            if not classical.provenance.protected_hashes_verified:
                raise ProvenanceMismatchError("Frozen classical source hashes differ")
            truth = _ground_truth_record(sample, classical.image_shape[:2])
            classical_set = _proposal_set(classical, classical_method.method_id, str(classical_method.ranking_definition))
            classical_evaluation = evaluate_image(
                classical_set, truth, category=sample.category,
                acquisition_group_id=sample.acquisition_group_id,
                processing_time_seconds=time.perf_counter() - classical_started,
                hardware_context=specification.hardware_metadata.canonical_payload,
                cache_state="controlled_single_process_order", policy=policy,
            )
            rows.append(_row(
                specification=specification, sample=sample, method=classical_method,
                executed_hash=executed_hashes[classical_method.method_id], attempt_id=execution_attempt_id,
                result_details=classical.to_dict(include_heatmap=False), proposal_set=classical_set,
                evaluation=classical_evaluation,
            ))
            analyses.append(HybridExecutionAnalysis(sample.image_id, classical_method.method_id, classical, classical_evaluation))
            patchcore = self.baseline_normal_detector.analyse(
                sample.image, model_artifact=baseline_model_artifact,
                calibration_artifact=baseline_calibration_artifact,
                operating_point_id=BASELINE_PATCHCORE_OPERATING_POINT_ID,
                image_id=sample.image_id,
            )
            patchcore_set = _learned_proposal_set(patchcore, str(patchcore_method.ranking_definition))
            patchcore_evaluation = evaluate_image(
                patchcore_set, truth, category=sample.category,
                acquisition_group_id=sample.acquisition_group_id,
                processing_time_seconds=patchcore.inference_seconds,
                hardware_context=specification.hardware_metadata.canonical_payload,
                cache_state="controlled_single_process_order", policy=policy,
            )
            rows.append(_row(
                specification=specification, sample=sample, method=patchcore_method,
                executed_hash=executed_hashes[patchcore_method.method_id], attempt_id=execution_attempt_id,
                result_details=patchcore.to_dict(include_map=False, include_masks=False),
                proposal_set=patchcore_set, evaluation=patchcore_evaluation,
            ))
            analyses.append(HybridExecutionAnalysis(sample.image_id, patchcore_method.method_id, patchcore, patchcore_evaluation))
            hybrid = self.hybrid_detector.analyse(
                sample.image, image_id=sample.image_id,
                false_proposal_budget=PRIMARY_BUDGET, metadata=sample.metadata,
            )
            hybrid_set = _hybrid_proposal_set(hybrid, str(hybrid_method.ranking_definition))
            hybrid_evaluation = evaluate_image(
                hybrid_set, truth, category=sample.category,
                acquisition_group_id=sample.acquisition_group_id,
                processing_time_seconds=dict(hybrid.timing_breakdown_seconds)["total_seconds"],
                hardware_context=specification.hardware_metadata.canonical_payload,
                cache_state="controlled_single_process_order", policy=policy,
            )
            rows.append(_row(
                specification=specification, sample=sample, method=hybrid_method,
                executed_hash=executed_hashes[hybrid_method.method_id], attempt_id=execution_attempt_id,
                result_details=hybrid.to_dict(include_masks=False), proposal_set=hybrid_set,
                evaluation=hybrid_evaluation,
            ))
            analyses.append(HybridExecutionAnalysis(sample.image_id, hybrid_method.method_id, hybrid, hybrid_evaluation))
            for budget in DECLARED_BUDGETS:
                budget_result = hybrid if budget == PRIMARY_BUDGET else self.hybrid_detector.reselect(
                    hybrid, false_proposal_budget=budget,
                )
                budget_set = _hybrid_proposal_set(budget_result, str(hybrid_method.ranking_definition))
                auxiliary[budget].append(evaluate_image(
                    budget_set, truth, category=sample.category,
                    acquisition_group_id=sample.acquisition_group_id, policy=policy,
                ))
        summary = ExecutionAttemptSummary(
            execution_attempt_id, specification.specification_hash, "completed",
            specification.expected_pair_count, len(rows), len(rows), 0, 0, len(rows),
            started, datetime.now(timezone.utc).isoformat(),
        )
        report = HybridExecutionReport(
            specification, summary, tuple(rows), tuple(analyses),
            tuple((budget, tuple(auxiliary[budget])) for budget in DECLARED_BUDGETS),
        )
        sink.write(report)
        return report
