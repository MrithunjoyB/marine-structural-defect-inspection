"""Protected proposal-guided hybrid development workflow.

Fusion fitting uses train-role capability objects only.  The validation holdout
is loaded only after the fusion artifact and experiment specification are
frozen and the one-shot attempt ledger has been started.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import sqlite3
import subprocess

from scientific_contract.hashing import canonical_json
from scientific_contract.metrics import AggregateMetricsV2, aggregate_metrics
from structvision.api import StructuralAnomalyDetector
from structvision.configuration import DetectorConfig
from structvision.hybrid.artifact import DirectoryHybridFusionArtifactSink, PRIMARY_BUDGET
from structvision.hybrid.detector import ProposalGuidedHybridDetector
from structvision.hybrid.experiment import (
    HYBRID_EXPERIMENT_ID,
    HYBRID_EXPERIMENT_VERSION,
    HoldoutAttemptLedger,
    HybridDevelopmentExperimentExecutorV2,
    create_hybrid_experiment_specification,
    holdout_samples,
)
from structvision.hybrid.protocol import (
    create_hybrid_development_manifest,
    fusion_fit_view,
    hybrid_normal_fit_samples,
    load_hybrid_manifest,
    write_hybrid_manifest,
)
from structvision.hybrid.selection import fit_hybrid_fusion
from structvision.normal_feature.calibration import load_calibration_artifact
from structvision.normal_feature.configuration import WEIGHT_SHA256
from structvision.normal_feature.model_artifact import DirectoryModelArtifactSink, load_model_artifact
from structvision.normal_feature.patchcore import NormalFeatureAnomalyDetector
from structvision.sinks import V2SQLiteResultSink


BASELINE_COMMIT = "6688d2f43b8a514f435d8df87c87861b478756de"
BASELINE_MODEL_ID = "4542d063a64eb22d795f7a7faabb3cad592f69bd1fe753abdda0e5428f4961e7"
BASELINE_CALIBRATION_ID = "a5a434281d7e16ffb5c0a9af65f5b27d100cd447f1d024b7cbc5199805a21a6f"
PROTECTED_CLASSICAL_FILES = (
    "preprocess.py", "feature_extraction.py", "region_proposal.py", "scoring.py",
    "severity.py", "explain.py", "report.py",
)
HISTORICAL_DATABASES = (
    "outputs/research_evaluation.sqlite3",
    "outputs/registered_experiment_results.sqlite3",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _automatic_rows(path: Path) -> int:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return int(connection.execute("SELECT count(*) FROM automatic_results").fetchone()[0])
    finally:
        connection.close()


def protected_snapshot(root: Path) -> dict[str, object]:
    baseline_root = root / "outputs/normal-feature-development/SYN-NORMAL-FEATURE-DEV-001-v1"
    files = {
        "historical_database_hashes": {item: _sha(root / item) for item in HISTORICAL_DATABASES},
        "historical_automatic_rows": _automatic_rows(root / "outputs/registered_experiment_results.sqlite3"),
        "registry_database_hash": _sha(root / "research_data/registry/datasets.sqlite"),
        "registry_manifest_hash": _sha(root / "research_data/registry/dataset_manifest.json"),
        "classical_source_hashes": {item: _sha(root / item) for item in PROTECTED_CLASSICAL_FILES},
        "patchcore_model_metadata_file_hash": _sha(baseline_root / f"model-artifacts/{BASELINE_MODEL_ID}.json"),
        "patchcore_model_bank_file_hash": _sha(baseline_root / f"model-artifacts/{BASELINE_MODEL_ID}.npz"),
        "patchcore_calibration_file_hash": _sha(baseline_root / f"calibration-artifacts/{BASELINE_CALIBRATION_ID}.json"),
        "normal_feature_manifest_file_hash": _sha(root / "development_data/normal-feature-development-manifest-v1.json"),
        "environment_lock_hash": _sha(root / "requirements/pylock.normal-feature-macos-arm64.toml"),
    }
    return files


def _git_pre_state(root: Path) -> dict[str, object]:
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    return {"branch": branch, "baseline_commit": BASELINE_COMMIT, "head_at_execution": commit}


def _metric_payload(metric: AggregateMetricsV2) -> dict[str, object]:
    return metric.to_dict()


def _holdout_preservation_failures(hybrid: AggregateMetricsV2, classical: AggregateMetricsV2) -> tuple[str, ...]:
    failures = []
    if float(hybrid.micro_component_sensitivity) < float(classical.micro_component_sensitivity) - 0.02 - 1e-12:
        failures.append("overall_micro_sensitivity_decrease_exceeds_0.02")
    hybrid_categories = dict(hybrid.category_component_sensitivity)
    classical_categories = dict(classical.category_component_sensitivity)
    for category in ("thin_crack", "pitting_cluster", "weld_disturbance"):
        if float(hybrid_categories[category]) + 1e-12 < float(classical_categories[category]):
            failures.append(f"{category}_sensitivity_decreased")
    if float(hybrid.image_level_detection_sensitivity) + 1e-12 < float(classical.image_level_detection_sensitivity):
        failures.append("image_level_sensitivity_decreased")
    if float(hybrid.assigned_pair_iou_mean or 0.0) + 1e-12 < float(classical.assigned_pair_iou_mean or 0.0) - 0.02:
        failures.append("mean_assigned_pair_iou_decrease_exceeds_0.02")
    return tuple(failures)


def _paired_effects(report) -> dict[str, object]:
    by_key = {(item.image_id, item.method_id): item.evaluation for item in report.analyses}
    classical_id = "structvision-classical-baseline-v1-frozen"
    hybrid_id = "structvision-proposal-guided-hybrid-v1-dev"
    rows = []
    for image_id in sorted({item.image_id for item in report.analyses}):
        classical = by_key[(image_id, classical_id)]
        hybrid = by_key[(image_id, hybrid_id)]
        rows.append({
            "image_id": image_id,
            "category": classical.category,
            "ground_truth_status": classical.ground_truth_status,
            "proposal_count_difference_hybrid_minus_classical": hybrid.proposal_count - classical.proposal_count,
            "matched_truth_difference_hybrid_minus_classical": (
                hybrid.result_at(0.25).matched_truth_count - classical.result_at(0.25).matched_truth_count
            ),
        })
    return {
        "image_count": len(rows),
        "proposal_count_difference_counts": dict(sorted(Counter(
            item["proposal_count_difference_hybrid_minus_classical"] for item in rows
        ).items())),
        "matched_truth_difference_counts": dict(sorted(Counter(
            item["matched_truth_difference_hybrid_minus_classical"] for item in rows
        ).items())),
        "per_image": rows,
    }


def prepare_manifest(root: Path, destination: Path) -> dict[str, object]:
    manifest = create_hybrid_development_manifest(
        repository_root=root,
        registry_database=root / "research_data/registry/datasets.sqlite",
        historical_result_database=root / "outputs/registered_experiment_results.sqlite3",
    )
    write_hybrid_manifest(manifest, destination)
    return manifest.to_dict()


def run(*, root: Path, weight_file: Path, output_directory: Path, manifest_path: Path) -> dict[str, object]:
    root = root.resolve()
    output = output_directory.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Hybrid development output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    before = protected_snapshot(root)
    manifest = load_hybrid_manifest(manifest_path)
    regenerated = create_hybrid_development_manifest(
        repository_root=root,
        registry_database=root / "research_data/registry/datasets.sqlite",
        historical_result_database=root / "outputs/registered_experiment_results.sqlite3",
    )
    if regenerated != manifest:
        raise RuntimeError("Committed hybrid manifest differs from deterministic regeneration")
    lock_hash = _sha(root / "requirements/pylock.normal-feature-macos-arm64.toml")
    if _sha(weight_file) != WEIGHT_SHA256:
        raise RuntimeError("Official PatchCore backbone weight differs")
    normal_detector = NormalFeatureAnomalyDetector(
        weight_file=weight_file, environment_lock_hash=lock_hash, repository_root=root,
    )
    hybrid_model = normal_detector.fit_normal(
        hybrid_normal_fit_samples(manifest, root),
        normal_fit_manifest_hash=manifest.manifest_hash,
        artifact_sink=DirectoryModelArtifactSink(output / "hybrid-model-artifacts"),
    )
    classical_detector = StructuralAnomalyDetector(DetectorConfig())
    fusion_fit = fit_hybrid_fusion(
        fusion_fit_view(manifest), repository_root=root,
        classical_detector=classical_detector, normal_detector=normal_detector,
        model_artifact=hybrid_model, environment_lock_hash=lock_hash,
        artifact_sink=DirectoryHybridFusionArtifactSink(output / "fusion-artifacts"),
    )
    fusion_artifact = fusion_fit.artifact
    fusion_summary = {
        "classical_baseline": _metric_payload(fusion_fit.baseline_metrics),
        "selection_status": fusion_artifact.selection_status,
        "selected_configuration_id": fusion_artifact.selected_configuration_id,
        "selected_operating_points": [item.to_dict() for item in fusion_artifact.selected_operating_points],
        "evaluated_configurations": [item.to_dict() for item in fusion_artifact.evaluated_configurations],
    }
    if fusion_artifact.selection_status != "selected":
        summary = {
            "classification": manifest.evidence_classification,
            "experiment_id": HYBRID_EXPERIMENT_ID,
            "experiment_version": HYBRID_EXPERIMENT_VERSION,
            "manifest_hash": manifest.manifest_hash,
            "normal_model_artifact_hash": hybrid_model.artifact_hash,
            "fusion_artifact_hash": fusion_artifact.artifact_hash,
            "fusion_fit": fusion_summary,
            "holdout_executed": False,
            "development_decision": "development candidate rejected under the predeclared protocol",
            "rejection_reason": "no fusion-fit configuration satisfied all preservation constraints and the primary clean-FP budget",
            "protected_before": before,
            "protected_after": protected_snapshot(root),
        }
        (output / "development-summary.json").write_text(canonical_json(summary) + "\n", encoding="utf-8")
        return summary
    baseline_root = root / "outputs/normal-feature-development/SYN-NORMAL-FEATURE-DEV-001-v1"
    baseline_model = load_model_artifact(baseline_root / f"model-artifacts/{BASELINE_MODEL_ID}.json")
    baseline_calibration = load_calibration_artifact(baseline_root / f"calibration-artifacts/{BASELINE_CALIBRATION_ID}.json")
    hybrid_detector = ProposalGuidedHybridDetector(
        classical_detector=classical_detector,
        normal_feature_detector=normal_detector,
        normal_feature_model_artifact=hybrid_model,
        fusion_artifact=fusion_artifact,
    )
    specification = create_hybrid_experiment_specification(
        manifest=manifest,
        baseline_model_artifact=baseline_model,
        baseline_calibration_artifact=baseline_calibration,
        hybrid_model_artifact=hybrid_model,
        fusion_artifact=fusion_artifact,
        dependency_lock_hash=lock_hash,
        repository_root=root,
    )
    (output / "experiment-specification-v2.json").write_text(specification.to_json() + "\n", encoding="utf-8")
    attempt_id = "SYN-PROPOSAL-HYBRID-DEV-001-v1-primary-cpu"
    ledger = HoldoutAttemptLedger(output / "holdout-attempts.jsonl")
    ledger.start(
        attempt_id=attempt_id, fusion_artifact_hash=fusion_artifact.artifact_hash,
        specification_hash=specification.specification_hash,
    )
    try:
        samples = holdout_samples(manifest, root)
        report = HybridDevelopmentExperimentExecutorV2(
            classical_detector=classical_detector,
            baseline_normal_detector=normal_detector,
            hybrid_detector=hybrid_detector,
        ).execute(
            specification, samples,
            baseline_model_artifact=baseline_model,
            baseline_calibration_artifact=baseline_calibration,
            sink=V2SQLiteResultSink(output / "v2-hybrid-development-results.sqlite3"),
            execution_attempt_id=attempt_id,
        )
    except Exception as error:
        ledger.finish(attempt_id=attempt_id, status="failed_software", detail=type(error).__name__)
        raise
    ledger.finish(attempt_id=attempt_id, status="completed")
    methods = tuple(item.method_id for item in specification.methods)
    aggregates = {
        method: aggregate_metrics(tuple(
            item.evaluation for item in report.analyses if item.method_id == method
        )) for method in methods
    }
    budget_metrics = {
        f"{budget:.2f}": aggregate_metrics(evaluations).to_dict()
        for budget, evaluations in report.hybrid_budget_evaluations
    }
    pairing = Counter((row.image_id, row.method_implementation_id) for row in report.rows)
    expected_rows = len(manifest.development_holdout) * 3
    pairing_complete = (
        len(report.rows) == expected_rows and len(pairing) == expected_rows
        and all(value == 1 for value in pairing.values())
    )
    classical = aggregates["structvision-classical-baseline-v1-frozen"]
    hybrid = aggregates["structvision-proposal-guided-hybrid-v1-dev"]
    preservation_failures = _holdout_preservation_failures(hybrid, classical)
    clean_improved = (
        float(hybrid.clean_false_proposals_per_image) < float(classical.clean_false_proposals_per_image)
        and float(hybrid.clean_images_with_any_proposal) < float(classical.clean_images_with_any_proposal)
    )
    primary_budget_met = float(hybrid.clean_false_proposals_per_image) <= PRIMARY_BUDGET + 1e-12
    passed = not preservation_failures and clean_improved and primary_budget_met and pairing_complete
    after = protected_snapshot(root)
    protected_unchanged = before == after
    summary = {
        "classification": manifest.evidence_classification,
        "experiment_id": HYBRID_EXPERIMENT_ID,
        "experiment_version": HYBRID_EXPERIMENT_VERSION,
        "specification_hash": specification.specification_hash,
        "manifest_hash": manifest.manifest_hash,
        "role_counts": {
            "hybrid_normal_fit": len(manifest.normal_fit),
            "hybrid_fusion_fit": len(manifest.fusion_fit),
            "hybrid_development_holdout": len(manifest.development_holdout),
        },
        "normal_model_artifact_hash": hybrid_model.artifact_hash,
        "normal_model_memory_bank_shape": list(hybrid_model.memory_bank_shape),
        "normal_model_memory_bank_hash": hybrid_model.memory_bank_hash,
        "fusion_artifact_hash": fusion_artifact.artifact_hash,
        "fusion_fit": fusion_summary,
        "holdout_executed": True,
        "holdout_primary_attempt_count": 1,
        "expected_result_rows": expected_rows,
        "actual_result_rows": len(report.rows),
        "pairing_complete": pairing_complete,
        "aggregate_metrics": {method: value.to_dict() for method, value in aggregates.items()},
        "hybrid_false_proposal_budget_metrics": budget_metrics,
        "paired_effects": _paired_effects(report),
        "holdout_preservation_failures": list(preservation_failures),
        "holdout_clean_burden_improved": clean_improved,
        "holdout_primary_budget_met": primary_budget_met,
        "development_decision": (
            "promising development candidate" if passed else
            "development candidate rejected under the predeclared protocol"
        ),
        "peak_resident_memory_bytes_macos": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "dependency_lock_hash": lock_hash,
        "weight_hash": WEIGHT_SHA256,
        "pre_task_git_state": _git_pre_state(root),
        "protected_before": before,
        "protected_after": after,
        "protected_unchanged": protected_unchanged,
        "historical_test_access": False,
        "professor_data_access": False,
        "deprecated_balanced_score_used": False,
        "api_key_required": False,
    }
    if not protected_unchanged or before["historical_automatic_rows"] != 888:
        raise RuntimeError("A protected historical identity changed during hybrid development")
    if not pairing_complete:
        raise RuntimeError("Three-method v2 pairing is incomplete")
    (output / "development-summary.json").write_text(canonical_json(summary) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-manifest", action="store_true")
    parser.add_argument("--weight-file", type=Path)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("development_data/hybrid-development-manifest-v1.json"),
    )
    parser.add_argument(
        "--output-directory", type=Path,
        default=Path("outputs/proposal-guided-hybrid/SYN-PROPOSAL-HYBRID-DEV-001-v1"),
    )
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parent
    if arguments.prepare_manifest:
        print(canonical_json(prepare_manifest(root, (root / arguments.manifest).resolve())))
        return
    if arguments.weight_file is None:
        parser.error("--weight-file is required unless --prepare-manifest is used")
    summary = run(
        root=root, weight_file=arguments.weight_file,
        output_directory=arguments.output_directory,
        manifest_path=(root / arguments.manifest).resolve(),
    )
    print(canonical_json(summary))


if __name__ == "__main__":
    main()
