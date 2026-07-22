"""Reproduce SYN-NORMAL-FEATURE-DEV-001 without accessing any test split."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import resource

import cv2
import numpy as np

from scientific_contract.hashing import canonical_json
from scientific_contract.metrics import aggregate_metrics
from structvision.development_protocol import load_development_manifest, normal_fit_samples
from structvision.learned_executor import (
    DevelopmentExperimentExecutorV2,
    create_development_experiment_specification,
    development_experiment_samples,
)
from structvision.normal_feature.calibration import (
    CalibrationSample,
    DirectoryCalibrationArtifactSink,
    calibrate,
)
from structvision.normal_feature.evaluation import dense_development_metrics
from structvision.normal_feature.model_artifact import DirectoryModelArtifactSink
from structvision.normal_feature.patchcore import NormalFeatureAnomalyDetector
from structvision.sinks import V2SQLiteResultSink


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _truth(identity: object, root: Path) -> np.ndarray:
    if identity.ground_truth_kind == "implicit_verified_zero_mask":
        image = cv2.imdecode(np.frombuffer((root / identity.image_path).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not decode protected image {identity.image_id}")
        return np.zeros(image.shape[:2], dtype=np.uint8)
    path = root / str(identity.ground_truth_path)
    truth = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if truth is None:
        raise RuntimeError(f"Could not decode protected truth {identity.image_id}")
    return np.ascontiguousarray((truth > 0).astype(np.uint8) * 255)


def run(*, repository_root: Path, weight_file: Path, output_directory: Path) -> dict[str, object]:
    root = Path(repository_root).resolve()
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Development output directory must be new or empty; artifacts are immutable")
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_development_manifest(root / "development_data/normal-feature-development-manifest-v1.json")
    if {item.split_role for item in manifest.selected_images} - {"train", "validation"}:
        raise RuntimeError("Protected manifest contains a forbidden split")
    lock = root / "requirements/pylock.normal-feature-macos-arm64.toml"
    lock_hash = _sha256(lock)
    detector = NormalFeatureAnomalyDetector(
        weight_file=weight_file,
        environment_lock_hash=lock_hash,
        repository_root=root,
    )
    model_sink = DirectoryModelArtifactSink(output / "model-artifacts")
    model = detector.fit_normal(
        normal_fit_samples(manifest, root),
        normal_fit_manifest_hash=manifest.manifest_hash,
        artifact_sink=model_sink,
    )
    calibration_samples = []
    scored = []
    truths = []
    for identity in manifest.calibration_validation:
        result = detector.score(
            root / identity.image_path,
            model_artifact=model,
            image_id=identity.image_id,
        )
        truth = _truth(identity, root)
        scored.append(result)
        truths.append(truth)
        calibration_samples.append(CalibrationSample(
            identity.image_id,
            identity.role,
            identity.category,
            identity.image_outcome,
            result.anomaly_map,
            truth,
        ))
    calibration = calibrate(
        calibration_samples,
        model_artifact_hash=model.artifact_hash,
        calibration_manifest_hash=manifest.manifest_hash,
        proposal_config=detector.config.proposal,
    )
    DirectoryCalibrationArtifactSink(output / "calibration-artifacts").write(calibration)
    specification = create_development_experiment_specification(
        manifest=manifest,
        model_artifact=model,
        calibration_artifact=calibration,
        dependency_lock_hash=lock_hash,
        repository_root=root,
    )
    samples = development_experiment_samples(manifest, root)
    sink = V2SQLiteResultSink(output / "v2-development-results.sqlite3")
    report = DevelopmentExperimentExecutorV2(detector).execute(
        specification,
        samples,
        model_artifact=model,
        calibration_artifact=calibration,
        sink=sink,
        execution_attempt_id="SYN-NORMAL-FEATURE-DEV-001-v1-reference-cpu",
    )
    methods = [method.method_id for method in specification.methods]
    aggregate = {
        method: aggregate_metrics([
            item.evaluation for item in report.analyses if item.method_id == method
        ]).to_dict()
        for method in methods
    }
    budget_metrics = {
        identity: aggregate_metrics(evaluations).to_dict()
        for identity, evaluations in report.learned_budget_evaluations
    }
    dense = dense_development_metrics(
        anomaly_maps=tuple(item.anomaly_map for item in scored),
        image_scores=tuple(item.image_anomaly_score for item in scored),
        ground_truths=tuple(truths),
        outcomes=tuple(item.image_outcome for item in manifest.calibration_validation),
        categories=tuple(item.category for item in manifest.calibration_validation),
    )
    pairs = Counter((row.image_id, row.method_implementation_id) for row in report.rows)
    pairing_complete = (
        len(report.rows) == len(manifest.calibration_validation) * 2
        and len(pairs) == len(report.rows)
        and all(value == 1 for value in pairs.values())
    )
    summary = {
        "classification": "development-only — non-confirmatory",
        "experiment_id": specification.experiment_id,
        "experiment_version": specification.experiment_version,
        "specification_hash": specification.specification_hash,
        "manifest_hash": manifest.manifest_hash,
        "normal_fit_count": len(manifest.normal_fit),
        "calibration_validation_count": len(manifest.calibration_validation),
        "expected_result_rows": len(manifest.calibration_validation) * 2,
        "actual_result_rows": len(report.rows),
        "pairing_complete": pairing_complete,
        "model_artifact_hash": model.artifact_hash,
        "memory_bank_shape": list(model.memory_bank_shape),
        "memory_bank_hash": model.memory_bank_hash,
        "coreset_hash": model.coreset_hash,
        "calibration_artifact_hash": calibration.artifact_hash,
        "operating_points": [item.to_dict() for item in calibration.operating_points],
        "aggregate_proposal_metrics": aggregate,
        "learned_fp_budget_metrics": budget_metrics,
        "dense_patchcore_metrics": dense.to_dict(),
        "peak_resident_memory_bytes_macos": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "dependency_lock_hash": lock_hash,
        "weight_hash": detector.config.pretrained_weight_sha256,
        "historical_test_access": False,
        "professor_data_access": False,
        "hybrid_method_implemented": False,
        "deprecated_balanced_score_used": False,
        "scientific_device": "cpu",
        "mps_scientific_reference": False,
    }
    if not pairing_complete:
        raise RuntimeError("Protected two-method v2 pairing is incomplete")
    (output / "experiment-specification-v2.json").write_text(specification.to_json() + "\n", encoding="utf-8")
    (output / "development-summary.json").write_text(canonical_json(summary) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-file", type=Path, required=True)
    parser.add_argument(
        "--output-directory", type=Path,
        default=Path("outputs/normal-feature-development/SYN-NORMAL-FEATURE-DEV-001-v1"),
    )
    arguments = parser.parse_args()
    summary = run(
        repository_root=Path(__file__).resolve().parent,
        weight_file=arguments.weight_file,
        output_directory=arguments.output_directory,
    )
    print(canonical_json(summary))


if __name__ == "__main__":
    main()
