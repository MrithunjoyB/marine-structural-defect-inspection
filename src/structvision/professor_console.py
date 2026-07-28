"""Offline professor-console orchestration over the public StructVision facade.

This module contains presentation, filesystem, and integrity handling only. It
does not implement image preprocessing, feature extraction, proposal
generation, scoring, ranking, thresholding, or evaluation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO, StringIO
import csv
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from typing import Iterable

from PIL import Image, PngImagePlugin

from structvision import (
    CLASSICAL_METHOD,
    DemonstrationInputError,
    analyse_demonstration_image,
    analysis_json_bytes,
    annotated_png_bytes,
    binary_mask_png_bytes,
    candidate_rows,
    decode_image_bytes,
    export_payload,
    method_status,
    pipeline_stages,
    proposal_csv_bytes,
    render_anomaly_overlay,
    technical_summary_bytes,
)


EXIT_SUCCESS = 0
EXIT_USAGE = 2
EXIT_INPUT = 3
EXIT_OUTPUT_EXISTS = 4
EXIT_EXECUTION = 5
EXIT_OUTPUT = 6

SCORE_WARNING = (
    "Review proposals only — not confirmed defects or engineering diagnosis."
)
TIMING_FIELDS = (
    ("input_normalisation", "input_normalisation"),
    ("preprocessing", "preprocessing"),
    ("feature_extraction", "feature_extraction"),
    ("proposal_generation", "proposal_generation"),
    ("result_conversion", "adapter_conversion"),
    ("total", "total"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="structvision-professor-demo",
        description=(
            "Run the stable frozen StructVision baseline once and create an "
            "explicit, professor-facing INPUT/PROCESSING/OUTPUT record."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Local PNG, JPEG, or TIFF image",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New explicit run directory",
    )
    parser.add_argument(
        "--alpha-handling",
        choices=("drop", "composite_black", "composite_white"),
        help="Required policy for alpha-bearing input",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace this exact existing run directory after successful analysis",
    )
    return parser


def _sha256_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes, *, executable: bool = False) -> None:
    """Write one payload through a temporary file in the same output folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if executable:
            temporary.chmod(0o755)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _png_bytes(
    image_bgr: object,
    *,
    metadata: dict[str, str],
) -> bytes:
    """Encode an already-exposed BGR presentation image as a lossless PNG."""
    rgb = image_bgr[:, :, ::-1]
    png_metadata = PngImagePlugin.PngInfo()
    for name, value in sorted(metadata.items()):
        png_metadata.add_text(f"structvision.{name}", value)
    buffer = BytesIO()
    Image.fromarray(rgb).save(
        buffer,
        format="PNG",
        pnginfo=png_metadata,
        optimize=False,
        compress_level=6,
    )
    return buffer.getvalue()


def _timings(analysis: object) -> tuple[tuple[str, str, float], ...]:
    available = dict(getattr(analysis.result, "timing_breakdown_seconds"))
    rows = []
    for presentation_name, source_name in TIMING_FIELDS:
        if source_name not in available:
            raise RuntimeError(
                f"Frozen result did not expose required timing field {source_name}"
            )
        rows.append(
            (presentation_name, source_name, float(available[source_name]))
        )
    return tuple(rows)


def _timing_csv_bytes(
    rows: Iterable[tuple[str, str, float]],
) -> bytes:
    buffer = StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(("stage", "source_timing_field", "seconds", "measurement"))
    for stage, source, seconds in rows:
        writer.writerow((stage, source, f"{seconds:.9f}", "measured"))
    return buffer.getvalue().encode("utf-8")


def _safe_output_target(path: Path) -> Path:
    target = path.expanduser().resolve(strict=False)
    anchor = Path(target.anchor)
    if target == anchor or target == Path.home().resolve():
        raise ValueError("The output directory must be a dedicated child folder")
    if target.is_symlink():
        raise ValueError("A symbolic-link output directory is not accepted")
    if target.exists() and not target.is_dir():
        raise ValueError("The output path exists and is not a directory")
    return target


def _file_records(root: Path, *, exclude: frozenset[str]) -> list[dict[str, object]]:
    records = []
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        if relative in exclude:
            continue
        records.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return records


def _platform_record() -> dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }


def _console_lines(
    *,
    input_name: str,
    decoded: object,
    status: object,
    timings: tuple[tuple[str, str, float], ...] | None = None,
    proposal_count: int | None = None,
    generated: Iterable[str] = (),
) -> list[str]:
    lines = [
        "StructVision-AI — Professor Console Demonstration",
        "--------------------------------------------------",
        f"Input       : {input_name}",
        f"Format      : {decoded.source_format} / {decoded.source_mode}",
        f"Dimensions  : {decoded.width} x {decoded.height}",
        f"Colour path : {decoded.colour_handling}",
        f"Method      : {status.method_id}",
        f"Status      : {status.status}",
        "",
        "Running analysis...",
    ]
    if timings is None:
        return lines
    lines.extend(("", "Measured processing stages:"))
    for stage, _source, seconds in timings:
        lines.append(f"  {stage:<22} {seconds:.6f} s")
    lines.extend(("", f"Selected proposals: {proposal_count}", "", "Generated:"))
    lines.extend(f"  {path}" for path in generated)
    lines.extend(("", SCORE_WARNING))
    return lines


def _prepare_output_directory(target: Path, *, overwrite: bool) -> None:
    if target.exists():
        if not overwrite:
            raise FileExistsError(
                "Output directory already exists; pass --overwrite to replace it"
            )
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=False)


def _write_run(
    *,
    target: Path,
    input_name: str,
    decoded: object,
    analysis: object,
    timing_rows: tuple[tuple[str, str, float], ...],
    initial_lines: list[str],
) -> list[str]:
    try:
        input_png = _png_bytes(
            decoded.image_bgr,
            metadata={
                "artifact_role": "lossless normalised input copy",
                "encoded_source_sha256": decoded.encoded_sha256,
                "fixture_status": (
                    "synthetic demonstration fixture; excluded from research cohorts; "
                    "not real inspection evidence"
                    if decoded.fixture_label
                    else "caller-supplied demonstration input"
                ),
            },
        )
        input_metadata = {
            "schema_version": "structvision-professor-input-v1",
            "source_filename": input_name,
            "source_format": decoded.source_format,
            "source_mode": decoded.source_mode,
            "source_encoded_sha256": decoded.encoded_sha256,
            "source_encoded_bytes_preserved": False,
            "normalised_copy": "INPUT/original.png",
            "normalised_copy_sha256": _sha256_bytes(input_png),
            "normalised_copy_encoding": "lossless PNG / uint8 BGR represented as RGB",
            "width": decoded.width,
            "height": decoded.height,
            "colour_handling": decoded.colour_handling,
            "warnings": list(decoded.warnings),
            "absolute_path_recorded": False,
        }
        _atomic_write(target / "INPUT" / "original.png", input_png)
        _atomic_write(
            target / "INPUT" / "input-metadata.json",
            _json_bytes(input_metadata),
        )

        stages = {
            "schema_version": "structvision-professor-pipeline-v1",
            "method_identity": analysis.method_id,
            "method_status": analysis.method.status,
            "stages": list(pipeline_stages(CLASSICAL_METHOD)),
            "artifact_policy": (
                "Only artifacts exposed through the current public demonstration "
                "facade are exported. Unexposed internal stages are not invented."
            ),
        }
        _atomic_write(
            target / "PROCESSING" / "pipeline-stages.json",
            _json_bytes(stages),
        )
        _atomic_write(
            target / "PROCESSING" / "stage-timings.csv",
            _timing_csv_bytes(timing_rows),
        )
        anomaly = render_anomaly_overlay(analysis)
        if anomaly is None:
            raise RuntimeError(
                "Frozen classical result did not expose anomaly evidence"
            )
        _atomic_write(
            target / "PROCESSING" / "anomaly-evidence.png",
            _png_bytes(
                anomaly,
                metadata={
                    "artifact_role": "exposed anomaly evidence visualisation",
                    "method_identity": analysis.method_id,
                    "warning": SCORE_WARNING,
                },
            ),
        )
        processing_readme = (
            "StructVision-AI processing artifacts\n"
            "=====================================\n\n"
            "pipeline-stages.json describes exposed and unavailable stages.\n"
            "stage-timings.csv contains measured detector-returned timings.\n"
            "anomaly-evidence.png visualises the exposed anomaly heatmap.\n\n"
            "Candidate-mask generation internals and intermediate mask-refinement "
            "images are not exposed by the current frozen API. They are not "
            "invented for this demonstration.\n\n"
            f"{SCORE_WARNING}\n"
        ).encode("utf-8")
        _atomic_write(target / "PROCESSING" / "README.txt", processing_readme)

        _atomic_write(
            target / "OUTPUT" / "overlay.png",
            annotated_png_bytes(analysis),
        )
        _atomic_write(
            target / "OUTPUT" / "proposals.csv",
            proposal_csv_bytes(analysis),
        )
        _atomic_write(
            target / "OUTPUT" / "result.json",
            analysis_json_bytes(analysis),
        )
        _atomic_write(
            target / "OUTPUT" / "technical-summary.txt",
            technical_summary_bytes(analysis)
            + f"\n{SCORE_WARNING}\n".encode("utf-8"),
        )
        selected_rows = tuple(
            row for row in candidate_rows(analysis) if bool(row["selected"])
        )
        for row in selected_rows:
            proposal_id = str(row["proposal_id"])
            _atomic_write(
                target / "OUTPUT" / "masks" / f"{proposal_id}.png",
                binary_mask_png_bytes(analysis, proposal_id),
            )

        generated = [
            "INPUT/original.png",
            "INPUT/input-metadata.json",
            "PROCESSING/pipeline-stages.json",
            "PROCESSING/stage-timings.csv",
            "PROCESSING/anomaly-evidence.png",
            "PROCESSING/README.txt",
            "OUTPUT/overlay.png",
            "OUTPUT/proposals.csv",
            "OUTPUT/result.json",
            "OUTPUT/technical-summary.txt",
        ]
        generated.extend(
            f"OUTPUT/masks/{row['proposal_id']}.png" for row in selected_rows
        )
        generated.extend(("RUN_MANIFEST.json", "CONSOLE_LOG.txt"))
        final_lines = _console_lines(
            input_name=input_name,
            decoded=decoded,
            status=analysis.method,
            timings=timing_rows,
            proposal_count=len(selected_rows),
            generated=generated,
        )
        if final_lines[: len(initial_lines)] != initial_lines:
            raise RuntimeError("Console transcript prefix changed unexpectedly")
        _atomic_write(
            target / "CONSOLE_LOG.txt",
            ("\n".join(final_lines) + "\n").encode("utf-8"),
        )

        exported = export_payload(analysis)
        provenance = getattr(analysis.result, "provenance").to_dict()
        manifest = {
            "schema_version": "structvision-professor-run-manifest-v1",
            "created_timestamp_utc": analysis.created_timestamp_utc,
            "method": analysis.method.to_dict(),
            "detector_execution_count": 1,
            "processing_status": "completed",
            "input": input_metadata,
            "implementation": {
                "implementation_identity": analysis.method_id,
                "implementation_version": analysis.method.version,
                "configuration_hash": analysis.configuration_hash,
                "protected_source_hashes": provenance[
                    "protected_source_hashes"
                ],
                "protected_source_hashes_verified": provenance[
                    "protected_hashes_verified"
                ],
                "artifact_identities": exported["analysis"][
                    "artifact_identities"
                ],
            },
            "timings": [
                {
                    "stage": stage,
                    "source_timing_field": source,
                    "seconds": seconds,
                    "measurement": "measured",
                }
                for stage, source, seconds in timing_rows
            ],
            "selected_proposal_count": len(selected_rows),
            "coordinate_convention": (
                "half-open (x_min, y_min, x_max, y_max) in analysed-image pixels"
            ),
            "score_semantics": analysis.score_semantics,
            "warnings": list(dict.fromkeys((*analysis.warnings, SCORE_WARNING))),
            "platform": _platform_record(),
            "privacy": exported["privacy"],
            "path_policy": (
                "All recorded paths are relative to this run directory; no "
                "absolute source path or private environment variable is stored."
            ),
            "file_manifest_scope": (
                "Every run payload file except RUN_MANIFEST.json itself; a file "
                "cannot contain its own final SHA-256 without circularity."
            ),
            "files": _file_records(
                target,
                exclude=frozenset({"RUN_MANIFEST.json"}),
            ),
        }
        _atomic_write(
            target / "RUN_MANIFEST.json",
            _json_bytes(manifest),
        )
        return final_lines
    except Exception:
        if target.is_dir():
            shutil.rmtree(target)
        raise


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    status = method_status(CLASSICAL_METHOD)
    input_name = arguments.input.name
    try:
        encoded = arguments.input.read_bytes()
        decoded = decode_image_bytes(
            encoded,
            filename=input_name,
            alpha_handling=arguments.alpha_handling,
        )
        target = _safe_output_target(arguments.output_dir)
        if target.exists() and not arguments.overwrite:
            raise FileExistsError(
                "Output directory already exists; pass --overwrite to replace it"
            )
    except FileExistsError as error:
        print(f"Output protection: {error}", file=sys.stderr)
        return EXIT_OUTPUT_EXISTS
    except (OSError, ValueError, DemonstrationInputError) as error:
        print(f"Input/output validation error: {error}", file=sys.stderr)
        return EXIT_INPUT

    initial_lines = _console_lines(
        input_name=input_name,
        decoded=decoded,
        status=status,
    )
    try:
        _prepare_output_directory(target, overwrite=arguments.overwrite)
    except FileExistsError as error:
        print(f"Output protection: {error}", file=sys.stderr)
        return EXIT_OUTPUT_EXISTS
    except Exception as error:
        print(
            f"Output error: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return EXIT_OUTPUT

    print("\n".join(initial_lines), flush=True)
    runtime_temporary = target / "PROCESSING" / ".runtime"
    runtime_temporary.mkdir(parents=True, exist_ok=False)
    prior_tempdir = tempfile.tempdir
    try:
        tempfile.tempdir = str(runtime_temporary)
        analysis = analyse_demonstration_image(
            decoded,
            method_id=CLASSICAL_METHOD,
        )
        timing_rows = _timings(analysis)
    except Exception as error:
        if target.is_dir():
            shutil.rmtree(target)
        print(
            f"Execution error: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return EXIT_EXECUTION
    finally:
        tempfile.tempdir = prior_tempdir
        if runtime_temporary.is_dir():
            shutil.rmtree(runtime_temporary)

    try:
        final_lines = _write_run(
            target=target,
            input_name=input_name,
            decoded=decoded,
            analysis=analysis,
            timing_rows=timing_rows,
            initial_lines=initial_lines,
        )
    except Exception as error:
        print(
            f"Output error: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return EXIT_OUTPUT

    print("\n".join(final_lines[len(initial_lines) :]), flush=True)
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
