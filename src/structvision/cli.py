"""No-write-by-default command line interface for one-image analysis."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from .demonstration import (
    ALPHA_HANDLING_OPTIONS,
    CLASSICAL_METHOD,
    DEFAULT_METHOD,
    HYBRID_METHOD,
    PATCHCORE_METHOD,
    DemonstrationArtifactError,
    DemonstrationInputError,
    LearnedEnvironmentUnavailableError,
    LearnedRuntimePaths,
    analyse_demonstration_image,
    analysis_json_bytes,
    annotated_png_bytes,
    binary_mask_png_bytes,
    candidate_rows,
    decode_image_bytes,
    proposal_csv_bytes,
    technical_summary_bytes,
)
from .storage import (
    CONFIG_ENVIRONMENT_VARIABLE,
    LogicalRoot,
    PathIntent,
    StorageConfig,
    StorageConfigurationError,
    load_storage_config,
)


EXIT_SUCCESS = 0
EXIT_USAGE = 2
EXIT_INPUT = 3
EXIT_LEARNED_ENVIRONMENT = 4
EXIT_ARTIFACT = 5
EXIT_EXECUTION = 6
EXIT_OUTPUT = 7
EXIT_STORAGE_CONFIGURATION = 8

METHOD_ALIASES = {
    "classical": CLASSICAL_METHOD,
    "patchcore": PATCHCORE_METHOD,
    "hybrid": HYBRID_METHOD,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="structvision-analyse",
        description=(
            "Analyse one local image with an existing StructVision method. "
            "No file is written unless an explicit output option is supplied."
        ),
    )
    parser.add_argument("--input", required=True, type=Path, help="PNG, JPEG, or TIFF input")
    parser.add_argument(
        "--method",
        choices=tuple(METHOD_ALIASES),
        default="classical",
        help="classical is the stable frozen default",
    )
    parser.add_argument("--alpha-handling", choices=ALPHA_HANDLING_OPTIONS)
    parser.add_argument("--stdout-json", action="store_true", help="emit analysis JSON to stdout")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument("--overlay-out", type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--mask-out-dir", type=Path)
    parser.add_argument("--environment-lock", type=Path)
    parser.add_argument("--weight", type=Path)
    parser.add_argument("--patchcore-model", type=Path)
    parser.add_argument("--patchcore-calibration", type=Path)
    parser.add_argument("--hybrid-model", type=Path)
    parser.add_argument("--hybrid-fusion", type=Path)
    parser.add_argument(
        "--storage-config",
        type=Path,
        help=(
            "local TOML configuration for named external roots; when enabled, "
            "learned artifacts and explicit outputs are restricted to their named roots"
        ),
    )
    return parser


def _runtime(
    arguments: argparse.Namespace,
    storage: StorageConfig | None = None,
) -> LearnedRuntimePaths:
    environment = LearnedRuntimePaths.from_environment()
    runtime = LearnedRuntimePaths(
        environment_lock=arguments.environment_lock or environment.environment_lock,
        weight=arguments.weight or environment.weight,
        patchcore_model=arguments.patchcore_model or environment.patchcore_model,
        patchcore_calibration=arguments.patchcore_calibration or environment.patchcore_calibration,
        hybrid_model=arguments.hybrid_model or environment.hybrid_model,
        hybrid_fusion=arguments.hybrid_fusion or environment.hybrid_fusion,
    )
    if storage is None:
        return runtime
    storage.require_external()
    roles = {
        "environment_lock": LogicalRoot.SOURCE,
        "weight": LogicalRoot.LEARNED_ARTIFACT,
        "patchcore_model": LogicalRoot.LEARNED_ARTIFACT,
        "patchcore_calibration": LogicalRoot.LEARNED_ARTIFACT,
        "hybrid_model": LogicalRoot.LEARNED_ARTIFACT,
        "hybrid_fusion": LogicalRoot.LEARNED_ARTIFACT,
    }
    for field, root in roles.items():
        selected = getattr(runtime, field)
        if selected is not None:
            storage.authorise_path(root, selected, intent=PathIntent.READ)
    return runtime


def _write_explicit(
    path: Path,
    payload: bytes,
    storage: StorageConfig | None = None,
) -> None:
    if storage is not None:
        storage.authorise_path(LogicalRoot.RUNS, path, intent=PathIntent.WRITE)
    if not path.parent.is_dir():
        raise OSError(f"Output parent directory does not exist: {path.parent}")
    path.write_bytes(payload)


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    method_id = METHOD_ALIASES[arguments.method]
    try:
        storage = None
        if (
            arguments.storage_config is not None
            or os.environ.get(CONFIG_ENVIRONMENT_VARIABLE)
        ):
            storage = load_storage_config(arguments.storage_config, required=True)
        if storage is not None:
            storage.require_external()
    except StorageConfigurationError as error:
        print(f"Storage configuration error: {error}", file=sys.stderr)
        return EXIT_STORAGE_CONFIGURATION
    try:
        runtime = _runtime(arguments, storage)
        encoded = arguments.input.read_bytes()
        decoded = decode_image_bytes(
            encoded,
            filename=arguments.input.name,
            alpha_handling=arguments.alpha_handling,
        )
        analysis = analyse_demonstration_image(
            decoded,
            method_id=method_id,
            runtime=runtime,
        )
    except StorageConfigurationError as error:
        print(f"Storage configuration error: {error}", file=sys.stderr)
        return EXIT_STORAGE_CONFIGURATION
    except (OSError, DemonstrationInputError) as error:
        print(f"Input error: {error}", file=sys.stderr)
        return EXIT_INPUT
    except LearnedEnvironmentUnavailableError as error:
        print(f"Learned environment unavailable: {error}", file=sys.stderr)
        return EXIT_LEARNED_ENVIRONMENT
    except DemonstrationArtifactError as error:
        print(f"Artifact error: {error}", file=sys.stderr)
        return EXIT_ARTIFACT
    except Exception as error:
        print(f"Execution error: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_EXECUTION

    try:
        if arguments.json_out is not None:
            _write_explicit(arguments.json_out, analysis_json_bytes(analysis), storage)
        if arguments.csv_out is not None:
            _write_explicit(arguments.csv_out, proposal_csv_bytes(analysis), storage)
        if arguments.overlay_out is not None:
            _write_explicit(arguments.overlay_out, annotated_png_bytes(analysis), storage)
        if arguments.summary_out is not None:
            _write_explicit(arguments.summary_out, technical_summary_bytes(analysis), storage)
        if arguments.mask_out_dir is not None:
            if storage is not None:
                storage.authorise_path(
                    LogicalRoot.RUNS,
                    arguments.mask_out_dir,
                    intent=PathIntent.WRITE,
                )
            if not arguments.mask_out_dir.is_dir():
                raise OSError(
                    f"Mask output directory does not exist: {arguments.mask_out_dir}"
                )
            for row in candidate_rows(analysis):
                if bool(row["selected"]):
                    target = arguments.mask_out_dir / f"{row['proposal_id']}.png"
                    _write_explicit(
                        target,
                        binary_mask_png_bytes(analysis, str(row["proposal_id"])),
                        storage,
                    )
    except (OSError, StorageConfigurationError) as error:
        print(f"Output error: {error}", file=sys.stderr)
        return EXIT_OUTPUT

    if arguments.stdout_json:
        sys.stdout.buffer.write(analysis_json_bytes(analysis))
    else:
        sys.stdout.buffer.write(technical_summary_bytes(analysis))
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
