"""No-write-by-default command line interface for one-image analysis."""

from __future__ import annotations

import argparse
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
from .operational_storage import OperationalStorageContext
from .resources import ProtectedResourceCatalog
from .storage import ResourceRole, StorageConfigurationError


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
    storage_context: OperationalStorageContext,
) -> LearnedRuntimePaths:
    if not storage_context.is_external:
        return LearnedRuntimePaths()
    environment = LearnedRuntimePaths.from_environment(storage_context)
    catalog = ProtectedResourceCatalog(storage_context)
    roles = {
        "environment_lock": ResourceRole.LEARNED_ENVIRONMENT_LOCK,
        "weight": ResourceRole.OFFICIAL_WEIGHT,
        "patchcore_model": ResourceRole.PATCHCORE_MODEL,
        "patchcore_calibration": ResourceRole.PATCHCORE_CALIBRATION,
        "hybrid_model": ResourceRole.HYBRID_MODEL,
        "hybrid_fusion": ResourceRole.HYBRID_FUSION,
    }
    selected: dict[str, Path | None] = {}
    for field, role in roles.items():
        argument = getattr(arguments, field)
        if argument is None:
            selected[field] = getattr(environment, field)
        else:
            selected[field] = catalog.resolve_selected(role, argument).path
    runtime = LearnedRuntimePaths(
        **selected,
    )
    return runtime


def _write_explicit(
    path: Path,
    payload: bytes,
    storage_context: OperationalStorageContext,
) -> None:
    if storage_context.is_external:
        storage_context.authorise_run_output(path)
    if not path.parent.is_dir():
        raise OSError(f"Output parent directory does not exist: {path.parent}")
    path.write_bytes(payload)


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    method_id = METHOD_ALIASES[arguments.method]
    try:
        storage_context = OperationalStorageContext.discover(
            arguments.storage_config
        )
        input_path = (
            storage_context.authorise_private_input(arguments.input).path
            if storage_context.is_external
            else arguments.input
        )
    except StorageConfigurationError as error:
        print(f"Storage configuration error: {error}", file=sys.stderr)
        return EXIT_STORAGE_CONFIGURATION
    try:
        runtime = _runtime(arguments, storage_context)
        encoded = input_path.read_bytes()
        decoded = decode_image_bytes(
            encoded,
            filename=input_path.name,
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
            _write_explicit(
                arguments.json_out,
                analysis_json_bytes(analysis),
                storage_context,
            )
        if arguments.csv_out is not None:
            _write_explicit(
                arguments.csv_out,
                proposal_csv_bytes(analysis),
                storage_context,
            )
        if arguments.overlay_out is not None:
            _write_explicit(
                arguments.overlay_out,
                annotated_png_bytes(analysis),
                storage_context,
            )
        if arguments.summary_out is not None:
            _write_explicit(
                arguments.summary_out,
                technical_summary_bytes(analysis),
                storage_context,
            )
        if arguments.mask_out_dir is not None:
            if storage_context.is_external:
                storage_context.authorise_run_output(arguments.mask_out_dir)
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
                        storage_context,
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
