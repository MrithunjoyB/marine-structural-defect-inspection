"""Offline live-console orchestration over the public StructVision facade.

This module contains presentation, filesystem, and integrity handling only. It
does not implement image preprocessing, feature extraction, proposal
generation, scoring, ranking, thresholding, or evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO, StringIO
import csv
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import sys
import tempfile
from typing import Iterable

from PIL import Image, PngImagePlugin

from structvision import (
    CLASSICAL_METHOD,
    DemonstrationInputError,
    OperationalStorageContext,
    StorageConfigurationError,
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
EXIT_UNSAFE_TARGET = 7
EXIT_STORAGE_CONFIGURATION = 8

SCORE_WARNING = (
    "Review proposals only — not confirmed defects or engineering diagnosis."
)
OWNERSHIP_MARKER_NAME = ".structvision-live-console-owner.json"
OWNERSHIP_SCHEMA = "structvision-live-console-run-owner-v1"
OWNERSHIP_TOOL = "structvision-live-demo"
OWNERSHIP_VERSION = 1
RUN_MANIFEST_SCHEMA = "structvision-live-run-manifest-v1"
STANDARD_USER_FOLDERS = (
    "Documents",
    "Desktop",
    "Downloads",
    "Developer",
)
STAGING_NAME_TOKEN = ".structvision-live-stage-"
BACKUP_NAME_TOKEN = ".structvision-live-backup-"
TIMING_FIELDS = (
    ("input_normalisation", "input_normalisation"),
    ("preprocessing", "preprocessing"),
    ("feature_extraction", "feature_extraction"),
    ("proposal_generation", "proposal_generation"),
    ("result_conversion", "adapter_conversion"),
    ("total", "total"),
)


class UnsafeOutputTargetError(ValueError):
    """A caller-selected output path is unsafe for creation or replacement."""


@dataclass(frozen=True)
class OwnedRunIdentity:
    directory_device: int
    directory_inode: int
    marker_sha256: str
    manifest_sha256: str
    payload_file_count: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="structvision-live-demo",
        description=(
            "Run the stable frozen StructVision baseline once and create an "
            "explicit technical-review INPUT/PROCESSING/OUTPUT record."
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
    parser.add_argument(
        "--storage-config",
        type=Path,
        help=(
            "optional external-storage configuration override; otherwise the "
            "preferred local configuration is discovered automatically"
        ),
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


def _home_directory() -> Path:
    return Path.home().resolve()


def _repository_roots() -> tuple[Path, ...]:
    roots = set()
    for origin in (Path.cwd(), Path(__file__).absolute()):
        candidates = (origin,) + tuple(origin.parents)
        for candidate in candidates:
            if (candidate / ".git").exists():
                roots.add(candidate.resolve())
                break
    return tuple(sorted(roots, key=lambda item: item.as_posix()))


def _is_filesystem_root(path: Path) -> bool:
    return path == Path(path.anchor)


def _is_mount_root(path: Path) -> bool:
    return (
        (path.exists() and os.path.ismount(path))
        or path.parent == Path("/Volumes")
    )


def _lexical_absolute(path: Path, *, label: str) -> Path:
    """Return an absolute path without resolving away symlink intent."""
    if any(part == ".." for part in path.parts):
        raise UnsafeOutputTargetError(
            f"{label} must not contain parent-directory traversal"
        )
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return expanded.absolute()


def _reject_existing_symlink_components(path: Path) -> None:
    """Reject the target or any already-existing ancestor that is a symlink."""
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            break
        except OSError as error:
            raise UnsafeOutputTargetError(
                "Output path ancestry could not be inspected safely"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise UnsafeOutputTargetError(
                "Output path must not be a symlink or traverse a symlink"
            )


def _input_ancestors(path: Path) -> frozenset[Path]:
    lexical = _lexical_absolute(path, label="Input path")
    values = set(lexical.parents)
    try:
        resolved = lexical.resolve(strict=True)
    except OSError:
        resolved = lexical.resolve(strict=False)
    values.update(resolved.parents)
    return frozenset(values)


def _safe_output_target(path: Path, *, input_path: Path) -> Path:
    """Validate raw path intent before returning a resolved safe target."""
    lexical = _lexical_absolute(path, label="Output directory")
    _reject_existing_symlink_components(lexical)
    target = lexical.resolve(strict=False)
    home = _home_directory()

    if _is_filesystem_root(target):
        raise UnsafeOutputTargetError("Filesystem roots cannot be output targets")
    if _is_mount_root(target):
        raise UnsafeOutputTargetError("Mount or external-volume roots are unsafe")
    if target == home:
        raise UnsafeOutputTargetError("The home directory cannot be an output target")
    if target in {home / name for name in STANDARD_USER_FOLDERS}:
        raise UnsafeOutputTargetError(
            "Standard user folders cannot be output targets"
        )
    for repository in _repository_roots():
        if target == repository or target in repository.parents:
            raise UnsafeOutputTargetError(
                "The repository or any of its ancestors cannot be an output target"
            )
    if target in _input_ancestors(input_path):
        raise UnsafeOutputTargetError(
            "The input parent or any input ancestor cannot be an output target"
        )
    if target.exists() and not target.is_dir():
        raise UnsafeOutputTargetError(
            "The output path exists and is not a directory"
        )
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


def _ownership_digest(manifest_sha256: str) -> str:
    payload = "\0".join(
        (
            OWNERSHIP_SCHEMA,
            OWNERSHIP_TOOL,
            str(OWNERSHIP_VERSION),
            CLASSICAL_METHOD,
            RUN_MANIFEST_SCHEMA,
            manifest_sha256,
        )
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _ownership_marker_bytes(manifest_payload: bytes) -> bytes:
    manifest_sha256 = _sha256_bytes(manifest_payload)
    return _json_bytes(
        {
            "schema_version": OWNERSHIP_SCHEMA,
            "tool_identity": OWNERSHIP_TOOL,
            "tool_version": OWNERSHIP_VERSION,
            "method_identity": CLASSICAL_METHOD,
            "run_manifest_path": "RUN_MANIFEST.json",
            "run_manifest_schema": RUN_MANIFEST_SCHEMA,
            "run_manifest_sha256": manifest_sha256,
            "ownership_digest": _ownership_digest(manifest_sha256),
            "completed": True,
        }
    )


def _regular_file_metadata(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise UnsafeOutputTargetError(f"{label} is missing or unreadable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise UnsafeOutputTargetError(f"{label} must be a regular non-symlink file")
    return metadata


def _manifest_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise UnsafeOutputTargetError("Ownership manifest contains an invalid path")
    if (
        value.startswith("/")
        or "\\" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise UnsafeOutputTargetError(
            "Ownership manifest path is absolute or traverses directories"
        )
    parsed = PurePosixPath(value)
    if parsed.is_absolute():
        raise UnsafeOutputTargetError("Ownership manifest path must be relative")
    return parsed.as_posix()


def _actual_run_entries(root: Path) -> tuple[set[str], set[str]]:
    files = set()
    directories_seen = set()
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            path = current_path / name
            directories_seen.add(path.relative_to(root).as_posix())
        for name in tuple(directories) + tuple(filenames):
            path = current_path / name
            try:
                metadata = os.lstat(path)
            except OSError as error:
                raise UnsafeOutputTargetError(
                    "Owned run contents changed during validation"
                ) from error
            if stat.S_ISLNK(metadata.st_mode):
                raise UnsafeOutputTargetError(
                    "Owned run must not contain symbolic links"
                )
        for name in filenames:
            path = current_path / name
            metadata = os.lstat(path)
            if not stat.S_ISREG(metadata.st_mode):
                raise UnsafeOutputTargetError(
                    "Owned run contains a non-regular file"
                )
            files.add(path.relative_to(root).as_posix())
    return files, directories_seen


def _validate_owned_run(root: Path) -> OwnedRunIdentity:
    """Verify a complete prior run before it is eligible for replacement."""
    try:
        root_metadata = os.lstat(root)
    except OSError as error:
        raise UnsafeOutputTargetError(
            "Existing output directory could not be inspected"
        ) from error
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise UnsafeOutputTargetError(
            "Existing output target must be a non-symlink directory"
        )

    marker_path = root / OWNERSHIP_MARKER_NAME
    marker_metadata = _regular_file_metadata(
        marker_path,
        label="StructVision ownership marker",
    )
    if marker_metadata.st_size > 64 * 1024:
        raise UnsafeOutputTargetError("StructVision ownership marker is oversized")
    try:
        marker_payload = marker_path.read_bytes()
        marker = json.loads(marker_payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise UnsafeOutputTargetError(
            "StructVision ownership marker is malformed"
        ) from error
    expected_marker_keys = {
        "schema_version",
        "tool_identity",
        "tool_version",
        "method_identity",
        "run_manifest_path",
        "run_manifest_schema",
        "run_manifest_sha256",
        "ownership_digest",
        "completed",
    }
    if not isinstance(marker, dict) or set(marker) != expected_marker_keys:
        raise UnsafeOutputTargetError(
            "StructVision ownership marker has an unexpected schema"
        )
    fixed_values = {
        "schema_version": OWNERSHIP_SCHEMA,
        "tool_identity": OWNERSHIP_TOOL,
        "tool_version": OWNERSHIP_VERSION,
        "method_identity": CLASSICAL_METHOD,
        "run_manifest_path": "RUN_MANIFEST.json",
        "run_manifest_schema": RUN_MANIFEST_SCHEMA,
        "completed": True,
    }
    if any(marker.get(name) != value for name, value in fixed_values.items()):
        raise UnsafeOutputTargetError(
            "StructVision ownership marker belongs to another tool or version"
        )
    manifest_digest = marker.get("run_manifest_sha256")
    if (
        not isinstance(manifest_digest, str)
        or len(manifest_digest) != 64
        or any(character not in "0123456789abcdef" for character in manifest_digest)
        or marker.get("ownership_digest") != _ownership_digest(manifest_digest)
    ):
        raise UnsafeOutputTargetError(
            "StructVision ownership marker identity is forged or invalid"
        )

    manifest_path = root / "RUN_MANIFEST.json"
    manifest_metadata = _regular_file_metadata(
        manifest_path,
        label="StructVision run manifest",
    )
    if manifest_metadata.st_size > 16 * 1024 * 1024:
        raise UnsafeOutputTargetError("StructVision run manifest is oversized")
    try:
        manifest_payload = manifest_path.read_bytes()
        manifest = json.loads(manifest_payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise UnsafeOutputTargetError(
            "StructVision run manifest is malformed"
        ) from error
    actual_manifest_digest = _sha256_bytes(manifest_payload)
    if actual_manifest_digest != manifest_digest:
        raise UnsafeOutputTargetError(
            "StructVision ownership marker does not match the run manifest"
        )
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != RUN_MANIFEST_SCHEMA
        or manifest.get("processing_status") != "completed"
        or manifest.get("detector_execution_count") != 1
        or not isinstance(manifest.get("method"), dict)
        or manifest["method"].get("method_id") != CLASSICAL_METHOD
    ):
        raise UnsafeOutputTargetError(
            "StructVision run manifest is not a completed live-console run"
        )

    records = manifest.get("files")
    if not isinstance(records, list):
        raise UnsafeOutputTargetError("StructVision run manifest file list is invalid")
    expected_files = {"RUN_MANIFEST.json", OWNERSHIP_MARKER_NAME}
    seen = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise UnsafeOutputTargetError(
                "StructVision run manifest contains a malformed file record"
            )
        relative = _manifest_relative_path(record["path"])
        if relative in seen or relative in expected_files:
            raise UnsafeOutputTargetError(
                "StructVision run manifest contains a duplicate/reserved path"
            )
        seen.add(relative)
        expected_files.add(relative)
        path = root / relative
        metadata = _regular_file_metadata(path, label=f"Run payload {relative}")
        if (
            type(record["size_bytes"]) is not int
            or record["size_bytes"] < 0
            or metadata.st_size != record["size_bytes"]
            or not isinstance(record["sha256"], str)
            or _sha256_file(path) != record["sha256"]
        ):
            raise UnsafeOutputTargetError(
                f"Run payload identity mismatch: {relative}"
            )
    actual_files, actual_directories = _actual_run_entries(root)
    if actual_files != expected_files:
        raise UnsafeOutputTargetError(
            "Existing run contains missing or unexpected files"
        )
    expected_directories = {
        parent.as_posix()
        for relative in expected_files
        for parent in PurePosixPath(relative).parents
        if parent.as_posix() != "."
    }
    if actual_directories != expected_directories:
        raise UnsafeOutputTargetError(
            "Existing run contains missing or unexpected directories"
        )
    return OwnedRunIdentity(
        directory_device=root_metadata.st_dev,
        directory_inode=root_metadata.st_ino,
        marker_sha256=_sha256_bytes(marker_payload),
        manifest_sha256=actual_manifest_digest,
        payload_file_count=len(records),
    )


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
        "StructVision-AI — Live Inspection Console",
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


def _create_private_sibling(target: Path, *, token: str) -> Path:
    path = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}{token}",
            dir=target.parent,
        )
    )
    path.chmod(0o700)
    return path


def _remove_private_directory(
    path: Path,
    *,
    expected_parent: Path,
    expected_prefix: str,
) -> None:
    """Remove only a private temporary directory created by this process."""
    if path.parent != expected_parent or not path.name.startswith(expected_prefix):
        raise UnsafeOutputTargetError(
            "Refusing to remove a directory not identified as private staging"
        )
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise UnsafeOutputTargetError(
            "Private staging identity changed before cleanup"
        )
    shutil.rmtree(path)


def _discard_staging(path: Path, *, target: Path) -> None:
    """Best-effort cleanup that never expands the caller-selected scope."""
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return
    except OSError:
        return
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        return
    try:
        _remove_private_directory(
            path,
            expected_parent=target.parent,
            expected_prefix=f".{target.name}{STAGING_NAME_TOKEN}",
        )
    except (OSError, UnsafeOutputTargetError):
        return


def _prepare_target(
    *,
    raw_target: Path,
    input_path: Path,
    overwrite: bool,
) -> tuple[Path, OwnedRunIdentity | None]:
    """Create only missing parents and capture an owned target identity."""
    target = _safe_output_target(raw_target, input_path=input_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target = _safe_output_target(raw_target, input_path=input_path)
    if not target.exists():
        return target, None
    if not overwrite:
        raise FileExistsError(
            "Output directory already exists; pass --overwrite only to replace "
            "a marker-owned live-console run"
        )
    return target, _validate_owned_run(target)


def _revalidate_target(
    *,
    raw_target: Path,
    input_path: Path,
    expected_target: Path,
) -> Path:
    target = _safe_output_target(raw_target, input_path=input_path)
    if target != expected_target:
        raise UnsafeOutputTargetError(
            "Output target changed between validation and replacement"
        )
    return target


def _install_staged_run(
    *,
    staging: Path,
    target: Path,
    raw_target: Path,
    input_path: Path,
    previous: OwnedRunIdentity | None,
) -> None:
    """Install a validated sibling run, rolling back an owned prior run."""
    staged_identity = _validate_owned_run(staging)
    _revalidate_target(
        raw_target=raw_target,
        input_path=input_path,
        expected_target=target,
    )

    if previous is None:
        if target.exists():
            raise UnsafeOutputTargetError(
                "Output target appeared after validation; it was not replaced"
            )
        os.rename(staging, target)
        _validate_owned_run(target)
        return

    current = _validate_owned_run(target)
    if current != previous:
        raise UnsafeOutputTargetError(
            "Existing owned run changed between validation and replacement"
        )

    backup_container = _create_private_sibling(
        target,
        token=BACKUP_NAME_TOKEN,
    )
    backup = backup_container / "previous"
    failed_install = backup_container / "failed-install"
    staging_installed = False
    try:
        os.rename(target, backup)
        if _validate_owned_run(backup) != previous:
            raise UnsafeOutputTargetError(
                "Owned run identity changed while entering the replacement transaction"
            )
        if target.exists():
            raise UnsafeOutputTargetError(
                "Output target reappeared during replacement; prior run retained"
        )
        try:
            os.rename(staging, target)
            staging_installed = True
            if _validate_owned_run(target) != staged_identity:
                raise UnsafeOutputTargetError(
                    "Staged run identity changed during installation"
                )
        except Exception:
            installed_target_is_owned = False
            if staging_installed and target.exists():
                try:
                    installed_target_is_owned = (
                        _validate_owned_run(target) == staged_identity
                    )
                except UnsafeOutputTargetError:
                    installed_target_is_owned = False
            if installed_target_is_owned:
                os.rename(target, failed_install)
            if not target.exists() and backup.exists():
                os.rename(backup, target)
            if failed_install.exists():
                _remove_private_directory(
                    failed_install,
                    expected_parent=backup_container,
                    expected_prefix="failed-install",
                )
            raise

        try:
            if _validate_owned_run(backup) == previous:
                shutil.rmtree(backup)
                backup_container.rmdir()
        except (OSError, UnsafeOutputTargetError):
            # The new run is already complete and validated. Leave an
            # undeleted private backup in place rather than broadening cleanup.
            pass
    except Exception:
        if not target.exists() and backup.exists():
            os.rename(backup, target)
        if backup_container.exists():
            try:
                backup_container.rmdir()
            except OSError:
                pass
        raise


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
            "schema_version": "structvision-live-input-v1",
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
            "schema_version": "structvision-live-pipeline-v1",
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
        generated.extend(
            ("RUN_MANIFEST.json", OWNERSHIP_MARKER_NAME, "CONSOLE_LOG.txt")
        )
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
            "schema_version": RUN_MANIFEST_SCHEMA,
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
                "Every run payload file except RUN_MANIFEST.json and its linked "
                "ownership marker; a file cannot contain its own final SHA-256 "
                "without circularity."
            ),
            "files": _file_records(
                target,
                exclude=frozenset({"RUN_MANIFEST.json"}),
            ),
        }
        manifest_payload = _json_bytes(manifest)
        _atomic_write(target / "RUN_MANIFEST.json", manifest_payload)
        _atomic_write(
            target / OWNERSHIP_MARKER_NAME,
            _ownership_marker_bytes(manifest_payload),
        )
        return final_lines
    except Exception:
        raise


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    status = method_status(CLASSICAL_METHOD)
    staging: Path | None = None
    target: Path | None = None
    try:
        storage_context = OperationalStorageContext.discover(
            arguments.storage_config
        )
        input_path = (
            storage_context.authorise_private_input(arguments.input).path
            if storage_context.is_external
            else arguments.input
        )
        output_path = (
            storage_context.authorise_run_output(arguments.output_dir).path
            if storage_context.is_external
            else arguments.output_dir
        )
    except StorageConfigurationError as error:
        print(f"Storage configuration error: {error}", file=sys.stderr)
        return EXIT_STORAGE_CONFIGURATION
    input_name = input_path.name
    try:
        encoded = input_path.read_bytes()
        decoded = decode_image_bytes(
            encoded,
            filename=input_name,
            alpha_handling=arguments.alpha_handling,
        )
        target, previous = _prepare_target(
            raw_target=output_path,
            input_path=input_path,
            overwrite=arguments.overwrite,
        )
    except FileExistsError as error:
        print(f"Output protection: {error}", file=sys.stderr)
        return EXIT_OUTPUT_EXISTS
    except UnsafeOutputTargetError as error:
        print(f"Unsafe output target: {error}", file=sys.stderr)
        return EXIT_UNSAFE_TARGET
    except (OSError, ValueError, DemonstrationInputError) as error:
        print(f"Input validation error: {error}", file=sys.stderr)
        return EXIT_INPUT

    initial_lines = _console_lines(
        input_name=input_name,
        decoded=decoded,
        status=status,
    )
    try:
        staging = _create_private_sibling(
            target,
            token=STAGING_NAME_TOKEN,
        )
    except Exception as error:
        print(
            f"Output error: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return EXIT_OUTPUT

    print("\n".join(initial_lines), flush=True)
    runtime_temporary = staging / "PROCESSING" / ".runtime"
    prior_tempdir = tempfile.tempdir
    try:
        runtime_temporary.mkdir(parents=True, exist_ok=False)
        tempfile.tempdir = str(runtime_temporary)
        analysis = analyse_demonstration_image(
            decoded,
            method_id=CLASSICAL_METHOD,
        )
        timing_rows = _timings(analysis)
    except Exception as error:
        _discard_staging(staging, target=target)
        print(
            f"Execution error: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return EXIT_EXECUTION
    finally:
        tempfile.tempdir = prior_tempdir
        if runtime_temporary.is_dir():
            try:
                shutil.rmtree(runtime_temporary)
            except OSError:
                pass

    try:
        final_lines = _write_run(
            target=staging,
            input_name=input_name,
            decoded=decoded,
            analysis=analysis,
            timing_rows=timing_rows,
            initial_lines=initial_lines,
        )
        _install_staged_run(
            staging=staging,
            target=target,
            raw_target=output_path,
            input_path=input_path,
            previous=previous,
        )
    except UnsafeOutputTargetError as error:
        _discard_staging(staging, target=target)
        print(f"Unsafe output target: {error}", file=sys.stderr)
        return EXIT_UNSAFE_TARGET
    except Exception as error:
        _discard_staging(staging, target=target)
        print(
            f"Output error: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return EXIT_OUTPUT

    print("\n".join(final_lines[len(initial_lines) :]), flush=True)
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
