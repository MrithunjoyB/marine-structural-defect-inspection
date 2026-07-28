#!/usr/bin/env python3
"""Build and verify the clean offline StructVision-AI professor handoff.

The source ZIP is produced by ``git archive HEAD``. Runtime folders, ignored
data, environments, learned artifacts, and historical stores are never copied
recursively into the handoff.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Iterable
import zipfile

from PIL import Image, PngImagePlugin

from structvision import __version__, demonstration_fixture


BUNDLE_NAME = "StructVision-AI-Professor-Handoff"
CHECKSUM_FILE = "CHECKSUMS.sha256"
VERSION_FILE = "VERSION.txt"
SOURCE_ARCHIVE_PREFIX = "structvision-source-"
FIXTURE_LABEL = "thin structural indication"
REQUIRED_RELATIVE_FILES = frozenset(
    {
        "README_FIRST.html",
        "README_FIRST.md",
        VERSION_FILE,
        CHECKSUM_FILE,
        "RUN_DEMO.sh",
        "INPUT/demonstration-fixture.png",
        "INPUT/YOUR_IMAGE_HERE.txt",
        "EXAMPLE_OUTPUT/RUN_MANIFEST.json",
        "EXAMPLE_OUTPUT/CONSOLE_LOG.txt",
        "DOCUMENTATION/block-diagram.svg",
        "DOCUMENTATION/block-diagram.html",
        "DOCUMENTATION/professor-console-handoff.html",
        "DOCUMENTATION/algorithm-specification.md",
        "DOCUMENTATION/algorithm-pseudocode.md",
        "DOCUMENTATION/code-structure-guide.md",
        "DOCUMENTATION/professor-handoff.md",
        "SOURCE/source-manifest.txt",
        "INSTALLATION/macOS.md",
        "INSTALLATION/Linux.md",
        "INSTALLATION/requirements-note.md",
    }
)
PROHIBITED_PARTS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        "node_modules",
    }
)
PROHIBITED_NAMES = frozenset(
    {
        ".DS_Store",
        ".env",
        "Thumbs.db",
        "desktop.ini",
    }
)
PROHIBITED_SUFFIXES = frozenset(
    {
        ".db",
        ".sqlite",
        ".sqlite3",
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".onnx",
        ".npz",
        ".npy",
        ".pkl",
        ".pickle",
    }
)
LOCAL_PATH_PATTERNS = (
    re.compile(re.escape(b"/" + b"Users" + b"/") + rb"[^/\s]+/"),
    re.compile(re.escape(b"/" + b"home" + b"/") + rb"[^/\s]+/"),
    re.compile(rb"[A-Za-z]:\\Users\\[^\\\s]+\\"),
)
TEXT_SUFFIXES = frozenset(
    {
        ".cfg",
        ".csv",
        ".html",
        ".ini",
        ".json",
        ".md",
        ".py",
        ".sh",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)


class HandoffError(RuntimeError):
    """The handoff could not be built or verified safely."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a small committed-source professor handoff or verify an "
            "existing handoff without network access."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="new handoff directory")
    mode.add_argument("--verify", type=Path, help="existing handoff directory")
    return parser


def _run(
    arguments: list[str],
    *,
    cwd: Path,
    capture: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            arguments,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = ""
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            detail = ": " + error.stderr.decode("utf-8", errors="replace").strip()
        raise HandoffError(f"Command failed: {' '.join(arguments)}{detail}") from error


def _git_text(root: Path, *arguments: str) -> str:
    result = _run(["git", *arguments], cwd=root)
    return result.stdout.decode("utf-8").strip()


def _repository_root() -> Path:
    script = Path(__file__).resolve()
    root_text = _git_text(script.parent, "rev-parse", "--show-toplevel")
    root = Path(root_text).resolve()
    try:
        script.relative_to(root)
    except ValueError as error:
        raise HandoffError("Builder is not inside the selected Git repository") from error
    return root


def _sha256_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, payload: bytes | str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    path.write_bytes(data)
    path.chmod(0o755 if executable else 0o644)


def _copy_text(source: Path, target: Path, *, replace: dict[str, str] | None = None) -> None:
    text = source.read_text(encoding="utf-8")
    for before, after in (replace or {}).items():
        text = text.replace(before, after)
    _write(target, text)


def _safe_target(path: Path, repo_root: Path) -> Path:
    target = path.expanduser().resolve(strict=False)
    if target == Path(target.anchor) or target == Path.home().resolve():
        raise HandoffError("Select a dedicated child directory for the handoff")
    try:
        target.relative_to(repo_root)
    except ValueError:
        pass
    else:
        raise HandoffError(
            "The generated handoff must be outside the active Git repository"
        )
    if target.exists():
        raise HandoffError(
            "Output already exists; choose a new path so no handoff is overwritten"
        )
    return target


def _build_timestamp() -> str:
    source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if source_date_epoch is not None:
        try:
            value = int(source_date_epoch)
        except ValueError as error:
            raise HandoffError("SOURCE_DATE_EPOCH must be an integer") from error
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


def _platform_text() -> str:
    return (
        f"{platform.system()} {platform.release()} "
        f"{platform.machine()} / Python {platform.python_version()}"
    )


def _fixture_png() -> bytes:
    fixture = demonstration_fixture(FIXTURE_LABEL)
    metadata = PngImagePlugin.PngInfo()
    values = {
        "artifact_role": "synthetic professor demonstration fixture",
        "fixture_label": FIXTURE_LABEL,
        "research_cohort_status": "excluded from every research cohort",
        "evidence_status": "not real inspection evidence",
        "generator": "structvision.demonstration_fixture",
        "fixture_pixel_identity": fixture.encoded_sha256,
    }
    for name, value in sorted(values.items()):
        metadata.add_text(f"structvision.{name}", value)
    buffer = BytesIO()
    Image.fromarray(fixture.image_bgr[:, :, ::-1]).save(
        buffer,
        format="PNG",
        pnginfo=metadata,
        optimize=False,
        compress_level=6,
    )
    return buffer.getvalue()


def _readme_markdown(commit: str) -> str:
    return f"""# StructVision-AI Professor Handoff

Open `README_FIRST.html` for the offline visual guide.

This package records source commit `{commit}` and package version `{__version__}`.
It contains one deterministic synthetic demonstration fixture and one completed
example run of `structvision-classical-baseline-v1-frozen`.

> Review proposals only — not confirmed defects or engineering diagnosis.

## Four distinct uses

1. **Live demonstration on Mrithunjoy's prepared Mac** — activate the prepared
   environment and run `RUN_DEMO.sh`, or use `structvision-professor-demo`
   directly. This is the reliable presentation route.
2. **Source/code review from this drive** — inspect `SOURCE/` and
   `DOCUMENTATION/`; verify hashes first.
3. **Installation on another machine** — extract the source ZIP, create a new
   environment, and install compatible dependencies using `INSTALLATION/`.
   This bundle is not universally executable without platform installation.
4. **Optional future macOS-arm64 offline wheelhouse** — not included. It should
   be a separately licensed, size-audited, Python/OS/architecture-specific
   artifact.

## Verify before use

From the matching checked-out repository:

```bash
python scripts/build_professor_handoff.py \
  --verify "/path/to/StructVision-AI-Professor-Handoff"
```

Expected: `Verification passed`, the commit identity, file count, and bundle
size. If verification fails, stop using the copy and recopy it from a verified
source.

## Live prepared-Mac command

```bash
source "/path/to/marine-structural-defect-inspection/venv/bin/activate"
STRUCTVISION_DEMO_OUTPUT_BASE="/path/to/presentation-runs" \
  ./RUN_DEMO.sh
```

`RUN_DEMO.sh` never installs packages, downloads a model, or uses the network.
Set the output base outside this verified bundle so demonstration runs do not
alter the handoff.

## Safe-copy checklist

- Run the verifier before copying.
- Copy the one handoff directory, not the working repository or virtual
  environment.
- Eject the drive safely and verify the copied directory again.
- Confirm no private/professor image was substituted into `INPUT/`.
- Keep temporary non-fixture runs outside the immutable handoff.

## Retention

The included fixture is synthetic. Delete temporary runs containing
caller-authorised images after the meeting according to the agreed retention
policy, including trash where required. Do not delete source history or reviewed
scientific stores as part of presentation cleanup.
"""


def _readme_html(commit: str) -> str:
    short = commit[:12]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>StructVision-AI Professor Handoff — Read First</title>
  <style>
    *{{box-sizing:border-box}} body{{margin:0;background:#f2f6f6;color:#173941;font:16px/1.55 Arial,Helvetica,sans-serif}}
    main{{width:min(980px,calc(100% - 32px));margin:36px auto}} header,section{{background:#fff;border:1px solid #ccd8da;border-radius:14px;padding:26px;margin:0 0 18px}}
    header{{border-top:7px solid #216577}} h1,h2{{margin-top:0}} code{{background:#eaf1f2;padding:.12rem .35rem;border-radius:4px}}
    .warning{{border-left:5px solid #963c32;padding:12px 16px;background:#fff0ee}} .modes{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}
    .mode{{border:1px solid #ccd8da;border-radius:9px;padding:14px}} a{{color:#216577}} pre{{overflow:auto;background:#15343c;color:#eef7f8;padding:14px;border-radius:8px}}
    @media(max-width:650px){{.modes{{grid-template-columns:1fr}}}}
  </style>
</head>
<body><main>
  <header><p>OFFLINE ENGINEERING/RESEARCH HANDOFF</p><h1>StructVision-AI — Read First</h1>
    <p>Recorded source commit <code>{short}</code>; package <code>{__version__}</code>.</p>
    <p class="warning"><strong>Review proposals only — not confirmed defects or engineering diagnosis.</strong></p>
  </header>
  <section><h2>Choose the intended use</h2><div class="modes">
    <div class="mode"><strong>1. Prepared-Mac live demo</strong><p>Activate the prepared environment and run <code>RUN_DEMO.sh</code>. This is the dependable presentation route.</p></div>
    <div class="mode"><strong>2. Drive source review</strong><p>Read <code>DOCUMENTATION/</code> and inspect the committed-source ZIP and manifest in <code>SOURCE/</code>.</p></div>
    <div class="mode"><strong>3. Another machine</strong><p>Follow <code>INSTALLATION/</code>. Compatible dependencies must be installed; no virtual environment is bundled.</p></div>
    <div class="mode"><strong>4. Future wheelhouse</strong><p>A macOS-arm64 offline wheelhouse is not included and requires a separate licence, size, Python, OS, and architecture audit.</p></div>
  </div></section>
  <section><h2>Verify first</h2><pre><code>python scripts/build_professor_handoff.py \\
  --verify "/path/to/StructVision-AI-Professor-Handoff"</code></pre>
    <p>Stop if verification reports a changed, missing, unexpected, prohibited, or wrong-commit file.</p></section>
  <section><h2>Start here</h2><ul>
    <li><a href="DOCUMENTATION/professor-console-handoff.html">Open the complete offline professor guide</a></li>
    <li><a href="DOCUMENTATION/block-diagram.html">Open the block diagram</a></li>
    <li>Review the completed run in <code>EXAMPLE_OUTPUT/</code></li>
    <li>Read <code>VERSION.txt</code>, <code>CHECKSUMS.sha256</code>, and <code>SOURCE/source-manifest.txt</code></li>
  </ul></section>
</main></body></html>
"""


def _run_demo_script() -> str:
    return """#!/bin/sh
set -eu

bundle_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
fixture="$bundle_dir/INPUT/demonstration-fixture.png"
output_base=${STRUCTVISION_DEMO_OUTPUT_BASE:-"$PWD/StructVision-Demo-Runs"}

command -v structvision-professor-demo >/dev/null 2>&1 || {
  echo "structvision-professor-demo is not installed in the active environment." >&2
  echo "Use Mrithunjoy's prepared Mac environment or follow INSTALLATION/." >&2
  exit 4
}

mkdir -p -- "$output_base"
timestamp=$(date -u '+%Y%m%dT%H%M%SZ')
output_dir="$output_base/demo-run-$timestamp"
structvision-professor-demo --input "$fixture" --output-dir "$output_dir"
echo
echo "Completed output folder: $output_dir"
"""


def _input_note() -> str:
    return """Place no private image in the immutable verified handoff.

For an authorised local demonstration, copy a PNG/JPEG/TIFF to a separate
working folder and pass its path to `structvision-professor-demo --input`.

For alpha-bearing input, explicitly select one:
  --alpha-handling drop
  --alpha-handling composite_black
  --alpha-handling composite_white

The included `demonstration-fixture.png` is synthetic, excluded from every
research cohort, and is not real inspection evidence.
"""


def _installation_macos() -> str:
    return """# macOS installation

This is source installation, not a universal offline binary.

```bash
unzip SOURCE/structvision-source-<commit>.zip -d StructVision-source
cd StructVision-source
python3 -m venv venv
source venv/bin/activate
python -m pip install .
structvision-professor-demo --help
```

Dependency resolution may require network access on a new machine. The
installed classical command itself is local/offline and requires no API key.
Use a Python release compatible with `pyproject.toml` and platform-compatible
NumPy, Pillow, and OpenCV wheels.

No wheelhouse is included. A future macOS-arm64 wheelhouse must be built and
licensed independently for an exact Python/macOS/architecture target.
"""


def _installation_linux() -> str:
    return """# Linux installation

This bundle includes source, not a Linux container or prebuilt environment.

```bash
unzip SOURCE/structvision-source-<commit>.zip -d StructVision-source
cd StructVision-source
python3 -m venv venv
. venv/bin/activate
python -m pip install .
structvision-professor-demo --help
```

Use platform-compatible Python and OpenCV system/runtime dependencies. Initial
dependency installation may require network access. The installed frozen
classical analysis is local and needs no API key, model weight, or cloud
service. A Linux container is an independent future deliverable.
"""


def _requirements_note() -> str:
    return """# Requirements note

- Package version: recorded in `../VERSION.txt`.
- Base dependencies: authoritative declarations are in the source
  `pyproject.toml`.
- No virtual environment, package cache, model weight, learned memory bank,
  database, private image, or professor data is included.
- The prepared Mac and a newly installed machine are different operating
  contexts. Source availability does not guarantee immediate execution.
- PatchCore and hybrid require a separate exact Python 3.12 learned environment
  and immutable local artifacts; neither is needed or recommended for the live
  handoff.
- No paid API, API key, automatic installer, automatic download, or network
  service is used by the live classical command.
"""


def _create_source_archive(root: Path, target: Path, commit: str) -> None:
    # The archive contains only paths tracked by the exact commit. No recursive
    # copy of the working tree is used.
    _run(
        [
            "git",
            "archive",
            "--format=zip",
            f"--output={target}",
            "HEAD",
        ],
        cwd=root,
    )
    if not target.is_file() or not zipfile.is_zipfile(target):
        raise HandoffError("git archive did not produce a readable source ZIP")
    if _git_text(root, "rev-parse", "HEAD") != commit:
        raise HandoffError("Git HEAD changed while the source archive was built")


def _zip_file_records(archive: Path) -> list[tuple[str, int, str]]:
    records = []
    with zipfile.ZipFile(archive) as handle:
        bad = handle.testzip()
        if bad is not None:
            raise HandoffError(f"Source ZIP CRC validation failed at {bad}")
        for info in sorted(handle.infolist(), key=lambda item: item.filename):
            if info.is_dir():
                continue
            payload = handle.read(info.filename)
            records.append((info.filename, len(payload), _sha256_bytes(payload)))
    return records


def _source_manifest(archive: Path, commit: str, branch: str) -> str:
    rows = [
        "StructVision-AI committed-source manifest",
        "==========================================",
        f"source_commit={commit}",
        f"source_branch={branch}",
        "archive_method=git archive --format=zip --output=<target> HEAD",
        f"archive_file={archive.name}",
        f"archive_size_bytes={archive.stat().st_size}",
        f"archive_sha256={_sha256_file(archive)}",
        "working_tree_recursively_copied=false",
        "ignored_runtime_content_included=false",
        "",
        "EXCLUSION AUDIT: PASS",
        "Excluded by construction: .git; virtual environments; Python/test caches;",
        "database files; historical result stores; model weights; learned memories;",
        "learned caches; ignored synthetic bulk images; private images; professor data;",
        "OS metadata; unrelated logs; unrelated projects.",
        "",
        "ARCHIVE MEMBERS (sorted)",
        "sha256  size_bytes  path",
    ]
    rows.extend(
        f"{digest}  {size}  {name}"
        for name, size, digest in _zip_file_records(archive)
    )
    return "\n".join(rows) + "\n"


def _run_example(root: Path, fixture: Path, output: Path) -> None:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "structvision.professor_console",
            "--input",
            str(fixture),
            "--output-dir",
            str(output),
        ],
        cwd=root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise HandoffError(f"Deterministic example run failed: {message}")
    transcript = (output / "CONSOLE_LOG.txt").read_bytes()
    if transcript != result.stdout:
        raise HandoffError("Example console transcript differs from captured stdout")


def _iter_files(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def _checksum_text(root: Path) -> str:
    rows = []
    for path in _iter_files(root):
        relative = path.relative_to(root).as_posix()
        if relative == CHECKSUM_FILE:
            continue
        rows.append(f"{_sha256_file(path)}  {relative}")
    return "\n".join(rows) + "\n"


def _bundle_size(root: Path) -> int:
    return sum(path.stat().st_size for path in _iter_files(root))


def _version_text(
    *,
    commit: str,
    branch: str,
    clean: bool,
    timestamp: str,
    file_count: int,
    bundle_size: int,
) -> str:
    return (
        "StructVision-AI Professor Handoff\n"
        f"package_version={__version__}\n"
        f"git_commit={commit}\n"
        f"git_branch={branch}\n"
        f"working_tree_state={'clean' if clean else 'dirty'}\n"
        f"build_timestamp_utc={timestamp}\n"
        f"build_platform={_platform_text()}\n"
        "source_archive_method=git archive HEAD\n"
        "exclusion_audit=PASS\n"
        f"bundle_file_count={file_count:020d}\n"
        f"bundle_size_bytes={bundle_size:020d}\n"
        "checksum_scope=every payload file except CHECKSUMS.sha256 itself\n"
        "live_method=structvision-classical-baseline-v1-frozen\n"
        "live_method_status=stable frozen baseline\n"
        "hybrid_status=rejected development candidate\n"
    )


def _finalise_integrity(
    root: Path,
    *,
    commit: str,
    branch: str,
    clean: bool,
    timestamp: str,
) -> tuple[int, int]:
    _write(
        root / VERSION_FILE,
        _version_text(
            commit=commit,
            branch=branch,
            clean=clean,
            timestamp=timestamp,
            file_count=0,
            bundle_size=0,
        ),
    )
    _write(root / CHECKSUM_FILE, _checksum_text(root))
    file_count = len(_iter_files(root))
    size = _bundle_size(root)
    _write(
        root / VERSION_FILE,
        _version_text(
            commit=commit,
            branch=branch,
            clean=clean,
            timestamp=timestamp,
            file_count=file_count,
            bundle_size=size,
        ),
    )
    _write(root / CHECKSUM_FILE, _checksum_text(root))
    final_size = _bundle_size(root)
    if final_size != size:
        raise HandoffError("Fixed-width integrity metadata did not stabilise")
    return file_count, final_size


def _is_prohibited(path: PurePosixPath) -> str | None:
    if any(part in PROHIBITED_PARTS for part in path.parts):
        return "prohibited directory"
    if path.name in PROHIBITED_NAMES:
        return "OS/private metadata"
    if path.suffix.lower() in PROHIBITED_SUFFIXES:
        return "database/model/cache artifact"
    if path.name.endswith("~") or path.name.startswith("._"):
        return "temporary/OS metadata"
    return None


def _scan_local_paths(payload: bytes) -> bool:
    return any(pattern.search(payload) for pattern in LOCAL_PATH_PATTERNS)


def _audit_tree(root: Path) -> list[str]:
    issues = []
    for path in _iter_files(root):
        relative = PurePosixPath(path.relative_to(root).as_posix())
        reason = _is_prohibited(relative)
        if reason:
            issues.append(f"{relative}: {reason}")
        if path.suffix.lower() in TEXT_SUFFIXES and _scan_local_paths(path.read_bytes()):
            issues.append(f"{relative}: absolute user-home path")
    archives = tuple((root / "SOURCE").glob(f"{SOURCE_ARCHIVE_PREFIX}*.zip"))
    if len(archives) != 1:
        issues.append("SOURCE must contain exactly one commit-named source ZIP")
        return issues
    try:
        with zipfile.ZipFile(archives[0]) as handle:
            bad = handle.testzip()
            if bad is not None:
                issues.append(f"source ZIP CRC error: {bad}")
            for info in handle.infolist():
                if info.is_dir():
                    continue
                member = PurePosixPath(info.filename)
                reason = _is_prohibited(member)
                if reason:
                    issues.append(f"source ZIP {member}: {reason}")
                if (
                    member.suffix.lower() in TEXT_SUFFIXES
                    and _scan_local_paths(handle.read(info.filename))
                ):
                    issues.append(
                        f"source ZIP {member}: absolute user-home path"
                    )
    except (OSError, zipfile.BadZipFile) as error:
        issues.append(f"source ZIP is unreadable: {error}")
    return issues


def _parse_version(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def _parse_checksums(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line:
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as error:
            raise HandoffError(
                f"Malformed checksum line {number}"
            ) from error
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not relative
            or relative in records
        ):
            raise HandoffError(f"Invalid checksum record at line {number}")
        records[relative] = digest
    return records


def verify_handoff(root: Path, *, expected_commit: str) -> tuple[int, int]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise HandoffError("Verification target is not a directory")
    actual_paths = {
        path.relative_to(root).as_posix() for path in _iter_files(root)
    }
    missing_required = sorted(REQUIRED_RELATIVE_FILES - actual_paths)
    if missing_required:
        raise HandoffError(
            "Missing required files: " + ", ".join(missing_required)
        )
    checksums = _parse_checksums(root / CHECKSUM_FILE)
    expected_paths = set(checksums) | {CHECKSUM_FILE}
    missing = sorted(expected_paths - actual_paths)
    unexpected = sorted(actual_paths - expected_paths)
    if missing:
        raise HandoffError("Missing checksummed files: " + ", ".join(missing))
    if unexpected:
        raise HandoffError("Unexpected files: " + ", ".join(unexpected))
    mismatches = [
        relative
        for relative, expected in sorted(checksums.items())
        if _sha256_file(root / relative) != expected
    ]
    if mismatches:
        raise HandoffError(
            "Checksum mismatch: " + ", ".join(mismatches)
        )
    audit_issues = _audit_tree(root)
    if audit_issues:
        raise HandoffError("Exclusion audit failed: " + "; ".join(audit_issues))
    version = _parse_version(root / VERSION_FILE)
    if version.get("git_commit") != expected_commit:
        raise HandoffError(
            "Wrong source commit: "
            f"bundle={version.get('git_commit', 'missing')} "
            f"expected={expected_commit}"
        )
    archives = tuple(
        (root / "SOURCE").glob(
            f"{SOURCE_ARCHIVE_PREFIX}{expected_commit}.zip"
        )
    )
    if len(archives) != 1:
        raise HandoffError("Source archive filename does not match the commit")
    source_manifest = (root / "SOURCE" / "source-manifest.txt").read_text(
        encoding="utf-8"
    )
    if (
        f"source_commit={expected_commit}\n" not in source_manifest
        or f"archive_sha256={_sha256_file(archives[0])}\n"
        not in source_manifest
    ):
        raise HandoffError("Source manifest commit/archive identity mismatch")
    file_count = len(actual_paths)
    size = _bundle_size(root)
    try:
        recorded_count = int(version.get("bundle_file_count", ""))
        recorded_size = int(version.get("bundle_size_bytes", ""))
    except ValueError as error:
        raise HandoffError("Invalid bundle count/size metadata") from error
    if recorded_count != file_count or recorded_size != size:
        raise HandoffError(
            "Bundle file count or size differs from VERSION.txt"
        )
    return file_count, size


def build_handoff(output: Path) -> tuple[Path, str, int, int]:
    root = _repository_root()
    commit = _git_text(root, "rev-parse", "HEAD")
    branch = _git_text(root, "branch", "--show-current")
    status = _git_text(root, "status", "--porcelain=v1", "--untracked-files=all")
    clean = not status
    if not clean:
        raise HandoffError(
            "Working tree must be clean so every handoff document and source "
            "file belongs to the recorded commit"
        )
    target = _safe_target(output, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=".structvision-professor-handoff-",
            dir=target.parent,
        )
    )
    bundle = temporary / BUNDLE_NAME
    bundle.mkdir()
    try:
        _write(bundle / "README_FIRST.md", _readme_markdown(commit))
        _write(bundle / "README_FIRST.html", _readme_html(commit))
        _write(bundle / "RUN_DEMO.sh", _run_demo_script(), executable=True)
        fixture_path = bundle / "INPUT" / "demonstration-fixture.png"
        _write(fixture_path, _fixture_png())
        _write(bundle / "INPUT" / "YOUR_IMAGE_HERE.txt", _input_note())

        documentation = {
            "professor-console-block-diagram.svg": "block-diagram.svg",
            "professor-console-block-diagram.html": "block-diagram.html",
            "professor-console-handoff.html": "professor-console-handoff.html",
            "algorithm-specification.md": "algorithm-specification.md",
            "algorithm-pseudocode.md": "algorithm-pseudocode.md",
            "code-structure-guide.md": "code-structure-guide.md",
            "professor-handoff.md": "professor-handoff.md",
        }
        for source_name, target_name in documentation.items():
            replacements = {
                "professor-console-block-diagram.svg": "block-diagram.svg"
            }
            _copy_text(
                root / "docs" / source_name,
                bundle / "DOCUMENTATION" / target_name,
                replace=replacements,
            )

        archive = (
            bundle
            / "SOURCE"
            / f"{SOURCE_ARCHIVE_PREFIX}{commit}.zip"
        )
        archive.parent.mkdir(parents=True, exist_ok=True)
        _create_source_archive(root, archive, commit)
        _write(
            bundle / "SOURCE" / "source-manifest.txt",
            _source_manifest(archive, commit, branch),
        )

        _write(bundle / "INSTALLATION" / "macOS.md", _installation_macos())
        _write(bundle / "INSTALLATION" / "Linux.md", _installation_linux())
        _write(
            bundle / "INSTALLATION" / "requirements-note.md",
            _requirements_note(),
        )

        _run_example(root, fixture_path, bundle / "EXAMPLE_OUTPUT")
        audit_issues = _audit_tree(bundle)
        if audit_issues:
            raise HandoffError(
                "Pre-check exclusion audit failed: " + "; ".join(audit_issues)
            )
        timestamp = _build_timestamp()
        file_count, size = _finalise_integrity(
            bundle,
            commit=commit,
            branch=branch,
            clean=clean,
            timestamp=timestamp,
        )
        if bundle.name != target.name:
            # The caller may use a different explicit directory name.
            pass
        os.replace(bundle, target)
        temporary.rmdir()
        verified_count, verified_size = verify_handoff(
            target,
            expected_commit=commit,
        )
        if (verified_count, verified_size) != (file_count, size):
            raise HandoffError("Post-build verification count/size mismatch")
        return target, commit, file_count, size
    except Exception:
        if target.exists():
            shutil.rmtree(target)
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        root = _repository_root()
        if arguments.verify is not None:
            commit = _git_text(root, "rev-parse", "HEAD")
            count, size = verify_handoff(
                arguments.verify,
                expected_commit=commit,
            )
            print("Verification passed")
            print(f"Git commit : {commit}")
            print(f"File count : {count}")
            print(f"Bundle size: {size} bytes")
            return 0
        target, commit, count, size = build_handoff(arguments.output)
        print("Professor handoff built and verified")
        print(f"Output     : {target}")
        print(f"Git commit : {commit}")
        print(f"File count : {count}")
        print(f"Bundle size: {size} bytes")
        return 0
    except HandoffError as error:
        print(f"Handoff error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
