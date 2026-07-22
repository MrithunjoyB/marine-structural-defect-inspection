"""Read-only capture of code and runtime provenance for future specifications."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import sys


@dataclass(frozen=True)
class GitState:
    commit: str
    tree_state: str
    uncommitted_diff_hash: str | None

    @property
    def clean(self) -> bool:
        return self.tree_state == "clean"


def _git(repo: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments], cwd=Path(repo), check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return completed.stdout


def capture_git_state(repo: Path) -> GitState:
    """Capture HEAD and a content hash covering tracked and untracked changes."""
    root = Path(repo).resolve()
    commit = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
    status = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    if not status:
        return GitState(commit, "clean", None)
    digest = hashlib.sha256()
    digest.update(b"status\0")
    digest.update(status)
    digest.update(b"diff-head\0")
    digest.update(_git(root, "diff", "--binary", "HEAD", "--"))
    entries = [entry for entry in status.split(b"\0") if entry]
    untracked = sorted(entry[3:].decode("utf-8", "surrogateescape") for entry in entries if entry.startswith(b"?? "))
    for relative in untracked:
        path = root / relative
        digest.update(relative.encode("utf-8", "surrogateescape"))
        digest.update(b"\0")
        if path.is_file():
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        elif path.is_symlink():
            digest.update(path.readlink().as_posix().encode("utf-8"))
    return GitState(commit, "dirty", digest.hexdigest())


@dataclass(frozen=True)
class RuntimeEnvironment:
    python_version: str
    dependencies: tuple[tuple[str, str], ...]
    operating_system: tuple[tuple[str, str], ...]
    hardware: tuple[tuple[str, str], ...]
    opencv_version: str
    opencv_backend: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def capture_runtime_environment() -> RuntimeEnvironment:
    packages = sorted(
        (str(distribution.metadata.get("Name") or distribution.metadata.get("Summary") or "unknown"), str(distribution.version))
        for distribution in importlib.metadata.distributions()
    )
    operating_system = (
        ("platform", platform.platform()),
        ("system", platform.system()),
        ("release", platform.release()),
    )
    hardware = (
        ("machine", platform.machine()),
        ("processor", platform.processor() or "unknown"),
        ("node", platform.node() or "unknown"),
    )
    try:
        import cv2
        opencv_version = cv2.__version__
        info = cv2.getBuildInformation()
        backend = next((line.strip() for line in info.splitlines() if "Parallel framework" in line), "unreported")
    except Exception:
        opencv_version, backend = "unavailable", "unavailable"
    return RuntimeEnvironment(sys.version, tuple(packages), operating_system, hardware, opencv_version, backend)
