from __future__ import annotations

import os
from pathlib import Path

import pytest


PROTECTED_TEST_ROOT_ENV = "STRUCTVISION_PROTECTED_TEST_ROOT"


def protected_test_root(default_root: Path) -> Path:
    """Return an explicitly selected authorised integration root, if provided."""
    configured = os.environ.get(PROTECTED_TEST_ROOT_ENV)
    if not configured:
        return default_root.resolve()
    candidate = Path(configured).expanduser()
    if not candidate.is_absolute():
        pytest.fail(f"{PROTECTED_TEST_ROOT_ENV} must be an absolute path")
    return candidate.resolve()


def require_protected_files(
    default_root: Path,
    *repository_relative_paths: str,
) -> Path:
    """Skip precisely when an optional protected integration input is absent."""
    root = protected_test_root(default_root)
    missing = [
        relative
        for relative in repository_relative_paths
        if not (root / relative).is_file()
    ]
    if missing:
        pytest.skip(
            "optional protected store unavailable: " + ", ".join(missing)
        )
    return root
