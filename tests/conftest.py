from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys

import pytest


LEGACY_WRITE_TEST_MODULES = frozenset(
    {
        "test_clean_checkout_isolation",
        "test_contextual_safety",
        "test_experiment_tracking",
        "test_navigation_state",
        "test_region_proposal",
        "test_registered_experiment",
        "test_specular_suppression",
    }
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_RUNTIME_ROOTS = ("outputs", "uploads", "reports")


@dataclass(frozen=True)
class LegacyArtifactSandbox:
    root: Path
    outputs: Path
    masks: Path
    feature_maps: Path
    uploads: Path
    reports: Path


class _GuardedCV2:
    def __init__(self, wrapped, authorised_root: Path) -> None:
        self._wrapped = wrapped
        self._authorised_root = authorised_root.resolve()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def imwrite(self, filename, image, *args, **kwargs):
        candidate = Path(os.fspath(filename)).resolve()
        if (
            candidate != self._authorised_root
            and self._authorised_root not in candidate.parents
        ):
            raise AssertionError(
                "legacy test artifact write escaped authorised sandbox: "
                f"{candidate}"
            )
        return self._wrapped.imwrite(
            os.fspath(filename),
            image,
            *args,
            **kwargs,
        )


def _runtime_inventory() -> tuple[tuple[object, ...], ...]:
    records: list[tuple[object, ...]] = []
    for relative_root in REPOSITORY_RUNTIME_ROOTS:
        root = REPOSITORY_ROOT / relative_root
        if not root.exists():
            records.append((relative_root, "missing"))
            continue
        for path in sorted((root, *root.rglob("*"))):
            stat = path.lstat()
            relative = path.relative_to(REPOSITORY_ROOT).as_posix()
            if path.is_symlink():
                records.append(
                    (
                        relative,
                        "symlink",
                        os.readlink(path),
                        stat.st_mode,
                        stat.st_mtime_ns,
                    )
                )
            else:
                records.append(
                    (
                        relative,
                        "directory" if path.is_dir() else "file",
                        stat.st_size,
                        stat.st_mode,
                        stat.st_mtime_ns,
                    )
                )
    return tuple(records)


def _patch_loaded_aliases(
    monkeypatch: pytest.MonkeyPatch,
    sandbox: LegacyArtifactSandbox,
) -> None:
    aliases = {
        "config": {
            "OUTPUT_DIR": sandbox.outputs,
            "MASK_DIR": sandbox.masks,
            "FEATURE_DIR": sandbox.feature_maps,
            "UPLOAD_DIR": sandbox.uploads,
            "REPORT_DIR": sandbox.reports,
        },
        "feature_extraction": {"FEATURE_DIR": sandbox.feature_maps},
        "region_proposal": {
            "OUTPUT_DIR": sandbox.outputs,
            "MASK_DIR": sandbox.masks,
        },
        "app": {
            "OUTPUT_DIR": sandbox.outputs,
            "UPLOAD_DIR": sandbox.uploads,
            "REPORT_DIR": sandbox.reports,
        },
    }
    for module_name, replacements in aliases.items():
        module = (
            importlib.import_module(module_name)
            if module_name in {"config", "feature_extraction", "region_proposal"}
            else sys.modules.get(module_name)
        )
        if module is None:
            continue
        for name, value in replacements.items():
            if hasattr(module, name):
                monkeypatch.setattr(module, name, value)


@pytest.fixture(scope="module", autouse=True)
def legacy_artifact_sandbox(request, tmp_path_factory):
    """Confine exact write-bearing legacy tests without changing production code."""
    module_name = request.module.__name__.rsplit(".", 1)[-1]
    if module_name not in LEGACY_WRITE_TEST_MODULES:
        yield None
        return

    root = tmp_path_factory.mktemp(f"{module_name}-artifacts")
    sandbox = LegacyArtifactSandbox(
        root=root,
        outputs=root / "outputs",
        masks=root / "outputs" / "masks",
        feature_maps=root / "outputs" / "feature_maps",
        uploads=root / "uploads",
        reports=root / "reports",
    )
    before = _runtime_inventory()
    patcher = pytest.MonkeyPatch()
    matplotlib_config = root / "matplotlib-config"
    matplotlib_config.mkdir()
    patcher.setenv("MPLCONFIGDIR", str(matplotlib_config))
    if module_name == "test_navigation_state":
        # Build the first-run font cache outside Streamlit's 20-second app
        # execution timeout, while keeping the cache in the test sandbox.
        importlib.import_module("matplotlib.font_manager")
    _patch_loaded_aliases(patcher, sandbox)

    region_proposal = importlib.import_module("region_proposal")
    feature_extraction = importlib.import_module("feature_extraction")
    patcher.setattr(
        region_proposal,
        "cv2",
        _GuardedCV2(region_proposal.cv2, sandbox.root),
    )
    patcher.setattr(
        feature_extraction,
        "cv2",
        _GuardedCV2(feature_extraction.cv2, sandbox.root),
    )

    try:
        yield sandbox
    finally:
        escaped = [
            path
            for path in root.rglob("*")
            if root.resolve() not in path.resolve().parents
        ]
        after = _runtime_inventory()
        patcher.undo()
        assert not escaped, f"test artifacts escaped sandbox: {escaped}"
        assert after == before, "repository runtime inventory changed during test"
