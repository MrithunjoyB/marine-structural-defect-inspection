from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import config
from feature_extraction import extract_feature_maps
import feature_extraction
from region_proposal import propose_regions
import region_proposal


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _under(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    authorised = root.resolve()
    return resolved == authorised or authorised in resolved.parents


def test_legacy_generated_artifacts_remain_under_module_sandbox(
    legacy_artifact_sandbox,
):
    sandbox = legacy_artifact_sandbox
    assert sandbox is not None
    assert region_proposal.OUTPUT_DIR == sandbox.outputs
    assert region_proposal.MASK_DIR == sandbox.masks
    assert feature_extraction.FEATURE_DIR == sandbox.feature_maps
    assert config.UPLOAD_DIR == sandbox.uploads
    assert config.REPORT_DIR == sandbox.reports

    image = np.full((96, 144, 3), 150, np.uint8)
    cv2.rectangle(image, (48, 30), (96, 66), (65, 125, 195), -1)
    result = propose_regions(
        image,
        extract_feature_maps(image),
        "clean_checkout_guard",
        min_area=20,
        max_regions=3,
    )
    generated = (
        result.combined_mask_path,
        result.overlay_path,
        *result.visualization_paths.values(),
        *result.comparison_paths.values(),
        *result.diagnostics.stage_overlay_paths.values(),
        *(
            path
            for proposal in result.proposals
            for path in (
                proposal.mask_path,
                proposal.raw_mask_path,
                proposal.context_mask_path,
            )
        ),
    )
    assert generated
    assert all(path.is_file() for path in generated)
    assert all(_under(path, sandbox.root) for path in generated)


def test_legacy_write_guard_rejects_repository_escape(
    legacy_artifact_sandbox,
):
    forbidden = REPOSITORY_ROOT / "outputs" / "forbidden-test-artifact.png"
    with pytest.raises(
        AssertionError,
        match="escaped authorised sandbox",
    ):
        region_proposal.cv2.imwrite(
            str(forbidden),
            np.zeros((4, 4), np.uint8),
        )
    assert not forbidden.exists()
