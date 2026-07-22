"""Write-free provenance records for reusable detector results."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


PROTECTED_SOURCE_HASHES = (
    ("preprocess.py", "fcd5da2b563e420b18f5baaf6a73c276457b4b6c65b33531cfeaf917ffefcf48"),
    ("feature_extraction.py", "1ae26484de02f4d5764d2ee90ee519babe307192c12fa8deecfc50d96ff1976c"),
    ("region_proposal.py", "65815b84dd8078b11776ccb70e81688e47f4e7afe1624534d6872bec1e46f80a"),
    ("scoring.py", "d284c8012464003a0ddc5a697c4d85303fbe73a356f8ee7f649c5d75ebcd3a79"),
)


@dataclass(frozen=True)
class ProvenanceRecord:
    adapter_id: str
    source_type: str
    source_content_hash: str
    protected_source_hashes: tuple[tuple[str, str], ...]
    protected_hashes_verified: bool
    numpy_version: str
    opencv_version: str
    temporary_artifact_count: int
    side_effect_boundary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "adapter_id": self.adapter_id,
            "source_type": self.source_type,
            "source_content_hash": self.source_content_hash,
            "protected_source_hashes": [list(item) for item in self.protected_source_hashes],
            "protected_hashes_verified": self.protected_hashes_verified,
            "numpy_version": self.numpy_version,
            "opencv_version": self.opencv_version,
            "temporary_artifact_count": self.temporary_artifact_count,
            "side_effect_boundary": self.side_effect_boundary,
        }


def hash_module_sources(modules: Iterable[object]) -> tuple[tuple[str, str], ...]:
    values = []
    for module in modules:
        path_value = getattr(module, "__file__", None)
        if not path_value:
            continue
        path = Path(path_value)
        if path.suffix == ".pyc" and path.with_suffix(".py").exists():
            path = path.with_suffix(".py")
        values.append((path.name, hashlib.sha256(path.read_bytes()).hexdigest()))
    return tuple(sorted(values))


def build_provenance(
    *, source_type: str, source_content_hash: str,
    actual_source_hashes: tuple[tuple[str, str], ...], temporary_artifact_count: int,
) -> ProvenanceRecord:
    expected = dict(PROTECTED_SOURCE_HASHES)
    actual = dict(actual_source_hashes)
    verified = all(actual.get(name) == digest for name, digest in expected.items())
    return ProvenanceRecord(
        adapter_id="legacy-compatibility-adapter-v1",
        source_type=source_type,
        source_content_hash=source_content_hash,
        protected_source_hashes=actual_source_hashes,
        protected_hashes_verified=verified,
        numpy_version=np.__version__,
        opencv_version=cv2.__version__,
        temporary_artifact_count=temporary_artifact_count,
        side_effect_boundary=(
            "Frozen legacy artifacts are redirected to one controlled TemporaryDirectory, "
            "read back for parity, and deleted before analyse returns."
        ),
    )
