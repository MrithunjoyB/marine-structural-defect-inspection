"""Explicitly cache and verify the one official PatchCore backbone weight."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from structvision.normal_feature.configuration import (
    WEIGHT_FILENAME,
    WEIGHT_LICENCE,
    WEIGHT_MODEL_ID,
    WEIGHT_REVISION,
    WEIGHT_SHA256,
    WEIGHT_SOURCE,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-directory", type=Path,
        default=Path("outputs/normal-feature-cache/huggingface/hub"),
    )
    arguments = parser.parse_args()
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(
        repo_id=WEIGHT_MODEL_ID,
        filename=WEIGHT_FILENAME,
        revision=WEIGHT_REVISION,
        cache_dir=arguments.cache_directory,
    )).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != WEIGHT_SHA256:
        raise RuntimeError("Official pretrained-weight SHA-256 mismatch; refusing use")
    print(json.dumps({
        "source": WEIGHT_SOURCE,
        "model_id": WEIGHT_MODEL_ID,
        "revision": WEIGHT_REVISION,
        "filename": WEIGHT_FILENAME,
        "sha256": digest,
        "licence": WEIGHT_LICENCE,
        "local_path": path.as_posix(),
        "api_key_required": False,
    }, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
