"""Stable, write-free public API for the frozen classical detector."""

from __future__ import annotations

from dataclasses import replace
import time
from typing import Iterable, Mapping

from .classical import run_frozen_classical
from .configuration import DetectorConfig
from .errors import DuplicateImageIDError, SinkError
from .inputs import normalise_input
from .sinks import ArtifactSink
from .types import AnalysisResult, AnalysisSample, BatchAnalysisResult, BatchFailure


class StructuralAnomalyDetector:
    """Reusable detector with no database, UI, API-key, or output-directory dependency."""

    def __init__(self, config: DetectorConfig) -> None:
        if not isinstance(config, DetectorConfig):
            raise TypeError("config must be DetectorConfig")
        self._config = config

    @property
    def config(self) -> DetectorConfig:
        return self._config

    def analyse(
        self,
        image: object,
        *,
        image_id: str,
        colour_space: str | None = None,
        alpha_handling: str | None = None,
        metadata: Mapping[str, object] | None = None,
        artifact_sink: ArtifactSink | None = None,
    ) -> AnalysisResult:
        if not isinstance(image_id, str) or not image_id.strip():
            raise ValueError("image_id must be a non-empty string")
        normalisation_started = time.perf_counter()
        normalised = normalise_input(image, colour_space=colour_space, alpha_handling=alpha_handling)
        normalisation_elapsed = time.perf_counter() - normalisation_started
        result = run_frozen_classical(
            normalised.image_bgr,
            image_id=image_id,
            input_hash=normalised.input_hash,
            source_hash=normalised.source_hash,
            source_type=normalised.source_type,
            metadata=metadata,
            config=self._config,
        )
        core_total = dict(result.timing_breakdown_seconds)["core_total"]
        timings = (("input_normalisation", normalisation_elapsed),) + result.timing_breakdown_seconds + (
            ("total", normalisation_elapsed + core_total),
        )
        result = replace(result, timing_breakdown_seconds=timings)
        if artifact_sink is not None:
            try:
                artifact_sink.write(result)
            except Exception as error:
                raise SinkError(f"Artifact sink failed for image {image_id}") from error
        return result

    def analyse_batch(
        self,
        samples: Iterable[AnalysisSample | Mapping[str, object]],
        *,
        fail_fast: bool = True,
        worker_count: int = 1,
        artifact_sink: ArtifactSink | None = None,
    ) -> BatchAnalysisResult:
        if type(fail_fast) is not bool:
            raise TypeError("fail_fast must be boolean")
        if worker_count != 1:
            raise ValueError("The frozen adapter supports explicit worker_count=1 only")
        prepared = []
        for item in samples:
            if isinstance(item, AnalysisSample):
                prepared.append(item)
            elif isinstance(item, Mapping):
                prepared.append(AnalysisSample(**dict(item)))
            else:
                raise TypeError("Batch items must be AnalysisSample or equivalent mappings")
        identifiers = [item.image_id for item in prepared]
        duplicates = sorted({identifier for identifier in identifiers if identifiers.count(identifier) > 1})
        if duplicates:
            raise DuplicateImageIDError(f"Duplicate image IDs: {duplicates}")
        results = []
        failures = []
        identities = set()
        for index, sample in enumerate(prepared):
            try:
                result = self.analyse(
                    sample.image,
                    image_id=sample.image_id,
                    colour_space=sample.colour_space,
                    alpha_handling=sample.alpha_handling,
                    metadata=sample.metadata,
                    artifact_sink=artifact_sink,
                )
                if result.identity in identities:
                    raise DuplicateImageIDError(f"Duplicate result identity for {sample.image_id}")
                identities.add(result.identity)
                results.append(result)
            except Exception as error:
                if fail_fast:
                    raise
                failures.append(BatchFailure(index, sample.image_id, type(error).__name__, str(error)))
        return BatchAnalysisResult(len(prepared), tuple(results), tuple(failures), fail_fast, worker_count)
