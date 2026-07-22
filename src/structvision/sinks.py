"""Explicit sink interfaces; the default API path instantiates none of them."""

from __future__ import annotations

from threading import Lock
from typing import Any, Protocol

from .errors import DuplicateResultIdentityError, SinkError


class ArtifactSink(Protocol):
    """Receives an in-memory analysis result only when explicitly injected."""

    def write(self, result: Any) -> None:
        ...


class ResultSink(Protocol):
    """Receives a complete v2 execution report only when explicitly injected."""

    def write(self, report: Any) -> None:
        ...


class NullArtifactSink:
    """Explicitly discard optional artifacts without touching the filesystem."""

    def write(self, result: Any) -> None:
        return None


class NullResultSink:
    """Explicitly discard a v2 report without touching the filesystem."""

    def write(self, report: Any) -> None:
        return None


class MemoryResultSink:
    """Append unique records to process memory for testing or orchestration."""

    def __init__(self) -> None:
        self._records: list[Any] = []
        self._identities: set[str] = set()
        self._lock = Lock()

    @property
    def records(self) -> tuple[Any, ...]:
        with self._lock:
            return tuple(self._records)

    def write(self, record: Any) -> None:
        identity = getattr(record, "identity", None)
        if identity is None:
            identity = getattr(record, "execution_attempt_id", None)
        if not isinstance(identity, str) or not identity:
            raise SinkError("MemoryResultSink records must expose a non-empty identity")
        with self._lock:
            if identity in self._identities:
                raise DuplicateResultIdentityError(f"Duplicate result identity: {identity}")
            self._identities.add(identity)
            self._records.append(record)


class V2SQLiteResultSink:
    """Explicit adapter around the append-only scientific v2 SQLite store."""

    def __init__(self, path: object) -> None:
        from scientific_contract.result_store import V2ResultStore

        self._store = V2ResultStore(path)

    @property
    def store(self) -> Any:
        return self._store

    def write(self, report: Any) -> None:
        try:
            with self._store.connect() as connection:
                present = connection.execute(
                    "SELECT 1 FROM experiment_specifications WHERE specification_hash=?",
                    (report.specification.specification_hash,),
                ).fetchone()
            if present is None:
                self._store.register_specification(report.specification)
            self._store.append_attempt(report.specification, report.summary, report.rows)
        except Exception as error:
            raise SinkError(f"Could not append v2 execution {report.execution_attempt_id}") from error
