"""Append-only SQLite storage for prospective v2 experiment evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from .hashing import canonical_json, is_sha256, sha256_json
from .specification import ExperimentSpecificationV2


RESULT_SCHEMA_VERSION = "scientific-result-v2"
DATABASE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ResultRowV2:
    result_id: str
    experiment_specification_hash: str
    executed_configuration_hash: str
    method_implementation_id: str
    method_implementation_version: str
    evaluation_policy_id: str
    evaluation_policy_version: int
    evaluation_policy_hash: str
    image_id: str
    image_content_hash: str
    ground_truth_content_hash: str
    proposal_output_artifact_hash: str | None
    proposal_output_details_json: str | None
    matching_policy_hash: str
    result_schema_version: str
    execution_attempt_id: str
    deterministic_mode: bool
    recorded_timestamp: str
    completion_status: str
    matching_details_json: str
    metrics_json: str

    def __post_init__(self) -> None:
        if not self.result_id.strip() or not self.execution_attempt_id.strip():
            raise ValueError("Result and attempt identifiers are required")
        hashes = (
            self.experiment_specification_hash, self.executed_configuration_hash,
            self.evaluation_policy_hash, self.image_content_hash,
            self.ground_truth_content_hash, self.matching_policy_hash,
        )
        if any(not is_sha256(value) for value in hashes):
            raise ValueError("Result provenance hashes must be SHA-256")
        if self.proposal_output_artifact_hash is not None and not is_sha256(self.proposal_output_artifact_hash):
            raise ValueError("Proposal artifact hash must be SHA-256")
        if self.proposal_output_artifact_hash is None and not self.proposal_output_details_json:
            raise ValueError("A proposal artifact hash or complete stored proposal details are required")
        if self.result_schema_version != RESULT_SCHEMA_VERSION:
            raise ValueError("Unsupported result schema version")
        if self.completion_status not in {"completed", "failed", "skipped"}:
            raise ValueError("Invalid completion status")
        decoded_json: dict[str, object] = {}
        for name, value in (
            ("proposal_output_details_json", self.proposal_output_details_json),
            ("matching_details_json", self.matching_details_json),
            ("metrics_json", self.metrics_json),
        ):
            if value is None:
                continue
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError as error:
                raise ValueError(f"{name} must contain JSON") from error
            if canonical_json(decoded) != value:
                raise ValueError(f"{name} must use canonical JSON")
            decoded_json[name] = decoded
        proposal_details = decoded_json.get("proposal_output_details_json")
        if self.proposal_output_artifact_hash is None and (not isinstance(proposal_details, dict) or "proposals" not in proposal_details):
            raise ValueError("Stored proposal details must contain the complete proposal list")
        if self.completion_status == "completed":
            matching = decoded_json.get("matching_details_json")
            required = {"proposals", "truths", "similarity_matrix", "proposal_decisions", "unmatched_truth_ids"}
            if not isinstance(matching, dict) or not required.issubset(matching):
                raise ValueError("Completed rows require reconstructible matching details")
            metrics = decoded_json.get("metrics_json")
            if not isinstance(metrics, dict) or not metrics:
                raise ValueError("Completed rows require explicit metric details")
        try:
            timestamp = datetime.fromisoformat(self.recorded_timestamp)
        except ValueError as error:
            raise ValueError("Result timestamp must be ISO-8601") from error
        if timestamp.tzinfo is None:
            raise ValueError("Result timestamp must include a timezone")


@dataclass(frozen=True)
class ExecutionAttemptSummary:
    execution_attempt_id: str
    experiment_specification_hash: str
    status: str
    expected_pairs: int
    attempted_pairs: int
    completed_pairs: int
    failed_pairs: int
    skipped_pairs: int
    unique_stored_pairs: int
    started_timestamp: str
    completed_timestamp: str

    def __post_init__(self) -> None:
        if self.status not in {"completed", "partially_completed", "failed", "cancelled"}:
            raise ValueError("Invalid immutable attempt status")
        counts = (self.expected_pairs, self.attempted_pairs, self.completed_pairs, self.failed_pairs, self.skipped_pairs, self.unique_stored_pairs)
        if any(value < 0 for value in counts):
            raise ValueError("Attempt counters cannot be negative")
        if self.attempted_pairs != self.completed_pairs + self.failed_pairs + self.skipped_pairs:
            raise ValueError("Attempted pairs must equal completed + failed + skipped")
        if self.unique_stored_pairs != self.attempted_pairs:
            raise ValueError("Every attempted pair must have one immutable stored row")
        if self.attempted_pairs > self.expected_pairs:
            raise ValueError("Attempted pairs cannot exceed the immutable expected matrix")
        if self.status == "completed" and (self.attempted_pairs != self.expected_pairs or self.failed_pairs or self.skipped_pairs):
            raise ValueError("Completed attempts require every expected pair to complete")


class V2ResultStore:
    """An explicit sink; construction creates only the caller-selected database."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
            connection.close()
            raise RuntimeError("SQLite foreign-key enforcement is unavailable")
        return connection

    def _initialize(self) -> None:
        applied = datetime.now(timezone.utc).isoformat()
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_versions (
                    schema_version INTEGER PRIMARY KEY,
                    description TEXT NOT NULL,
                    applied_timestamp TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS migration_history (
                    migration_id TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    migration_hash TEXT NOT NULL,
                    applied_timestamp TEXT NOT NULL,
                    FOREIGN KEY(schema_version) REFERENCES schema_versions(schema_version)
                );
                CREATE TABLE IF NOT EXISTS experiment_specifications (
                    specification_hash TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    experiment_version INTEGER NOT NULL,
                    dataset_id TEXT NOT NULL,
                    dataset_version TEXT NOT NULL,
                    specification_json TEXT NOT NULL,
                    registered_timestamp TEXT NOT NULL,
                    UNIQUE(experiment_id, experiment_version)
                );
                CREATE TABLE IF NOT EXISTS execution_attempts (
                    execution_attempt_id TEXT PRIMARY KEY,
                    experiment_specification_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    expected_pairs INTEGER NOT NULL,
                    attempted_pairs INTEGER NOT NULL,
                    completed_pairs INTEGER NOT NULL,
                    failed_pairs INTEGER NOT NULL,
                    skipped_pairs INTEGER NOT NULL,
                    unique_stored_pairs INTEGER NOT NULL,
                    started_timestamp TEXT NOT NULL,
                    completed_timestamp TEXT NOT NULL,
                    FOREIGN KEY(experiment_specification_hash) REFERENCES experiment_specifications(specification_hash),
                    CHECK(attempted_pairs = completed_pairs + failed_pairs + skipped_pairs),
                    CHECK(unique_stored_pairs = attempted_pairs),
                    CHECK(attempted_pairs <= expected_pairs)
                );
                CREATE TABLE IF NOT EXISTS result_rows (
                    result_id TEXT PRIMARY KEY,
                    experiment_specification_hash TEXT NOT NULL,
                    executed_configuration_hash TEXT NOT NULL,
                    method_implementation_id TEXT NOT NULL,
                    method_implementation_version TEXT NOT NULL,
                    evaluation_policy_id TEXT NOT NULL,
                    evaluation_policy_version INTEGER NOT NULL,
                    evaluation_policy_hash TEXT NOT NULL,
                    image_id TEXT NOT NULL,
                    image_content_hash TEXT NOT NULL,
                    ground_truth_content_hash TEXT NOT NULL,
                    proposal_output_artifact_hash TEXT,
                    proposal_output_details_json TEXT,
                    matching_policy_hash TEXT NOT NULL,
                    result_schema_version TEXT NOT NULL,
                    execution_attempt_id TEXT NOT NULL,
                    deterministic_mode INTEGER NOT NULL,
                    recorded_timestamp TEXT NOT NULL,
                    completion_status TEXT NOT NULL,
                    matching_details_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    FOREIGN KEY(experiment_specification_hash) REFERENCES experiment_specifications(specification_hash),
                    FOREIGN KEY(execution_attempt_id) REFERENCES execution_attempts(execution_attempt_id),
                    UNIQUE(execution_attempt_id, image_id, method_implementation_id),
                    CHECK(proposal_output_artifact_hash IS NOT NULL OR proposal_output_details_json IS NOT NULL)
                );
                CREATE TABLE IF NOT EXISTS result_supersessions (
                    supersession_id TEXT PRIMARY KEY,
                    prior_result_id TEXT NOT NULL,
                    successor_result_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    recorded_timestamp TEXT NOT NULL,
                    FOREIGN KEY(prior_result_id) REFERENCES result_rows(result_id),
                    FOREIGN KEY(successor_result_id) REFERENCES result_rows(result_id),
                    UNIQUE(prior_result_id, successor_result_id)
                );
                """
            )
            if connection.execute("SELECT 1 FROM schema_versions WHERE schema_version=?", (DATABASE_SCHEMA_VERSION,)).fetchone() is None:
                connection.execute("INSERT INTO schema_versions VALUES(?,?,?)", (DATABASE_SCHEMA_VERSION, "Initial append-only scientific-result-v2 schema", applied))
            migration_id = "scientific-result-v2-initial"
            migration_hash = sha256_json({"migration_id": migration_id, "schema_version": DATABASE_SCHEMA_VERSION})
            if connection.execute("SELECT 1 FROM migration_history WHERE migration_id=?", (migration_id,)).fetchone() is None:
                connection.execute("INSERT INTO migration_history VALUES(?,?,?,?)", (migration_id, DATABASE_SCHEMA_VERSION, migration_hash, applied))
            connection.execute(f"PRAGMA user_version = {DATABASE_SCHEMA_VERSION}")

    def register_specification(self, specification: ExperimentSpecificationV2) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO experiment_specifications VALUES(?,?,?,?,?,?,?)",
                (
                    specification.specification_hash, specification.experiment_id,
                    specification.experiment_version, specification.dataset_id,
                    specification.dataset_version, specification.to_json(),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    @staticmethod
    def _validate_row(row: ResultRowV2, specification: ExperimentSpecificationV2) -> None:
        if row.experiment_specification_hash != specification.specification_hash:
            raise ValueError("Result references a different experiment specification")
        if row.evaluation_policy_id != specification.evaluation_policy_id or row.evaluation_policy_version != specification.evaluation_policy_version or row.evaluation_policy_hash != specification.evaluation_policy_hash:
            raise ValueError("Result evaluation policy differs from the specification")
        if row.deterministic_mode != specification.deterministic_mode:
            raise ValueError("Result deterministic-mode state differs from the specification")
        images = {image.image_id: image for image in specification.selected_images}
        if row.image_id not in images:
            raise ValueError("Result image is not selected by the specification")
        if row.image_content_hash != images[row.image_id].image_sha256 or row.ground_truth_content_hash != images[row.image_id].ground_truth_sha256:
            raise ValueError("Result image or ground-truth hash differs from the specification")
        method = specification.method(row.method_implementation_id)
        if row.method_implementation_version != method.implementation_version:
            raise ValueError("Result method implementation version differs from the specification")
        if row.executed_configuration_hash != dict(specification.expected_executed_configuration_hashes)[method.method_id]:
            raise ValueError("Result executed-configuration hash differs from the specification")

    def append_attempt(
        self,
        specification: ExperimentSpecificationV2,
        summary: ExecutionAttemptSummary,
        rows: tuple[ResultRowV2, ...] | list[ResultRowV2],
    ) -> None:
        rows = tuple(rows)
        if summary.experiment_specification_hash != specification.specification_hash:
            raise ValueError("Attempt references a different specification")
        if summary.expected_pairs != specification.expected_pair_count:
            raise ValueError("Expected pair count must equal selected images x selected methods")
        if len(rows) != summary.unique_stored_pairs:
            raise ValueError("Attempt row count differs from its immutable counters")
        pairs = {(row.image_id, row.method_implementation_id) for row in rows}
        if len(pairs) != len(rows):
            raise ValueError("Attempt contains duplicate image-method pairs")
        observed = {status: sum(row.completion_status == status for row in rows) for status in ("completed", "failed", "skipped")}
        if (observed["completed"], observed["failed"], observed["skipped"]) != (summary.completed_pairs, summary.failed_pairs, summary.skipped_pairs):
            raise ValueError("Attempt counters do not match result completion statuses")
        for row in rows:
            if row.execution_attempt_id != summary.execution_attempt_id:
                raise ValueError("Result references a different execution attempt")
            self._validate_row(row, specification)
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT 1 FROM experiment_specifications WHERE specification_hash=?", (specification.specification_hash,)).fetchone() is None:
                raise ValueError("Specification must be registered before appending an attempt")
            connection.execute(
                "INSERT INTO execution_attempts VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    summary.execution_attempt_id, summary.experiment_specification_hash,
                    summary.status, summary.expected_pairs, summary.attempted_pairs,
                    summary.completed_pairs, summary.failed_pairs, summary.skipped_pairs,
                    summary.unique_stored_pairs, summary.started_timestamp,
                    summary.completed_timestamp,
                ),
            )
            fields = tuple(ResultRowV2.__annotations__)
            sql = f"INSERT INTO result_rows ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})"
            for row in rows:
                values = [int(value) if isinstance(value, bool) else value for value in (getattr(row, field) for field in fields)]
                connection.execute(sql, values)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def append_supersession(self, supersession_id: str, prior_result_id: str, successor_result_id: str, reason: str, timestamp: str) -> None:
        if prior_result_id == successor_result_id or not reason.strip():
            raise ValueError("Supersession requires distinct rows and a reason")
        with self.connect() as connection:
            connection.execute("INSERT INTO result_supersessions VALUES(?,?,?,?,?)", (supersession_id, prior_result_id, successor_result_id, reason, timestamp))

    def counts(self) -> dict[str, int]:
        with self.connect() as connection:
            return {
                table: int(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
                for table in ("experiment_specifications", "execution_attempts", "result_rows", "result_supersessions", "schema_versions", "migration_history")
            }
