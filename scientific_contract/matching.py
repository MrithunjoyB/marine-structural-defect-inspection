"""Deterministic one-to-one mask matching and annotation semantics for v2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Mapping

import numpy as np

from .evaluation_policy import EvaluationPolicyV2, default_evaluation_policy
from .hashing import sha256_bytes


@dataclass(frozen=True)
class EncodedMask:
    """Lossless binary run-length encoding, starting with a zero-valued run."""

    height: int
    width: int
    runs: tuple[int, ...]
    content_sha256: str

    @classmethod
    def from_array(cls, mask: np.ndarray) -> "EncodedMask":
        array = np.asarray(mask)
        if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
            raise ValueError("Masks must be non-empty two-dimensional arrays")
        binary = np.ascontiguousarray(array > 0, dtype=np.uint8)
        flat = binary.ravel(order="C")
        runs: list[int] = []
        current = 0
        length = 0
        for value in flat:
            bit = int(value)
            if bit == current:
                length += 1
            else:
                runs.append(length)
                current = bit
                length = 1
        runs.append(length)
        shape_prefix = int(array.shape[0]).to_bytes(8, "big") + int(array.shape[1]).to_bytes(8, "big")
        return cls(int(array.shape[0]), int(array.shape[1]), tuple(runs), sha256_bytes(shape_prefix + binary.tobytes()))

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0 or not self.runs:
            raise ValueError("Encoded mask dimensions and runs are required")
        if any(value < 0 for value in self.runs) or sum(self.runs) != self.height * self.width:
            raise ValueError("Encoded mask runs do not match its dimensions")
        if self.to_array().sum() == 0:
            raise ValueError("V2 truth and proposal masks must be non-empty")
        array = np.ascontiguousarray(self.to_array(), dtype=np.uint8)
        shape_prefix = self.height.to_bytes(8, "big") + self.width.to_bytes(8, "big")
        if sha256_bytes(shape_prefix + array.tobytes()) != self.content_sha256:
            raise ValueError("Encoded mask content hash mismatch")

    def to_array(self) -> np.ndarray:
        values = np.empty(self.height * self.width, dtype=np.uint8)
        offset = 0
        bit = 0
        for length in self.runs:
            values[offset:offset + length] = bit
            offset += length
            bit = 1 - bit
        return values.reshape((self.height, self.width))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "EncodedMask":
        return cls(int(payload["height"]), int(payload["width"]), tuple(int(v) for v in payload["runs"]), str(payload["content_sha256"]))


@dataclass(frozen=True)
class ProposalRecord:
    proposal_id: str
    mask: EncodedMask
    score: float | None = None
    rank: int | None = None

    def __post_init__(self) -> None:
        if not self.proposal_id.strip():
            raise ValueError("proposal_id is required")
        if self.score is not None and not math.isfinite(float(self.score)):
            raise ValueError("Proposal scores must be finite")
        if self.rank is not None and self.rank <= 0:
            raise ValueError("Proposal ranks are one-based positive integers")

    def to_dict(self) -> dict[str, object]:
        return {"proposal_id": self.proposal_id, "mask": self.mask.to_dict(), "score": self.score, "rank": self.rank}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ProposalRecord":
        return cls(str(payload["proposal_id"]), EncodedMask.from_dict(payload["mask"]), payload.get("score"), payload.get("rank"))


@dataclass(frozen=True)
class ProposalSet:
    method_id: str
    proposals: tuple[ProposalRecord, ...]
    ranking_eligible: bool
    ranking_definition: str | None = None

    def __post_init__(self) -> None:
        if not self.method_id.strip():
            raise ValueError("method_id is required")
        identifiers = [proposal.proposal_id for proposal in self.proposals]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Proposal identifiers must be unique")
        dimensions = {(proposal.mask.height, proposal.mask.width) for proposal in self.proposals}
        if len(dimensions) > 1:
            raise ValueError("All proposal masks must have identical dimensions")
        validate_ranking(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "proposals": [proposal.to_dict() for proposal in self.proposals],
            "ranking_eligible": self.ranking_eligible,
            "ranking_definition": self.ranking_definition,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ProposalSet":
        return cls(
            str(payload["method_id"]),
            tuple(ProposalRecord.from_dict(item) for item in payload["proposals"]),
            bool(payload["ranking_eligible"]),
            payload.get("ranking_definition"),
        )


def validate_ranking(proposal_set: ProposalSet) -> None:
    if not proposal_set.ranking_eligible:
        return
    if not proposal_set.ranking_definition or not proposal_set.ranking_definition.strip():
        raise ValueError("Rank-eligible output requires a documented ranking definition")
    proposals = proposal_set.proposals
    if any(proposal.score is None or proposal.rank is None for proposal in proposals):
        raise ValueError("Rank-eligible output requires a numeric score and rank for every proposal")
    ranks = [int(proposal.rank) for proposal in proposals]
    if sorted(ranks) != list(range(1, len(proposals) + 1)):
        raise ValueError("Ranks must be unique, contiguous, and one-based")
    expected = sorted(proposals, key=lambda item: (-float(item.score), item.proposal_id))
    if any(proposal.rank != index for index, proposal in enumerate(expected, 1)):
        raise ValueError("Stored ranks are inconsistent with score ordering and deterministic tie-breaking")


@dataclass(frozen=True)
class TruthInstance:
    truth_id: str
    mask: EncodedMask
    category: str = ""

    def __post_init__(self) -> None:
        if not self.truth_id.strip():
            raise ValueError("truth_id is required")

    def to_dict(self) -> dict[str, object]:
        return {"truth_id": self.truth_id, "mask": self.mask.to_dict(), "category": self.category}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TruthInstance":
        return cls(str(payload["truth_id"]), EncodedMask.from_dict(payload["mask"]), str(payload.get("category", "")))


@dataclass(frozen=True)
class GroundTruthRecord:
    image_id: str
    ground_truth_status: str
    truth_instances: tuple[TruthInstance, ...]
    legacy_warning: str | None = None

    def __post_init__(self) -> None:
        if not self.image_id.strip():
            raise ValueError("image_id is required")
        if self.ground_truth_status not in {"anomaly_present", "no_anomaly"}:
            raise ValueError("Ground-truth status must be anomaly_present or no_anomaly")
        identifiers = [truth.truth_id for truth in self.truth_instances]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Truth-instance identifiers must be unique")
        if self.ground_truth_status == "anomaly_present" and not self.truth_instances:
            raise ValueError("Anomaly-present images require at least one non-empty truth instance")
        if self.ground_truth_status == "no_anomaly" and self.truth_instances:
            raise ValueError("Clean images must have zero truth instances")
        dimensions = {(truth.mask.height, truth.mask.width) for truth in self.truth_instances}
        if len(dimensions) > 1:
            raise ValueError("All truth masks must have identical dimensions")

    def to_dict(self) -> dict[str, object]:
        return {
            "image_id": self.image_id,
            "ground_truth_status": self.ground_truth_status,
            "truth_instances": [truth.to_dict() for truth in self.truth_instances],
            "legacy_warning": self.legacy_warning,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "GroundTruthRecord":
        return cls(
            str(payload["image_id"]), str(payload["ground_truth_status"]),
            tuple(TruthInstance.from_dict(item) for item in payload["truth_instances"]),
            payload.get("legacy_warning"),
        )


def canonical_ground_truth(
    image_id: str,
    status: str,
    truth_masks: Iterable[tuple[str, np.ndarray, str]],
    *,
    allow_legacy_empty_clean_mask: bool = False,
) -> GroundTruthRecord:
    """Validate future annotations; optionally read a legacy empty clean mask."""
    items = list(truth_masks)
    warning = None
    if status == "no_anomaly" and items:
        if allow_legacy_empty_clean_mask and all(not np.any(mask) for _, mask, _ in items):
            warning = "legacy empty clean mask accepted as no_anomaly; do not register this representation in v2"
            items = []
        else:
            raise ValueError("Clean status cannot be combined with annotation objects")
    truths = tuple(TruthInstance(identifier, EncodedMask.from_array(mask), category) for identifier, mask, category in items)
    return GroundTruthRecord(image_id, status, truths, warning)


@dataclass(frozen=True)
class PairMeasurement:
    proposal_id: str
    truth_id: str
    mask_iou: float
    mask_dice: float
    truth_overlap: float
    proposal_overlap: float
    centroid_inside_truth: bool


@dataclass(frozen=True)
class ProposalDecision:
    proposal_id: str
    score: float | None
    rank: int | None
    assigned_truth_id: str | None
    matched: bool
    match_metric: str
    match_threshold: float
    mask_iou: float
    mask_dice: float
    truth_overlap: float
    proposal_overlap: float
    centroid_inside_truth_diagnostic: bool
    decision_reason: str


@dataclass(frozen=True)
class MatchingResult:
    image_id: str
    method_id: str
    evaluation_policy_id: str
    evaluation_policy_version: int
    evaluation_policy_hash: str
    matching_policy_hash: str
    match_metric: str
    match_threshold: float
    ranking_eligible: bool
    ranking_definition: str | None
    proposals: tuple[ProposalRecord, ...]
    truths: tuple[TruthInstance, ...]
    similarity_matrix: tuple[PairMeasurement, ...]
    proposal_decisions: tuple[ProposalDecision, ...]
    unmatched_truth_ids: tuple[str, ...]

    @property
    def matched_proposal_count(self) -> int:
        return sum(decision.matched for decision in self.proposal_decisions)

    @property
    def matched_truth_count(self) -> int:
        return len(self.truths) - len(self.unmatched_truth_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "image_id": self.image_id,
            "method_id": self.method_id,
            "evaluation_policy_id": self.evaluation_policy_id,
            "evaluation_policy_version": self.evaluation_policy_version,
            "evaluation_policy_hash": self.evaluation_policy_hash,
            "matching_policy_hash": self.matching_policy_hash,
            "match_metric": self.match_metric,
            "match_threshold": self.match_threshold,
            "ranking_eligible": self.ranking_eligible,
            "ranking_definition": self.ranking_definition,
            "proposals": [proposal.to_dict() for proposal in self.proposals],
            "truths": [truth.to_dict() for truth in self.truths],
            "similarity_matrix": [asdict(item) for item in self.similarity_matrix],
            "proposal_decisions": [asdict(item) for item in self.proposal_decisions],
            "unmatched_truth_ids": list(self.unmatched_truth_ids),
        }


def _measurement(proposal: ProposalRecord, truth: TruthInstance) -> PairMeasurement:
    left = proposal.mask.to_array().astype(bool)
    right = truth.mask.to_array().astype(bool)
    if left.shape != right.shape:
        raise ValueError("Proposal and truth masks must have identical dimensions")
    intersection = int(np.count_nonzero(left & right))
    proposal_area = int(np.count_nonzero(left))
    truth_area = int(np.count_nonzero(right))
    union = proposal_area + truth_area - intersection
    positions = np.argwhere(left)
    centroid_inside = False
    if len(positions):
        y, x = np.floor(positions.mean(axis=0)).astype(int)
        centroid_inside = bool(right[y, x])
    return PairMeasurement(
        proposal.proposal_id, truth.truth_id,
        intersection / union if union else 0.0,
        (2.0 * intersection) / (proposal_area + truth_area) if proposal_area + truth_area else 0.0,
        intersection / truth_area if truth_area else 0.0,
        intersection / proposal_area if proposal_area else 0.0,
        centroid_inside,
    )


def _maximum_weight_assignment(weights: np.ndarray) -> dict[int, int]:
    """Square Hungarian assignment with stable ascending-index tie handling."""
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("Assignment matrix must be square")
    size = weights.shape[0]
    if size == 0:
        return {}
    maximum = float(weights.max(initial=0.0))
    cost = maximum - weights
    u = np.zeros(size + 1, dtype=float)
    v = np.zeros(size + 1, dtype=float)
    p = np.zeros(size + 1, dtype=int)
    way = np.zeros(size + 1, dtype=int)
    epsilon = 1e-12
    for row in range(1, size + 1):
        p[0] = row
        column0 = 0
        minimum = np.full(size + 1, np.inf)
        used = np.zeros(size + 1, dtype=bool)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = np.inf
            column1 = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                current = cost[row0 - 1, column - 1] - u[row0] - v[column]
                if current < minimum[column] - epsilon:
                    minimum[column] = current
                    way[column] = column0
                if minimum[column] < delta - epsilon:
                    delta = minimum[column]
                    column1 = column
            for column in range(size + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    return {int(p[column] - 1): column - 1 for column in range(1, size + 1) if p[column]}


def match_one_to_one(
    proposal_set: ProposalSet,
    ground_truth: GroundTruthRecord,
    threshold: float | None = None,
    policy: EvaluationPolicyV2 | None = None,
) -> MatchingResult:
    policy = policy or default_evaluation_policy()
    threshold = policy.primary_match_threshold if threshold is None else float(threshold)
    if threshold not in policy.threshold_analyses:
        raise ValueError("Matching threshold must be an explicitly named policy analysis")
    proposals = tuple(sorted(proposal_set.proposals, key=lambda item: item.proposal_id))
    truths = tuple(sorted(ground_truth.truth_instances, key=lambda item: item.truth_id))
    if proposals and truths:
        dimensions = (proposals[0].mask.height, proposals[0].mask.width)
        if dimensions != (truths[0].mask.height, truths[0].mask.width):
            raise ValueError("Proposal and truth dimensions differ")
    matrix = tuple(_measurement(proposal, truth) for proposal in proposals for truth in truths)
    lookup = {(item.proposal_id, item.truth_id): item for item in matrix}
    size = max(len(proposals), len(truths))
    weights = np.zeros((size, size), dtype=float)
    cardinality_weight = float(size + 1)
    for row, proposal in enumerate(proposals):
        for column, truth in enumerate(truths):
            value = lookup[(proposal.proposal_id, truth.truth_id)].mask_iou
            if value + 1e-12 >= threshold:
                weights[row, column] = cardinality_weight + value
    assignment = _maximum_weight_assignment(weights)
    assigned_truths: set[str] = set()
    decisions: list[ProposalDecision] = []
    for row, proposal in enumerate(proposals):
        column = assignment.get(row)
        pair = None
        if column is not None and column < len(truths) and weights[row, column] > 0:
            pair = lookup[(proposal.proposal_id, truths[column].truth_id)]
            assigned_truths.add(pair.truth_id)
        candidates = [lookup[(proposal.proposal_id, truth.truth_id)] for truth in truths]
        best = max(candidates, key=lambda item: item.mask_iou, default=None)
        evidence = pair or best
        if pair is not None:
            reason = "matched_primary_mask_iou_at_or_above_threshold"
        elif not truths:
            reason = "unmatched_clean_image_has_no_truth_instances"
        elif best is not None and best.mask_iou + 1e-12 >= threshold:
            reason = "unmatched_one_to_one_competition"
        else:
            reason = "unmatched_primary_mask_iou_below_threshold"
        decisions.append(ProposalDecision(
            proposal.proposal_id, proposal.score, proposal.rank,
            pair.truth_id if pair else None, pair is not None,
            policy.primary_match_metric, threshold,
            evidence.mask_iou if evidence else 0.0,
            evidence.mask_dice if evidence else 0.0,
            evidence.truth_overlap if evidence else 0.0,
            evidence.proposal_overlap if evidence else 0.0,
            evidence.centroid_inside_truth if evidence else False,
            reason,
        ))
    return MatchingResult(
        ground_truth.image_id, proposal_set.method_id, policy.policy_id,
        policy.policy_version, policy.configuration_hash, policy.matching_policy_hash,
        policy.primary_match_metric, threshold, proposal_set.ranking_eligible,
        proposal_set.ranking_definition, proposals, truths, matrix, tuple(decisions),
        tuple(truth.truth_id for truth in truths if truth.truth_id not in assigned_truths),
    )


def reconstruct_matching(payload: Mapping[str, object], policy: EvaluationPolicyV2 | None = None) -> MatchingResult:
    """Re-evaluate solely from stored masks and identities, never from inference."""
    policy = policy or default_evaluation_policy()
    if (
        payload.get("evaluation_policy_id") != policy.policy_id
        or int(payload.get("evaluation_policy_version", -1)) != policy.policy_version
        or payload.get("evaluation_policy_hash") != policy.configuration_hash
        or payload.get("matching_policy_hash") != policy.matching_policy_hash
    ):
        raise ValueError("Stored matching evidence references a different evaluation policy")
    proposals = ProposalSet(
        str(payload["method_id"]),
        tuple(ProposalRecord.from_dict(item) for item in payload["proposals"]),
        bool(payload["ranking_eligible"]),
        payload.get("ranking_definition"),
    )
    truth = GroundTruthRecord(
        str(payload["image_id"]),
        "anomaly_present" if payload["truths"] else "no_anomaly",
        tuple(TruthInstance.from_dict(item) for item in payload["truths"]),
    )
    return match_one_to_one(proposals, truth, float(payload["match_threshold"]), policy)
