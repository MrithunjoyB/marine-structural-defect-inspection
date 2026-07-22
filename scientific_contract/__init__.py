"""Prospective scientific evaluation and provenance contract.

The modules in this package implement ``structvision-eval-v2`` independently
of the historical evaluation and result stores.  Importing the package has no
filesystem side effects.
"""

from .evaluation_policy import EvaluationPolicyV2, default_evaluation_policy
from .matching import (
    EncodedMask,
    GroundTruthRecord,
    ProposalRecord,
    ProposalSet,
    TruthInstance,
    match_one_to_one,
)

__all__ = [
    "EncodedMask",
    "EvaluationPolicyV2",
    "GroundTruthRecord",
    "ProposalRecord",
    "ProposalSet",
    "TruthInstance",
    "default_evaluation_policy",
    "match_one_to_one",
]
