"""
Module 11 - Cross-Encoder Re-ranking: data schema definitions.

Defines RankedCandidate, the module's primary output type: one
FusedCandidate (Module 9/10) re-scored by a cross-encoder and assigned
its final rank in the shortlist handed to Module 12.

Reference: Capstone Project Guide, Section 3L ("Re-ranking"), Phase 4
Module 11 contract, Phase 5 file structure (reranking/schema.py).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from retrieval.schema import FusedCandidate

logger = logging.getLogger(__name__)


class SchemaValidationError(Exception):
    """Raised when a RankedCandidate is constructed with invalid field values."""


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    """
    Immutable representation of one candidate after cross-encoder
    re-ranking, produced by CrossEncoderReranker.rerank().

    Attributes:
        candidate_id: Inherited unchanged from the source
            FusedCandidate (chunk_id or fact_id).
        text: Inherited unchanged from the source FusedCandidate --
            the exact text the cross-encoder scored.
        rerank_score: The cross-encoder's joint query-candidate
            relevance score. Not assumed bounded to [0, 1] in general
            (depends on the model's activation function), though the
            default sigmoid activation for a single-label
            cross-encoder does produce scores in that range.
        original_candidate: The full source FusedCandidate, preserved
            for provenance -- Module 12's prompt/citation builder needs
            source, vector_score, graph_score, fused_score, and
            metadata, none of which re-ranking should discard.
        rank: 1-based position in the final re-ranked shortlist (1 =
            most relevant).

    Raises:
        SchemaValidationError: If candidate_id or text is empty, rank
            is not >= 1, or rerank_score is not finite.
    """

    candidate_id: str
    text: str
    rerank_score: float
    original_candidate: FusedCandidate
    rank: int

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise SchemaValidationError(
                "RankedCandidate was constructed with an empty candidate_id."
            )
        if not self.text:
            raise SchemaValidationError(
                f"RankedCandidate {self.candidate_id!r} was constructed with empty text."
            )
        if not math.isfinite(self.rerank_score):
            raise SchemaValidationError(
                f"RankedCandidate {self.candidate_id!r} has a non-finite "
                f"rerank_score ({self.rerank_score})."
            )
        if self.rank < 1:
            raise SchemaValidationError(
                f"RankedCandidate {self.candidate_id!r} has rank {self.rank}, must be >= 1."
            )
