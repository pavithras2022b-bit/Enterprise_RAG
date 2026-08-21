"""
Module 11 - Cross-Encoder Re-ranking: re-ranking.

Provides score_pair(...), rerank(...), and the CrossEncoderReranker
class per the Phase 4 module contract: a second-stage, higher-
precision relevance scorer applied to Module 9/10's fused candidate
shortlist, jointly encoding query and candidate (unlike the bi-encoder
used for initial vector retrieval, which encodes them independently).

Reference: Capstone Project Guide, Section 3L ("Re-ranking"), Phase 4
Module 11 contract, Phase 5 file structure (reranking/model.py).
"""

from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

from reranking.schema import RankedCandidate
from retrieval.schema import FusedCandidate

logger = logging.getLogger(__name__)


class CrossEncoderRerankerError(Exception):
    """Raised when CrossEncoderReranker cannot load its checkpoint or score a pair."""


class CrossEncoderReranker:
    """
    Wraps a cross-encoder checkpoint (Section 12.3: base
    ms-marco-MiniLM-L-6-v2, further fine-tuned per Section 8.5) for
    precise, joint query-candidate relevance scoring.

    Unlike the bi-encoder used for initial vector retrieval (Module
    7), which encodes query and candidate independently so their
    vectors can be pre-computed and indexed, a cross-encoder
    concatenates [CLS] query [SEP] candidate [SEP] into one input and
    lets the transformer's self-attention run across both jointly --
    far more accurate at judging true relevance, but too slow to run
    over an entire candidate pool. This is why it is restricted to
    Module 9/10's already-fused, already-narrowed shortlist (typically
    20-50 candidates), not the full corpus.

    Construct via the constructor for a checkpoint on disk or the Hub,
    or via from_cross_encoder() to inject an already-loaded CrossEncoder
    (used by fine_tune_cross_encoder() and tests).
    """

    def __init__(self, model_name_or_path: str, device: str = "cpu", max_length: int = 512) -> None:
        """
        Args:
            model_name_or_path: A local checkpoint directory (e.g. the
                output_dir of a prior fine_tune_cross_encoder() run) or
                a Hugging Face Hub model id (e.g.
                "cross-encoder/ms-marco-MiniLM-L-6-v2").
            device: torch device string ("cpu" or "cuda").
            max_length: Maximum token sequence length for the
                concatenated query+candidate input.

        Raises:
            CrossEncoderRerankerError: If the model cannot be loaded.
        """
        try:
            model = CrossEncoder(model_name_or_path, device=device, max_length=max_length)
        except Exception as exc:  # noqa: BLE001
            raise CrossEncoderRerankerError(
                f"Failed to load CrossEncoder from {model_name_or_path!r}: {exc}"
            ) from exc
        self._model = model

    @classmethod
    def from_cross_encoder(cls, model: CrossEncoder) -> "CrossEncoderReranker":
        """
        Construct a CrossEncoderReranker from an already-loaded
        CrossEncoder, bypassing constructor-style loading. Used by
        fine_tune_cross_encoder() (which already holds the freshly
        fine-tuned model in memory) and by tests using a small local
        checkpoint.

        Args:
            model: A loaded CrossEncoder.

        Returns:
            A ready-to-use CrossEncoderReranker.
        """
        instance = cls.__new__(cls)
        instance._model = model
        return instance

    def score_pair(self, query: str, candidate_text: str) -> float:
        """
        Score a single (query, candidate) pair.

        Args:
            query: The user query.
            candidate_text: The candidate's text.

        Returns:
            The cross-encoder's relevance score for this pair.

        Raises:
            ValueError: If query or candidate_text is empty/whitespace-only.
            CrossEncoderRerankerError: If inference fails.
        """
        if not query or not query.strip():
            raise ValueError("score_pair() requires a non-empty query.")
        if not candidate_text or not candidate_text.strip():
            raise ValueError("score_pair() requires non-empty candidate_text.")

        try:
            scores = self._model.predict([(query, candidate_text)])
        except Exception as exc:  # noqa: BLE001
            raise CrossEncoderRerankerError(f"Cross-encoder scoring failed: {exc}") from exc
        return float(scores[0])

    def rerank(
        self, query: str, candidates: list[FusedCandidate], top_n: int = 5
    ) -> list[RankedCandidate]:
        """
        Score every candidate against the query in one batched forward
        pass, and return the top_n highest-scoring, ranked.

        Args:
            query: The user query.
            candidates: The fused candidate shortlist from Module
                9/10 (typically 20-50 items).
            top_n: Number of top candidates to return.

        Returns:
            Up to top_n RankedCandidate objects, sorted descending by
            rerank_score, rank fields set 1..len(result). Empty list
            if candidates is empty.

        Raises:
            ValueError: If query is empty/whitespace-only, or top_n is
                not positive.
            CrossEncoderRerankerError: If inference fails.
        """
        if not query or not query.strip():
            raise ValueError("rerank() requires a non-empty query.")
        if top_n <= 0:
            raise ValueError(f"top_n must be positive, got {top_n}")
        if not candidates:
            return []

        pairs = [(query, candidate.text) for candidate in candidates]
        try:
            scores = self._model.predict(pairs)
        except Exception as exc:  # noqa: BLE001
            raise CrossEncoderRerankerError(f"Cross-encoder scoring failed: {exc}") from exc

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        top = scored[:top_n]

        ranked = [
            RankedCandidate(
                candidate_id=candidate.candidate_id,
                text=candidate.text,
                rerank_score=float(score),
                original_candidate=candidate,
                rank=position,
            )
            for position, (candidate, score) in enumerate(top, start=1)
        ]
        logger.debug(
            "rerank() scored %d candidate(s), returning top %d for query %r",
            len(candidates),
            len(ranked),
            query,
        )
        return ranked


def score_pair(model: CrossEncoderReranker, query: str, candidate_text: str) -> float:
    """
    Module-level convenience wrapper around
    CrossEncoderReranker.score_pair(), matching the Phase 4 module
    contract's function signature
    (score_pair(query, candidate_text) -> float).

    Args:
        model: A loaded CrossEncoderReranker.
        query: The user query.
        candidate_text: The candidate's text.

    Returns:
        The cross-encoder's relevance score.
    """
    return model.score_pair(query, candidate_text)


def rerank(
    model: CrossEncoderReranker,
    query: str,
    candidates: list[FusedCandidate],
    top_n: int = 5,
) -> list[RankedCandidate]:
    """
    Module-level convenience wrapper around
    CrossEncoderReranker.rerank(), matching the Phase 4 module
    contract's function signature
    (rerank(query, candidates, top_n) -> list[RankedCandidate]).

    Args:
        model: A loaded CrossEncoderReranker.
        query: The user query.
        candidates: The fused candidate shortlist.
        top_n: Number of top candidates to return.

    Returns:
        Up to top_n RankedCandidate objects, sorted descending by
        rerank_score.
    """
    return model.rerank(query, candidates, top_n=top_n)
