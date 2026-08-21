"""
Module 10 - Adaptive Retrieval: rule-based classifier baseline.

Provides RuleBasedAlphaClassifier (Tier A, Section 7.2a) and
rule_based_alpha(...) per the Phase 4 module contract: a
zero-training-data lexicon-ratio baseline, fast and fully
interpretable, but brittle to phrasing outside the keyword lists.

Reference: Capstone Project Guide, Section 7.2(a) ("Rule-based /
lexical classification (baseline)"), Phase 4 Module 10 contract,
Phase 5 file structure (adaptive_retrieval/rules.py).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from adaptive_retrieval.classifier import AlphaClassifier

logger = logging.getLogger(__name__)

# Section 7.2(a)'s exact example lexicons: relational-indicator phrases
# vs. descriptive/procedural-indicator phrases.
_DEFAULT_RELATIONAL_INDICATORS: tuple[str, ...] = (
    "who",
    "which department",
    "which team",
    "reports to",
    "manager",
    "belongs to",
    "how many",
    "who manages",
    "who is",
    "which policy applies",
)
_DEFAULT_DESCRIPTIVE_INDICATORS: tuple[str, ...] = (
    "how do i",
    "explain",
    "what is the process",
    "why",
    "describe",
    "walk me through",
    "what is",
    "how does",
)


class RuleBasedAlphaClassifier(AlphaClassifier):
    """
    Tier A: counts matches against a relational-indicator lexicon
    versus a descriptive-indicator lexicon in the query text, and sets
    alpha from the match ratio. Ignores query_embedding entirely
    (fully text-based, zero training data, Section 7.2a).

    alpha is defined as the vector-retrieval weight (Section 7.3), so
    a query dominated by descriptive-indicator matches should score
    HIGH (favor vector search), and one dominated by
    relational-indicator matches should score LOW (favor graph
    traversal):

        alpha = descriptive_count / (relational_count + descriptive_count)

    A query with no matches from either lexicon (brittle to unseen
    phrasing, per Section 7.2a's documented limitation) returns the
    neutral default 0.5, splitting fusion weight evenly rather than
    guessing.
    """

    def __init__(
        self,
        relational_indicators: Optional[tuple[str, ...]] = None,
        descriptive_indicators: Optional[tuple[str, ...]] = None,
    ) -> None:
        """
        Args:
            relational_indicators: Phrases indicating a relational
                query. Defaults to Section 7.2(a)'s example lexicon.
            descriptive_indicators: Phrases indicating a
                descriptive/procedural query. Defaults to Section
                7.2(a)'s example lexicon.
        """
        self._relational_indicators = tuple(
            phrase.lower() for phrase in (relational_indicators or _DEFAULT_RELATIONAL_INDICATORS)
        )
        self._descriptive_indicators = tuple(
            phrase.lower() for phrase in (descriptive_indicators or _DEFAULT_DESCRIPTIVE_INDICATORS)
        )

    def predict_alpha(self, query: str, query_embedding: np.ndarray) -> float:
        """
        Args:
            query: Raw query text.
            query_embedding: Unused by this tier (kept for interface
                uniformity with LearnedAlphaClassifier).

        Returns:
            The lexicon-ratio alpha estimate, in [0, 1].

        Raises:
            ValueError: If query is empty/whitespace-only.
        """
        if not query or not query.strip():
            raise ValueError("RuleBasedAlphaClassifier.predict_alpha() requires non-empty query.")

        lowered = query.lower()
        relational_count = sum(lowered.count(phrase) for phrase in self._relational_indicators)
        descriptive_count = sum(lowered.count(phrase) for phrase in self._descriptive_indicators)

        total = relational_count + descriptive_count
        if total == 0:
            logger.debug("No lexicon match for query %r; returning neutral alpha=0.5", query)
            return 0.5

        alpha = descriptive_count / total
        logger.debug(
            "Rule-based alpha for %r: relational=%d descriptive=%d -> alpha=%.3f",
            query,
            relational_count,
            descriptive_count,
            alpha,
        )
        return alpha


def rule_based_alpha(
    query: str,
    relational_indicators: Optional[tuple[str, ...]] = None,
    descriptive_indicators: Optional[tuple[str, ...]] = None,
) -> float:
    """
    Module-level convenience wrapper around
    RuleBasedAlphaClassifier.predict_alpha(), matching the Phase 4
    module contract's function signature
    (rule_based_alpha(query_text) -> float). Constructs a
    default-config classifier for one-off use; callers scoring many
    queries should instantiate RuleBasedAlphaClassifier directly
    instead, to avoid rebuilding the lexicon tuples per call.

    Args:
        query: Raw query text.
        relational_indicators: Optional custom relational lexicon.
        descriptive_indicators: Optional custom descriptive lexicon.

    Returns:
        The lexicon-ratio alpha estimate, in [0, 1].
    """
    classifier = RuleBasedAlphaClassifier(
        relational_indicators=relational_indicators,
        descriptive_indicators=descriptive_indicators,
    )
    # query_embedding is unused by this tier; pass a harmless placeholder
    # rather than requiring callers to compute an embedding they don't need.
    return classifier.predict_alpha(query, np.empty(0, dtype=np.float32))
