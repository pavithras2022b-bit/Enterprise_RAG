"""
Module 10 - Adaptive Retrieval: alpha classifier base + learned tier.

Provides AlphaClassifier (abstract base shared by both tiers),
LabeledQuery/AlphaLabel (Section 7.6's distant-supervision training
data contract), LearnedAlphaClassifier (Tier B, Section 7.2b), and
predict_alpha(...) per the Phase 4 module contract.

This is the project's central research contribution: replacing a
static fusion weight with a learned function of the query embedding,
alpha(q) = sigma(w^T e_q + b) -- the simplest well-posed form, per
Section 7.7's mathematical formulation.

Reference: Capstone Project Guide, Section 7 ("Adaptive Retrieval Deep
Dive"), Phase 4 Module 10 contract, Phase 5 file structure
(adaptive_retrieval/classifier.py).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class AlphaClassifierError(Exception):
    """Raised when an AlphaClassifier is used incorrectly (e.g. before training)."""


class SchemaValidationError(Exception):
    """Raised when a LabeledQuery is constructed with invalid values."""


class AlphaLabel(str, Enum):
    """
    Discrete query-type label used for distant-supervision training
    data (Section 7.6): tag a query GRAPH_FAVORING if its gold-answer
    evidence came from the knowledge graph, VECTOR_FAVORING if from
    vector chunks, or MIXED if both contributed.
    """

    GRAPH_FAVORING = "GRAPH_FAVORING"
    VECTOR_FAVORING = "VECTOR_FAVORING"
    MIXED = "MIXED"


# Preset alpha values for each discrete label, per Section 7.2(b):
# "map preset alpha values" to a discrete relational/descriptive/mixed
# classification. Used only by LabeledQuery.target_alpha, a reference
# value for evaluation/reporting -- LearnedAlphaClassifier itself is
# trained as a binary classifier (see train.py) and never consults
# this mapping directly.
_PRESET_ALPHA_BY_LABEL: dict[AlphaLabel, float] = {
    AlphaLabel.GRAPH_FAVORING: 0.1,
    AlphaLabel.VECTOR_FAVORING: 0.9,
    AlphaLabel.MIXED: 0.5,
}


@dataclass(frozen=True, slots=True)
class LabeledQuery:
    """
    One distant-supervision training/evaluation example for the
    adaptive fusion classifier (Section 7.6, Phase 1's roadmap: 150-300
    labeled queries, derived from which retrieval mode's evidence
    matched the gold answer).

    Attributes:
        query: The query text.
        label: Its discrete query-type label.

    Raises:
        SchemaValidationError: If query is empty/whitespace-only.
    """

    query: str
    label: AlphaLabel

    def __post_init__(self) -> None:
        if not self.query or not self.query.strip():
            raise SchemaValidationError("LabeledQuery.query must not be empty.")

    @property
    def target_alpha(self) -> float:
        """Returns this label's preset reference alpha value (Section 7.2b)."""
        return _PRESET_ALPHA_BY_LABEL[self.label]


class AlphaClassifier(ABC):
    """
    Common interface for both fusion-weight classifier tiers
    (Section 7.2): RuleBasedAlphaClassifier (Tier A, rules.py) and
    LearnedAlphaClassifier (Tier B, below). AdaptiveFusionEngine
    (fusion.py) is written against this interface so either tier can
    be swapped in without changing any other code -- exactly what
    Section 13.3's Ablation A5 (static vs. rule-based-adaptive vs.
    learned-adaptive) requires: the only thing that differs between
    variants is which AlphaClassifier is injected.

    Both the query text and its embedding are always passed to
    predict_alpha(), even though a given implementation may only use
    one of them (rule-based ignores the embedding; learned ignores the
    text) -- consistent with Section 7.5's logic flow, where the query
    embedding is computed once and reused regardless of which tier is
    active.
    """

    @abstractmethod
    def predict_alpha(self, query: str, query_embedding: np.ndarray) -> float:
        """
        Args:
            query: Raw query text.
            query_embedding: The query's embedding vector (from Module
                7's EmbeddingModel), 1-D.

        Returns:
            The predicted vector-retrieval weight alpha, in [0, 1].
        """
        raise NotImplementedError


class LearnedAlphaClassifier(AlphaClassifier):
    """
    Tier B (Section 7.2b, the main contribution): a logistic regression
    over the query embedding, alpha(q) = sigma(w^T e_q + b) (Section
    7.7). Implemented as a binary classifier over
    {GRAPH_FAVORING: 0, VECTOR_FAVORING: 1} (see train.py for why
    MIXED-labeled examples are excluded from training); predict_proba's
    probability of the VECTOR_FAVORING class *is* sigma(w^T e_q + b)
    for a binary logistic regression, so it is returned directly as
    alpha with no further transformation.
    """

    def __init__(self, model: Optional[LogisticRegression] = None) -> None:
        """
        Args:
            model: A fitted scikit-learn LogisticRegression, or None to
                construct an untrained classifier (predict_alpha raises
                AlphaClassifierError until a trained model is supplied
                via from_sklearn_model()).
        """
        self._model = model

    @classmethod
    def from_sklearn_model(cls, model: LogisticRegression) -> "LearnedAlphaClassifier":
        """
        Wraps an already-fitted scikit-learn LogisticRegression (used
        by train.py's train_classifier(), and by tests/deployment code
        loading a persisted model).

        Args:
            model: A fitted LogisticRegression.

        Returns:
            A ready-to-use LearnedAlphaClassifier.
        """
        return cls(model=model)

    @property
    def is_fitted(self) -> bool:
        """True if this classifier wraps a trained model."""
        return self._model is not None and hasattr(self._model, "coef_")

    def predict_alpha(self, query: str, query_embedding: np.ndarray) -> float:
        """
        Args:
            query: Unused by this tier (kept for interface uniformity
                with RuleBasedAlphaClassifier).
            query_embedding: The query's embedding vector, 1-D.

        Returns:
            predict_proba's probability of the VECTOR_FAVORING class,
            in [0, 1].

        Raises:
            AlphaClassifierError: If this classifier has not been
                trained (is_fitted is False).
            ValueError: If query_embedding is not 1-D.
        """
        if not self.is_fitted:
            raise AlphaClassifierError(
                "LearnedAlphaClassifier has not been trained. Train via "
                "adaptive_retrieval.train.train_classifier() or construct with "
                "from_sklearn_model(fitted_model)."
            )
        if query_embedding.ndim != 1:
            raise ValueError(
                f"query_embedding must be 1-D, got shape {query_embedding.shape}."
            )

        probabilities = self._model.predict_proba(query_embedding.reshape(1, -1))[0]
        vector_favoring_index = list(self._model.classes_).index(1)
        return float(probabilities[vector_favoring_index])


def predict_alpha(
    classifier: AlphaClassifier, query: str, query_embedding: np.ndarray
) -> float:
    """
    Module-level convenience wrapper around AlphaClassifier.predict_alpha(),
    matching the Phase 4 module contract's function signature
    (predict_alpha(query_embedding) -> float). The classifier is
    threaded through explicitly, mirroring the convention established
    in Modules 4/5/7/9.

    Args:
        classifier: Either tier of AlphaClassifier.
        query: Raw query text.
        query_embedding: The query's embedding vector.

    Returns:
        The predicted alpha, in [0, 1].
    """
    return classifier.predict_alpha(query, query_embedding)
