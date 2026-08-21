"""
Module 10 - Adaptive Retrieval: classifier training entrypoint.

Provides train_classifier(...) per the Phase 4 module contract:
trains LearnedAlphaClassifier's logistic regression on distant-
supervision-labeled queries, per Section 7.6's training procedure.

Reference: Capstone Project Guide, Section 7.6 ("Training Procedure"),
Phase 4 Module 10 contract, Phase 5 file structure
(adaptive_retrieval/train.py).
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import LogisticRegression

from adaptive_retrieval.classifier import AlphaLabel, LabeledQuery, LearnedAlphaClassifier
from embeddings.model import EmbeddingModel, EmbeddingModelError

logger = logging.getLogger(__name__)


class AlphaTrainingError(Exception):
    """Raised when classifier training cannot proceed for the given input."""


def train_classifier(
    labeled_queries: list[LabeledQuery],
    embedding_model: EmbeddingModel,
    max_iter: int = 1000,
    seed: int = 42,
) -> LearnedAlphaClassifier:
    """
    Train LearnedAlphaClassifier's logistic regression on labeled
    queries, per Section 7.6's training procedure.

    MIXED-labeled queries are excluded from the binary training set:
    a logistic regression fit on {GRAPH_FAVORING: 0, VECTOR_FAVORING: 1}
    naturally produces probabilities near 0.5 for genuinely ambiguous
    queries at inference time, as a consequence of decision-boundary
    uncertainty -- forcing an explicit 0.5 training target for MIXED
    examples is not necessary to achieve this, and would need a
    continuous-regression objective (Section 7.6's alternative MSE
    formulation) rather than the binary logistic regression this
    project's simplest well-posed form (Section 7.7) specifies. MIXED
    examples remain valid for held-out evaluation elsewhere (Module
    13); they are simply not part of this function's training set.

    Args:
        labeled_queries: Labeled queries, ideally from Phase 1's
            roadmap (150-300 queries, distantly supervised from which
            retrieval mode's evidence matched the gold answer).
        embedding_model: Module 7's EmbeddingModel, used to embed every
            training query (the same embedding space
            LearnedAlphaClassifier will see at inference time).
        max_iter: Maximum solver iterations for LogisticRegression.
        seed: Random seed for LogisticRegression's solver.

    Returns:
        A LearnedAlphaClassifier wrapping the fitted model.

    Raises:
        AlphaTrainingError: If labeled_queries is empty, contains no
            usable (non-MIXED) examples, or contains examples of only
            one class (a binary classifier needs both).
    """
    if not labeled_queries:
        raise AlphaTrainingError("train_classifier() received an empty labeled_queries list.")

    binary_examples = [lq for lq in labeled_queries if lq.label != AlphaLabel.MIXED]
    if not binary_examples:
        raise AlphaTrainingError(
            "train_classifier() found no GRAPH_FAVORING/VECTOR_FAVORING examples "
            "to train on -- labeled_queries contained only MIXED-labeled entries, "
            "which are excluded from binary training (see this function's "
            "docstring)."
        )

    present_labels = {lq.label for lq in binary_examples}
    if len(present_labels) < 2:
        raise AlphaTrainingError(
            f"train_classifier() requires both GRAPH_FAVORING and VECTOR_FAVORING "
            f"examples to fit a binary classifier; only found: "
            f"{sorted(label.value for label in present_labels)}."
        )

    try:
        embeddings = np.stack(
            [embedding_model.embed(labeled_query.query) for labeled_query in binary_examples]
        )
    except EmbeddingModelError as exc:
        raise AlphaTrainingError(f"Failed to embed training queries: {exc}") from exc

    targets = np.array(
        [1 if lq.label == AlphaLabel.VECTOR_FAVORING else 0 for lq in binary_examples]
    )

    model = LogisticRegression(max_iter=max_iter, random_state=seed)
    model.fit(embeddings, targets)

    logger.info(
        "Trained LearnedAlphaClassifier on %d example(s) (%d graph-favoring, "
        "%d vector-favoring); %d MIXED example(s) excluded from training.",
        len(binary_examples),
        int((targets == 0).sum()),
        int((targets == 1).sum()),
        len(labeled_queries) - len(binary_examples),
    )

    return LearnedAlphaClassifier.from_sklearn_model(model)
