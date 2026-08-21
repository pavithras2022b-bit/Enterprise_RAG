"""
Module 10 - Adaptive Retrieval: adaptive fusion engine.

Provides AdaptiveFusionEngine and adaptive_fuse(...) per the Phase 4
module contract. This is the integration point with Module 9: per
Module 9's hard architectural requirement (its own spec, Section 8),
HybridRetrievalController.static_fuse() is the SINGLE SOURCE OF TRUTH
for the fusion formula, and Module 10 must reuse it verbatim with a
learned per-query alpha instead of reimplementing any fusion logic.

Reference: Capstone Project Guide, Section 7.1/7.3/7.5 ("Adaptive
Retrieval Deep Dive"), Phase 4 Module 10 contract, Phase 5 file
structure (adaptive_retrieval/fusion.py).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from embeddings.model import EmbeddingModel, EmbeddingModelError
from retrieval.entity_linker import EntityLinker
from retrieval.hybrid_controller import HybridRetrievalController, RetrievalError
from retrieval.schema import FusedCandidate, GraphFact
from vector_db.schema import VectorSearchResult

from adaptive_retrieval.classifier import AlphaClassifier

logger = logging.getLogger(__name__)


class AdaptiveFusionError(Exception):
    """Raised when the adaptive fusion pipeline fails."""


class AdaptiveFusionEngine:
    """
    Computes a query-conditional fusion weight alpha(q) and produces
    the adaptively-fused candidate list, per Section 7.1's definition
    of adaptive retrieval.

    Wraps a Module 9 HybridRetrievalController (constructor-injected,
    not constructed internally -- Module 9 has no dependency on Module
    10, only the reverse) and calls its run_vector_retrieval() /
    run_graph_retrieval() / static_fuse() methods directly rather than
    its retrieve() convenience entrypoint, since retrieve() bundles in
    a fixed alpha. entity_linker is injected separately from the
    controller (typically the same EntityLinker instance the
    controller itself was built with) rather than reaching into the
    controller's private state, keeping Module 9's public API
    untouched.

    Any AlphaClassifier (RuleBasedAlphaClassifier or
    LearnedAlphaClassifier) can be injected interchangeably -- this is
    what makes Section 13.3's Ablation A5 (static / rule-based-adaptive
    / learned-adaptive) a clean, isolated comparison: only the
    classifier changes between variants, nothing else in the pipeline
    does.
    """

    def __init__(
        self,
        controller: HybridRetrievalController,
        embedding_model: EmbeddingModel,
        entity_linker: EntityLinker,
        classifier: AlphaClassifier,
    ) -> None:
        """
        Args:
            controller: Module 9's HybridRetrievalController, already
                constructed and pointed at a populated vector store and
                graph.
            embedding_model: Module 7's EmbeddingModel, used to embed
                the incoming query once (Section 7.5: computed once,
                reused for both retrieval and classification).
            entity_linker: Module 9's EntityLinker (typically the same
                instance `controller` was constructed with).
            classifier: Either AlphaClassifier tier.
        """
        self._controller = controller
        self._embedding_model = embedding_model
        self._entity_linker = entity_linker
        self._classifier = classifier

    def compute_alpha(self, query: str) -> float:
        """
        Embeds the query and runs it through the injected classifier.

        Args:
            query: Free-text user query.

        Returns:
            The predicted vector-retrieval weight alpha, in [0, 1].

        Raises:
            ValueError: If query is empty/whitespace-only.
            AdaptiveFusionError: If embedding fails, or the classifier
                returns a value outside [0, 1] (a defensive check --
                well-behaved classifiers should never do this, but a
                bad custom AlphaClassifier implementation must not
                silently corrupt static_fuse()'s own validation
                downstream with a confusing error far from its cause).
        """
        if not query or not query.strip():
            raise ValueError("compute_alpha() requires a non-empty query.")

        try:
            query_embedding = self._embedding_model.embed(query)
        except EmbeddingModelError as exc:
            raise AdaptiveFusionError(f"Failed to embed query: {exc}") from exc

        alpha = self._classifier.predict_alpha(query, query_embedding)

        if not 0.0 <= alpha <= 1.0:
            raise AdaptiveFusionError(
                f"{type(self._classifier).__name__} returned alpha={alpha}, "
                f"outside the required [0, 1] range."
            )
        return alpha

    def retrieve(
        self,
        query: str,
        k_vector: int = 20,
        k_graph: int = 20,
        where: Optional[dict[str, Any]] = None,
    ) -> tuple[float, list[FusedCandidate]]:
        """
        Full adaptive retrieval pipeline: compute alpha(q), run both
        retrieval branches via the injected controller, and fuse via
        controller.static_fuse() -- Module 9's fusion method, reused
        verbatim (see this module's docstring).

        Args:
            query: Free-text user query.
            k_vector: Passed to controller.run_vector_retrieval().
            k_graph: Passed to controller.run_graph_retrieval().
            where: Passed to controller.run_vector_retrieval().

        Returns:
            A (alpha, candidates) tuple: the classifier's predicted
            alpha for this query, and the resulting fused candidate
            list. Returning alpha alongside the candidates matters for
            Section 13.1's evaluation design, which reports classifier
            accuracy/behavior *alongside*, not instead of, end-to-end
            retrieval accuracy.

        Raises:
            ValueError: Propagated from compute_alpha() or the
                underlying controller calls (empty query, non-positive
                k_vector/k_graph).
            AdaptiveFusionError: If embedding, retrieval, or fusion
                fails.
        """
        alpha = self.compute_alpha(query)
        entities = self._entity_linker.link(query)

        try:
            vector_hits = self._controller.run_vector_retrieval(query, k=k_vector, where=where)
            graph_hits = self._controller.run_graph_retrieval(entities, k=k_graph)
        except RetrievalError as exc:
            raise AdaptiveFusionError(f"Retrieval failed: {exc}") from exc

        candidates = HybridRetrievalController.static_fuse(vector_hits, graph_hits, alpha=alpha)
        return alpha, candidates


def adaptive_fuse(
    vector_hits: list[VectorSearchResult],
    graph_hits: list[GraphFact],
    alpha: float,
) -> list[FusedCandidate]:
    """
    Thin alias for HybridRetrievalController.static_fuse(), exposed
    under adaptive_retrieval's own namespace to match the Phase 4
    module contract's function signature
    (adaptive_fuse(vector_hits, graph_hits, alpha) -> list[FusedCandidate]).

    Delegates entirely -- per Module 9's hard requirement (its spec,
    Section 8), the fusion formula must appear in exactly one place in
    the codebase, and this is not it.

    Args:
        vector_hits: From HybridRetrievalController.run_vector_retrieval().
        graph_hits: From HybridRetrievalController.run_graph_retrieval().
        alpha: The vector-retrieval weight to fuse with, in [0, 1]
            (typically AdaptiveFusionEngine.compute_alpha()'s output).

    Returns:
        The fused, deduplicated, descending-sorted candidate list.
    """
    return HybridRetrievalController.static_fuse(vector_hits, graph_hits, alpha=alpha)
