"""
retrieval.hybrid_controller (Module 9)

HybridRetrievalController is the first point where the graph and vector
retrieval mechanisms meet: it runs both in parallel for a query, normalizes
their differently-shaped scores onto [0, 1], and fuses them into one
ranked FusedCandidate list.

Public methods (`run_vector_retrieval`, `run_graph_retrieval`,
`normalize_scores`, `static_fuse`) are each independently callable -- this
is a hard requirement (Section 8 of the spec) so Module 10 (Adaptive
Retrieval) can reuse them with its own learned per-query alpha without
duplicating any fusion logic. `static_fuse` is the SINGLE SOURCE OF TRUTH
for the fusion formula; it must not be reimplemented anywhere else.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from embeddings.model import EmbeddingModel, EmbeddingModelError
from knowledge_graph.connection import Neo4jConnection, Neo4jConnectionError
from knowledge_graph.schema import GraphSchema
from vector_db.chroma_store import VectorStore, VectorStoreError
from vector_db.schema import VectorSearchResult

from .entity_linker import EntityLinker
from .schema import CandidateSource, FusedCandidate, GraphFact, GraphNodeRef

_TRAVERSAL_QUERY_TEMPLATE = """
MATCH path = (start {{canonical_name: $start_name}})-[*1..{max_hops}]->(end)
RETURN [n IN nodes(path) | n.canonical_name] AS node_names,
       [n IN nodes(path) | labels(n)[0]] AS node_labels,
       [r IN relationships(path) | type(r)] AS rel_types,
       [r IN relationships(path) | r.source_chunk_id] AS source_chunk_ids,
       length(path) AS hops
""".strip()


class RetrievalError(Exception):
    """Wraps failures from the underlying vector/graph/embedding backends."""


class HybridRetrievalController:
    def __init__(
        self,
        vector_store: VectorStore,
        graph_connection: Neo4jConnection,
        embedding_model: EmbeddingModel,
        entity_linker: EntityLinker,
        max_hops: int = 3,
        graph_schema: Optional[GraphSchema] = None,
    ) -> None:
        self._vector_store = vector_store
        self._graph_connection = graph_connection
        self._embedding_model = embedding_model
        self._entity_linker = entity_linker
        self._max_hops = max_hops
        self._graph_schema = graph_schema if graph_schema is not None else GraphSchema.default()

    # ------------------------------------------------------------------
    # Vector branch
    # ------------------------------------------------------------------
    def run_vector_retrieval(
        self,
        query: str,
        k: int = 20,
        where: Optional[dict[str, Any]] = None,
    ) -> list[VectorSearchResult]:
        if query is None or not query.strip():
            raise ValueError("query must be non-empty.")
        if k <= 0:
            raise ValueError("k must be positive.")

        try:
            query_vector = self._embedding_model.embed(query)
        except EmbeddingModelError as exc:
            raise RetrievalError(f"Failed to embed query: {exc}") from exc

        try:
            return self._vector_store.search(query_vector, k=k, where=where)
        except VectorStoreError as exc:
            raise RetrievalError(f"Vector store search failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Graph branch
    # ------------------------------------------------------------------
    def run_graph_retrieval(
        self,
        entities: list[GraphNodeRef],
        max_hops: Optional[int] = None,
        k: int = 20,
    ) -> list[GraphFact]:
        effective_max_hops = self._max_hops if max_hops is None else max_hops
        if effective_max_hops <= 0:
            raise ValueError("max_hops must be positive.")
        if k <= 0:
            raise ValueError("k must be positive.")

        if not entities:
            return []

        # max_hops is bounds-checked above BEFORE string interpolation --
        # never interpolate a value that hasn't first been type/range
        # checked in Python (Section 9's Cypher safety note).
        query = _TRAVERSAL_QUERY_TEMPLATE.format(max_hops=int(effective_max_hops))

        best_by_fact_id: dict[str, GraphFact] = {}

        for entity_ref in entities:
            try:
                rows = self._graph_connection.execute_read(
                    query, {"start_name": entity_ref.canonical_name}
                )
            except Neo4jConnectionError as exc:
                raise RetrievalError(f"Graph traversal failed: {exc}") from exc

            for row in rows:
                node_names: list[str] = row.get("node_names") or []
                rel_types: list[str] = row.get("rel_types") or []
                source_chunk_ids: list[Optional[str]] = row.get("source_chunk_ids") or []

                for i, rel_type in enumerate(rel_types):
                    if rel_type not in self._graph_schema.allowed_relationship_types:
                        raise RetrievalError(
                            "Graph traversal returned an out-of-schema "
                            f"relationship type: {rel_type!r}."
                        )

                    subject_name = node_names[i]
                    object_name = node_names[i + 1]
                    hop_count = i + 1
                    source_chunk_id = (
                        source_chunk_ids[i] if i < len(source_chunk_ids) else None
                    )

                    fact = GraphFact.build(
                        subject=subject_name,
                        predicate=rel_type,
                        obj=object_name,
                        hop_count=hop_count,
                        source_chunk_id=source_chunk_id,
                    )

                    existing = best_by_fact_id.get(fact.fact_id)
                    if existing is None or fact.hop_count < existing.hop_count:
                        best_by_fact_id[fact.fact_id] = fact

        ranked = sorted(
            best_by_fact_id.values(), key=lambda f: f.raw_relevance, reverse=True
        )
        return ranked[:k]

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_scores(raw_scores: list[float]) -> list[float]:
        if not raw_scores:
            return []

        for value in raw_scores:
            if not math.isfinite(value):
                raise ValueError(f"raw_scores must all be finite, got {value}.")

        lo = min(raw_scores)
        hi = max(raw_scores)

        if hi == lo:
            # A lone candidate, or a set of equally-scored candidates, has
            # nothing to be ranked below within its own set -- it must not
            # be punished to 0.0 by a degenerate min-max range.
            return [1.0 for _ in raw_scores]

        return [(value - lo) / (hi - lo) for value in raw_scores]

    # ------------------------------------------------------------------
    # Fusion (single source of truth)
    # ------------------------------------------------------------------
    @staticmethod
    def static_fuse(
        vector_hits: list[VectorSearchResult],
        graph_hits: list[GraphFact],
        alpha: float = 0.5,
    ) -> list[FusedCandidate]:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}.")

        if not vector_hits and not graph_hits:
            return []

        vector_scores_norm = HybridRetrievalController.normalize_scores(
            [hit.score for hit in vector_hits]
        )
        graph_scores_norm = HybridRetrievalController.normalize_scores(
            [fact.raw_relevance for fact in graph_hits]
        )

        # chunk_id -> (VectorSearchResult, normalized score)
        vector_by_id: dict[str, tuple[VectorSearchResult, float]] = {
            hit.chunk_id: (hit, norm)
            for hit, norm in zip(vector_hits, vector_scores_norm)
        }
        # fact_id -> (GraphFact, normalized score)
        graph_by_id: dict[str, tuple[GraphFact, float]] = {
            fact.fact_id: (fact, norm)
            for fact, norm in zip(graph_hits, graph_scores_norm)
        }

        # Dedup rule (Section 7.3): merge a vector hit and a graph fact iff
        # the graph fact's source_chunk_id equals the vector hit's chunk_id.
        chunk_id_to_fact_ids: dict[str, list[str]] = {}
        for fact_id, (fact, _norm) in graph_by_id.items():
            if fact.source_chunk_id is not None and fact.source_chunk_id in vector_by_id:
                chunk_id_to_fact_ids.setdefault(fact.source_chunk_id, []).append(fact_id)

        merged_fact_ids: set[str] = set()
        candidates: list[FusedCandidate] = []

        for chunk_id, (hit, vec_norm) in vector_by_id.items():
            matching_fact_ids = chunk_id_to_fact_ids.get(chunk_id)
            if matching_fact_ids:
                # Merge with EVERY graph fact sharing this source_chunk_id
                # into one BOTH candidate -- a chunk can have had more than
                # one relation extracted from it (Module 5), and each is
                # equally "evidence from this exact passage" per the
                # Section 7.3 dedup rule. Using only the first and letting
                # the rest re-emit as separate GRAPH candidates would
                # under-merge and misrepresent how much graph support this
                # chunk actually has.
                matching_facts = [graph_by_id[fid] for fid in matching_fact_ids]
                graph_norm = max(norm for _fact, norm in matching_facts)
                merged_fact_ids.update(matching_fact_ids)

                metadata = dict(hit.metadata)
                for fact, _norm in matching_facts:
                    for key, value in fact.metadata.items():
                        metadata.setdefault(key, value)

                fused_score = alpha * vec_norm + (1 - alpha) * graph_norm
                candidates.append(
                    FusedCandidate(
                        candidate_id=chunk_id,
                        source=CandidateSource.BOTH,
                        vector_score=vec_norm,
                        graph_score=graph_norm,
                        fused_score=fused_score,
                        text=hit.text,
                        metadata=metadata,
                    )
                )
            else:
                fused_score = alpha * vec_norm + (1 - alpha) * 0.0
                candidates.append(
                    FusedCandidate(
                        candidate_id=chunk_id,
                        source=CandidateSource.VECTOR,
                        vector_score=vec_norm,
                        graph_score=None,
                        fused_score=fused_score,
                        text=hit.text,
                        metadata=dict(hit.metadata),
                    )
                )

        for fact_id, (fact, graph_norm) in graph_by_id.items():
            if fact_id in merged_fact_ids:
                continue
            fused_score = alpha * 0.0 + (1 - alpha) * graph_norm
            candidates.append(
                FusedCandidate(
                    candidate_id=fact_id,
                    source=CandidateSource.GRAPH,
                    vector_score=None,
                    graph_score=graph_norm,
                    fused_score=fused_score,
                    text=fact.text,
                    metadata=dict(fact.metadata),
                )
            )

        candidates.sort(key=lambda c: c.fused_score, reverse=True)
        return candidates

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        k_vector: int = 20,
        k_graph: int = 20,
        alpha: float = 0.5,
        where: Optional[dict[str, Any]] = None,
    ) -> list[FusedCandidate]:
        entities = self._entity_linker.link(query)
        vector_hits = self.run_vector_retrieval(query, k=k_vector, where=where)
        graph_hits = self.run_graph_retrieval(entities, k=k_graph)
        return self.static_fuse(vector_hits, graph_hits, alpha=alpha)
