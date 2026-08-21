"""
retrieval.entity_linker (Module 9)

EntityLinker maps free-text mentions in a query to known graph node
canonical names, giving graph traversal a starting point. This is
deliberately lightweight -- fuzzy substring matching against a cached
index of (entity_type, canonical_name) pairs pulled from Neo4j -- and is
NOT a second NER pass (Module 4's NERModel is not invoked here).
"""

from __future__ import annotations

from rapidfuzz import fuzz

from knowledge_graph.connection import Neo4jConnection, Neo4jConnectionError
from ner.schema import EntityType
from .schema import GraphNodeRef, SchemaValidationError

_INDEX_QUERY = "MATCH (n) RETURN labels(n)[0] AS label, n.canonical_name AS canonical_name"


class EntityLinkerError(Exception):
    """Raised when index refresh fails, or link() is called on an
    index that has never been populated."""


class EntityLinker:
    def __init__(
        self,
        connection: Neo4jConnection,
        similarity_threshold: float = 80.0,
        refresh_on_init: bool = True,
    ) -> None:
        if not (0.0 <= similarity_threshold <= 100.0):
            raise ValueError(
                f"similarity_threshold must be in [0, 100], got {similarity_threshold}."
            )

        self._connection = connection
        self._similarity_threshold = similarity_threshold
        # Index entries: list of (label, canonical_name). None until the
        # first successful refresh_index() call.
        self._index: list[tuple[str, str]] | None = None

        if refresh_on_init:
            self.refresh_index()

    def refresh_index(self) -> None:
        """Re-fetches (label, canonical_name) pairs from Neo4j and rebuilds
        the in-memory index."""
        try:
            rows = self._connection.execute_read(_INDEX_QUERY)
        except Neo4jConnectionError as exc:
            raise EntityLinkerError(
                f"Failed to refresh entity index: {exc}"
            ) from exc

        index: list[tuple[str, str]] = []
        for row in rows:
            label = row.get("label")
            canonical_name = row.get("canonical_name")
            if label is None or canonical_name is None:
                # A node with no label or no canonical_name is not linkable;
                # skip it rather than failing the whole refresh.
                continue
            index.append((label, canonical_name))

        self._index = index

    def link(self, query: str) -> list[GraphNodeRef]:
        """
        Fuzzy-matches `query` against every cached canonical_name using
        rapidfuzz.fuzz.partial_ratio (appropriate here because the query is
        a full sentence and the canonical name is typically a short
        substring of it -- see Section 5.2 of the spec). Returns one
        GraphNodeRef per canonical_name scoring >= similarity_threshold,
        deduplicated by canonical_name (keeping the highest-scoring match),
        sorted by descending match_score.
        
        If the index is not populated (e.g., Neo4j unavailable), returns
        an empty list for graceful degradation to vector-only retrieval.
        """
        if self._index is None:
            # Graceful degradation: return empty list if index not populated
            return []

        if not query or not query.strip():
            return []

        best_by_name: dict[str, GraphNodeRef] = {}

        for label, canonical_name in self._index:
            score = fuzz.partial_ratio(canonical_name.lower(), query.lower())
            if score < self._similarity_threshold:
                continue

            try:
                entity_type = EntityType(label)
            except ValueError:
                # A node label outside the known EntityType enum isn't
                # something Module 9 can meaningfully link to; skip it
                # rather than raising, consistent with the "never crash on
                # a stray node" spirit of the empty-index behavior rule.
                continue

            try:
                candidate = GraphNodeRef(
                    mention_text=canonical_name,
                    canonical_name=canonical_name,
                    entity_type=entity_type,
                    match_score=float(score),
                )
            except SchemaValidationError:
                continue

            existing = best_by_name.get(canonical_name)
            if existing is None or candidate.match_score > existing.match_score:
                best_by_name[canonical_name] = candidate

        return sorted(
            best_by_name.values(), key=lambda ref: ref.match_score, reverse=True
        )
