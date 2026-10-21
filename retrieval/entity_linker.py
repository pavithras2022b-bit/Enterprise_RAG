"""
retrieval.entity_linker (Module 9)

EntityLinker maps free-text mentions in a query to known graph node
canonical names, giving graph traversal a starting point. This is
deliberately lightweight -- fuzzy substring matching against a cached
index of (entity_type, canonical_name) pairs pulled from Neo4j -- and is
NOT a second NER pass (Module 4's NERModel is not invoked here).
"""

from __future__ import annotations

import logging
from rapidfuzz import fuzz

from knowledge_graph.connection import Neo4jConnection, Neo4jConnectionError
from ner.schema import EntityType
from .schema import GraphNodeRef, SchemaValidationError

logger = logging.getLogger(__name__)

_INDEX_QUERY = "MATCH (n) RETURN labels(n)[0] AS label, n.canonical_name AS canonical_name"
_SCOPED_INDEX_QUERY = (
    "MATCH (n) WHERE n.dataset_id = $dataset_id "
    "RETURN labels(n)[0] AS label, n.canonical_name AS canonical_name"
)


class EntityLinkerError(Exception):
    """Raised when index refresh fails, or link() is called on an
    index that has never been populated."""


class EntityLinker:
    def __init__(
        self,
        connection: Neo4jConnection,
        similarity_threshold: float = 80.0,
        refresh_on_init: bool = True,
        dataset_id: str | None = None,
    ) -> None:
        if not (0.0 <= similarity_threshold <= 100.0):
            raise ValueError(
                f"similarity_threshold must be in [0, 100], got {similarity_threshold}."
            )

        self._connection = connection
        self._similarity_threshold = similarity_threshold
        self._dataset_id = dataset_id
        # Index entries: list of (label, canonical_name). None until the
        # first successful refresh_index() call.
        self._index: list[tuple[str, str]] | None = None

        if refresh_on_init:
            self.refresh_index()

    def refresh_index(self) -> None:
        """Re-fetches (label, canonical_name) pairs from Neo4j and rebuilds
        the in-memory index."""
        try:
            if self._dataset_id is None:
                rows = self._connection.execute_read(_INDEX_QUERY)
            else:
                rows = self._connection.execute_read(
                    _SCOPED_INDEX_QUERY, {"dataset_id": self._dataset_id}
                )
        except Exception as exc:
            logger.warning("Failed to refresh entity index: %s; falling back to empty index", exc)
            self._index = []
            return

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
        Link entity mentions in a query to known Neo4j canonical names.

        Exact canonical-name matches are preferred. If no exact match
        exists, fuzzy matching is used as a fallback. Very short graph
        fragments are ignored to avoid partial_ratio false positives.
        """
        if self._index is None:
            return []

        if not query or not query.strip():
            return []

        query_lower = query.lower()

        # --------------------------------------------------------------
        # 1. Prefer exact canonical-name occurrences in the query.
        # --------------------------------------------------------------
        exact_matches: list[GraphNodeRef] = []

        for label, canonical_name in self._index:
            clean_name = canonical_name.strip()

            # Ignore tiny fragmented graph nodes such as "A", "S", "Ka".
            if len(clean_name) < 3:
                continue

            try:
                entity_type = EntityType(label)
            except ValueError:
                continue

            if clean_name.lower() in query_lower:
                try:
                    exact_matches.append(
                        GraphNodeRef(
                            mention_text=clean_name,
                            canonical_name=clean_name,
                            entity_type=entity_type,
                            match_score=100.0,
                        )
                    )
                except SchemaValidationError:
                    continue

        if exact_matches:
            # Prefer the longest exact canonical name.
            max_length = max(
                len(ref.canonical_name)
                for ref in exact_matches
            )

            longest_matches = [
                ref
                for ref in exact_matches
                if len(ref.canonical_name) == max_length
            ]

            return sorted(
                longest_matches,
                key=lambda ref: ref.match_score,
                reverse=True,
            )

        # --------------------------------------------------------------
        # 2. Fuzzy fallback when no exact entity occurs in the query.
        # --------------------------------------------------------------
        best_by_name: dict[str, GraphNodeRef] = {}

        for label, canonical_name in self._index:
            clean_name = canonical_name.strip()

            # Fuzzy matching short strings is extremely noisy.
            if len(clean_name) < 5:
                continue

            score = fuzz.partial_ratio(
                clean_name.lower(),
                query_lower,
            )

            if score < self._similarity_threshold:
                continue

            try:
                entity_type = EntityType(label)
            except ValueError:
                continue

            try:
                candidate = GraphNodeRef(
                    mention_text=clean_name,
                    canonical_name=clean_name,
                    entity_type=entity_type,
                    match_score=float(score),
                )
            except SchemaValidationError:
                continue

            existing = best_by_name.get(clean_name)

            if (
                existing is None
                or candidate.match_score > existing.match_score
            ):
                best_by_name[clean_name] = candidate

        return sorted(
            best_by_name.values(),
            key=lambda ref: ref.match_score,
            reverse=True,
        )
