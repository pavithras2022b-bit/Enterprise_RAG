"""
Module 4 - NER: entity resolution / deduplication.

Provides resolve_entity(...) and the EntityResolver class per the
Phase 4 module contract, merging entity mentions that refer to the
same real-world thing under different surface text (e.g. "Finance",
"Finance Dept.", "the Finance team" all resolving to "Finance") via
fuzzy string matching, per Section 4.2's documented approach.

Reference: Capstone Project Guide, Section 4.2 ("How Nodes Are
Created" -- entity resolution/deduplication), Phase 4 Module 4
contract, Phase 5 file structure (ner/resolver.py).
"""

from __future__ import annotations

import logging
from dataclasses import replace

from rapidfuzz import fuzz, process

from ner.schema import Entity

logger = logging.getLogger(__name__)


class EntityResolutionError(Exception):
    """Raised when entity resolution cannot proceed for the given input."""


def resolve_entity(
    entity: Entity,
    existing_nodes: list[Entity],
    similarity_threshold: float = 85.0,
) -> Entity:
    """
    Resolve a single entity against a list of already-known entities,
    assigning it a canonical_name -- either merging into an existing
    node's canonical name (if a same-type match scores at or above
    similarity_threshold) or establishing itself as a new canonical
    node (using its own text).

    Uses rapidfuzz's token_sort_ratio, which is robust to word-order
    differences and minor phrasing variation (e.g. "Finance
    department" vs "department of Finance" still score highly), per
    Section 4.2's "fuzzy-matching, via string similarity like
    Levenshtein distance" guidance.

    Args:
        entity: The entity to resolve.
        existing_nodes: Already-resolved entities to compare against.
            Only entities of the same entity_type are considered
            candidates (an Employee should never fuzzy-match a
            Department, however similar the text).
        similarity_threshold: Minimum rapidfuzz token_sort_ratio score
            (0-100) for a match to be accepted. Higher is stricter.

    Returns:
        A new Entity (the input is immutable) with canonical_name set.

    Raises:
        ValueError: If similarity_threshold is not in [0, 100].
    """
    if not 0.0 <= similarity_threshold <= 100.0:
        raise ValueError(
            f"similarity_threshold must be in [0, 100], got {similarity_threshold}"
        )

    same_type_candidates = [
        node for node in existing_nodes if node.entity_type == entity.entity_type
    ]

    if same_type_candidates:
        candidate_names = [
            node.canonical_name or node.text for node in same_type_candidates
        ]
        match_result = process.extractOne(
            entity.text, candidate_names, scorer=fuzz.token_sort_ratio
        )
        if match_result is not None:
            matched_name, score, _ = match_result
            if score >= similarity_threshold:
                logger.debug(
                    "Resolved %r -> existing canonical name %r (score=%.1f)",
                    entity.text,
                    matched_name,
                    score,
                )
                return replace(entity, canonical_name=matched_name)

    canonical_name = entity.text.strip()
    logger.debug(
        "Resolved %r as a new canonical node %r (no match >= %.1f)",
        entity.text,
        canonical_name,
        similarity_threshold,
    )
    return replace(entity, canonical_name=canonical_name)


class EntityResolver:
    """
    Stateful, batch-oriented wrapper around resolve_entity(): processes
    a stream of entities, incrementally growing the set of known
    canonical nodes so each new entity is resolved against everything
    seen so far.
    """

    def __init__(self, similarity_threshold: float = 85.0) -> None:
        """
        Args:
            similarity_threshold: Forwarded to resolve_entity() for
                every entity processed by this resolver instance.

        Raises:
            ValueError: If similarity_threshold is not in [0, 100].
        """
        if not 0.0 <= similarity_threshold <= 100.0:
            raise ValueError(
                f"similarity_threshold must be in [0, 100], got {similarity_threshold}"
            )
        self._similarity_threshold = similarity_threshold
        self._resolved: list[Entity] = []

    @property
    def resolved_count(self) -> int:
        """Returns the number of entities resolved so far by this instance."""
        return len(self._resolved)

    def resolve(self, entity: Entity) -> Entity:
        """
        Resolve a single entity against every entity this resolver has
        already processed, then record the result for future calls to
        match against.

        Args:
            entity: The entity to resolve.

        Returns:
            The resolved Entity (with canonical_name set).
        """
        resolved_entity = resolve_entity(
            entity, self._resolved, similarity_threshold=self._similarity_threshold
        )
        self._resolved.append(resolved_entity)
        return resolved_entity

    def resolve_all(self, entities: list[Entity]) -> list[Entity]:
        """
        Resolve a batch of entities in order, each against everything
        resolved before it (within this call and any prior calls to
        this resolver instance).

        Args:
            entities: Entities to resolve, in processing order.

        Returns:
            A list of resolved Entity objects, same length and order
            as entities.
        """
        return [self.resolve(entity) for entity in entities]

    def canonical_names(self) -> dict[str, list[str]]:
        """
        Returns a mapping of entity_type value -> sorted list of
        distinct canonical names established so far, useful for a
        Module 6 (Knowledge Graph) sanity-check report of how many
        distinct nodes were actually created per type.
        """
        names_by_type: dict[str, set[str]] = {}
        for entity in self._resolved:
            if entity.canonical_name is None:
                continue
            names_by_type.setdefault(entity.entity_type.value, set()).add(
                entity.canonical_name
            )
        return {
            entity_type: sorted(names) for entity_type, names in names_by_type.items()
        }
