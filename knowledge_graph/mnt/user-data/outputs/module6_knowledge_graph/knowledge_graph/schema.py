"""
Module 6 - Knowledge Graph: data schema definitions.

Defines GraphSchema (the closed set of allowed node labels and
relationship types the graph will accept -- Section 4.4's "define the
schema up front, because it directly determines what questions the
graph can answer") and GraphBuildReport (a summary of one
build_graph() run, per Module 6's contract: "no in-memory return value
-- side effect on the DB; a GraphBuildReport (node/edge counts,
sample-validated precision)").

Reference: Capstone Project Guide, Section 3F/3I/4, Phase 4 Module 6
contract, Phase 5 file structure (knowledge_graph/schema.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ner.schema import Entity, EntityType
from relation_extraction.schema import Relation, RelationType

logger = logging.getLogger(__name__)


class GraphSchemaError(Exception):
    """Raised when an entity or relation falls outside the allowed graph schema."""


@dataclass(frozen=True, slots=True)
class GraphSchema:
    """
    The fixed set of node labels and relationship types the graph
    accepts. Constructing nodes/relationships outside this schema is
    rejected up front, before ever reaching Neo4j -- catching a
    misconfigured entity/relation type extractor cheaply, in Python,
    rather than as a silent bad write.

    Attributes:
        allowed_node_labels: Node labels (EntityType values) this
            schema accepts.
        allowed_relationship_types: Relationship types (RelationType
            values) this schema accepts.
    """

    allowed_node_labels: frozenset[str]
    allowed_relationship_types: frozenset[str]

    @classmethod
    def default(cls) -> "GraphSchema":
        """
        Builds the default schema covering every EntityType and
        RelationType defined in Modules 4 and 5 -- the representative
        schema from Section 4.4.
        """
        return cls(
            allowed_node_labels=frozenset(entity_type.value for entity_type in EntityType),
            allowed_relationship_types=frozenset(
                relation_type.value for relation_type in RelationType
            ),
        )

    def validate_entity(self, entity: Entity) -> None:
        """
        Args:
            entity: The entity to validate against this schema.

        Raises:
            GraphSchemaError: If entity.entity_type is not in
                allowed_node_labels.
        """
        if entity.entity_type.value not in self.allowed_node_labels:
            raise GraphSchemaError(
                f"Entity type {entity.entity_type.value!r} is not in the "
                f"allowed node labels: {sorted(self.allowed_node_labels)}"
            )

    def validate_relation(self, relation: Relation) -> None:
        """
        Args:
            relation: The relation to validate against this schema.

        Raises:
            GraphSchemaError: If relation.predicate is not in
                allowed_relationship_types, or if either endpoint
                entity fails validate_entity.
        """
        if relation.predicate.value not in self.allowed_relationship_types:
            raise GraphSchemaError(
                f"Relationship type {relation.predicate.value!r} is not in "
                f"the allowed relationship types: "
                f"{sorted(self.allowed_relationship_types)}"
            )
        self.validate_entity(relation.subject_entity)
        self.validate_entity(relation.object_entity)


@dataclass(frozen=True, slots=True)
class GraphBuildReport:
    """
    Summary of one Neo4jGraphBuilder.build_graph() run. The build
    itself has no in-memory return value -- its effect is entirely on
    the Neo4j database -- so this report is the only signal the caller
    gets about what happened, per Module 6's contract.

    Attributes:
        nodes_created: Count of entities that resulted in a genuinely
            new node (MERGE created rather than matched).
        nodes_merged: Count of entities that matched an existing node
            (already present, so MERGE was a no-op create-wise).
        relationships_created: Count of relations that resulted in a
            genuinely new edge.
        relationships_merged: Count of relations that matched an
            existing edge.
        errors: Human-readable messages for entities/relations that
            were rejected (schema violation) or failed to write,
            keeping one bad record from aborting the whole batch.
    """

    nodes_created: int
    nodes_merged: int
    relationships_created: int
    relationships_merged: int
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def total_nodes_processed(self) -> int:
        """Total entities successfully processed (created + merged)."""
        return self.nodes_created + self.nodes_merged

    @property
    def total_relationships_processed(self) -> int:
        """Total relations successfully processed (created + merged)."""
        return self.relationships_created + self.relationships_merged

    @property
    def has_errors(self) -> bool:
        """True if any entity/relation was rejected or failed to write."""
        return len(self.errors) > 0
