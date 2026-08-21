"""
Module 6 - Knowledge Graph: schema definitions.

Defines the RelationType enum, Relation dataclass, GraphSchema,
GraphSchemaError, and GraphBuildReport.

Reference: Capstone Project Guide, Section 3F/4.2/4.3, Phase 4 Module 6 contract.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ner.schema import Entity, EntityType
from relation_extraction.schema import Relation, RelationType, SchemaValidationError

logger = logging.getLogger(__name__)


class GraphSchemaError(Exception):
    """Raised when an Entity or Relation fails GraphSchema validation."""


@dataclass(frozen=True, slots=True)
class GraphSchema:
    """
    Schema validation rules for nodes and relationships written to Neo4j.
    """

    allowed_node_labels: frozenset[str]
    allowed_relationship_types: frozenset[str]

    @classmethod
    def default(cls) -> GraphSchema:
        """
        Returns a GraphSchema allowing all EntityType values as node labels
        and all RelationType values as relationship types.
        """
        return cls(
            allowed_node_labels=frozenset(e.value for e in EntityType),
            allowed_relationship_types=frozenset(r.value for r in RelationType),
        )

    def validate_entity(self, entity: Entity) -> None:
        """
        Validates that entity.entity_type is in allowed_node_labels.

        Raises:
            GraphSchemaError: If entity type is disallowed.
        """
        label = entity.entity_type.value
        if label not in self.allowed_node_labels:
            raise GraphSchemaError(
                f"Entity type {label!r} is not in allowed node labels: "
                f"{sorted(self.allowed_node_labels)}"
            )

    def validate_relation(self, relation: Relation) -> None:
        """
        Validates that relation.predicate is in allowed_relationship_types
        and both endpoint entities are valid.

        Raises:
            GraphSchemaError: If relation predicate or endpoint entities are disallowed.
        """
        rel_type = relation.predicate.value
        if rel_type not in self.allowed_relationship_types:
            raise GraphSchemaError(
                f"Relation type {rel_type!r} is not in allowed relationship types: "
                f"{sorted(self.allowed_relationship_types)}"
            )
        self.validate_entity(relation.subject_entity)
        self.validate_entity(relation.object_entity)


@dataclass(frozen=True, slots=True)
class GraphBuildReport:
    """
    Summary metrics and diagnostic logs returned by Neo4jGraphBuilder.build_graph.
    """

    nodes_created: int = 0
    nodes_merged: int = 0
    relationships_created: int = 0
    relationships_merged: int = 0
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def total_nodes_processed(self) -> int:
        return self.nodes_created + self.nodes_merged

    @property
    def total_relationships_processed(self) -> int:
        return self.relationships_created + self.relationships_merged

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0


# Re-export Relation, RelationType, SchemaValidationError for backward compatibility
__all__ = [
    "RelationType",
    "SchemaValidationError",
    "Relation",
    "GraphSchemaError",
    "GraphSchema",
    "GraphBuildReport",
]
