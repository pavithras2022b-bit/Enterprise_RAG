"""
Module 6 - Knowledge Graph: graph population.

Provides Neo4jGraphBuilder, which turns Entity/Relation objects
(Modules 4 and 5) into nodes and relationships in Neo4j, using
dedup-safe MERGE semantics throughout so re-running ingestion never
creates duplicate nodes for the same canonical entity.

Reference: Capstone Project Guide, Section 3F/4.2/4.3 (node/relationship
creation, entity resolution/dedup), Phase 4 Module 6 contract, Phase 5
file structure (knowledge_graph/builder.py).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from knowledge_graph.connection import Neo4jConnection, Neo4jConnectionError
from knowledge_graph.schema import GraphBuildReport, GraphSchema, GraphSchemaError
from ner.schema import Entity
from relation_extraction.schema import Relation

logger = logging.getLogger(__name__)


class GraphBuildError(Exception):
    """Raised when a node or relationship write fails for a reason other than schema validation."""


class Neo4jGraphBuilder:
    """
    Writes Entity and Relation objects into Neo4j as a property graph,
    validating each against a GraphSchema before writing.

    Node labels and relationship types are interpolated directly into
    the generated Cypher text rather than passed as parameters.
    This is intentional and safe: Cypher does not support parameterized
    labels/relationship types at all, and the values interpolated here
    are always `EntityType`/`RelationType` enum members validated by
    GraphSchema first -- never raw, unvalidated user input. All *data*
    values (names, properties) are passed as bound parameters, which is
    where injection risk actually lives.
    """

    def __init__(
        self,
        connection: Neo4jConnection,
        schema: Optional[GraphSchema] = None,
    ) -> None:
        """
        Args:
            connection: An open Neo4jConnection.
            schema: The GraphSchema to validate against. Defaults to
                GraphSchema.default() (every EntityType/RelationType).
        """
        self._connection = connection
        self._schema = schema or GraphSchema.default()

    def create_node(self, entity: Entity) -> bool:
        """
        MERGE a node for the given entity, keyed by (label,
        canonical_name) so re-processing the same real-world entity
        never creates a duplicate node (Section 4.2's entity
        resolution/dedup requirement).

        Args:
            entity: The entity to write. Should already be resolved
                (entity.canonical_name set) by Module 4's
                EntityResolver; falls back to entity.text if not.

        Returns:
            True if this call created a genuinely new node, False if
            it matched an existing node.

        Raises:
            GraphSchemaError: If entity.entity_type is outside the
                configured GraphSchema.
            GraphBuildError: If the underlying write fails.
        """
        self._schema.validate_entity(entity)

        query = (
            f"MERGE (n:{entity.entity_type.value} {{canonical_name: $canonical_name}}) "
            f"ON CREATE SET n.text = $text, n.source_chunk_id = $source_chunk_id "
            f"RETURN n"
        )
        params = {
            "canonical_name": entity.canonical_name if entity.canonical_name is not None else entity.text,
            "text": entity.text,
            "source_chunk_id": entity.source_chunk_id,
        }
        try:
            _, counters = self._connection.execute_write(query, params)
        except Neo4jConnectionError as exc:
            raise GraphBuildError(f"Failed to create node for {entity.text!r}: {exc}") from exc

        created = counters.get("nodes_created", 0) > 0
        logger.debug(
            "create_node(%s:%s) -> %s",
            entity.entity_type.value,
            entity.canonical_name if entity.canonical_name is not None else entity.text,
            "created" if created else "matched existing",
        )
        return created

    def merge_or_create(self, entity: Entity) -> bool:
        """
        Alias for create_node(), named to match the Phase 4 module
        contract's explicit "dedup-safe upsert" function name. MERGE
        semantics already make create_node() dedup-safe, so this is a
        pure alias, not separate logic.
        """
        return self.create_node(entity)

    def create_relationship(self, relation: Relation) -> bool:
        """
        MERGE a relationship between two entities, creating both
        endpoint nodes first (idempotent, since create_node() also
        uses MERGE) so a relationship can never reference a missing
        node.

        Args:
            relation: The relation to write.

        Returns:
            True if this call created a genuinely new relationship,
            False if it matched an existing one.

        Raises:
            GraphSchemaError: If relation.predicate or either endpoint
                entity is outside the configured GraphSchema.
            GraphBuildError: If the underlying write fails.
        """
        self._schema.validate_relation(relation)

        # Ensure both endpoint nodes exist. Safe to call unconditionally:
        # create_node() is itself a MERGE, so this never duplicates a
        # node that create_graph()/build_graph() already wrote.
        self.create_node(relation.subject_entity)
        self.create_node(relation.object_entity)

        set_clause = ""
        if relation.properties:
            assignments = ", ".join(f"r.{key} = ${key}" for key in relation.properties)
            set_clause = f"ON CREATE SET {assignments} "

        query = (
            f"MATCH (s:{relation.subject_entity.entity_type.value} "
            f"{{canonical_name: $subject_name}}), "
            f"(o:{relation.object_entity.entity_type.value} "
            f"{{canonical_name: $object_name}}) "
            f"MERGE (s)-[r:{relation.predicate.value}]->(o) "
            f"{set_clause}"
            f"RETURN r"
        )
        params: dict[str, Any] = {
            "subject_name": (
                relation.subject_entity.canonical_name
                if relation.subject_entity.canonical_name is not None
                else relation.subject_entity.text
            ),
            "object_name": (
                relation.object_entity.canonical_name
                if relation.object_entity.canonical_name is not None
                else relation.object_entity.text
            ),
            **relation.properties,
        }
        try:
            _, counters = self._connection.execute_write(query, params)
        except Neo4jConnectionError as exc:
            raise GraphBuildError(
                f"Failed to create relationship {relation}: {exc}"
            ) from exc

        created = counters.get("relationships_created", 0) > 0
        logger.debug("create_relationship(%s) -> %s", relation, "created" if created else "matched existing")
        return created

    def run_cypher(self, query: str, parameters: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """
        Escape hatch for arbitrary read queries against the graph
        (e.g. manual validation during development), per the Phase 4
        module contract's run_cypher(query) -> list[dict] function.

        Args:
            query: Cypher query text.
            parameters: Query parameters.

        Returns:
            Result rows as plain dicts.

        Raises:
            GraphBuildError: If the query fails.
        """
        try:
            return self._connection.execute_read(query, parameters)
        except Neo4jConnectionError as exc:
            raise GraphBuildError(f"run_cypher failed: {exc}") from exc

    def build_graph(self, entities: list[Entity], relations: list[Relation]) -> GraphBuildReport:
        """
        Write a full batch of entities and relations, continuing past
        individual failures rather than aborting the whole batch (one
        malformed extraction result should not lose an otherwise-valid
        batch), and returns a GraphBuildReport summarizing the outcome.

        Args:
            entities: Entities to write as nodes (typically Module 4's
                resolved output).
            relations: Relations to write as relationships (typically
                Module 5's output). Endpoint entities are written
                automatically if not already present in `entities`.

        Returns:
            A GraphBuildReport with created/merged counts and any
            per-record error messages.
        """
        nodes_created = 0
        nodes_merged = 0
        relationships_created = 0
        relationships_merged = 0
        errors: list[str] = []

        for entity in entities:
            try:
                if self.create_node(entity):
                    nodes_created += 1
                else:
                    nodes_merged += 1
            except (GraphSchemaError, GraphBuildError) as exc:
                logger.error("Skipping entity %r: %s", entity.text, exc)
                errors.append(f"entity {entity.text!r}: {exc}")

        for relation in relations:
            try:
                if self.create_relationship(relation):
                    relationships_created += 1
                else:
                    relationships_merged += 1
            except (GraphSchemaError, GraphBuildError) as exc:
                logger.error("Skipping relation %r: %s", relation, exc)
                errors.append(f"relation {relation}: {exc}")

        report = GraphBuildReport(
            nodes_created=nodes_created,
            nodes_merged=nodes_merged,
            relationships_created=relationships_created,
            relationships_merged=relationships_merged,
            errors=tuple(errors),
        )
        logger.info(
            "build_graph complete: %d nodes created, %d matched, %d relationships "
            "created, %d matched, %d error(s)",
            nodes_created,
            nodes_merged,
            relationships_created,
            relationships_merged,
            len(errors),
        )
        return report
