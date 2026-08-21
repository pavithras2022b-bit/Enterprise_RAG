"""
Unit tests for Module 6 - Knowledge Graph.

Covers knowledge_graph/schema.py, connection.py, builder.py, and
cypher_queries.py, plus relation_extraction/schema.py (the Relation
contract Module 6 consumes). Uses the fake_neo4j_connection fixture
(see conftest.py) -- an in-memory graph store standing in for a real
Neo4j server, which this sandbox's network cannot reach.
"""

from __future__ import annotations

import pytest

from knowledge_graph.builder import GraphBuildError, Neo4jGraphBuilder
from knowledge_graph.connection import Neo4jConnection, Neo4jConnectionError
from knowledge_graph.cypher_queries import CypherQueryLibrary, CypherQueryLibraryError
from knowledge_graph.schema import GraphBuildReport, GraphSchema, GraphSchemaError
from ner.schema import Entity, EntityType
from relation_extraction.schema import Relation, RelationType, SchemaValidationError


def _entity(text: str, entity_type: EntityType, canonical_name: str | None = None) -> Entity:
    return Entity(
        text=text,
        entity_type=entity_type,
        start_offset=0,
        end_offset=len(text),
        source_chunk_id="chunk_0001",
        canonical_name=canonical_name or text,
    )


# --------------------------------------------------------------------------
# relation_extraction/schema.py
# --------------------------------------------------------------------------


class TestRelation:
    def test_construction(self) -> None:
        subject = _entity("Asha Rao", EntityType.EMPLOYEE)
        obj = _entity("Finance", EntityType.DEPARTMENT)
        relation = Relation(
            subject_entity=subject,
            predicate=RelationType.BELONGS_TO,
            object_entity=obj,
            source_chunk_id="chunk_0001",
        )
        assert relation.predicate == RelationType.BELONGS_TO
        assert "BELONGS_TO" in repr(relation)

    def test_self_relation_raises(self) -> None:
        entity = _entity("Asha Rao", EntityType.EMPLOYEE)
        with pytest.raises(SchemaValidationError):
            Relation(
                subject_entity=entity,
                predicate=RelationType.MANAGES,
                object_entity=entity,
                source_chunk_id="chunk_0001",
            )

    def test_empty_source_chunk_id_raises(self) -> None:
        subject = _entity("Asha Rao", EntityType.EMPLOYEE)
        obj = _entity("Finance", EntityType.DEPARTMENT)
        with pytest.raises(SchemaValidationError):
            Relation(
                subject_entity=subject,
                predicate=RelationType.BELONGS_TO,
                object_entity=obj,
                source_chunk_id="",
            )

    def test_confidence_out_of_range_raises(self) -> None:
        subject = _entity("Asha Rao", EntityType.EMPLOYEE)
        obj = _entity("Finance", EntityType.DEPARTMENT)
        with pytest.raises(SchemaValidationError):
            Relation(
                subject_entity=subject,
                predicate=RelationType.BELONGS_TO,
                object_entity=obj,
                source_chunk_id="chunk_0001",
                confidence=2.0,
            )

    def test_properties_carried(self) -> None:
        subject = _entity("Finance", EntityType.DEPARTMENT)
        obj = _entity("SAP FI Certification", EntityType.SOFTWARE)
        relation = Relation(
            subject_entity=subject,
            predicate=RelationType.REQUIRES_TRAINING,
            object_entity=obj,
            source_chunk_id="chunk_0001",
            properties={"within_days": "90"},
        )
        assert relation.properties["within_days"] == "90"


# --------------------------------------------------------------------------
# knowledge_graph/schema.py
# --------------------------------------------------------------------------


class TestGraphSchema:
    def test_default_covers_all_entity_and_relation_types(self) -> None:
        schema = GraphSchema.default()
        assert "EMPLOYEE" in schema.allowed_node_labels
        assert "BELONGS_TO" in schema.allowed_relationship_types

    def test_validate_entity_passes_for_allowed_type(self) -> None:
        schema = GraphSchema.default()
        schema.validate_entity(_entity("Asha Rao", EntityType.EMPLOYEE))  # no raise

    def test_validate_entity_rejects_restricted_schema(self) -> None:
        schema = GraphSchema(
            allowed_node_labels=frozenset({"Employee"}),
            allowed_relationship_types=frozenset({"BELONGS_TO"}),
        )
        with pytest.raises(GraphSchemaError):
            schema.validate_entity(_entity("Finance", EntityType.DEPARTMENT))

    def test_validate_relation_rejects_disallowed_predicate(self) -> None:
        schema = GraphSchema(
            allowed_node_labels=frozenset({"Employee", "Department"}),
            allowed_relationship_types=frozenset({"BELONGS_TO"}),
        )
        relation = Relation(
            subject_entity=_entity("Asha Rao", EntityType.EMPLOYEE),
            predicate=RelationType.MANAGES,
            object_entity=_entity("Ravi Iyer", EntityType.EMPLOYEE),
            source_chunk_id="chunk_0001",
        )
        with pytest.raises(GraphSchemaError):
            schema.validate_relation(relation)


class TestGraphBuildReport:
    def test_totals_and_has_errors(self) -> None:
        report = GraphBuildReport(
            nodes_created=3, nodes_merged=1, relationships_created=2, relationships_merged=0,
            errors=("bad record",),
        )
        assert report.total_nodes_processed == 4
        assert report.total_relationships_processed == 2
        assert report.has_errors is True

    def test_no_errors(self) -> None:
        report = GraphBuildReport(nodes_created=1, nodes_merged=0, relationships_created=0, relationships_merged=0)
        assert report.has_errors is False


# --------------------------------------------------------------------------
# connection.py
# --------------------------------------------------------------------------


class TestNeo4jConnection:
    def test_empty_uri_raises(self) -> None:
        with pytest.raises(Neo4jConnectionError):
            Neo4jConnection(uri="", username="u", password="p", driver_factory=lambda *a, **k: None)

    def test_driver_factory_failure_raises_connection_error(self) -> None:
        def failing_factory(*args, **kwargs):
            raise RuntimeError("boom")

        with pytest.raises(Neo4jConnectionError):
            Neo4jConnection(uri="bolt://x", username="u", password="p", driver_factory=failing_factory)

    def test_execute_write_returns_records_and_counters(self, fake_neo4j_connection: Neo4jConnection) -> None:
        query = (
            "MERGE (n:Employee {canonical_name: $canonical_name}) "
            "ON CREATE SET n.text = $text, n.source_chunk_id = $source_chunk_id "
            "RETURN n"
        )
        params = {"canonical_name": "Asha Rao", "text": "Asha Rao", "source_chunk_id": "c1"}
        records, counters = fake_neo4j_connection.execute_write(query, params)
        assert counters["nodes_created"] == 1
        assert len(records) == 1

    def test_execute_write_second_call_matches_not_creates(self, fake_neo4j_connection: Neo4jConnection) -> None:
        query = (
            "MERGE (n:Employee {canonical_name: $canonical_name}) "
            "ON CREATE SET n.text = $text, n.source_chunk_id = $source_chunk_id "
            "RETURN n"
        )
        params = {"canonical_name": "Asha Rao", "text": "Asha Rao", "source_chunk_id": "c1"}
        fake_neo4j_connection.execute_write(query, params)
        _, counters = fake_neo4j_connection.execute_write(query, params)
        assert counters["nodes_created"] == 0

    def test_execute_read_unrecognized_query_raises_connection_error(
        self, fake_neo4j_connection: Neo4jConnection
    ) -> None:
        with pytest.raises(Neo4jConnectionError):
            fake_neo4j_connection.execute_read("MATCH (x) RETURN x")

    def test_context_manager_closes_driver(self) -> None:
        closed = {"value": False}

        class _Driver:
            def close(self):
                closed["value"] = True

        with Neo4jConnection(uri="bolt://x", username="u", password="p", driver_factory=lambda *a, **k: _Driver()) as conn:
            assert conn is not None
        assert closed["value"] is True


# --------------------------------------------------------------------------
# builder.py
# --------------------------------------------------------------------------


class TestNeo4jGraphBuilder:
    def test_create_node_first_time_returns_true(self, fake_neo4j_connection: Neo4jConnection) -> None:
        builder = Neo4jGraphBuilder(fake_neo4j_connection)
        created = builder.create_node(_entity("Asha Rao", EntityType.EMPLOYEE))
        assert created is True

    def test_create_node_second_time_returns_false(self, fake_neo4j_connection: Neo4jConnection) -> None:
        builder = Neo4jGraphBuilder(fake_neo4j_connection)
        entity = _entity("Asha Rao", EntityType.EMPLOYEE)
        builder.create_node(entity)
        assert builder.create_node(entity) is False

    def test_merge_or_create_is_alias_for_create_node(self, fake_neo4j_connection: Neo4jConnection) -> None:
        builder = Neo4jGraphBuilder(fake_neo4j_connection)
        entity = _entity("Asha Rao", EntityType.EMPLOYEE)
        assert builder.merge_or_create(entity) is True
        assert builder.merge_or_create(entity) is False

    def test_create_node_rejects_disallowed_schema(self, fake_neo4j_connection: Neo4jConnection) -> None:
        schema = GraphSchema(
            allowed_node_labels=frozenset({"Employee"}),
            allowed_relationship_types=frozenset(),
        )
        builder = Neo4jGraphBuilder(fake_neo4j_connection, schema=schema)
        with pytest.raises(GraphSchemaError):
            builder.create_node(_entity("Finance", EntityType.DEPARTMENT))

    def test_create_relationship_creates_both_endpoint_nodes(self, fake_neo4j_connection: Neo4jConnection) -> None:
        builder = Neo4jGraphBuilder(fake_neo4j_connection)
        relation = Relation(
            subject_entity=_entity("Asha Rao", EntityType.EMPLOYEE),
            predicate=RelationType.BELONGS_TO,
            object_entity=_entity("Finance", EntityType.DEPARTMENT),
            source_chunk_id="chunk_0001",
        )
        created = builder.create_relationship(relation)
        assert created is True
        # Endpoint nodes should now exist -- re-creating them should
        # report "matched", not "created".
        assert builder.create_node(relation.subject_entity) is False
        assert builder.create_node(relation.object_entity) is False

    def test_create_relationship_second_time_returns_false(self, fake_neo4j_connection: Neo4jConnection) -> None:
        builder = Neo4jGraphBuilder(fake_neo4j_connection)
        relation = Relation(
            subject_entity=_entity("Asha Rao", EntityType.EMPLOYEE),
            predicate=RelationType.BELONGS_TO,
            object_entity=_entity("Finance", EntityType.DEPARTMENT),
            source_chunk_id="chunk_0001",
        )
        builder.create_relationship(relation)
        assert builder.create_relationship(relation) is False

    def test_create_relationship_with_properties(self, fake_neo4j_connection: Neo4jConnection) -> None:
        builder = Neo4jGraphBuilder(fake_neo4j_connection)
        relation = Relation(
            subject_entity=_entity("Finance", EntityType.DEPARTMENT),
            predicate=RelationType.REQUIRES_TRAINING,
            object_entity=_entity("SAP FI Certification", EntityType.SOFTWARE),
            source_chunk_id="chunk_0001",
            properties={"within_days": "90"},
        )
        assert builder.create_relationship(relation) is True

    def test_build_graph_end_to_end(self, fake_neo4j_connection: Neo4jConnection) -> None:
        builder = Neo4jGraphBuilder(fake_neo4j_connection)
        asha = _entity("Asha Rao", EntityType.EMPLOYEE)
        finance = _entity("Finance", EntityType.DEPARTMENT)
        leave_policy = _entity("Leave Policy", EntityType.POLICY)
        casual_leave = _entity("Casual Leave", EntityType.LEAVE_TYPE)

        entities = [asha, finance, leave_policy, casual_leave]
        relations = [
            Relation(asha, RelationType.BELONGS_TO, finance, source_chunk_id="c1"),
            Relation(finance, RelationType.FOLLOWS, leave_policy, source_chunk_id="c1"),
            Relation(leave_policy, RelationType.GOVERNS, casual_leave, source_chunk_id="c1"),
        ]

        report = builder.build_graph(entities, relations)

        assert isinstance(report, GraphBuildReport)
        assert report.nodes_created == 4
        assert report.relationships_created == 3
        assert report.has_errors is False

    def test_build_graph_continues_past_schema_errors(self, fake_neo4j_connection: Neo4jConnection) -> None:
        schema = GraphSchema(
            allowed_node_labels=frozenset({"EMPLOYEE"}),
            allowed_relationship_types=frozenset(),
        )
        builder = Neo4jGraphBuilder(fake_neo4j_connection, schema=schema)
        entities = [_entity("Asha Rao", EntityType.EMPLOYEE), _entity("Finance", EntityType.DEPARTMENT)]

        report = builder.build_graph(entities, relations=[])

        assert report.nodes_created == 1
        assert report.has_errors is True
        assert len(report.errors) == 1

    def test_run_cypher_delegates_to_connection(self, fake_neo4j_connection: Neo4jConnection) -> None:
        builder = Neo4jGraphBuilder(fake_neo4j_connection)
        builder.create_node(_entity("Asha Rao", EntityType.EMPLOYEE))
        finance = _entity("Finance", EntityType.DEPARTMENT)
        builder.create_relationship(
            Relation(
                subject_entity=_entity("Asha Rao", EntityType.EMPLOYEE),
                predicate=RelationType.BELONGS_TO,
                object_entity=finance,
                source_chunk_id="c1",
            )
        )
        rows = builder.run_cypher(
            "MATCH (e:EMPLOYEE {canonical_name: $employee_name})-[:BELONGS_TO]->(d:DEPARTMENT) "
            "RETURN d.canonical_name AS department",
            {"employee_name": "Asha Rao"},
        )
        assert rows == [{"department": "Finance"}]


# --------------------------------------------------------------------------
# cypher_queries.py
# --------------------------------------------------------------------------


class TestCypherQueryLibrary:
    def _build_sample_graph(self, connection: Neo4jConnection) -> None:
        builder = Neo4jGraphBuilder(connection)
        asha = _entity("Asha Rao", EntityType.EMPLOYEE)
        finance = _entity("Finance", EntityType.DEPARTMENT)
        leave_policy = _entity("Leave Policy", EntityType.POLICY)
        casual_leave = _entity("Casual Leave", EntityType.LEAVE_TYPE)
        sap_fi = _entity("SAP FI Certification", EntityType.SOFTWARE)
        ravi = _entity("Ravi Iyer", EntityType.EMPLOYEE)

        builder.build_graph(
            entities=[asha, finance, leave_policy, casual_leave, sap_fi, ravi],
            relations=[
                Relation(asha, RelationType.BELONGS_TO, finance, source_chunk_id="c1"),
                Relation(ravi, RelationType.BELONGS_TO, finance, source_chunk_id="c1"),
                Relation(finance, RelationType.FOLLOWS, leave_policy, source_chunk_id="c1"),
                Relation(leave_policy, RelationType.GOVERNS, casual_leave, source_chunk_id="c1"),
                Relation(asha, RelationType.REQUIRES_TRAINING, sap_fi, source_chunk_id="c1"),
            ],
        )

    def test_department_of_employee(self, fake_neo4j_connection: Neo4jConnection) -> None:
        self._build_sample_graph(fake_neo4j_connection)
        library = CypherQueryLibrary(fake_neo4j_connection)
        rows = library.department_of_employee("Asha Rao")
        assert rows == [{"department": "Finance"}]

    def test_department_of_employee_not_found_returns_empty(self, fake_neo4j_connection: Neo4jConnection) -> None:
        library = CypherQueryLibrary(fake_neo4j_connection)
        assert library.department_of_employee("Nobody") == []

    def test_department_of_employee_empty_name_raises(self, fake_neo4j_connection: Neo4jConnection) -> None:
        library = CypherQueryLibrary(fake_neo4j_connection)
        with pytest.raises(CypherQueryLibraryError):
            library.department_of_employee("")

    def test_leave_policy_for_employee_multi_hop(self, fake_neo4j_connection: Neo4jConnection) -> None:
        self._build_sample_graph(fake_neo4j_connection)
        library = CypherQueryLibrary(fake_neo4j_connection)
        rows = library.leave_policy_for_employee("Asha Rao")
        assert rows == [{"leave_type": "Casual Leave", "policy": "Leave Policy"}]

    def test_employees_missing_training(self, fake_neo4j_connection: Neo4jConnection) -> None:
        self._build_sample_graph(fake_neo4j_connection)
        library = CypherQueryLibrary(fake_neo4j_connection)
        rows = library.employees_missing_training("Finance", "SAP FI Certification")
        # Asha has the training; Ravi does not.
        assert rows == [{"employee": "Ravi Iyer"}]

    def test_employees_missing_training_empty_args_raise(self, fake_neo4j_connection: Neo4jConnection) -> None:
        library = CypherQueryLibrary(fake_neo4j_connection)
        with pytest.raises(CypherQueryLibraryError):
            library.employees_missing_training("", "SAP FI Certification")
        with pytest.raises(CypherQueryLibraryError):
            library.employees_missing_training("Finance", "")
