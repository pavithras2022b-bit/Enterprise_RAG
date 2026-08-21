"""
Module 6 - Knowledge Graph: reusable, named Cypher queries.

Provides a small library of parametrized read queries mirroring
Section 4.5/4.6's worked examples -- single-hop lookup, multi-hop
traversal, and a relational filter query -- as both raw NamedQuery
constants (for direct use/testing) and a CypherQueryLibrary wrapper
that runs them against a live Neo4jConnection.

Reference: Capstone Project Guide, Section 4.5 ("How Graph Queries
Work"), Section 4.6 (worked single-hop/multi-hop comparison), Phase 5
file structure (knowledge_graph/cypher_queries.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from knowledge_graph.connection import Neo4jConnection
from ner.schema import EntityType
from relation_extraction.schema import RelationType

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NamedQuery:
    """A reusable Cypher query with a stable name and human-readable description."""

    name: str
    cypher: str
    description: str


# Node labels and relationship types are pulled from EntityType/RelationType
# rather than hardcoded strings, so these queries can never drift out of
# sync with the labels builder.py actually writes to Neo4j (both derive
# from the same enums).
_EMPLOYEE = EntityType.EMPLOYEE.value
_DEPARTMENT = EntityType.DEPARTMENT.value
_POLICY = EntityType.POLICY.value
_LEAVE_TYPE = EntityType.LEAVE_TYPE.value
_SOFTWARE = EntityType.SOFTWARE.value

_BELONGS_TO = RelationType.BELONGS_TO.value
_FOLLOWS = RelationType.FOLLOWS.value
_GOVERNS = RelationType.GOVERNS.value
_REQUIRES_TRAINING = RelationType.REQUIRES_TRAINING.value


DEPARTMENT_OF_EMPLOYEE = NamedQuery(
    name="department_of_employee",
    cypher=(
        f"MATCH (e:{_EMPLOYEE} {{canonical_name: $employee_name}})-[:{_BELONGS_TO}]->(d:{_DEPARTMENT}) "
        f"RETURN d.canonical_name AS department"
    ),
    description=(
        "Single-hop: which department does a given employee belong to? "
        "(Section 4.5's single-hop example.)"
    ),
)

LEAVE_POLICY_FOR_EMPLOYEE = NamedQuery(
    name="leave_policy_for_employee",
    cypher=(
        f"MATCH (e:{_EMPLOYEE} {{canonical_name: $employee_name}})-[:{_BELONGS_TO}]->(:{_DEPARTMENT})"
        f"-[:{_FOLLOWS}]->(p:{_POLICY})-[:{_GOVERNS}]->(lt:{_LEAVE_TYPE}) "
        f"RETURN lt.canonical_name AS leave_type, p.canonical_name AS policy"
    ),
    description=(
        "Multi-hop (3 hops): what leave types apply to an employee, and "
        "under which policy? A class of query vector search cannot answer "
        "(Section 4.5's multi-hop example)."
    ),
)

EMPLOYEES_MISSING_TRAINING = NamedQuery(
    name="employees_missing_training",
    cypher=(
        f"MATCH (e:{_EMPLOYEE})-[:{_BELONGS_TO}]->(d:{_DEPARTMENT} {{canonical_name: $department_name}}) "
        f"WHERE NOT (e)-[:{_REQUIRES_TRAINING}]->(:{_SOFTWARE} {{canonical_name: $software_name}}) "
        f"RETURN e.canonical_name AS employee"
    ),
    description=(
        "Relational filter: employees in a given department who are "
        "missing a given required training/certification (Section 4.6's "
        "worked relational-filter example)."
    ),
)


class CypherQueryLibraryError(Exception):
    """Raised when a named query is called with invalid/missing parameters."""


class CypherQueryLibrary:
    """
    Runs the module's NamedQuery constants against a live
    Neo4jConnection, with per-query parameter validation so a missing
    required parameter fails fast in Python rather than as a confusing
    empty Cypher result.
    """

    def __init__(self, connection: Neo4jConnection) -> None:
        """
        Args:
            connection: An open Neo4jConnection.
        """
        self._connection = connection

    def department_of_employee(self, employee_name: str) -> list[dict[str, str]]:
        """
        Args:
            employee_name: canonical_name of the Employee node.

        Returns:
            A list of rows with a "department" key. Empty if the
            employee has no BELONGS_TO relationship (or doesn't
            exist).

        Raises:
            CypherQueryLibraryError: If employee_name is empty.
        """
        if not employee_name:
            raise CypherQueryLibraryError("employee_name must not be empty.")
        return self._connection.execute_read(
            DEPARTMENT_OF_EMPLOYEE.cypher, {"employee_name": employee_name}
        )

    def leave_policy_for_employee(self, employee_name: str) -> list[dict[str, str]]:
        """
        Args:
            employee_name: canonical_name of the Employee node.

        Returns:
            A list of rows with "leave_type" and "policy" keys, one
            per (leave type, governing policy) pair reachable via the
            employee's department.

        Raises:
            CypherQueryLibraryError: If employee_name is empty.
        """
        if not employee_name:
            raise CypherQueryLibraryError("employee_name must not be empty.")
        return self._connection.execute_read(
            LEAVE_POLICY_FOR_EMPLOYEE.cypher, {"employee_name": employee_name}
        )

    def employees_missing_training(
        self, department_name: str, software_name: str
    ) -> list[dict[str, str]]:
        """
        Args:
            department_name: canonical_name of the Department node.
            software_name: canonical_name of the required Software
                node.

        Returns:
            A list of rows with an "employee" key, one per employee in
            the department lacking a REQUIRES_TRAINING relationship to
            the given software.

        Raises:
            CypherQueryLibraryError: If either argument is empty.
        """
        if not department_name or not software_name:
            raise CypherQueryLibraryError(
                "Both department_name and software_name must be non-empty."
            )
        return self._connection.execute_read(
            EMPLOYEES_MISSING_TRAINING.cypher,
            {"department_name": department_name, "software_name": software_name},
        )
