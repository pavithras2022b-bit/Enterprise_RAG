"""
Module 6 - Knowledge Graph: Neo4j driver connection wrapper.

Provides Neo4jConnection, a thin, dependency-injectable wrapper around
the official neo4j Python driver, separating write operations (which
report mutation counters, needed by Neo4jGraphBuilder's
GraphBuildReport) from read-only operations (used by
CypherQueryLibrary).

Reference: Capstone Project Guide, Section 3I ("Neo4j"), Phase 4
Module 6 contract, Phase 5 file structure
(knowledge_graph/connection.py).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from neo4j import Driver, GraphDatabase

logger = logging.getLogger(__name__)

# Signature of neo4j.GraphDatabase.driver, used as the default factory.
# Accepting an injectable factory (rather than calling GraphDatabase.driver
# directly) is what lets tests substitute a fake driver without a live
# Neo4j server -- see tests/conftest.py's fake_neo4j_driver_factory.
DriverFactory = Callable[..., Driver]


class Neo4jConnectionError(Exception):
    """Raised when a Neo4j connection cannot be established or a query fails."""


class Neo4jConnection:
    """
    Thin wrapper around a neo4j.Driver, providing:
      - connection lifecycle management (connect, verify, close),
      - execute_write(): runs a mutating Cypher statement and returns
        both result records and mutation counters,
      - execute_read(): runs a read-only Cypher statement and returns
        just the result records.

    Used as a context manager to guarantee the underlying driver is
    closed even if an exception is raised mid-pipeline.
    """

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        driver_factory: DriverFactory = GraphDatabase.driver,
    ) -> None:
        """
        Args:
            uri: Bolt connection URI (e.g. "bolt://localhost:7687").
            username: Neo4j username.
            password: Neo4j password.
            database: Target database name within the Neo4j instance.
            driver_factory: Callable used to construct the underlying
                driver, defaulting to neo4j.GraphDatabase.driver.
                Overridable for testing with a fake driver.

        Raises:
            Neo4jConnectionError: If the driver cannot be constructed
                (e.g. malformed URI).
        """
        if not uri:
            raise Neo4jConnectionError("Neo4jConnection requires a non-empty uri.")
        try:
            self._driver: Driver = driver_factory(uri, auth=(username, password))
        except Exception as exc:  # noqa: BLE001
            raise Neo4jConnectionError(
                f"Failed to construct Neo4j driver for uri={uri!r}: {exc}"
            ) from exc
        self._database = database
        self._uri = uri

    def verify_connectivity(self) -> None:
        """
        Eagerly checks that the driver can reach the Neo4j instance.

        Raises:
            Neo4jConnectionError: If connectivity verification fails.
        """
        try:
            self._driver.verify_connectivity()
        except Exception as exc:  # noqa: BLE001
            raise Neo4jConnectionError(
                f"Failed to verify connectivity to {self._uri!r}: {exc}"
            ) from exc

    def execute_write(
        self, query: str, parameters: Optional[dict[str, Any]] = None
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """
        Run a mutating Cypher statement (typically containing MERGE or
        CREATE) and return both its result records and mutation
        counters, so callers (Neo4jGraphBuilder) can distinguish
        "created new" from "matched existing" for accurate reporting.

        Args:
            query: Cypher query text.
            parameters: Query parameters.

        Returns:
            A (records, counters) tuple. records is a list of
            plain-dict row data. counters contains
            "nodes_created", "relationships_created", and
            "properties_set" (0 if the query performed none of that
            kind of mutation).

        Raises:
            Neo4jConnectionError: If the query fails.
        """
        params = parameters or {}
        try:
            with self._driver.session(database=self._database) as session:
                result = session.run(query, params)
                records = [record.data() for record in result]
                summary = result.consume()
                counters = {
                    "nodes_created": summary.counters.nodes_created,
                    "relationships_created": summary.counters.relationships_created,
                    "properties_set": summary.counters.properties_set,
                }
                return records, counters
        except Neo4jConnectionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise Neo4jConnectionError(
                f"execute_write failed for query {query!r}: {exc}"
            ) from exc

    def execute_read(
        self, query: str, parameters: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        """
        Run a read-only Cypher statement and return its result records.

        Args:
            query: Cypher query text.
            parameters: Query parameters.

        Returns:
            A list of plain-dict row data.

        Raises:
            Neo4jConnectionError: If the query fails.
        """
        params = parameters or {}
        try:
            with self._driver.session(database=self._database) as session:
                result = session.run(query, params)
                return [record.data() for record in result]
        except Neo4jConnectionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise Neo4jConnectionError(
                f"execute_read failed for query {query!r}: {exc}"
            ) from exc

    def close(self) -> None:
        """Closes the underlying driver and releases its connection pool."""
        self._driver.close()

    def __enter__(self) -> "Neo4jConnection":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
