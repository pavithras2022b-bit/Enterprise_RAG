"""
Shared pytest fixtures.

tiny_bert_checkpoint_dir builds a miniature BERT-architecture
token-classification checkpoint entirely from local objects (no
Hugging Face Hub download), so ner/model.py and ner/train.py can be
exercised end-to-end against a real tokenizer + real transformer
forward/backward pass, fully offline.
"""

from __future__ import annotations

import pytest
from transformers import BertConfig, BertForTokenClassification, BertTokenizerFast

from ner.schema import build_bio_label_maps

_VOCAB_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "asha",
    "rao",
    "belongs",
    "to",
    "the",
    "finance",
    "department",
    "employee",
    "handbook",
    "policy",
    "vpn",
    "access",
    "requires",
    "manager",
    "approval",
    "leave",
    "casual",
    "sick",
    "days",
    "year",
    "project",
    "payments",
    "modernization",
    "uses",
    "sap",
    "fi",
    "role",
    "analyst",
    "network",
    "gateway",
    "software",
    "engineering",
    ".",
    ",",
]


@pytest.fixture(scope="session")
def tiny_bert_checkpoint_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """
    Builds and saves a tiny, randomly-initialized BERT token-classification
    checkpoint to a temp directory, using a hand-written vocab (no
    network access). Session-scoped since building/saving it is not
    free and its content is never mutated by tests.
    """
    vocab_dir = tmp_path_factory.mktemp("tiny_bert_vocab")
    vocab_file = vocab_dir / "vocab.txt"
    vocab_file.write_text("\n".join(_VOCAB_TOKENS), encoding="utf-8")

    tokenizer = BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=True)

    id2label, label2id = build_bio_label_maps()
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=128,
        num_labels=len(id2label),
    )
    config.id2label = id2label
    config.label2id = label2id
    model = BertForTokenClassification(config)

    checkpoint_dir = tmp_path_factory.mktemp("tiny_bert_checkpoint")
    tokenizer.save_pretrained(str(checkpoint_dir))
    model.save_pretrained(str(checkpoint_dir))
    return str(checkpoint_dir)


@pytest.fixture(scope="session")
def tiny_relation_checkpoint_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """
    Builds and saves a tiny, randomly-initialized BERT
    sequence-classification checkpoint (for Module 5's
    RelationClassifier/train_re), using the same hand-written vocab
    strategy as tiny_bert_checkpoint_dir, fully offline. Its config's
    id2label/label2id already match
    relation_extraction.model.build_relation_label_maps(), and its
    tokenizer already has the entity-marker special tokens
    ([SUBJ], [/SUBJ], [OBJ], [/OBJ]) added, so tests can load it
    directly without re-adding tokens/resizing embeddings themselves
    (RelationClassifier does this idempotently regardless).
    """
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    from relation_extraction.model import MARKER_TOKENS, build_relation_label_maps

    vocab_dir = tmp_path_factory.mktemp("tiny_relation_vocab")
    vocab_file = vocab_dir / "vocab.txt"
    vocab_file.write_text("\n".join(_VOCAB_TOKENS), encoding="utf-8")

    tokenizer = BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=True)
    tokenizer.add_special_tokens({"additional_special_tokens": list(MARKER_TOKENS)})

    id2label, label2id = build_relation_label_maps()
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,  # resized below to include marker tokens
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=256,
        num_labels=len(id2label),
    )
    config.id2label = id2label
    config.label2id = label2id
    model = BertForSequenceClassification(config)
    model.resize_token_embeddings(len(tokenizer))

    checkpoint_dir = tmp_path_factory.mktemp("tiny_relation_checkpoint")
    tokenizer.save_pretrained(str(checkpoint_dir))
    model.save_pretrained(str(checkpoint_dir))
    return str(checkpoint_dir)


# ---------------------------------------------------------------------------
# Fake Neo4j driver, for Module 6 tests.
#
# This sandbox's network allowlist has no route to a real Neo4j server (no
# JVM package repo, no container registry access), so knowledge_graph/*.py
# is tested against a small in-memory graph store that recognizes exactly
# the Cypher patterns builder.py and cypher_queries.py generate -- not a
# general Cypher interpreter. This validates the wrapper's query
# construction, MERGE dedup logic, and mutation-counter reporting; it is
# NOT a substitute for integration testing against a live Neo4j instance,
# which a real deployment should still run (e.g. via testcontainers in CI).
# ---------------------------------------------------------------------------

import re as _re
from typing import Any as _Any, Optional as _Optional

from knowledge_graph.cypher_queries import (
    DEPARTMENT_OF_EMPLOYEE as _DEPARTMENT_OF_EMPLOYEE,
    EMPLOYEES_MISSING_TRAINING as _EMPLOYEES_MISSING_TRAINING,
    LEAVE_POLICY_FOR_EMPLOYEE as _LEAVE_POLICY_FOR_EMPLOYEE,
)

_NODE_MERGE_PATTERN = _re.compile(
    r"^MERGE \(n:(\w+) \{canonical_name: \$canonical_name\}\) "
    r"ON CREATE SET n\.text = \$text, n\.source_chunk_id = \$source_chunk_id "
    r"RETURN n$"
)
_RELATIONSHIP_MERGE_PATTERN = _re.compile(
    r"^MATCH \(s:(\w+) \{canonical_name: \$subject_name\}\), "
    r"\(o:(\w+) \{canonical_name: \$object_name\}\) "
    r"MERGE \(s\)-\[r:(\w+)\]->\(o\) "
    r"(?:ON CREATE SET [^R]+)?"
    r"RETURN r$"
)

_NodeKey = tuple  # (label, canonical_name)
_RelKey = tuple  # (subject_label, subject_name, rel_type, object_label, object_name)


class _FakeGraphStore:
    """In-memory stand-in for a Neo4j graph's node/relationship set."""

    def __init__(self) -> None:
        self.nodes: dict[_NodeKey, dict[str, str]] = {}
        self.relationships: dict[_RelKey, dict[str, str]] = {}

    def run(self, query: str, params: dict[str, _Any]) -> tuple[list[dict[str, _Any]], dict[str, int]]:
        node_match = _NODE_MERGE_PATTERN.match(query)
        if node_match:
            return self._run_node_merge(node_match, params)

        rel_match = _RELATIONSHIP_MERGE_PATTERN.match(query)
        if rel_match:
            return self._run_relationship_merge(rel_match, params)

        if query == _DEPARTMENT_OF_EMPLOYEE.cypher:
            return self._department_of_employee(params), {}
        if query == _LEAVE_POLICY_FOR_EMPLOYEE.cypher:
            return self._leave_policy_for_employee(params), {}
        if query == _EMPLOYEES_MISSING_TRAINING.cypher:
            return self._employees_missing_training(params), {}

        raise ValueError(f"_FakeGraphStore does not recognize this query:\n{query}")

    def _run_node_merge(
        self, match: "_re.Match[str]", params: dict[str, _Any]
    ) -> tuple[list[dict[str, _Any]], dict[str, int]]:
        label = match.group(1)
        name = params["canonical_name"]
        key = (label, name)
        created = key not in self.nodes
        if created:
            self.nodes[key] = {"text": params["text"], "source_chunk_id": params["source_chunk_id"]}
        counters = {"nodes_created": 1 if created else 0, "relationships_created": 0, "properties_set": 2 if created else 0}
        return [{"n": {**self.nodes[key], "canonical_name": name}}], counters

    def _run_relationship_merge(
        self, match: "_re.Match[str]", params: dict[str, _Any]
    ) -> tuple[list[dict[str, _Any]], dict[str, int]]:
        subject_label, object_label, rel_type = match.group(1), match.group(2), match.group(3)
        subject_name, object_name = params["subject_name"], params["object_name"]
        key = (subject_label, subject_name, rel_type, object_label, object_name)
        created = key not in self.relationships
        if created:
            properties = {k: v for k, v in params.items() if k not in ("subject_name", "object_name")}
            self.relationships[key] = properties
        counters = {
            "nodes_created": 0,
            "relationships_created": 1 if created else 0,
            "properties_set": len(self.relationships[key]) if created else 0,
        }
        return [{"r": self.relationships[key]}], counters

    def _department_of_employee(self, params: dict[str, _Any]) -> list[dict[str, _Any]]:
        employee_name = params["employee_name"]
        for (slabel, sname, reltype, olabel, oname) in self.relationships:
            if (slabel, sname, reltype, olabel) == ("EMPLOYEE", employee_name, "BELONGS_TO", "DEPARTMENT"):
                return [{"department": oname}]
        return []

    def _leave_policy_for_employee(self, params: dict[str, _Any]) -> list[dict[str, _Any]]:
        employee_name = params["employee_name"]
        departments = [
            oname for (slabel, sname, reltype, olabel, oname) in self.relationships
            if (slabel, sname, reltype, olabel) == ("EMPLOYEE", employee_name, "BELONGS_TO", "DEPARTMENT")
        ]
        rows: list[dict[str, _Any]] = []
        for department in departments:
            policies = [
                oname for (slabel, sname, reltype, olabel, oname) in self.relationships
                if (slabel, sname, reltype, olabel) == ("DEPARTMENT", department, "FOLLOWS", "POLICY")
            ]
            for policy in policies:
                leave_types = [
                    oname for (slabel, sname, reltype, olabel, oname) in self.relationships
                    if (slabel, sname, reltype, olabel) == ("POLICY", policy, "GOVERNS", "LEAVE_TYPE")
                ]
                for leave_type in leave_types:
                    rows.append({"leave_type": leave_type, "policy": policy})
        return rows

    def _employees_missing_training(self, params: dict[str, _Any]) -> list[dict[str, _Any]]:
        department_name = params["department_name"]
        software_name = params["software_name"]
        employees = [
            sname for (slabel, sname, reltype, olabel, oname) in self.relationships
            if (slabel, reltype, olabel, oname) == ("EMPLOYEE", "BELONGS_TO", "DEPARTMENT", department_name)
        ]
        trained = {
            sname for (slabel, sname, reltype, olabel, oname) in self.relationships
            if (slabel, reltype, olabel, oname) == ("EMPLOYEE", "REQUIRES_TRAINING", "SOFTWARE", software_name)
        }
        return [{"employee": e} for e in employees if e not in trained]


class _FakeRecord:
    def __init__(self, data: dict[str, _Any]) -> None:
        self._data = data

    def data(self) -> dict[str, _Any]:
        return self._data


class _FakeCounters:
    def __init__(self, counters: dict[str, int]) -> None:
        self.nodes_created = counters.get("nodes_created", 0)
        self.relationships_created = counters.get("relationships_created", 0)
        self.properties_set = counters.get("properties_set", 0)


class _FakeSummary:
    def __init__(self, counters: dict[str, int]) -> None:
        self.counters = _FakeCounters(counters)


class _FakeResult:
    def __init__(self, records: list[dict[str, _Any]], counters: dict[str, int]) -> None:
        self._records = [_FakeRecord(r) for r in records]
        self._counters = counters

    def __iter__(self):
        return iter(self._records)

    def consume(self) -> _FakeSummary:
        return _FakeSummary(self._counters)


class _FakeSession:
    def __init__(self, store: _FakeGraphStore) -> None:
        self._store = store

    def run(self, query: str, parameters: _Optional[dict[str, _Any]] = None) -> _FakeResult:
        records, counters = self._store.run(query, parameters or {})
        return _FakeResult(records, counters)

    def __enter__(self) -> "_FakeSession":
        return self

    def __exit__(self, *exc_info: object) -> None:
        pass


class _FakeDriver:
    def __init__(self, store: _FakeGraphStore) -> None:
        self._store = store
        self.closed = False

    def session(self, database: _Optional[str] = None) -> _FakeSession:
        return _FakeSession(self._store)

    def verify_connectivity(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_neo4j_connection():
    """
    Builds a Neo4jConnection backed by a fresh, empty _FakeGraphStore
    (via dependency-injected driver_factory) -- one clean in-memory
    graph per test.
    """
    from knowledge_graph.connection import Neo4jConnection

    store = _FakeGraphStore()

    def factory(uri: str, auth: _Optional[tuple[str, str]] = None, **kwargs: _Any) -> _FakeDriver:
        return _FakeDriver(store)

    return Neo4jConnection(
        uri="bolt://fake-test-instance:7687",
        username="test",
        password="test",
        driver_factory=factory,
    )
