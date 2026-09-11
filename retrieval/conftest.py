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


@pytest.fixture(scope="session")
def tiny_sentence_transformer_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """
    Builds and saves a tiny, randomly-initialized Sentence-BERT
    checkpoint (Transformer + mean-Pooling modules), using the same
    hand-written vocab strategy as the other tiny checkpoints, fully
    offline. Used to exercise embeddings/model.py and
    embeddings/train.py's real sentence-transformers/torch code paths
    without Hugging Face Hub access.
    """
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import models as st_models
    from transformers import BertConfig, BertModel, BertTokenizerFast

    vocab_dir = tmp_path_factory.mktemp("tiny_st_vocab")
    vocab_file = vocab_dir / "vocab.txt"
    vocab_file.write_text("\n".join(_VOCAB_TOKENS), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=True)

    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=128,
    )
    bert_model = BertModel(config)

    transformer_dir = tmp_path_factory.mktemp("tiny_st_transformer")
    bert_model.save_pretrained(str(transformer_dir))
    tokenizer.save_pretrained(str(transformer_dir))

    word_embedding_model = st_models.Transformer(str(transformer_dir), max_seq_length=128)
    pooling_model = st_models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), pooling_mode="mean"
    )
    sentence_transformer = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    checkpoint_dir = tmp_path_factory.mktemp("tiny_sentence_transformer")
    sentence_transformer.save(str(checkpoint_dir))
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
    r"(?:SET .+? )?"
    r"RETURN r$"
)

_NodeKey = tuple  # (label, canonical_name)
_RelKey = tuple  # (subject_label, subject_name, rel_type, object_label, object_name)


class _FakeGraphStore:
    """In-memory stand-in for a Neo4j graph's node/relationship set."""

    def __init__(self) -> None:
        self.nodes: dict[_NodeKey, dict[str, str]] = {}
        self.relationships: dict[_RelKey, dict[str, str]] = {}

    def add_node(
        self, label: str, canonical_name: str, properties: _Optional[dict[str, _Any]] = None
    ) -> None:
        """
        Lightweight test-seeding helper, additive to (not a replacement
        for) the MERGE-Cypher dispatch below -- both populate the same
        self.nodes dict, so a test may freely mix Neo4jGraphBuilder-driven
        seeding (Module 6's convention) with direct add_node/
        add_relationship calls (Module 9's tests' convention) in the same
        fixture instance.
        """
        self.nodes[(label, canonical_name)] = dict(properties or {})

    def add_relationship(
        self,
        subject: str,
        predicate: str,
        obj: str,
        properties: _Optional[dict[str, _Any]] = None,
    ) -> None:
        """Both `subject` and `obj` must already exist as nodes (via
        add_node or prior MERGE-dispatch writes) so their labels can be
        resolved -- self.relationships' key format requires both labels,
        matching what the MERGE-relationship dispatch below produces."""
        subject_label = self._label_for_name(subject)
        object_label = self._label_for_name(obj)
        if subject_label is None or object_label is None:
            raise ValueError(
                f"add_relationship: both subject ({subject!r}) and object "
                f"({obj!r}) must already exist as nodes -- call add_node "
                f"first."
            )
        key = (subject_label, subject, predicate, object_label, obj)
        self.relationships[key] = dict(properties or {})

    def _label_for_name(self, canonical_name: str) -> _Optional[str]:
        for (label, name) in self.nodes:
            if name == canonical_name:
                return label
        return None

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

        # Module 9 (retrieval.entity_linker / retrieval.hybrid_controller)
        # read-only query shapes. Both operate on the SAME self.nodes /
        # self.relationships dicts the MERGE dispatch above already
        # populates -- Module 9's tests seed the graph via the real
        # Neo4jGraphBuilder (as Module 6's tests do), not via a separate
        # ad hoc seeding API, so a single fake store stays the source of
        # truth for both modules' tests.
        normalized_query = " ".join(query.split())
        if normalized_query.startswith("MATCH (n) RETURN labels(n)[0]"):
            return self._entity_index(), {}

        traversal_match = _re.search(r"\[\*1\.\.(\d+)\]", query)
        if traversal_match:
            max_hops = int(traversal_match.group(1))
            return self._variable_length_traversal(params, max_hops), {}

        raise ValueError(f"_FakeGraphStore does not recognize this query:\n{query}")

    def _entity_index(self) -> list[dict[str, _Any]]:
        """Backs EntityLinker.refresh_index()'s
        `MATCH (n) RETURN labels(n)[0] AS label, n.canonical_name AS canonical_name`."""
        return [{"label": label, "canonical_name": name} for (label, name) in self.nodes]

    def _variable_length_traversal(
        self, params: dict[str, _Any], max_hops: int
    ) -> list[dict[str, _Any]]:
        """
        Backs HybridRetrievalController.run_graph_retrieval()'s
        `MATCH path = (start {canonical_name: $start_name})-[*1..N]->(end) ...`.

        Mirrors real Neo4j's variable-length-path semantics: one row per
        distinct path of EACH length from 1 to max_hops (not just the
        longest path), since that is what `[*1..N]` actually returns
        against a live Neo4j instance -- the hybrid_controller's
        fact_id-keyed dedup (keeping the minimum hop_count) depends on
        this prefix-duplication shape being present, exactly as it does
        against a real graph with branching paths.
        """
        start_name = params.get("start_name")
        adjacency: dict[str, list[tuple[str, str, str, dict[str, str]]]] = {}
        label_by_name: dict[str, str] = {}
        for (subject_label, subject_name, rel_type, object_label, object_name), properties in (
            self.relationships.items()
        ):
            adjacency.setdefault(subject_name, []).append(
                (rel_type, object_name, object_label, properties)
            )
            label_by_name[subject_name] = subject_label
            label_by_name[object_name] = object_label

        if start_name is None or start_name not in label_by_name:
            return []

        rows: list[dict[str, _Any]] = []

        def _dfs(
            current: str,
            node_names: list[str],
            rel_types: list[str],
            source_chunk_ids: list[_Optional[str]],
            depth: int,
        ) -> None:
            if depth >= max_hops:
                return
            for rel_type, object_name, _object_label, properties in adjacency.get(current, []):
                new_node_names = node_names + [object_name]
                new_rel_types = rel_types + [rel_type]
                new_source_chunk_ids = source_chunk_ids + [properties.get("source_chunk_id")]
                rows.append(
                    {
                        "node_names": new_node_names,
                        "node_labels": [label_by_name.get(n, "") for n in new_node_names],
                        "rel_types": new_rel_types,
                        "source_chunk_ids": new_source_chunk_ids,
                        "hops": depth + 1,
                    }
                )
                _dfs(object_name, new_node_names, new_rel_types, new_source_chunk_ids, depth + 1)

        _dfs(start_name, [start_name], [], [], 0)
        return rows

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

    connection = Neo4jConnection(
        uri="bolt://fake-test-instance:7687",
        username="test",
        password="test",
        driver_factory=factory,
    )
    # Test-only escape hatch: exposes the underlying store so tests can
    # seed data directly via store.add_node()/add_relationship() as a
    # lighter-weight alternative to Neo4jGraphBuilder.build_graph(),
    # without adding anything to Neo4jConnection's real public API.
    connection._test_store = store  # type: ignore[attr-defined]
    return connection
