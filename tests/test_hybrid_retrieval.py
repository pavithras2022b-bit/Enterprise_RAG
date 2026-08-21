"""
tests/test_hybrid_retrieval.py

Covers Section 10 of the Module 9 spec:
    10.1 retrieval/schema.py
    10.2 entity_linker.py
    10.3 hybrid_controller.py -- run_vector_retrieval
    10.4 hybrid_controller.py -- run_graph_retrieval
    10.5 hybrid_controller.py -- normalize_scores
    10.6 hybrid_controller.py -- static_fuse
    10.7 hybrid_controller.py -- retrieve (end-to-end)

Uses a REAL in-memory VectorStore (Module 8, ChromaDB EphemeralClient)
and a REAL EmbeddingModel built on the session-scoped
tiny_sentence_transformer_dir fixture (Module 7's tiny offline
SentenceTransformer checkpoint -- see conftest.py), so this suite
exercises the actual production classes end-to-end rather than stubs.

Graph seeding uses fake_neo4j_connection's underlying store directly
(fake_neo4j_connection._test_store.add_node/add_relationship) as a
lightweight alternative to going through Neo4jGraphBuilder for every
test -- Module 6's own test suite already thoroughly covers that write
path; Module 9's tests exist to test retrieval logic, not graph
population.
"""

from __future__ import annotations

import math
import uuid

import pytest

from embeddings.model import EmbeddingModel
from embeddings.schema import EmbeddedChunk
from knowledge_graph.schema import GraphSchema
from ner.schema import EntityType
from preprocessing.schema import DocumentType
from relation_extraction.schema import RelationType
from vector_db.chroma_store import VectorStore

from retrieval.entity_linker import EntityLinker, EntityLinkerError
from retrieval.hybrid_controller import HybridRetrievalController, RetrievalError
from retrieval.schema import (
    CandidateSource,
    FusedCandidate,
    GraphFact,
    GraphNodeRef,
    SchemaValidationError,
    render_predicate_phrase,
)
from vector_db.schema import VectorSearchResult


# ----------------------------------------------------------------------
# Shared scenario helper: Asha Rao / Finance / Leave Policy / Casual Leave
# (reused across Modules 6/7/8's own tests, per Section 10.7's guidance)
# ----------------------------------------------------------------------
def seed_scenario_graph(fake_neo4j_connection) -> None:
    store = fake_neo4j_connection._test_store
    store.add_node("EMPLOYEE", "Asha Rao")
    store.add_node("DEPARTMENT", "Finance")
    store.add_node("POLICY", "Leave Policy")
    store.add_node("LEAVE_TYPE", "Casual Leave")

    store.add_relationship(
        "Asha Rao", RelationType.BELONGS_TO.value, "Finance",
        properties={"source_chunk_id": "chunk-1"},
    )
    store.add_relationship(
        "Finance", RelationType.FOLLOWS.value, "Leave Policy",
        properties={"source_chunk_id": "chunk-2"},
    )
    store.add_relationship(
        "Leave Policy", RelationType.GOVERNS.value, "Casual Leave",
        properties={"source_chunk_id": "chunk-3"},
    )


def make_embedded_chunk(chunk_id: str, text: str, model: EmbeddingModel) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=chunk_id,
        vector=model.embed(text),
        text=text,
        source_document_id="doc-1",
        document_type=DocumentType.POLICY,
        department="Finance",
        effective_date=None,
        source_path="/docs/leave-policy.md",
        extra_metadata={},
    )


def make_vector_store() -> VectorStore:
    """
    Fresh, isolated in-memory ChromaDB collection. A unique collection
    name per call is required, not optional: ChromaDB's in-process
    ephemeral client shares state across separate VectorStore instances
    within the same process when given the same collection_name (a
    real, empirically-verified behavior -- see Module 8's own test
    suite, which already works around it the same way). Constructing a
    "fresh" VectorStore with a REUSED name does not give an isolated
    collection.
    """
    return VectorStore(collection_name=f"test_module9_{uuid.uuid4().hex}", persist_directory=None)


# ========================================================================
# 10.1 retrieval/schema.py
# ========================================================================
class TestGraphNodeRef:
    def test_valid_construction(self):
        ref = GraphNodeRef(
            mention_text="Asha", canonical_name="Asha Rao",
            entity_type=EntityType.EMPLOYEE, match_score=95.0,
        )
        assert ref.canonical_name == "Asha Rao"

    def test_empty_mention_text_raises(self):
        with pytest.raises(SchemaValidationError):
            GraphNodeRef(
                mention_text="", canonical_name="Asha Rao",
                entity_type=EntityType.EMPLOYEE, match_score=95.0,
            )

    def test_empty_canonical_name_raises(self):
        with pytest.raises(SchemaValidationError):
            GraphNodeRef(
                mention_text="Asha", canonical_name="",
                entity_type=EntityType.EMPLOYEE, match_score=95.0,
            )

    @pytest.mark.parametrize("score", [-1.0, 100.1])
    def test_match_score_out_of_range_raises(self, score):
        with pytest.raises(SchemaValidationError):
            GraphNodeRef(
                mention_text="Asha", canonical_name="Asha Rao",
                entity_type=EntityType.EMPLOYEE, match_score=score,
            )


class TestGraphFact:
    def test_valid_construction(self):
        fact = GraphFact.build("Asha Rao", RelationType.BELONGS_TO.value, "Finance", hop_count=1)
        assert fact.fact_id == "asha rao|belongs_to|finance"
        assert fact.raw_relevance == 1.0

    @pytest.mark.parametrize("field_name", ["fact_id", "subject", "object", "text"])
    def test_empty_required_fields_raise(self, field_name):
        kwargs = dict(
            fact_id="a|belongs_to|b", subject="a", predicate=RelationType.BELONGS_TO.value,
            object="b", text="a belongs to b.", source_chunk_id=None, hop_count=1,
            raw_relevance=1.0,
        )
        kwargs[field_name] = ""
        with pytest.raises(SchemaValidationError):
            GraphFact(**kwargs)

    def test_invalid_predicate_raises(self):
        with pytest.raises(SchemaValidationError):
            GraphFact(
                fact_id="a|not_a_type|b", subject="a", predicate="NOT_A_TYPE", object="b",
                text="a foos b.", source_chunk_id=None, hop_count=1, raw_relevance=1.0,
            )

    def test_hop_count_less_than_one_raises(self):
        with pytest.raises(SchemaValidationError):
            GraphFact(
                fact_id="a|belongs_to|b", subject="a", predicate=RelationType.BELONGS_TO.value,
                object="b", text="a belongs to b.", source_chunk_id=None, hop_count=0,
                raw_relevance=1.0,
            )

    def test_raw_relevance_non_positive_raises(self):
        with pytest.raises(SchemaValidationError):
            GraphFact(
                fact_id="a|belongs_to|b", subject="a", predicate=RelationType.BELONGS_TO.value,
                object="b", text="a belongs to b.", source_chunk_id=None, hop_count=1,
                raw_relevance=0.0,
            )

    @pytest.mark.parametrize(
        "relation_type,expected_phrase",
        [
            (RelationType.BELONGS_TO, "belongs to"),
            (RelationType.FOLLOWS, "follows"),
            (RelationType.GOVERNS, "governs"),
            (RelationType.REQUIRES_TRAINING, "requires training in"),
        ],
    )
    def test_predicate_phrase_rendering(self, relation_type, expected_phrase):
        assert render_predicate_phrase(relation_type.value) == expected_phrase
        fact = GraphFact.build("X", relation_type.value, "Y", hop_count=1)
        assert fact.text == f"X {expected_phrase} Y."

    def test_predicate_phrase_fallback_for_unmapped_value(self):
        # Every real RelationType is in the table; this exercises the
        # fallback path directly against the rendering helper (not a
        # GraphFact, since GraphFact itself requires a valid RelationType
        # value).
        assert render_predicate_phrase("SOME_FUTURE_RELATION") == "some future relation"


class TestFusedCandidate:
    def test_valid_construction_vector(self):
        c = FusedCandidate(
            candidate_id="chunk-1", source=CandidateSource.VECTOR,
            vector_score=0.9, graph_score=None, fused_score=0.45, text="hello",
        )
        assert c.source == CandidateSource.VECTOR

    def test_valid_construction_graph(self):
        c = FusedCandidate(
            candidate_id="fact-1", source=CandidateSource.GRAPH,
            vector_score=None, graph_score=0.7, fused_score=0.35, text="hello",
        )
        assert c.source == CandidateSource.GRAPH

    def test_valid_construction_both(self):
        c = FusedCandidate(
            candidate_id="chunk-1", source=CandidateSource.BOTH,
            vector_score=0.9, graph_score=0.7, fused_score=0.8, text="hello",
        )
        assert c.source == CandidateSource.BOTH

    def test_both_scores_none_raises(self):
        with pytest.raises(SchemaValidationError):
            FusedCandidate(
                candidate_id="x", source=CandidateSource.VECTOR,
                vector_score=None, graph_score=None, fused_score=0.0, text="hello",
            )

    @pytest.mark.parametrize("bad_score", [-0.1, 1.1, math.nan, math.inf])
    def test_score_out_of_range_raises(self, bad_score):
        with pytest.raises(SchemaValidationError):
            FusedCandidate(
                candidate_id="x", source=CandidateSource.VECTOR,
                vector_score=bad_score, graph_score=None, fused_score=0.0, text="hello",
            )

    def test_vector_source_with_graph_score_raises(self):
        with pytest.raises(SchemaValidationError):
            FusedCandidate(
                candidate_id="x", source=CandidateSource.VECTOR,
                vector_score=0.5, graph_score=0.5, fused_score=0.5, text="hello",
            )

    def test_graph_source_with_vector_score_raises(self):
        with pytest.raises(SchemaValidationError):
            FusedCandidate(
                candidate_id="x", source=CandidateSource.GRAPH,
                vector_score=0.5, graph_score=0.5, fused_score=0.5, text="hello",
            )

    def test_both_source_with_missing_score_raises(self):
        with pytest.raises(SchemaValidationError):
            FusedCandidate(
                candidate_id="x", source=CandidateSource.BOTH,
                vector_score=0.5, graph_score=None, fused_score=0.5, text="hello",
            )


# ========================================================================
# 10.2 entity_linker.py
# ========================================================================
class TestEntityLinker:
    def test_init_threshold_out_of_range_raises(self, fake_neo4j_connection):
        with pytest.raises(ValueError):
            EntityLinker(fake_neo4j_connection, similarity_threshold=101.0)

    def test_refresh_index_populates_from_graph(self, fake_neo4j_connection):
        seed_scenario_graph(fake_neo4j_connection)
        linker = EntityLinker(fake_neo4j_connection, refresh_on_init=False)
        linker.refresh_index()
        refs = linker.link("Tell me about the Finance department")
        assert any(r.canonical_name == "Finance" for r in refs)

    def test_link_on_empty_graph_returns_empty_list(self, fake_neo4j_connection):
        linker = EntityLinker(fake_neo4j_connection)  # refresh_on_init=True, empty graph
        assert linker.link("anything at all") == []

    def test_link_finds_high_confidence_match(self, fake_neo4j_connection):
        seed_scenario_graph(fake_neo4j_connection)
        linker = EntityLinker(fake_neo4j_connection)
        refs = linker.link("What is the Finance department's leave policy?")
        names = [r.canonical_name for r in refs]
        assert "Finance" in names

    def test_link_below_threshold_returns_no_match(self, fake_neo4j_connection):
        seed_scenario_graph(fake_neo4j_connection)
        linker = EntityLinker(fake_neo4j_connection, similarity_threshold=95.0)
        refs = linker.link("completely unrelated gibberish query xyz")
        assert refs == []

    def test_link_deduplicates_same_canonical_name(self, fake_neo4j_connection):
        seed_scenario_graph(fake_neo4j_connection)
        linker = EntityLinker(fake_neo4j_connection)
        # "Finance" appears effectively multiple times in the sliding
        # match space of this query but must be reported once, at its
        # best score.
        refs = linker.link("Finance Finance department Finance policy")
        finance_refs = [r for r in refs if r.canonical_name == "Finance"]
        assert len(finance_refs) == 1

    def test_link_before_refresh_raises(self, fake_neo4j_connection):
        linker = EntityLinker(fake_neo4j_connection, refresh_on_init=False)
        with pytest.raises(EntityLinkerError):
            linker.link("Finance")


# ========================================================================
# 10.3 hybrid_controller.py -- run_vector_retrieval
# ========================================================================
class TestRunVectorRetrieval:
    def _make_controller(self, fake_neo4j_connection, tiny_sentence_transformer_dir):
        model = EmbeddingModel(tiny_sentence_transformer_dir)
        store = make_vector_store()
        store.add(
            [
                make_embedded_chunk("chunk-1", "Employees get casual leave under the Finance leave policy.", model),
                make_embedded_chunk("chunk-2", "The office coffee machine is on the third floor.", model),
            ]
        )
        linker = EntityLinker(fake_neo4j_connection)
        return HybridRetrievalController(store, fake_neo4j_connection, model, linker)

    def test_delegates_to_vector_store(self, fake_neo4j_connection, tiny_sentence_transformer_dir):
        controller = self._make_controller(fake_neo4j_connection, tiny_sentence_transformer_dir)
        results = controller.run_vector_retrieval("What is the casual leave policy?", k=2)
        assert results
        assert isinstance(results[0], VectorSearchResult)

    def test_empty_query_raises(self, fake_neo4j_connection, tiny_sentence_transformer_dir):
        controller = self._make_controller(fake_neo4j_connection, tiny_sentence_transformer_dir)
        with pytest.raises(ValueError):
            controller.run_vector_retrieval("   ", k=2)

    def test_non_positive_k_raises(self, fake_neo4j_connection, tiny_sentence_transformer_dir):
        controller = self._make_controller(fake_neo4j_connection, tiny_sentence_transformer_dir)
        with pytest.raises(ValueError):
            controller.run_vector_retrieval("leave policy", k=0)

    def test_where_filter_passed_through(self, fake_neo4j_connection, tiny_sentence_transformer_dir):
        controller = self._make_controller(fake_neo4j_connection, tiny_sentence_transformer_dir)
        results = controller.run_vector_retrieval(
            "leave policy", k=5, where={"department": "Finance"}
        )
        assert all(r.metadata.get("department") == "Finance" for r in results)


# ========================================================================
# 10.4 hybrid_controller.py -- run_graph_retrieval
# ========================================================================
class TestRunGraphRetrieval:
    def _make_controller(self, fake_neo4j_connection, tiny_sentence_transformer_dir, graph_schema=None):
        model = EmbeddingModel(tiny_sentence_transformer_dir)
        store = make_vector_store()
        linker = EntityLinker(fake_neo4j_connection)
        return HybridRetrievalController(
            store, fake_neo4j_connection, model, linker, graph_schema=graph_schema
        )

    def test_empty_entities_returns_empty_without_querying(self, fake_neo4j_connection, tiny_sentence_transformer_dir):
        seed_scenario_graph(fake_neo4j_connection)
        controller = self._make_controller(fake_neo4j_connection, tiny_sentence_transformer_dir)
        assert controller.run_graph_retrieval([]) == []

    def test_single_hop_traversal(self, fake_neo4j_connection, tiny_sentence_transformer_dir):
        seed_scenario_graph(fake_neo4j_connection)
        controller = self._make_controller(fake_neo4j_connection, tiny_sentence_transformer_dir)
        entity = GraphNodeRef(
            mention_text="Asha Rao", canonical_name="Asha Rao",
            entity_type=EntityType.EMPLOYEE, match_score=100.0,
        )
        facts = controller.run_graph_retrieval([entity], max_hops=1)
        assert len(facts) == 1
        assert facts[0].hop_count == 1
        assert facts[0].text == "Asha Rao belongs to Finance."

    def test_multi_hop_traversal(self, fake_neo4j_connection, tiny_sentence_transformer_dir):
        seed_scenario_graph(fake_neo4j_connection)
        controller = self._make_controller(fake_neo4j_connection, tiny_sentence_transformer_dir)
        entity = GraphNodeRef(
            mention_text="Asha Rao", canonical_name="Asha Rao",
            entity_type=EntityType.EMPLOYEE, match_score=100.0,
        )
        facts = controller.run_graph_retrieval([entity], max_hops=3, k=20)
        hop_counts = sorted(f.hop_count for f in facts)
        assert hop_counts == [1, 2, 3]
        by_hop = {f.hop_count: f for f in facts}
        assert by_hop[1].raw_relevance > by_hop[2].raw_relevance > by_hop[3].raw_relevance

    def test_k_caps_returned_length(self, fake_neo4j_connection, tiny_sentence_transformer_dir):
        seed_scenario_graph(fake_neo4j_connection)
        controller = self._make_controller(fake_neo4j_connection, tiny_sentence_transformer_dir)
        entity = GraphNodeRef(
            mention_text="Asha Rao", canonical_name="Asha Rao",
            entity_type=EntityType.EMPLOYEE, match_score=100.0,
        )
        facts = controller.run_graph_retrieval([entity], max_hops=3, k=1)
        assert len(facts) == 1
        assert facts[0].hop_count == 1  # highest raw_relevance kept

    def test_dedup_across_two_linked_entities(self, fake_neo4j_connection, tiny_sentence_transformer_dir):
        # Two different starting entities both reach Finance -> Leave Policy,
        # one at hop 1 (direct) and one at hop 2 (through an extra node).
        store = fake_neo4j_connection._test_store
        store.add_node("DEPARTMENT", "Finance")
        store.add_node("POLICY", "Leave Policy")
        store.add_node("PROJECT", "Onboarding")
        store.add_relationship("Finance", RelationType.FOLLOWS.value, "Leave Policy")
        store.add_relationship("Onboarding", RelationType.USES.value, "Finance")

        controller = self._make_controller(fake_neo4j_connection, tiny_sentence_transformer_dir)
        entities = [
            GraphNodeRef(mention_text="Finance", canonical_name="Finance", entity_type=EntityType.DEPARTMENT, match_score=100.0),
            GraphNodeRef(mention_text="Onboarding", canonical_name="Onboarding", entity_type=EntityType.PROJECT, match_score=100.0),
        ]
        facts = controller.run_graph_retrieval(entities, max_hops=3, k=20)
        matching = [f for f in facts if f.subject == "Finance" and f.object == "Leave Policy"]
        assert len(matching) == 1
        assert matching[0].hop_count == 1  # best (lowest) hop_count kept

    @pytest.mark.parametrize("bad_max_hops,bad_k", [(0, 1), (1, 0)])
    def test_non_positive_args_raise(self, fake_neo4j_connection, tiny_sentence_transformer_dir, bad_max_hops, bad_k):
        seed_scenario_graph(fake_neo4j_connection)
        controller = self._make_controller(fake_neo4j_connection, tiny_sentence_transformer_dir)
        entity = GraphNodeRef(
            mention_text="Asha Rao", canonical_name="Asha Rao",
            entity_type=EntityType.EMPLOYEE, match_score=100.0,
        )
        with pytest.raises(ValueError):
            controller.run_graph_retrieval([entity], max_hops=bad_max_hops, k=bad_k)

    def test_out_of_schema_relationship_raises_retrieval_error(self, fake_neo4j_connection, tiny_sentence_transformer_dir):
        seed_scenario_graph(fake_neo4j_connection)
        restricted_schema = GraphSchema(
            allowed_node_labels=frozenset(e.value for e in EntityType),
            allowed_relationship_types=frozenset({RelationType.BELONGS_TO.value}),
        )
        controller = self._make_controller(fake_neo4j_connection, tiny_sentence_transformer_dir, graph_schema=restricted_schema)
        entity = GraphNodeRef(
            mention_text="Asha Rao", canonical_name="Asha Rao",
            entity_type=EntityType.EMPLOYEE, match_score=100.0,
        )
        with pytest.raises(RetrievalError):
            controller.run_graph_retrieval([entity], max_hops=2)


# ========================================================================
# 10.5 hybrid_controller.py -- normalize_scores
# ========================================================================
class TestNormalizeScores:
    def test_empty_list(self):
        assert HybridRetrievalController.normalize_scores([]) == []

    def test_single_element(self):
        assert HybridRetrievalController.normalize_scores([0.42]) == [1.0]

    def test_all_equal_values(self):
        assert HybridRetrievalController.normalize_scores([0.5, 0.5, 0.5]) == [1.0, 1.0, 1.0]

    def test_distinct_values_hand_computed(self):
        result = HybridRetrievalController.normalize_scores([1.0, 2.0, 4.0])
        assert result == [0.0, 1 / 3, 1.0]

    @pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
    def test_nan_or_inf_raises(self, bad_value):
        with pytest.raises(ValueError):
            HybridRetrievalController.normalize_scores([1.0, bad_value])


# ========================================================================
# 10.6 hybrid_controller.py -- static_fuse
# ========================================================================
class TestStaticFuse:
    def test_vector_only(self):
        hits = [
            VectorSearchResult(chunk_id="c1", score=0.9, text="a"),
            VectorSearchResult(chunk_id="c2", score=0.1, text="b"),
        ]
        result = HybridRetrievalController.static_fuse(hits, [], alpha=0.5)
        assert all(c.source == CandidateSource.VECTOR for c in result)
        assert all(c.graph_score is None for c in result)
        top = result[0]
        assert top.candidate_id == "c1"
        assert math.isclose(top.fused_score, 0.5 * 1.0)

    def test_graph_only(self):
        facts = [
            GraphFact.build("A", RelationType.BELONGS_TO.value, "B", hop_count=1),
            GraphFact.build("C", RelationType.BELONGS_TO.value, "D", hop_count=2),
        ]
        result = HybridRetrievalController.static_fuse([], facts, alpha=0.5)
        assert all(c.source == CandidateSource.GRAPH for c in result)
        assert all(c.vector_score is None for c in result)
        top = result[0]
        assert math.isclose(top.fused_score, 0.5 * 1.0)

    def test_both_nonempty_no_overlap(self):
        hits = [VectorSearchResult(chunk_id="c1", score=0.8, text="a")]
        facts = [GraphFact.build("A", RelationType.BELONGS_TO.value, "B", hop_count=1, source_chunk_id="c-other")]
        result = HybridRetrievalController.static_fuse(hits, facts, alpha=0.5)
        sources = {c.source for c in result}
        assert sources == {CandidateSource.VECTOR, CandidateSource.GRAPH}

    def test_both_nonempty_with_overlap_merges(self):
        hits = [VectorSearchResult(chunk_id="c1", score=0.8, text="vector text")]
        facts = [GraphFact.build("A", RelationType.BELONGS_TO.value, "B", hop_count=1, source_chunk_id="c1")]
        result = HybridRetrievalController.static_fuse(hits, facts, alpha=0.5)
        assert len(result) == 1
        merged = result[0]
        assert merged.source == CandidateSource.BOTH
        assert merged.candidate_id == "c1"
        assert merged.vector_score is not None
        assert merged.graph_score is not None
        assert merged.text == "vector text"

    def test_multiple_facts_same_chunk_all_merge(self):
        # Two different relations were extracted from the same chunk --
        # both must contribute to the single BOTH candidate, not just
        # the first one encountered.
        hits = [VectorSearchResult(chunk_id="c1", score=0.8, text="vector text")]
        facts = [
            GraphFact.build("A", RelationType.BELONGS_TO.value, "B", hop_count=1, source_chunk_id="c1"),
            GraphFact.build("A", RelationType.HAS_ROLE.value, "C", hop_count=1, source_chunk_id="c1"),
        ]
        result = HybridRetrievalController.static_fuse(hits, facts, alpha=0.5)
        assert len(result) == 1
        assert result[0].source == CandidateSource.BOTH

    def test_alpha_one_is_vector_ranked_equivalent(self):
        hits = [
            VectorSearchResult(chunk_id="c1", score=0.2, text="a"),
            VectorSearchResult(chunk_id="c2", score=0.9, text="b"),
        ]
        facts = [GraphFact.build("A", RelationType.BELONGS_TO.value, "B", hop_count=1)]
        result = HybridRetrievalController.static_fuse(hits, facts, alpha=1.0)
        vector_candidates = [c for c in result if c.source != CandidateSource.GRAPH]
        assert [c.candidate_id for c in vector_candidates] == ["c2", "c1"]

    def test_alpha_zero_is_graph_ranked_equivalent(self):
        hits = [VectorSearchResult(chunk_id="c1", score=0.9, text="a")]
        facts = [
            GraphFact.build("A", RelationType.BELONGS_TO.value, "B", hop_count=1),
            GraphFact.build("C", RelationType.BELONGS_TO.value, "D", hop_count=3),
        ]
        result = HybridRetrievalController.static_fuse(hits, facts, alpha=0.0)
        graph_candidates = [c for c in result if c.source == CandidateSource.GRAPH]
        assert graph_candidates[0].fused_score >= graph_candidates[1].fused_score

    def test_output_sorted_descending(self):
        hits = [
            VectorSearchResult(chunk_id="c1", score=0.1, text="a"),
            VectorSearchResult(chunk_id="c2", score=0.9, text="b"),
            VectorSearchResult(chunk_id="c3", score=0.5, text="c"),
        ]
        result = HybridRetrievalController.static_fuse(hits, [], alpha=1.0)
        scores = [c.fused_score for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_alpha_out_of_range_raises(self):
        with pytest.raises(ValueError):
            HybridRetrievalController.static_fuse([], [], alpha=1.5)

    def test_both_inputs_empty_returns_empty(self):
        assert HybridRetrievalController.static_fuse([], [], alpha=0.5) == []


# ========================================================================
# 10.7 hybrid_controller.py -- retrieve (end-to-end)
# ========================================================================
class TestRetrieveEndToEnd:
    def _make_controller(self, fake_neo4j_connection, store, model):
        linker = EntityLinker(fake_neo4j_connection)
        return HybridRetrievalController(store, fake_neo4j_connection, model, linker)

    def test_links_matches_and_fuses(self, fake_neo4j_connection, tiny_sentence_transformer_dir):
        seed_scenario_graph(fake_neo4j_connection)
        model = EmbeddingModel(tiny_sentence_transformer_dir)
        store = make_vector_store()
        store.add(
            [
                make_embedded_chunk(
                    "chunk-1",
                    "Asha Rao belongs to Finance and follows the Leave Policy for casual leave.",
                    model,
                ),
                make_embedded_chunk("chunk-2", "The office coffee machine is broken again.", model),
            ]
        )
        controller = self._make_controller(fake_neo4j_connection, store, model)
        results = controller.retrieve("What is Asha Rao's leave policy in Finance?", alpha=0.5)
        assert results
        scores = [c.fused_score for c in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_links_no_entities_still_returns_vector_only(self, fake_neo4j_connection, tiny_sentence_transformer_dir):
        seed_scenario_graph(fake_neo4j_connection)
        model = EmbeddingModel(tiny_sentence_transformer_dir)
        store = make_vector_store()
        store.add([make_embedded_chunk("chunk-1", "The printer on the fourth floor needs toner.", model)])
        controller = self._make_controller(fake_neo4j_connection, store, model)
        results = controller.retrieve("printer toner fourth floor", alpha=0.5)
        assert results
        assert all(c.source == CandidateSource.VECTOR for c in results)

    def test_query_links_entity_but_no_relevant_chunks_returns_graph_only(self, fake_neo4j_connection, tiny_sentence_transformer_dir):
        seed_scenario_graph(fake_neo4j_connection)
        model = EmbeddingModel(tiny_sentence_transformer_dir)
        store = make_vector_store()  # empty vector store
        controller = self._make_controller(fake_neo4j_connection, store, model)
        results = controller.retrieve("Tell me about Asha Rao", alpha=0.5)
        assert results
        assert all(c.source == CandidateSource.GRAPH for c in results)

    def test_invalid_alpha_propagates(self, fake_neo4j_connection, tiny_sentence_transformer_dir):
        seed_scenario_graph(fake_neo4j_connection)
        model = EmbeddingModel(tiny_sentence_transformer_dir)
        store = make_vector_store()
        store.add([make_embedded_chunk("chunk-1", "Some text about Finance.", model)])
        controller = self._make_controller(fake_neo4j_connection, store, model)
        with pytest.raises(ValueError):
            controller.retrieve("Finance", alpha=2.0)

    def test_invalid_k_propagates(self, fake_neo4j_connection, tiny_sentence_transformer_dir):
        seed_scenario_graph(fake_neo4j_connection)
        model = EmbeddingModel(tiny_sentence_transformer_dir)
        store = make_vector_store()
        store.add([make_embedded_chunk("chunk-1", "Some text about Finance.", model)])
        controller = self._make_controller(fake_neo4j_connection, store, model)
        with pytest.raises(ValueError):
            controller.retrieve("Finance", k_vector=0)
