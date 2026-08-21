"""
Unit tests for Module 10 - Adaptive Retrieval.

Covers adaptive_retrieval/classifier.py, rules.py, train.py, and
fusion.py. Uses the same real-component testing pattern established in
Modules 7/8/9: a real EmbeddingModel (tiny_sentence_transformer_dir
fixture), a real in-memory VectorStore, and the fake_neo4j_connection
fixture for the graph side.
"""

from __future__ import annotations

import math
import uuid

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from adaptive_retrieval.classifier import (
    AlphaClassifierError,
    AlphaLabel,
    LabeledQuery,
    LearnedAlphaClassifier,
    SchemaValidationError,
    predict_alpha,
)
from adaptive_retrieval.fusion import AdaptiveFusionEngine, AdaptiveFusionError, adaptive_fuse
from adaptive_retrieval.rules import RuleBasedAlphaClassifier, rule_based_alpha
from adaptive_retrieval.train import AlphaTrainingError, train_classifier
from embeddings.model import EmbeddingModel
from embeddings.schema import EmbeddedChunk
from ner.schema import EntityType
from preprocessing.schema import DocumentType
from relation_extraction.schema import RelationType
from retrieval.entity_linker import EntityLinker
from retrieval.hybrid_controller import HybridRetrievalController
from retrieval.schema import CandidateSource, GraphFact
from vector_db.chroma_store import VectorStore
from vector_db.schema import VectorSearchResult


def make_vector_store() -> VectorStore:
    """Unique collection name per call -- see Module 9's own note on why
    a reused name is NOT isolated within ChromaDB's in-process ephemeral
    client (empirically verified during Module 9's own test-writing)."""
    return VectorStore(collection_name=f"test_module10_{uuid.uuid4().hex}", persist_directory=None)


def make_embedded_chunk(chunk_id: str, text: str, model: EmbeddingModel) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=chunk_id,
        vector=model.embed(text),
        text=text,
        source_document_id="doc-1",
        document_type=DocumentType.POLICY,
        department="Finance",
    )


def seed_scenario_graph(fake_neo4j_connection) -> None:
    store = fake_neo4j_connection._test_store
    store.add_node("EMPLOYEE", "Asha Rao")
    store.add_node("DEPARTMENT", "Finance")
    store.add_relationship(
        "Asha Rao", RelationType.BELONGS_TO.value, "Finance",
        properties={"source_chunk_id": "chunk-1"},
    )


# ========================================================================
# classifier.py
# ========================================================================
class TestLabeledQuery:
    def test_valid_construction(self):
        lq = LabeledQuery(query="Who manages Finance?", label=AlphaLabel.GRAPH_FAVORING)
        assert lq.target_alpha == pytest.approx(0.1)

    def test_vector_favoring_target_alpha(self):
        lq = LabeledQuery(query="Explain the leave policy.", label=AlphaLabel.VECTOR_FAVORING)
        assert lq.target_alpha == pytest.approx(0.9)

    def test_mixed_target_alpha(self):
        lq = LabeledQuery(query="Tell me about leave.", label=AlphaLabel.MIXED)
        assert lq.target_alpha == pytest.approx(0.5)

    def test_empty_query_raises(self):
        with pytest.raises(SchemaValidationError):
            LabeledQuery(query="   ", label=AlphaLabel.MIXED)


class TestLearnedAlphaClassifierUnfitted:
    def test_is_fitted_false_when_no_model(self):
        classifier = LearnedAlphaClassifier()
        assert classifier.is_fitted is False

    def test_predict_alpha_before_training_raises(self):
        classifier = LearnedAlphaClassifier()
        with pytest.raises(AlphaClassifierError):
            classifier.predict_alpha("some query", np.zeros(8, dtype=np.float32))


class TestLearnedAlphaClassifierFitted:
    @staticmethod
    def _fitted_classifier() -> LearnedAlphaClassifier:
        # Tiny synthetic 2-D embedding space: class 0 (graph-favoring)
        # clusters near [0,0], class 1 (vector-favoring) near [10,10].
        X = np.array([[0.0, 0.0], [0.1, -0.1], [10.0, 10.0], [9.9, 10.1]])
        y = np.array([0, 0, 1, 1])
        model = LogisticRegression().fit(X, y)
        return LearnedAlphaClassifier.from_sklearn_model(model)

    def test_is_fitted_true(self):
        classifier = self._fitted_classifier()
        assert classifier.is_fitted is True

    def test_predicts_high_alpha_for_vector_favoring_region(self):
        classifier = self._fitted_classifier()
        alpha = classifier.predict_alpha("irrelevant text", np.array([10.0, 10.0]))
        assert alpha > 0.9

    def test_predicts_low_alpha_for_graph_favoring_region(self):
        classifier = self._fitted_classifier()
        alpha = classifier.predict_alpha("irrelevant text", np.array([0.0, 0.0]))
        assert alpha < 0.1

    def test_non_1d_embedding_raises(self):
        classifier = self._fitted_classifier()
        with pytest.raises(ValueError):
            classifier.predict_alpha("text", np.zeros((2, 2)))

    def test_module_level_predict_alpha(self):
        classifier = self._fitted_classifier()
        alpha = predict_alpha(classifier, "text", np.array([10.0, 10.0]))
        assert 0.0 <= alpha <= 1.0


# ========================================================================
# rules.py
# ========================================================================
class TestRuleBasedAlphaClassifier:
    def test_relational_query_yields_low_alpha(self):
        classifier = RuleBasedAlphaClassifier()
        alpha = classifier.predict_alpha("Who manages the Finance department?", np.empty(0))
        assert alpha < 0.5

    def test_descriptive_query_yields_high_alpha(self):
        classifier = RuleBasedAlphaClassifier()
        alpha = classifier.predict_alpha("Explain how do I request VPN access?", np.empty(0))
        assert alpha > 0.5

    def test_no_match_returns_neutral_default(self):
        classifier = RuleBasedAlphaClassifier()
        alpha = classifier.predict_alpha("zzzz qqqq unrelated gibberish", np.empty(0))
        assert alpha == pytest.approx(0.5)

    def test_empty_query_raises(self):
        classifier = RuleBasedAlphaClassifier()
        with pytest.raises(ValueError):
            classifier.predict_alpha("   ", np.empty(0))

    def test_custom_lexicons(self):
        classifier = RuleBasedAlphaClassifier(
            relational_indicators=("foo",), descriptive_indicators=("bar",)
        )
        assert classifier.predict_alpha("foo foo", np.empty(0)) == pytest.approx(0.0)
        assert classifier.predict_alpha("bar", np.empty(0)) == pytest.approx(1.0)

    def test_module_level_rule_based_alpha(self):
        alpha = rule_based_alpha("Who manages Finance?")
        assert 0.0 <= alpha <= 1.0
        assert alpha < 0.5


# ========================================================================
# train.py
# ========================================================================
class TestTrainClassifier:
    def test_trains_and_returns_usable_classifier(self, tiny_sentence_transformer_dir):
        model = EmbeddingModel(tiny_sentence_transformer_dir)
        labeled_queries = [
            LabeledQuery("Who manages Finance?", AlphaLabel.GRAPH_FAVORING),
            LabeledQuery("Which department does Asha Rao belong to?", AlphaLabel.GRAPH_FAVORING),
            LabeledQuery("Explain the casual leave policy.", AlphaLabel.VECTOR_FAVORING),
            LabeledQuery("How do I request VPN access?", AlphaLabel.VECTOR_FAVORING),
        ]
        classifier = train_classifier(labeled_queries, model, max_iter=200)
        assert classifier.is_fitted
        alpha = classifier.predict_alpha("Explain leave.", model.embed("Explain leave."))
        assert 0.0 <= alpha <= 1.0

    def test_excludes_mixed_examples(self, tiny_sentence_transformer_dir):
        model = EmbeddingModel(tiny_sentence_transformer_dir)
        labeled_queries = [
            LabeledQuery("Who manages Finance?", AlphaLabel.GRAPH_FAVORING),
            LabeledQuery("Explain the casual leave policy.", AlphaLabel.VECTOR_FAVORING),
            LabeledQuery("Tell me about leave in Finance.", AlphaLabel.MIXED),
        ]
        # Should not raise despite the MIXED example being unusable for
        # binary training -- it's simply excluded, not an error.
        classifier = train_classifier(labeled_queries, model, max_iter=200)
        assert classifier.is_fitted

    def test_empty_labeled_queries_raises(self, tiny_sentence_transformer_dir):
        model = EmbeddingModel(tiny_sentence_transformer_dir)
        with pytest.raises(AlphaTrainingError):
            train_classifier([], model)

    def test_only_mixed_examples_raises(self, tiny_sentence_transformer_dir):
        model = EmbeddingModel(tiny_sentence_transformer_dir)
        labeled_queries = [LabeledQuery("Tell me about leave.", AlphaLabel.MIXED)]
        with pytest.raises(AlphaTrainingError):
            train_classifier(labeled_queries, model)

    def test_single_class_raises(self, tiny_sentence_transformer_dir):
        model = EmbeddingModel(tiny_sentence_transformer_dir)
        labeled_queries = [
            LabeledQuery("Who manages Finance?", AlphaLabel.GRAPH_FAVORING),
            LabeledQuery("Which department is this?", AlphaLabel.GRAPH_FAVORING),
        ]
        with pytest.raises(AlphaTrainingError):
            train_classifier(labeled_queries, model)


# ========================================================================
# fusion.py
# ========================================================================
class TestAdaptiveFusionEngine:
    def _make_engine(self, fake_neo4j_connection, tiny_sentence_transformer_dir, classifier):
        model = EmbeddingModel(tiny_sentence_transformer_dir)
        store = make_vector_store()
        store.add(
            [
                make_embedded_chunk("chunk-1", "Asha Rao belongs to the Finance department.", model),
            ]
        )
        linker = EntityLinker(fake_neo4j_connection)
        controller = HybridRetrievalController(store, fake_neo4j_connection, model, linker)
        engine = AdaptiveFusionEngine(controller, model, linker, classifier)
        return engine

    def test_compute_alpha_with_rule_based_classifier(self, fake_neo4j_connection, tiny_sentence_transformer_dir):
        seed_scenario_graph(fake_neo4j_connection)
        engine = self._make_engine(fake_neo4j_connection, tiny_sentence_transformer_dir, RuleBasedAlphaClassifier())
        alpha = engine.compute_alpha("Who manages Finance?")
        assert 0.0 <= alpha <= 1.0

    def test_compute_alpha_empty_query_raises(self, fake_neo4j_connection, tiny_sentence_transformer_dir):
        seed_scenario_graph(fake_neo4j_connection)
        engine = self._make_engine(fake_neo4j_connection, tiny_sentence_transformer_dir, RuleBasedAlphaClassifier())
        with pytest.raises(ValueError):
            engine.compute_alpha("   ")

    def test_classifier_returning_out_of_range_alpha_raises(self, fake_neo4j_connection, tiny_sentence_transformer_dir):
        class _BadClassifier:
            def predict_alpha(self, query, query_embedding):
                return 1.5

        seed_scenario_graph(fake_neo4j_connection)
        engine = self._make_engine(fake_neo4j_connection, tiny_sentence_transformer_dir, _BadClassifier())
        with pytest.raises(AdaptiveFusionError):
            engine.compute_alpha("anything")

    def test_retrieve_end_to_end_with_rule_based_classifier(self, fake_neo4j_connection, tiny_sentence_transformer_dir):
        seed_scenario_graph(fake_neo4j_connection)
        engine = self._make_engine(fake_neo4j_connection, tiny_sentence_transformer_dir, RuleBasedAlphaClassifier())
        alpha, candidates = engine.retrieve("Who does Asha Rao belong to in Finance?")
        assert 0.0 <= alpha <= 1.0
        assert candidates
        scores = [c.fused_score for c in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_uses_the_reported_alpha_in_fusion(self, fake_neo4j_connection, tiny_sentence_transformer_dir):
        # A classifier that always returns alpha=1.0 (vector-only) should
        # produce the same fusion outcome as calling static_fuse directly
        # with alpha=1.0 -- proving retrieve() doesn't silently use a
        # different alpha internally than the one it reports.
        class _AlwaysVectorClassifier:
            def predict_alpha(self, query, query_embedding):
                return 1.0

        seed_scenario_graph(fake_neo4j_connection)
        engine = self._make_engine(fake_neo4j_connection, tiny_sentence_transformer_dir, _AlwaysVectorClassifier())
        alpha, candidates = engine.retrieve("Who does Asha Rao belong to in Finance?")
        assert alpha == pytest.approx(1.0)
        assert all(c.source != CandidateSource.GRAPH or c.fused_score == pytest.approx(0.0) for c in candidates)

    def test_retrieve_propagates_invalid_k(self, fake_neo4j_connection, tiny_sentence_transformer_dir):
        seed_scenario_graph(fake_neo4j_connection)
        engine = self._make_engine(fake_neo4j_connection, tiny_sentence_transformer_dir, RuleBasedAlphaClassifier())
        with pytest.raises(ValueError):
            engine.retrieve("Finance", k_vector=0)


class TestAdaptiveFuseAlias:
    def test_delegates_to_static_fuse(self):
        hits = [VectorSearchResult(chunk_id="c1", score=0.9, text="a")]
        facts: list[GraphFact] = []
        result = adaptive_fuse(hits, facts, alpha=0.7)
        expected = HybridRetrievalController.static_fuse(hits, facts, alpha=0.7)
        assert [c.candidate_id for c in result] == [c.candidate_id for c in expected]
        assert [c.fused_score for c in result] == [c.fused_score for c in expected]

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            adaptive_fuse([], [], alpha=2.0)
