"""
Unit tests for Module 13: Evaluation & Ablation.
"""

from __future__ import annotations

import pytest

from evaluation.ablation_runner import AblationRunner, EvaluationExample
from evaluation.generation_metrics import (
    calculate_answer_relevance,
    calculate_citation_precision,
    calculate_faithfulness,
    evaluate_generation_batch,
)
from evaluation.retrieval_metrics import (
    calculate_hit_rate,
    calculate_mrr,
    calculate_ndcg,
    evaluate_retrieval_batch,
)


class TestRetrievalMetrics:
    def test_calculate_hit_rate(self):
        retrieved = ["chunk_1", "chunk_2", "chunk_3"]
        ground_truth = ["chunk_3"]
        assert calculate_hit_rate(retrieved, ground_truth, k=5) == 1.0
        assert calculate_hit_rate(retrieved, ["chunk_99"], k=5) == 0.0

    def test_calculate_mrr(self):
        retrieved = ["chunk_1", "chunk_2", "chunk_3"]
        assert calculate_mrr(retrieved, ["chunk_1"], k=5) == 1.0
        assert calculate_mrr(retrieved, ["chunk_2"], k=5) == 0.5
        assert calculate_mrr(retrieved, ["chunk_3"], k=5) == 1.0 / 3.0
        assert calculate_mrr(retrieved, ["chunk_99"], k=5) == 0.0

    def test_calculate_ndcg(self):
        retrieved = ["chunk_1", "chunk_2", "chunk_3"]
        assert calculate_ndcg(retrieved, ["chunk_1"], k=5) == 1.0
        assert calculate_ndcg(retrieved, ["chunk_99"], k=5) == 0.0

    def test_evaluate_retrieval_batch(self):
        cases = [
            {"retrieved": ["c1", "c2"], "ground_truth": ["c1"]},
            {"retrieved": ["c1", "c2"], "ground_truth": ["c2"]},
        ]
        res = evaluate_retrieval_batch(cases, k=2)
        assert res["hit_rate@2"] == 1.0
        assert res["mrr@2"] == 0.75


class TestGenerationMetrics:
    def test_citation_precision(self):
        citations = ["chunk_1", "chunk_2"]
        valid = ["chunk_1", "chunk_3"]
        assert calculate_citation_precision(citations, valid) == 0.5

    def test_faithfulness(self):
        answer = "The policy allows casual leave."
        evidence = ["Employees are granted casual leave under policy 5."]
        score = calculate_faithfulness(answer, evidence)
        assert score > 0.0

    def test_answer_relevance(self):
        query = "What is the annual leave policy?"
        answer = "The annual leave policy allows 15 paid days off per year."
        score = calculate_answer_relevance(answer, query)
        assert score > 0.0

    def test_evaluate_generation_batch(self):
        cases = [
            {
                "answer": "Test answer",
                "query": "Test query",
                "evidence_texts": ["Test evidence"],
                "citations": ["c1"],
                "valid_source_ids": ["c1"],
            }
        ]
        metrics = evaluate_generation_batch(cases)
        assert metrics["citation_precision"] == 1.0
        assert metrics["faithfulness"] > 0.0
        assert metrics["answer_relevance"] > 0.0


class TestAblationRunner:
    def test_ablation_run_all(self, fake_neo4j_connection, tiny_sentence_transformer_dir):
        from embeddings.model import EmbeddingModel
        from retrieval.entity_linker import EntityLinker
        from retrieval.hybrid_controller import HybridRetrievalController
        from vector_db.chroma_store import VectorStore

        embed_model = EmbeddingModel(model_name_or_path=tiny_sentence_transformer_dir)
        vector_store = VectorStore(collection_name="test_ablation")
        linker = EntityLinker(fake_neo4j_connection)
        linker.refresh_index()

        controller = HybridRetrievalController(
            graph_connection=fake_neo4j_connection,
            vector_store=vector_store,
            embedding_model=embed_model,
            entity_linker=linker,
        )

        runner = AblationRunner(hybrid_controller=controller)
        examples = [
            EvaluationExample(query="annual leave policy", ground_truth_chunk_ids=("c1",)),
        ]

        results = runner.run_all(examples, k=3)
        assert "vector_only" in results
        assert "graph_only" in results
        assert "static_hybrid" in results
        assert results["vector_only"].strategy_name == "vector_only"
