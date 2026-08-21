"""
End-to-End Integration Tests for Enterprise RAG System.

Validates complete pipeline flow: Document Ingestion -> Preprocessing -> Chunking -> Vector & Graph Indexing -> Adaptive Hybrid Retrieval -> Cross-Encoder Reranking -> LLM Answer Generation with Citations.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.pipeline import EnterpriseRAGPipeline
from api.routes import set_pipeline
from embeddings.model import EmbeddingModel
from vector_db.chroma_store import VectorStore


class TestEndToEndPipeline:
    @pytest.fixture
    def setup_pipeline(self, fake_neo4j_connection, tiny_sentence_transformer_dir, tmp_path):
        # 1. Initialize components
        embed_model = EmbeddingModel(model_name_or_path=tiny_sentence_transformer_dir)
        vector_store = VectorStore(collection_name="e2e_test_collection")

        pipeline = EnterpriseRAGPipeline(
            neo4j_connection=fake_neo4j_connection,
            vector_store=vector_store,
            embedding_model=embed_model,
        )
        set_pipeline(pipeline)

        # 2. Create sample enterprise policy documents
        doc1 = tmp_path / "hr_policy.txt"
        doc1.write_text(
            "Section 1.0 Leave Entitlement.\n"
            "All full-time employees receive 25 days of annual leave per calendar year.\n"
            "Casual leave allowance is fixed at 10 days annually.",
            encoding="utf-8",
        )

        doc2 = tmp_path / "finance_policy.txt"
        doc2.write_text(
            "Section 2.0 Expense Reimbursement.\n"
            "Employees may claim travel expenses up to $500 without prior manager approval.\n"
            "All claims above $500 require VP finance authorization.",
            encoding="utf-8",
        )

        # 3. Perform ingestion
        ingest_res = pipeline.ingest_files([str(doc1), str(doc2)])
        assert ingest_res.documents_processed == 2
        assert ingest_res.chunks_created >= 2

        return pipeline, TestClient(app)

    def test_end_to_end_query_flow(self, setup_pipeline):
        pipeline, client = setup_pipeline

        # Test direct pipeline query
        response = pipeline.answer_query(
            query="How many annual leave days do full-time employees get?",
            top_k=3,
            use_adaptive=True,
        )

        assert response.query == "How many annual leave days do full-time employees get?"
        assert isinstance(response.answer, str)
        assert len(response.answer) > 0
        assert isinstance(response.citations, list)
        assert response.latency_ms > 0
        assert len(response.source_candidates) > 0

    def test_end_to_end_api_endpoints(self, setup_pipeline):
        pipeline, client = setup_pipeline

        # Health endpoint
        health_res = client.get("/health")
        assert health_res.status_code == 200
        assert health_res.json()["status"] == "healthy"

        # Query endpoint
        query_payload = {
            "query": "What is the manager approval limit for travel expenses?",
            "top_k": 3,
            "use_adaptive": True,
            "use_reranking": True,
        }
        res = client.post("/api/v1/query", json=query_payload)
        assert res.status_code == 200
        data = res.json()

        assert data["query"] == query_payload["query"]
        assert "answer" in data
        assert isinstance(data["citations"], list)
        assert data["alpha_used"] >= 0.0
        assert data["latency_ms"] >= 0.0
