"""
Unit tests for Module 14: API & Service Layer.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.pipeline import EnterpriseRAGPipeline
from api.routes import set_pipeline
from api.schema import IngestRequest, QueryRequest


class TestAPISchemas:
    def test_query_request_valid(self):
        req = QueryRequest(query="What is the leave policy?", top_k=5)
        assert req.query == "What is the leave policy?"
        assert req.top_k == 5

    def test_ingest_request_valid(self):
        req = IngestRequest(file_paths=["doc1.txt", "doc2.pdf"])
        assert len(req.file_paths) == 2


class TestAPIEndpoints:
    @pytest.fixture
    def client(self, fake_neo4j_connection, tiny_sentence_transformer_dir):
        from embeddings.model import EmbeddingModel
        from vector_db.chroma_store import VectorStore

        embed_model = EmbeddingModel(model_name_or_path=tiny_sentence_transformer_dir)
        vector_store = VectorStore(collection_name="test_api")

        pipeline = EnterpriseRAGPipeline(
            neo4j_connection=fake_neo4j_connection,
            vector_store=vector_store,
            embedding_model=embed_model,
        )
        set_pipeline(pipeline)
        return TestClient(app)

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_query_endpoint(self, client):
        payload = {"query": "What is the annual leave allowance?", "top_k": 3}
        response = client.post("/api/v1/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == payload["query"]
        assert "answer" in data
        assert "latency_ms" in data

    def test_ingest_endpoint(self, client, tmp_path):
        doc_path = tmp_path / "sample.txt"
        doc_path.write_text("Finance department policy allows 15 days casual leave.", encoding="utf-8")

        payload = {"file_paths": [str(doc_path)]}
        response = client.post("/api/v1/ingest", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["documents_processed"] == 1
        assert data["chunks_created"] > 0
