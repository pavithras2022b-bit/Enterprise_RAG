"""
Unit tests for Module 8 - Vector Database.

Covers vector_db/schema.py, chroma_store.py, and faiss_index.py.
ChromaDB and FAISS both run fully in-process/in-memory here -- no
external service or network access needed for either.
"""

from __future__ import annotations

import numpy as np
import pytest

from embeddings.schema import EmbeddedChunk
from vector_db.chroma_store import VectorStore, VectorStoreError
from vector_db.faiss_index import FAISSBenchmarkIndex, FAISSIndexError
from vector_db.schema import SchemaValidationError, VectorSearchResult


def _chunk(chunk_id: str, vector: list[float], text: str, department: str = "HR") -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=chunk_id,
        vector=np.array(vector, dtype=np.float32),
        text=text,
        source_document_id="doc1",
        department=department,
    )


# --------------------------------------------------------------------------
# schema.py
# --------------------------------------------------------------------------


class TestVectorSearchResult:
    def test_valid_construction(self) -> None:
        result = VectorSearchResult(chunk_id="c1", score=0.92, text="Casual leave is 12 days.", metadata={"department": "HR"})
        assert result.score == 0.92

    def test_empty_chunk_id_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            VectorSearchResult(chunk_id="", score=0.5, text="text")

    def test_nan_score_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            VectorSearchResult(chunk_id="c1", score=float("nan"), text="text")

    def test_inf_score_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            VectorSearchResult(chunk_id="c1", score=float("inf"), text="text")

    def test_default_metadata_is_empty_dict(self) -> None:
        result = VectorSearchResult(chunk_id="c1", score=0.5, text="text")
        assert result.metadata == {}


# --------------------------------------------------------------------------
# chroma_store.py
# --------------------------------------------------------------------------


class TestVectorStore:
    def test_add_and_count(self) -> None:
        store = VectorStore(collection_name="test_add_and_count")
        store.add([_chunk("c1", [1.0, 0.0, 0.0], "doc one"), _chunk("c2", [0.0, 1.0, 0.0], "doc two")])
        assert store.count() == 2

    def test_add_empty_list_is_noop(self) -> None:
        store = VectorStore(collection_name="test_add_empty")
        store.add([])
        assert store.count() == 0

    def test_search_returns_most_similar_first(self) -> None:
        store = VectorStore(collection_name="test_search_order")
        store.add(
            [
                _chunk("close", [1.0, 0.0, 0.0], "close match"),
                _chunk("far", [0.0, 1.0, 0.0], "far match"),
            ]
        )
        results = store.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=2)
        assert results[0].chunk_id == "close"
        assert results[0].score > results[1].score

    def test_search_identical_vector_scores_near_one(self) -> None:
        store = VectorStore(collection_name="test_search_identical")
        store.add([_chunk("c1", [1.0, 0.0, 0.0], "text")])
        results = store.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1)
        assert results[0].score == pytest.approx(1.0, abs=1e-4)

    def test_search_orthogonal_vector_scores_near_zero(self) -> None:
        store = VectorStore(collection_name="test_search_orthogonal")
        store.add([_chunk("c1", [1.0, 0.0, 0.0], "text")])
        results = store.search(np.array([0.0, 1.0, 0.0], dtype=np.float32), k=1)
        assert results[0].score == pytest.approx(0.0, abs=1e-4)

    def test_search_returns_text_and_metadata(self) -> None:
        store = VectorStore(collection_name="test_search_metadata")
        store.add([_chunk("c1", [1.0, 0.0, 0.0], "Casual leave is 12 days.", department="HR")])
        results = store.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1)
        assert results[0].text == "Casual leave is 12 days."
        assert results[0].metadata["department"] == "HR"

    def test_search_with_metadata_filter(self) -> None:
        store = VectorStore(collection_name="test_search_filter")
        store.add(
            [
                _chunk("finance_doc", [1.0, 0.0, 0.0], "finance text", department="Finance"),
                _chunk("hr_doc", [1.0, 0.0, 0.0], "hr text", department="HR"),
            ]
        )
        results = store.search(
            np.array([1.0, 0.0, 0.0], dtype=np.float32), k=5, where={"department": "Finance"}
        )
        assert len(results) == 1
        assert results[0].chunk_id == "finance_doc"

    def test_search_invalid_k_raises(self) -> None:
        store = VectorStore(collection_name="test_search_invalid_k")
        with pytest.raises(ValueError):
            store.search(np.array([1.0, 0.0, 0.0]), k=0)

    def test_search_non_1d_vector_raises(self) -> None:
        store = VectorStore(collection_name="test_search_non_1d")
        with pytest.raises(ValueError):
            store.search(np.zeros((2, 3)), k=1)

    def test_add_upserts_by_chunk_id(self) -> None:
        store = VectorStore(collection_name="test_upsert")
        store.add([_chunk("c1", [1.0, 0.0, 0.0], "original text")])
        store.add([_chunk("c1", [1.0, 0.0, 0.0], "updated text")])
        assert store.count() == 1
        results = store.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1)
        assert results[0].text == "updated text"

    def test_delete_removes_chunk(self) -> None:
        store = VectorStore(collection_name="test_delete")
        store.add([_chunk("c1", [1.0, 0.0, 0.0], "text")])
        store.delete("c1")
        assert store.count() == 0

    def test_delete_empty_chunk_id_raises(self) -> None:
        store = VectorStore(collection_name="test_delete_empty")
        with pytest.raises(ValueError):
            store.delete("")

    def test_persistent_client_survives_reconnection(self, tmp_path) -> None:
        persist_dir = str(tmp_path / "chroma_persist")
        store_a = VectorStore(collection_name="persisted", persist_directory=persist_dir)
        store_a.add([_chunk("c1", [1.0, 0.0, 0.0], "persisted text")])

        store_b = VectorStore(collection_name="persisted", persist_directory=persist_dir)
        assert store_b.count() == 1
        results = store_b.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1)
        assert results[0].text == "persisted text"


# --------------------------------------------------------------------------
# faiss_index.py
# --------------------------------------------------------------------------


class TestFAISSBenchmarkIndex:
    def test_add_and_size(self) -> None:
        index = FAISSBenchmarkIndex(dimension=3)
        index.add([_chunk("c1", [1.0, 0.0, 0.0], "one"), _chunk("c2", [0.0, 1.0, 0.0], "two")])
        assert index.size == 2

    def test_add_empty_list_is_noop(self) -> None:
        index = FAISSBenchmarkIndex(dimension=3)
        index.add([])
        assert index.size == 0

    def test_search_returns_most_similar_first(self) -> None:
        index = FAISSBenchmarkIndex(dimension=3)
        index.add(
            [
                _chunk("close", [1.0, 0.0, 0.0], "close match"),
                _chunk("far", [0.0, 1.0, 0.0], "far match"),
            ]
        )
        results = index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=2)
        assert results[0].chunk_id == "close"
        assert results[0].score >= results[1].score

    def test_search_identical_vector_scores_near_one(self) -> None:
        index = FAISSBenchmarkIndex(dimension=3)
        index.add([_chunk("c1", [1.0, 0.0, 0.0], "text")])
        results = index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1)
        assert results[0].score == pytest.approx(1.0, abs=1e-4)

    def test_search_empty_index_returns_empty_list(self) -> None:
        index = FAISSBenchmarkIndex(dimension=3)
        results = index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=5)
        assert results == []

    def test_search_k_larger_than_index_size_capped(self) -> None:
        index = FAISSBenchmarkIndex(dimension=3)
        index.add([_chunk("c1", [1.0, 0.0, 0.0], "only one")])
        results = index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=10)
        assert len(results) == 1

    def test_search_no_metadata_stored(self) -> None:
        index = FAISSBenchmarkIndex(dimension=3)
        index.add([_chunk("c1", [1.0, 0.0, 0.0], "text", department="Finance")])
        results = index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1)
        assert results[0].metadata == {}

    def test_dimension_mismatch_on_add_raises(self) -> None:
        index = FAISSBenchmarkIndex(dimension=3)
        with pytest.raises(FAISSIndexError):
            index.add([_chunk("c1", [1.0, 0.0], "wrong dim")])

    def test_dimension_mismatch_on_search_raises(self) -> None:
        index = FAISSBenchmarkIndex(dimension=3)
        index.add([_chunk("c1", [1.0, 0.0, 0.0], "text")])
        with pytest.raises(ValueError):
            index.search(np.array([1.0, 0.0], dtype=np.float32), k=1)

    def test_invalid_dimension_raises(self) -> None:
        with pytest.raises(ValueError):
            FAISSBenchmarkIndex(dimension=0)

    def test_invalid_index_type_raises(self) -> None:
        with pytest.raises(ValueError):
            FAISSBenchmarkIndex(dimension=3, index_type="not_a_real_type")

    def test_hnsw_index_type(self) -> None:
        index = FAISSBenchmarkIndex(dimension=3, index_type="hnsw")
        index.add([_chunk("c1", [1.0, 0.0, 0.0], "one"), _chunk("c2", [0.0, 1.0, 0.0], "two")])
        results = index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1)
        assert results[0].chunk_id == "c1"

    def test_invalid_k_raises(self) -> None:
        index = FAISSBenchmarkIndex(dimension=3)
        index.add([_chunk("c1", [1.0, 0.0, 0.0], "text")])
        with pytest.raises(ValueError):
            index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=0)
