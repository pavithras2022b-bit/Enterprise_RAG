"""
Unit tests for Module 7 - Embedding Generation.

Covers embeddings/schema.py, model.py, and train.py. Model/training
tests use the session-scoped tiny_sentence_transformer_dir fixture
(see conftest.py) to exercise the real sentence-transformers/torch
code paths fully offline.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pytest

from chunking.schema import Chunk
from embeddings.model import EmbeddingModel, EmbeddingModelError, embed, embed_batch
from embeddings.schema import ContrastivePair, EmbeddedChunk, SchemaValidationError
from embeddings.train import EmbeddingTrainingError, fine_tune_embeddings
from ingestion.schema import DocumentFormat, SourceSystem
from preprocessing.schema import CleanDocument, DocumentType

logging.disable(logging.CRITICAL)


def _make_chunk(text: str = "Casual leave is 12 days per year.", position: int = 0) -> Chunk:
    document = CleanDocument(
        document_id="pdf_abc123",
        text=text,
        source_path="/data/raw/leave_policy.pdf",
        source_system=SourceSystem.LOCAL_FILESYSTEM,
        document_format=DocumentFormat.PDF,
        collected_at=datetime.now(timezone.utc),
        checksum="checksum-1",
        document_type=DocumentType.POLICY,
        department="HR",
    )
    return Chunk.from_clean_document(document, text, position=position)


# --------------------------------------------------------------------------
# schema.py - EmbeddedChunk
# --------------------------------------------------------------------------


class TestEmbeddedChunk:
    def test_from_chunk_inherits_metadata(self) -> None:
        chunk = _make_chunk()
        vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        embedded = EmbeddedChunk.from_chunk(chunk, vector)

        assert embedded.chunk_id == chunk.chunk_id
        assert embedded.source_document_id == chunk.source_document_id
        assert embedded.department == "HR"
        assert embedded.document_type == DocumentType.POLICY
        assert np.array_equal(embedded.vector, vector)

    def test_dimension_property(self) -> None:
        chunk = _make_chunk()
        vector = np.zeros(384, dtype=np.float32)
        embedded = EmbeddedChunk.from_chunk(chunk, vector)
        assert embedded.dimension == 384

    def test_empty_chunk_id_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            EmbeddedChunk(
                chunk_id="",
                vector=np.array([1.0, 2.0]),
                text="text",
                source_document_id="doc1",
            )

    def test_non_1d_vector_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            EmbeddedChunk(
                chunk_id="c1",
                vector=np.zeros((2, 2)),
                text="text",
                source_document_id="doc1",
            )

    def test_empty_vector_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            EmbeddedChunk(
                chunk_id="c1",
                vector=np.array([]),
                text="text",
                source_document_id="doc1",
            )

    def test_nan_vector_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            EmbeddedChunk(
                chunk_id="c1",
                vector=np.array([1.0, np.nan, 3.0]),
                text="text",
                source_document_id="doc1",
            )

    def test_inf_vector_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            EmbeddedChunk(
                chunk_id="c1",
                vector=np.array([1.0, np.inf]),
                text="text",
                source_document_id="doc1",
            )

    def test_identity_equality_not_value_equality(self) -> None:
        # eq=False is deliberate (see schema.py docstring) -- two
        # EmbeddedChunks with identical field values are NOT equal
        # unless they're the same object, avoiding numpy's ambiguous
        # array-equality pitfall.
        chunk = _make_chunk()
        vector = np.array([1.0, 2.0])
        a = EmbeddedChunk.from_chunk(chunk, vector)
        b = EmbeddedChunk.from_chunk(chunk, vector)
        assert a != b
        assert a == a

    def test_to_metadata_dict(self) -> None:
        chunk = _make_chunk()
        embedded = EmbeddedChunk.from_chunk(chunk, np.array([1.0, 2.0]))
        metadata = embedded.to_metadata_dict()
        assert metadata["department"] == "HR"
        assert "vector" not in metadata
        assert "chunk_id" not in metadata


class TestContrastivePair:
    def test_valid_construction(self) -> None:
        pair = ContrastivePair(query="How many casual leave days?", positive_passage="12 days per year.")
        assert pair.query.startswith("How many")

    def test_empty_query_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            ContrastivePair(query="   ", positive_passage="text")

    def test_empty_passage_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            ContrastivePair(query="query text", positive_passage="")


# --------------------------------------------------------------------------
# model.py
# --------------------------------------------------------------------------


class TestEmbeddingModel:
    def test_loads_from_local_checkpoint(self, tiny_sentence_transformer_dir: str) -> None:
        model = EmbeddingModel(tiny_sentence_transformer_dir)
        assert model.dimension == 16

    def test_invalid_checkpoint_path_raises(self, tmp_path) -> None:
        with pytest.raises(EmbeddingModelError):
            EmbeddingModel(str(tmp_path / "does_not_exist"))

    def test_embed_returns_correct_shape(self, tiny_sentence_transformer_dir: str) -> None:
        model = EmbeddingModel(tiny_sentence_transformer_dir)
        vector = model.embed("Asha Rao belongs to the Finance department.")
        assert vector.shape == (16,)
        assert vector.dtype == np.float32

    def test_embed_is_normalized_by_default(self, tiny_sentence_transformer_dir: str) -> None:
        model = EmbeddingModel(tiny_sentence_transformer_dir)
        vector = model.embed("Finance department leave policy.")
        norm = float(np.linalg.norm(vector))
        assert norm == pytest.approx(1.0, abs=1e-4)

    def test_embed_without_normalization(self, tiny_sentence_transformer_dir: str) -> None:
        model = EmbeddingModel(tiny_sentence_transformer_dir, normalize_embeddings=False)
        vector = model.embed("Finance department leave policy.")
        assert vector.shape == (16,)

    def test_embed_empty_text_raises(self, tiny_sentence_transformer_dir: str) -> None:
        model = EmbeddingModel(tiny_sentence_transformer_dir)
        with pytest.raises(EmbeddingModelError):
            model.embed("")
        with pytest.raises(EmbeddingModelError):
            model.embed("   ")

    def test_embed_batch_returns_correct_shape(self, tiny_sentence_transformer_dir: str) -> None:
        model = EmbeddingModel(tiny_sentence_transformer_dir)
        vectors = model.embed_batch(["Finance department.", "Casual leave policy.", "VPN access."])
        assert vectors.shape == (3, 16)

    def test_embed_batch_matches_individual_embed(self, tiny_sentence_transformer_dir: str) -> None:
        model = EmbeddingModel(tiny_sentence_transformer_dir)
        text = "Finance department leave policy."
        individual = model.embed(text)
        batch = model.embed_batch([text])
        assert np.allclose(individual, batch[0], atol=1e-5)

    def test_embed_batch_empty_list_returns_empty_array(self, tiny_sentence_transformer_dir: str) -> None:
        model = EmbeddingModel(tiny_sentence_transformer_dir)
        result = model.embed_batch([])
        assert result.shape == (0, 16)

    def test_embed_batch_with_empty_text_raises(self, tiny_sentence_transformer_dir: str) -> None:
        model = EmbeddingModel(tiny_sentence_transformer_dir)
        with pytest.raises(EmbeddingModelError):
            model.embed_batch(["valid text", ""])

    def test_embed_chunk(self, tiny_sentence_transformer_dir: str) -> None:
        model = EmbeddingModel(tiny_sentence_transformer_dir)
        chunk = _make_chunk()
        embedded = model.embed_chunk(chunk)
        assert isinstance(embedded, EmbeddedChunk)
        assert embedded.chunk_id == chunk.chunk_id
        assert embedded.dimension == 16

    def test_embed_chunks_batch(self, tiny_sentence_transformer_dir: str) -> None:
        model = EmbeddingModel(tiny_sentence_transformer_dir)
        chunks = [_make_chunk(text=f"Chunk number {i}.", position=i) for i in range(3)]
        embedded_chunks = model.embed_chunks(chunks)
        assert len(embedded_chunks) == 3
        assert [ec.chunk_id for ec in embedded_chunks] == [c.chunk_id for c in chunks]

    def test_embed_chunks_empty_list_returns_empty_list(self, tiny_sentence_transformer_dir: str) -> None:
        model = EmbeddingModel(tiny_sentence_transformer_dir)
        assert model.embed_chunks([]) == []

    def test_module_level_embed_function(self, tiny_sentence_transformer_dir: str) -> None:
        model = EmbeddingModel(tiny_sentence_transformer_dir)
        vector = embed(model, "Finance department.")
        assert vector.shape == (16,)

    def test_module_level_embed_batch_function(self, tiny_sentence_transformer_dir: str) -> None:
        model = EmbeddingModel(tiny_sentence_transformer_dir)
        vectors = embed_batch(model, ["Finance department.", "VPN access."])
        assert vectors.shape == (2, 16)


# --------------------------------------------------------------------------
# train.py
# --------------------------------------------------------------------------


_TRAINING_PAIRS = [
    ContrastivePair(
        query="How many casual leave days do I get?",
        positive_passage="Casual leave is 12 days per year.",
    ),
    ContrastivePair(
        query="What department does Asha Rao belong to?",
        positive_passage="Asha Rao belongs to the Finance department.",
    ),
    ContrastivePair(
        query="How do I request VPN access?",
        positive_passage="VPN access requires manager approval.",
    ),
]


class TestFineTuneEmbeddings:
    def test_trains_and_returns_usable_model(self, tiny_sentence_transformer_dir: str, tmp_path) -> None:
        output_dir = str(tmp_path / "trained_embeddings")
        model = fine_tune_embeddings(
            _TRAINING_PAIRS,
            base_model=tiny_sentence_transformer_dir,
            output_dir=output_dir,
            num_epochs=1,
            batch_size=2,
        )
        assert isinstance(model, EmbeddingModel)
        vector = model.embed("Casual leave is 12 days per year.")
        assert vector.shape == (16,)

    def test_saves_checkpoint_to_output_dir(self, tiny_sentence_transformer_dir: str, tmp_path) -> None:
        output_dir = tmp_path / "trained_embeddings_2"
        fine_tune_embeddings(
            _TRAINING_PAIRS,
            base_model=tiny_sentence_transformer_dir,
            output_dir=str(output_dir),
            num_epochs=1,
            batch_size=2,
        )
        assert (output_dir / "config_sentence_transformers.json").exists()

    def test_empty_pairs_raises(self, tiny_sentence_transformer_dir: str, tmp_path) -> None:
        with pytest.raises(EmbeddingTrainingError):
            fine_tune_embeddings([], base_model=tiny_sentence_transformer_dir, output_dir=str(tmp_path / "out"))

    def test_invalid_num_epochs_raises(self, tiny_sentence_transformer_dir: str, tmp_path) -> None:
        with pytest.raises(ValueError):
            fine_tune_embeddings(
                _TRAINING_PAIRS,
                base_model=tiny_sentence_transformer_dir,
                output_dir=str(tmp_path / "out"),
                num_epochs=0,
            )

    def test_invalid_batch_size_raises(self, tiny_sentence_transformer_dir: str, tmp_path) -> None:
        with pytest.raises(ValueError):
            fine_tune_embeddings(
                _TRAINING_PAIRS,
                base_model=tiny_sentence_transformer_dir,
                output_dir=str(tmp_path / "out"),
                batch_size=0,
            )

    def test_invalid_warmup_ratio_raises(self, tiny_sentence_transformer_dir: str, tmp_path) -> None:
        with pytest.raises(ValueError):
            fine_tune_embeddings(
                _TRAINING_PAIRS,
                base_model=tiny_sentence_transformer_dir,
                output_dir=str(tmp_path / "out"),
                warmup_ratio=1.5,
            )

    def test_invalid_base_model_raises(self, tmp_path) -> None:
        with pytest.raises(EmbeddingTrainingError):
            fine_tune_embeddings(
                _TRAINING_PAIRS,
                base_model=str(tmp_path / "does_not_exist"),
                output_dir=str(tmp_path / "out"),
            )
