"""
Unit tests for Module 3 - Chunking.

Covers chunking/schema.py and chunking/strategies.py: the three free
chunking functions, the ChunkingStrategy hierarchy, and the
create_chunker() factory.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from chunking.schema import Chunk, SchemaValidationError
from chunking.strategies import (
    ChunkingError,
    FixedSizeChunker,
    SentenceAwareChunker,
    StructureAwareChunker,
    chunk_fixed_size,
    chunk_sentence_aware,
    chunk_structure_aware,
    create_chunker,
)
from ingestion.schema import DocumentFormat, SourceSystem
from preprocessing.schema import CleanDocument, DocumentType


def _make_clean_document(text: str, document_id: str = "pdf_abc123") -> CleanDocument:
    return CleanDocument(
        document_id=document_id,
        text=text,
        source_path="/data/raw/leave_policy.pdf",
        source_system=SourceSystem.LOCAL_FILESYSTEM,
        document_format=DocumentFormat.PDF,
        collected_at=datetime.now(timezone.utc),
        checksum="checksum-1",
        document_type=DocumentType.POLICY,
        department="HR",
    )


_STRUCTURED_TEXT = (
    "Section 4.2: Casual Leave\n"
    "Employees get 12 days of casual leave per year. It carries forward "
    "up to 5 days into the next year.\n"
    "Section 4.3: Sick Leave\n"
    "Employees get 8 days of sick leave per year. A doctor's note is "
    "required beyond 2 consecutive days."
)


# --------------------------------------------------------------------------
# schema.py
# --------------------------------------------------------------------------


class TestChunk:
    def test_from_clean_document_inherits_metadata(self) -> None:
        document = _make_clean_document("Some cleaned text.")
        chunk = Chunk.from_clean_document(document, "Some cleaned text.", position=0)

        assert chunk.source_document_id == document.document_id
        assert chunk.document_type == DocumentType.POLICY
        assert chunk.department == "HR"
        assert chunk.chunk_id == f"{document.document_id}_chunk_0000"

    def test_chunk_id_reflects_position(self) -> None:
        document = _make_clean_document("text")
        chunk = Chunk.from_clean_document(document, "text", position=7)
        assert chunk.chunk_id == f"{document.document_id}_chunk_0007"

    def test_word_count(self) -> None:
        document = _make_clean_document("text")
        chunk = Chunk.from_clean_document(document, "one two three four", position=0)
        assert chunk.word_count == 4

    def test_empty_chunk_id_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            Chunk(chunk_id="", text="text", source_document_id="doc1", position=0)

    def test_empty_source_document_id_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            Chunk(chunk_id="c1", text="text", source_document_id="", position=0)

    def test_negative_position_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            Chunk(chunk_id="c1", text="text", source_document_id="doc1", position=-1)

    def test_empty_text_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            Chunk(chunk_id="c1", text="   ", source_document_id="doc1", position=0)

    def test_is_immutable(self) -> None:
        chunk = Chunk(chunk_id="c1", text="text", source_document_id="doc1", position=0)
        with pytest.raises(AttributeError):
            chunk.text = "mutated"  # type: ignore[misc]

    def test_to_metadata_dict(self) -> None:
        document = _make_clean_document("text")
        chunk = Chunk.from_clean_document(document, "text", position=0)
        metadata = chunk.to_metadata_dict()
        assert metadata["chunk_id"] == chunk.chunk_id
        assert metadata["department"] == "HR"
        assert metadata["document_type"] == "policy"


# --------------------------------------------------------------------------
# strategies.py - chunk_fixed_size
# --------------------------------------------------------------------------


class TestChunkFixedSize:
    def test_short_text_returns_single_chunk(self) -> None:
        result = chunk_fixed_size("one two three", size=10, overlap=2)
        assert result == ["one two three"]

    def test_splits_into_overlapping_windows(self) -> None:
        text = " ".join(f"word{i}" for i in range(1, 21))  # 20 words
        result = chunk_fixed_size(text, size=10, overlap=3)
        # step = size - overlap = 7; 20 words -> windows at 0, 7, 14 -> 3 chunks
        assert len(result) == 3
        for earlier, later in zip(result, result[1:]):
            # Last 3 words of one window == first 3 words of the next.
            assert earlier.split()[-3:] == later.split()[:3]

    def test_empty_text_returns_empty_list(self) -> None:
        assert chunk_fixed_size("") == []
        assert chunk_fixed_size("   ") == []

    def test_invalid_size_raises(self) -> None:
        with pytest.raises(ValueError):
            chunk_fixed_size("text", size=0)

    def test_overlap_equal_to_size_raises(self) -> None:
        with pytest.raises(ValueError):
            chunk_fixed_size("text", size=10, overlap=10)

    def test_negative_overlap_raises(self) -> None:
        with pytest.raises(ValueError):
            chunk_fixed_size("text", size=10, overlap=-1)

    def test_all_words_covered(self) -> None:
        text = " ".join(f"word{i}" for i in range(1, 51))  # 50 words
        result = chunk_fixed_size(text, size=10, overlap=2)
        rejoined_words: set[str] = set()
        for piece in result:
            rejoined_words.update(piece.split())
        original_words = set(text.split())
        assert rejoined_words == original_words


# --------------------------------------------------------------------------
# strategies.py - chunk_sentence_aware
# --------------------------------------------------------------------------


class TestChunkSentenceAware:
    def test_groups_sentences_up_to_target_size(self) -> None:
        text = "Short sentence one. Short sentence two. Short sentence three."
        result = chunk_sentence_aware(text, target_size=6)
        # "Short sentence one." = 3 words, so 2 sentences (6 words) fit,
        # third sentence starts a new chunk.
        assert len(result) == 2

    def test_never_splits_mid_sentence(self) -> None:
        text = "This is a complete sentence that should never be cut in half."
        result = chunk_sentence_aware(text, target_size=3)
        assert len(result) == 1
        assert result[0] == text

    def test_empty_text_returns_empty_list(self) -> None:
        assert chunk_sentence_aware("") == []

    def test_invalid_target_size_raises(self) -> None:
        with pytest.raises(ValueError):
            chunk_sentence_aware("text", target_size=0)

    def test_single_sentence_no_period(self) -> None:
        result = chunk_sentence_aware("No terminal punctuation here", target_size=10)
        assert result == ["No terminal punctuation here"]


# --------------------------------------------------------------------------
# strategies.py - chunk_structure_aware
# --------------------------------------------------------------------------


class TestChunkStructureAware:
    def test_splits_at_headings(self) -> None:
        result = chunk_structure_aware(_STRUCTURED_TEXT, target_size=300)
        assert len(result) == 2
        assert result[0].startswith("Section 4.2: Casual Leave")
        assert result[1].startswith("Section 4.3: Sick Leave")

    def test_keeps_clause_with_its_qualifier(self) -> None:
        result = chunk_structure_aware(_STRUCTURED_TEXT, target_size=300)
        assert "12 days of casual leave" in result[0]
        assert "carries forward up to 5 days" in result[0]

    def test_oversized_section_falls_back_to_sentence_aware(self) -> None:
        result = chunk_structure_aware(_STRUCTURED_TEXT, target_size=10)
        # Each section now exceeds 10 words, so it should be split
        # further, producing more pieces than there are headings.
        assert len(result) > 2

    def test_no_headings_returns_whole_text_as_single_chunk(self) -> None:
        # No heading-like lines at all -> no section boundaries detected,
        # so the whole text is returned as one chunk (per docstring).
        result = chunk_structure_aware("Just a flat paragraph with no headings.", target_size=300)
        assert result == ["Just a flat paragraph with no headings."]

    def test_empty_text_returns_empty_list(self) -> None:
        assert chunk_structure_aware("") == []

    def test_invalid_target_size_raises(self) -> None:
        with pytest.raises(ValueError):
            chunk_structure_aware("text", target_size=0)

    def test_markdown_headings_detected(self) -> None:
        text = "## Introduction\nSome intro text.\n## Details\nSome detail text."
        result = chunk_structure_aware(text, target_size=300)
        assert len(result) == 2
        assert result[0].startswith("## Introduction")


# --------------------------------------------------------------------------
# strategies.py - ChunkingStrategy classes
# --------------------------------------------------------------------------


class TestFixedSizeChunker:
    def test_chunk_produces_chunk_objects_with_correct_positions(self) -> None:
        document = _make_clean_document(" ".join(f"word{i}" for i in range(1, 31)))
        chunker = FixedSizeChunker(size=10, overlap=2)
        chunks = chunker.chunk(document)

        assert all(isinstance(c, Chunk) for c in chunks)
        assert [c.position for c in chunks] == list(range(len(chunks)))
        assert all(c.source_document_id == document.document_id for c in chunks)

    def test_invalid_constructor_args_raise(self) -> None:
        with pytest.raises(ValueError):
            FixedSizeChunker(size=0)
        with pytest.raises(ValueError):
            FixedSizeChunker(size=10, overlap=10)


class TestSentenceAwareChunker:
    def test_chunk_respects_sentence_boundaries(self) -> None:
        document = _make_clean_document(
            "First sentence here. Second sentence here. Third sentence here."
        )
        chunker = SentenceAwareChunker(target_size=5)
        chunks = chunker.chunk(document)
        assert len(chunks) >= 1
        for c in chunks:
            assert c.text.strip().endswith(".")


class TestStructureAwareChunker:
    def test_chunk_end_to_end(self) -> None:
        document = _make_clean_document(_STRUCTURED_TEXT)
        chunker = StructureAwareChunker(target_size=300)
        chunks = chunker.chunk(document)

        assert len(chunks) == 2
        assert chunks[0].position == 0
        assert chunks[1].position == 1
        assert chunks[0].department == "HR"
        assert chunks[0].document_type == DocumentType.POLICY

    def test_chunk_none_document_raises(self) -> None:
        chunker = StructureAwareChunker()
        with pytest.raises(ChunkingError):
            chunker.chunk(None)  # type: ignore[arg-type]


class TestCreateChunker:
    def test_creates_structure_aware(self) -> None:
        chunker = create_chunker("structure_aware", target_size=200)
        assert isinstance(chunker, StructureAwareChunker)

    def test_creates_fixed_size(self) -> None:
        chunker = create_chunker("fixed_size", size=100, overlap=10)
        assert isinstance(chunker, FixedSizeChunker)

    def test_creates_sentence_aware(self) -> None:
        chunker = create_chunker("sentence_aware", target_size=150)
        assert isinstance(chunker, SentenceAwareChunker)

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError):
            create_chunker("nonexistent_strategy")
