"""
Module 7 - Embedding Generation: data schema definitions.

Defines EmbeddedChunk (a Chunk paired with its dense embedding vector,
ready for Module 8's vector store) and ContrastivePair (one training
example for contrastive embedding fine-tuning, per Section 8.3/8.7).

Reference: Capstone Project Guide, Section 3D ("Embeddings"), Section
8.3 ("Transfer Learning Applied to Embeddings"), Section 8.7
("Contrastive Learning"), Phase 4 Module 7 contract, Phase 5 file
structure (embeddings/schema.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np

from chunking.schema import Chunk
from preprocessing.schema import DocumentType

logger = logging.getLogger(__name__)


class SchemaValidationError(Exception):
    """Raised when an EmbeddedChunk or ContrastivePair is constructed with invalid values."""


@dataclass(frozen=True, slots=True, eq=False)
class EmbeddedChunk:
    """
    Immutable representation of one Chunk paired with its dense
    embedding vector, produced by EmbeddingModel and consumed by
    Module 8 (Vector Database).

    Note: eq=False is deliberate. The default dataclass-generated
    __eq__ would compare `self.vector == other.vector`, which returns
    an element-wise boolean array rather than a single bool and raises
    ValueError wherever Python expects a plain boolean (e.g. `==` in
    an assert, or membership in a set) -- exactly the numpy pitfall
    this avoids. Compare vectors explicitly with np.array_equal() or
    np.allclose() instead.

    Attributes:
        chunk_id: Inherited unchanged from the source Chunk.
        vector: The dense embedding, a 1-D float array.
        text: The chunk's text (kept for reference/display alongside
            retrieval results; Module 8 also stores this as ChromaDB's
            "document" field).
        source_document_id: Inherited from the source Chunk.
        document_type: Inherited from the source Chunk.
        department: Inherited from the source Chunk, if any.
        effective_date: Inherited from the source Chunk, if any.
        source_path: Inherited from the source Chunk.
        extra_metadata: Inherited from the source Chunk.

    Raises:
        SchemaValidationError: If chunk_id is empty, vector is not a
            1-D array, vector is empty, or vector contains NaN/Inf
            values (which would silently corrupt downstream cosine
            similarity search in Module 8).
    """

    chunk_id: str
    vector: np.ndarray
    text: str
    source_document_id: str
    document_type: DocumentType = DocumentType.UNKNOWN
    department: Optional[str] = None
    effective_date: Optional[date] = None
    source_path: str = ""
    extra_metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise SchemaValidationError("EmbeddedChunk was constructed with an empty chunk_id.")
        if self.vector.ndim != 1:
            raise SchemaValidationError(
                f"EmbeddedChunk {self.chunk_id!r} vector must be 1-D, got shape "
                f"{self.vector.shape}."
            )
        if self.vector.size == 0:
            raise SchemaValidationError(f"EmbeddedChunk {self.chunk_id!r} vector is empty.")
        if not np.all(np.isfinite(self.vector)):
            raise SchemaValidationError(
                f"EmbeddedChunk {self.chunk_id!r} vector contains NaN or Inf values."
            )

    @property
    def dimension(self) -> int:
        """Returns the embedding vector's dimensionality."""
        return int(self.vector.shape[0])

    @classmethod
    def from_chunk(cls, chunk: Chunk, vector: np.ndarray) -> "EmbeddedChunk":
        """
        Construct an EmbeddedChunk by inheriting metadata from a Chunk
        and attaching its computed embedding vector.

        Args:
            chunk: The source Chunk (from Module 3).
            vector: The chunk's embedding vector, as computed by
                EmbeddingModel.embed().

        Returns:
            A populated, immutable EmbeddedChunk.
        """
        return cls(
            chunk_id=chunk.chunk_id,
            vector=vector,
            text=chunk.text,
            source_document_id=chunk.source_document_id,
            document_type=chunk.document_type,
            department=chunk.department,
            effective_date=chunk.effective_date,
            source_path=chunk.source_path,
            extra_metadata=dict(chunk.extra_metadata),
        )

    def to_metadata_dict(self) -> dict[str, str]:
        """
        Returns a flat, string-valued metadata dict (excluding the
        vector itself and chunk_id/text, which ChromaDB's API takes as
        separate arguments) suitable for Module 8's metadata filtering.
        """
        metadata = {
            "source_document_id": self.source_document_id,
            "document_type": self.document_type.value,
            "source_path": self.source_path,
        }
        if self.department is not None:
            metadata["department"] = self.department
        if self.effective_date is not None:
            metadata["effective_date"] = self.effective_date.isoformat()
        metadata.update(self.extra_metadata)
        return metadata


@dataclass(frozen=True, slots=True)
class ContrastivePair:
    """
    One training example for contrastive embedding fine-tuning
    (Section 8.3/8.7): a (query, positive_passage) pair whose
    embeddings should be pulled close together. Multiple Negatives
    Ranking Loss (used by fine_tune_embeddings()) treats every other
    passage in the same training batch as an implicit negative for
    this query, so no explicit negative examples need to be
    hand-curated.

    Attributes:
        query: A question or query-like text.
        positive_passage: A passage that correctly answers/is relevant
            to query.

    Raises:
        SchemaValidationError: If either field is empty or
            whitespace-only.
    """

    query: str
    positive_passage: str

    def __post_init__(self) -> None:
        if not self.query or not self.query.strip():
            raise SchemaValidationError("ContrastivePair.query must not be empty.")
        if not self.positive_passage or not self.positive_passage.strip():
            raise SchemaValidationError("ContrastivePair.positive_passage must not be empty.")
