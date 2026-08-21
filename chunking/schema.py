"""
Module 3 - Chunking: data schema definitions.

Defines the Chunk dataclass, produced from a CleanDocument by one of
the strategies in chunking/strategies.py, and consumed by Module 4
(NER), Module 7 (Embedding Generation), and beyond.

Reference: Capstone Project Guide, Section 3C ("Chunking"), Phase 4
Module 3 contract, Phase 5 file structure (chunking/schema.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from preprocessing.schema import CleanDocument, DocumentType

logger = logging.getLogger(__name__)


class SchemaValidationError(Exception):
    """Raised when a Chunk is constructed with invalid field values."""


@dataclass(frozen=True, slots=True)
class Chunk:
    """
    Immutable representation of a single retrieval-sized passage,
    produced by splitting a CleanDocument's text.

    Attributes:
        chunk_id: Deterministic identifier, derived from
            source_document_id and position, stable across repeated
            chunking runs with the same strategy/parameters.
        text: The chunk's text content.
        source_document_id: The CleanDocument.document_id this chunk
            was split from (== the original RawDocument.document_id,
            preserving the full provenance chain from Module 1).
        position: Zero-based index of this chunk within its source
            document, in document order.
        document_type: Inherited from the source CleanDocument.
        department: Inherited from the source CleanDocument, if any.
        effective_date: Inherited from the source CleanDocument, if
            any.
        source_path: Inherited from the source CleanDocument.
        checksum: Inherited from the source CleanDocument (the raw
            file's checksum, not a checksum of the chunk itself).
        extra_metadata: Inherited from the source CleanDocument's
            extra_metadata.

    Raises:
        SchemaValidationError: If constructed with empty text, an
            empty chunk_id/source_document_id, or a negative position.
    """

    chunk_id: str
    text: str
    source_document_id: str
    position: int
    document_type: DocumentType = DocumentType.UNKNOWN
    department: Optional[str] = None
    effective_date: Optional[date] = None
    source_path: str = ""
    checksum: str = ""
    extra_metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise SchemaValidationError("Chunk was constructed with an empty chunk_id.")
        if not self.source_document_id:
            raise SchemaValidationError(
                f"Chunk {self.chunk_id!r} was constructed with an empty "
                f"source_document_id."
            )
        if self.position < 0:
            raise SchemaValidationError(
                f"Chunk {self.chunk_id!r} was constructed with a negative "
                f"position ({self.position})."
            )
        if not self.text or not self.text.strip():
            raise SchemaValidationError(
                f"Chunk {self.chunk_id!r} (source_document_id="
                f"{self.source_document_id!r}, position={self.position}) was "
                f"constructed with empty text -- chunking strategies must "
                f"filter empty pieces before constructing a Chunk."
            )

    @property
    def char_count(self) -> int:
        """Returns the character length of the chunk's text."""
        return len(self.text)

    @property
    def word_count(self) -> int:
        """Returns the whitespace-delimited word count of the chunk's text."""
        return len(self.text.split())

    @classmethod
    def from_clean_document(
        cls, document: CleanDocument, text: str, position: int
    ) -> "Chunk":
        """
        Construct a Chunk by inheriting metadata from a CleanDocument
        and attaching a specific text piece and position.

        Args:
            document: The source CleanDocument (from Module 2).
            text: The chunk's text content (one element of a chunking
                strategy's split_text() output).
            position: Zero-based index of this chunk within the
                document.

        Returns:
            A populated, immutable Chunk.
        """
        chunk_id = f"{document.document_id}_chunk_{position:04d}"
        return cls(
            chunk_id=chunk_id,
            text=text,
            source_document_id=document.document_id,
            position=position,
            document_type=document.document_type,
            department=document.department,
            effective_date=document.effective_date,
            source_path=document.source_path,
            checksum=document.checksum,
            extra_metadata=dict(document.extra_metadata),
        )

    def to_metadata_dict(self) -> dict[str, str]:
        """
        Returns a flat, string-valued metadata dict suitable for
        attaching to downstream artifacts (e.g. ChromaDB metadata
        filters in Module 8, which require primitive values).
        """
        metadata = {
            "chunk_id": self.chunk_id,
            "source_document_id": self.source_document_id,
            "position": str(self.position),
            "document_type": self.document_type.value,
            "source_path": self.source_path,
            "checksum": self.checksum,
        }
        if self.department is not None:
            metadata["department"] = self.department
        if self.effective_date is not None:
            metadata["effective_date"] = self.effective_date.isoformat()
        metadata.update(self.extra_metadata)
        return metadata
