"""
Module 1 - Document Ingestion: data schema definitions.

Defines the RawDocument dataclass that represents a single ingested
document with its raw text content and provenance metadata, before
any cleaning or normalization (Module 2) has been applied.

Reference: Capstone Project Guide, Section 3A ("Data Collection"),
Phase 4 Module 1 contract, Phase 5 file structure (ingestion/schema.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SourceSystem(str, Enum):
    """Enumerates the systems a raw document can originate from."""

    LOCAL_FILESYSTEM = "local_filesystem"
    CONFLUENCE = "confluence"
    SHAREPOINT = "sharepoint"
    UNKNOWN = "unknown"


class DocumentFormat(str, Enum):
    """Enumerates the file formats supported by Module 1's loaders."""

    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    CSV = "csv"
    MARKDOWN = "markdown"
    TXT = "txt"


class SchemaValidationError(Exception):
    """Raised when a RawDocument is constructed with invalid field values."""


@dataclass(frozen=True, slots=True)
class RawDocument:
    """
    Immutable representation of a single raw document pulled into the
    ingestion pipeline, prior to any cleaning (Module 2).

    Attributes:
        document_id: Deterministic identifier derived from the content
            checksum, stable across re-ingestion runs of the same file.
        text: The raw extracted text content of the document.
        source_path: Filesystem path (or connector URI) the document
            was pulled from.
        source_system: The system of origin (local filesystem,
            Confluence, SharePoint, etc.).
        document_format: The original file format prior to extraction.
        collected_at: UTC timestamp of when this document was
            extracted by the pipeline.
        checksum: SHA-256 hex digest of the raw file bytes. Used by
            Module 1's dedupe step (dedupe.py) and as the seed for
            document_id.
        document_version: Optional version label/string. When the
            source system provides no explicit versioning, this
            defaults to None and downstream consumers should treat
            the document as unversioned.
        extra_metadata: Free-form dict for source-system-specific
            metadata that doesn't warrant a first-class field (e.g. a
            Confluence page ID or SharePoint drive item ID).

    Raises:
        SchemaValidationError: If constructed with an empty checksum
            or a non-positive document_id, since both are required
            for downstream dedupe and traceability to function.
    """

    document_id: str
    text: str
    source_path: str
    source_system: SourceSystem
    document_format: DocumentFormat
    collected_at: datetime
    checksum: str
    document_version: Optional[str] = None
    extra_metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.checksum:
            raise SchemaValidationError(
                f"RawDocument {self.document_id!r} was constructed without a "
                f"checksum; checksum is required for dedupe.py to function."
            )
        if not self.document_id:
            raise SchemaValidationError(
                "RawDocument was constructed with an empty document_id."
            )
        if not self.text or not self.text.strip():
            logger.warning(
                "RawDocument %s created with empty or whitespace-only text "
                "(source_path=%s)",
                self.document_id,
                self.source_path,
            )

    @property
    def char_count(self) -> int:
        """Returns the character length of the raw extracted text."""
        return len(self.text)

    @property
    def file_name(self) -> str:
        """Returns just the filename component of source_path."""
        return Path(self.source_path).name

    def to_metadata_dict(self) -> dict[str, str]:
        """
        Returns a flat, string-valued metadata dict suitable for
        attaching to downstream artifacts (e.g. ChromaDB metadata
        filters in Module 8), which require primitive values.
        """
        metadata = {
            "document_id": self.document_id,
            "source_path": self.source_path,
            "source_system": self.source_system.value,
            "document_format": self.document_format.value,
            "collected_at": self.collected_at.isoformat(),
            "checksum": self.checksum,
        }
        if self.document_version is not None:
            metadata["document_version"] = self.document_version
        metadata.update(self.extra_metadata)
        return metadata
