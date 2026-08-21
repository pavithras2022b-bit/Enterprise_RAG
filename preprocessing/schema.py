"""
Module 2 - Cleaning: data schema definitions.

Defines the CleanDocument dataclass, which carries a document's
cleaned text plus everything inherited from Module 1's RawDocument,
plus the new metadata tags this module attaches (document_type,
department, effective_date).

Reference: Capstone Project Guide, Section 3B ("Data Cleaning"),
Phase 4 Module 2 contract, Phase 5 file structure
(preprocessing/schema.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Optional

from ingestion.schema import DocumentFormat, RawDocument, SourceSystem

logger = logging.getLogger(__name__)


class DocumentType(str, Enum):
    """
    Enumerates the enterprise document categories MetadataTagger can
    infer. UNKNOWN is used when no category keyword is matched, per
    Section 3A's metadata-tagging discussion.
    """

    POLICY = "policy"
    HANDBOOK = "handbook"
    MANUAL = "manual"
    FAQ = "faq"
    UNKNOWN = "unknown"


class SchemaValidationError(Exception):
    """Raised when a CleanDocument is constructed with invalid field values."""


@dataclass(frozen=True, slots=True)
class CleanDocument:
    """
    Immutable representation of a document after Module 2's cleaning
    and metadata-tagging steps, ready to be handed to Module 3
    (Chunking).

    Attributes:
        document_id: Inherited unchanged from the source RawDocument,
            so a CleanDocument can always be traced back to its raw
            source (important for the audit trail the project guide
            emphasizes throughout Section 3A/3B).
        text: The cleaned, normalized text -- boilerplate stripped,
            encoding fixed, near-duplicate paragraphs removed.
        source_path: Inherited from RawDocument.
        source_system: Inherited from RawDocument.
        document_format: Inherited from RawDocument.
        collected_at: Inherited from RawDocument.
        checksum: Inherited from RawDocument (the checksum of the
            *raw* file, not the cleaned text -- this remains the
            stable identity anchor for dedupe/traceability).
        document_version: Inherited from RawDocument, if present.
        document_type: Inferred category (policy/handbook/manual/faq),
            defaults to DocumentType.UNKNOWN if no signal was found.
        department: Inferred owning department (e.g. "Finance", "HR"),
            or None if no department signal was found.
        effective_date: Inferred policy effective date, or None if not
            found/parseable in the text.
        extra_metadata: Inherited from RawDocument's extra_metadata,
            merged with any tagging-specific additions.

    Raises:
        SchemaValidationError: If constructed with empty text or an
            empty document_id, mirroring RawDocument's validation.
    """

    document_id: str
    text: str
    source_path: str
    source_system: SourceSystem
    document_format: DocumentFormat
    collected_at: datetime
    checksum: str
    document_version: Optional[str] = None
    document_type: DocumentType = DocumentType.UNKNOWN
    department: Optional[str] = None
    effective_date: Optional[date] = None
    extra_metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.document_id:
            raise SchemaValidationError(
                "CleanDocument was constructed with an empty document_id."
            )
        if not self.text or not self.text.strip():
            logger.warning(
                "CleanDocument %s has empty or whitespace-only text after "
                "cleaning -- check whether cleaning over-stripped the "
                "source content.",
                self.document_id,
            )

    @property
    def char_count(self) -> int:
        """Returns the character length of the cleaned text."""
        return len(self.text)

    @classmethod
    def from_raw_document(
        cls,
        raw: RawDocument,
        cleaned_text: str,
        *,
        document_type: DocumentType = DocumentType.UNKNOWN,
        department: Optional[str] = None,
        effective_date: Optional[date] = None,
    ) -> "CleanDocument":
        """
        Construct a CleanDocument by inheriting provenance fields from
        a RawDocument and attaching newly-cleaned text plus tags.

        Args:
            raw: The source RawDocument (from Module 1).
            cleaned_text: The text after DocumentCleaner.clean() has
                been applied.
            document_type: Inferred document category.
            department: Inferred owning department, if any.
            effective_date: Inferred policy effective date, if any.

        Returns:
            A populated, immutable CleanDocument.
        """
        return cls(
            document_id=raw.document_id,
            text=cleaned_text,
            source_path=raw.source_path,
            source_system=raw.source_system,
            document_format=raw.document_format,
            collected_at=raw.collected_at,
            checksum=raw.checksum,
            document_version=raw.document_version,
            document_type=document_type,
            department=department,
            effective_date=effective_date,
            extra_metadata=dict(raw.extra_metadata),
        )

    def to_metadata_dict(self) -> dict[str, str]:
        """
        Returns a flat, string-valued metadata dict suitable for
        attaching to downstream artifacts (e.g. ChromaDB metadata
        filters in Module 8, which need primitive values).
        """
        metadata = {
            "document_id": self.document_id,
            "source_path": self.source_path,
            "source_system": self.source_system.value,
            "document_format": self.document_format.value,
            "collected_at": self.collected_at.isoformat(),
            "checksum": self.checksum,
            "document_type": self.document_type.value,
        }
        if self.document_version is not None:
            metadata["document_version"] = self.document_version
        if self.department is not None:
            metadata["department"] = self.department
        if self.effective_date is not None:
            metadata["effective_date"] = self.effective_date.isoformat()
        metadata.update(self.extra_metadata)
        return metadata
