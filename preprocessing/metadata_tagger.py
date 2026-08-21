"""
Module 2 - Cleaning: metadata tagging.

Provides tag_metadata(...) and the MetadataTagger class per the
Phase 4 module contract, inferring document_type, department, and
effective_date from a cleaned document's filename and text content,
and producing the final CleanDocument.

Reference: Capstone Project Guide, Section 3B ("Data Cleaning") and
Section 1.2's department taxonomy (HR & People, IT & Technical,
Compliance & Legal, Project & Org Knowledge), Phase 4 Module 2
contract, Phase 5 file structure (preprocessing/metadata_tagger.py).
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Optional

from dateutil import parser as dateutil_parser

from ingestion.schema import RawDocument
from preprocessing.schema import CleanDocument, DocumentType

logger = logging.getLogger(__name__)

# Only the first N characters of the cleaned text are scanned for
# document-type keywords: type is almost always signaled in a title or
# opening section, and limiting the scan keeps tagging fast even on
# long documents.
_DOCUMENT_TYPE_SCAN_CHARS = 2000

_EFFECTIVE_DATE_PATTERN = re.compile(
    r"effective\s*(?:date)?\s*[:\-]?\s*"
    r"([A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
    re.IGNORECASE,
)


class MetadataTaggingError(Exception):
    """Raised when metadata tagging cannot proceed for the given input."""


class MetadataTagger:
    """
    Infers document_type, department, and effective_date tags from a
    document's filename and cleaned text, using keyword/regex
    heuristics -- a fast, zero-training-data baseline consistent with
    Section 3B's cleaning-stage scope (full NER-based tagging is
    Module 4's job, not Module 2's).
    """

    _DOCUMENT_TYPE_KEYWORDS: dict[DocumentType, tuple[str, ...]] = {
        DocumentType.HANDBOOK: ("handbook",),
        DocumentType.FAQ: ("faq", "frequently asked"),
        DocumentType.MANUAL: ("manual", "runbook", "technical guide", "user guide"),
        DocumentType.POLICY: ("policy", "policies"),
    }

    # Mirrors Section 1.2's department taxonomy (HR & People, IT &
    # Technical, Compliance & Legal, Project & Org Knowledge).
    _DEPARTMENT_KEYWORDS: dict[str, tuple[str, ...]] = {
        "Finance": ("finance", "expense", "reimbursement", "payroll", "sap fi"),
        "HR": (
            "human resources",
            "hr department",
            "leave policy",
            "onboarding",
            "appraisal",
            "code of conduct",
        ),
        "IT": (
            "vpn",
            "software provisioning",
            "incident response",
            "helpdesk",
            "cisco ios",
            "it department",
        ),
        "Legal": ("nda", "data-retention", "data retention", "audit procedure", "compliance policy"),
    }

    def __init__(self, scan_chars: int = _DOCUMENT_TYPE_SCAN_CHARS) -> None:
        """
        Args:
            scan_chars: Number of leading characters of cleaned text
                used for document_type inference.

        Raises:
            ValueError: If scan_chars is not positive.
        """
        if scan_chars <= 0:
            raise ValueError(f"scan_chars must be positive, got {scan_chars}")
        self._scan_chars = scan_chars

    def infer_document_type(self, raw: RawDocument, cleaned_text: str) -> DocumentType:
        """
        Infer the document's category from its filename and the
        leading portion of its cleaned text.

        Args:
            raw: The source RawDocument (its file_name is checked
                first, since filenames like "employee_handbook.pdf"
                are a strong, cheap signal).
            cleaned_text: The document's cleaned text.

        Returns:
            The first matching DocumentType, checked in the order
            HANDBOOK, FAQ, MANUAL, POLICY (most-specific first, since
            e.g. "policy" appears incidentally in many document types
            but "handbook" rarely does), or DocumentType.UNKNOWN if no
            keyword matches.
        """
        haystack = f"{raw.file_name} {cleaned_text[: self._scan_chars]}".lower()
        for document_type, keywords in self._DOCUMENT_TYPE_KEYWORDS.items():
            if any(keyword in haystack for keyword in keywords):
                return document_type
        return DocumentType.UNKNOWN

    def infer_department(self, cleaned_text: str) -> Optional[str]:
        """
        Infer the most likely owning department by keyword frequency.

        Args:
            cleaned_text: The document's cleaned text.

        Returns:
            The department name with the highest total keyword-match
            count, or None if no department keyword appears at all.
        """
        if not cleaned_text:
            return None
        lowered = cleaned_text.lower()
        scores = {
            department: sum(lowered.count(keyword) for keyword in keywords)
            for department, keywords in self._DEPARTMENT_KEYWORDS.items()
        }
        best_department, best_score = max(scores.items(), key=lambda item: item[1])
        return best_department if best_score > 0 else None

    def infer_effective_date(self, cleaned_text: str) -> Optional[date]:
        """
        Infer a policy effective date from an "Effective Date: ..."
        style phrase in the text.

        Args:
            cleaned_text: The document's cleaned text.

        Returns:
            The parsed date, or None if no matching phrase is found or
            the matched text cannot be parsed as a date.
        """
        if not cleaned_text:
            return None
        match = _EFFECTIVE_DATE_PATTERN.search(cleaned_text)
        if not match:
            return None
        candidate = match.group(1)
        try:
            return dateutil_parser.parse(candidate, fuzzy=True).date()
        except (ValueError, OverflowError) as exc:
            logger.warning(
                "Matched an effective-date phrase but failed to parse "
                "%r as a date: %s",
                candidate,
                exc,
            )
            return None

    def tag(self, raw: RawDocument, cleaned_text: str) -> CleanDocument:
        """
        Infer all metadata tags and produce the final CleanDocument.

        Args:
            raw: The source RawDocument.
            cleaned_text: The document's cleaned text (from
                DocumentCleaner.clean()).

        Returns:
            A populated CleanDocument with document_type, department,
            and effective_date tags attached.

        Raises:
            MetadataTaggingError: If raw is None.
        """
        if raw is None:
            raise MetadataTaggingError("tag() received raw=None, expected a RawDocument.")

        document_type = self.infer_document_type(raw, cleaned_text)
        department = self.infer_department(cleaned_text)
        effective_date = self.infer_effective_date(cleaned_text)

        logger.debug(
            "Tagged %s: document_type=%s department=%s effective_date=%s",
            raw.document_id,
            document_type.value,
            department,
            effective_date,
        )

        return CleanDocument.from_raw_document(
            raw,
            cleaned_text,
            document_type=document_type,
            department=department,
            effective_date=effective_date,
        )


def tag_metadata(raw: RawDocument, cleaned_text: str) -> CleanDocument:
    """
    Module-level convenience wrapper around MetadataTagger.tag(), per
    the Phase 4 module contract's function signature
    (tag_metadata(doc) -> CleanDocument). Constructs a default-config
    MetadataTagger for one-off use; callers that tag many documents in
    a loop should instantiate MetadataTagger directly instead, to
    avoid rebuilding it per call.

    Args:
        raw: The source RawDocument.
        cleaned_text: The document's cleaned text.

    Returns:
        A populated CleanDocument.
    """
    return MetadataTagger().tag(raw, cleaned_text)
