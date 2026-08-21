"""
Module 1 - Document Ingestion: checksum-based deduplication.

Provides compute_checksum and dedupe_by_checksum per the Phase 4
module contract, preventing the same physical document from being
ingested twice (e.g. re-running ingestion after adding new files to a
directory that already contains previously-ingested ones).

Reference: Capstone Project Guide, Section 3A ("checksum-based
deduplication (MD5/SHA-256 hashing of file contents)"), Phase 4
Module 1 contract, Phase 5 file structure (ingestion/dedupe.py).
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ingestion.schema import RawDocument

logger = logging.getLogger(__name__)

_SUPPORTED_ALGORITHMS = {"sha256", "sha1", "md5"}


def compute_checksum(content: bytes, algorithm: str = "sha256") -> str:
    """
    Compute a hex-digest checksum of raw file bytes.

    Args:
        content: Raw bytes of the file.
        algorithm: Hash algorithm name. Must be one of
            _SUPPORTED_ALGORITHMS. SHA-256 is the default per the
            project guide's stack (collision-resistant, negligible
            performance cost at capstone corpus scale).

    Returns:
        The lowercase hex digest string.

    Raises:
        ValueError: If content is empty (an empty-file checksum is
            not a meaningful document fingerprint and usually
            indicates an upstream extraction bug), or if algorithm is
            not one of the supported algorithms.
    """
    if not content:
        raise ValueError("Cannot compute checksum of empty byte content.")
    if algorithm not in _SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Unsupported checksum algorithm '{algorithm}'. "
            f"Supported: {sorted(_SUPPORTED_ALGORITHMS)}"
        )
    hasher = hashlib.new(algorithm)
    hasher.update(content)
    return hasher.hexdigest()


def dedupe_by_checksum(documents: list["RawDocument"]) -> list["RawDocument"]:
    """
    Remove duplicate RawDocuments that share an identical checksum.

    Two documents with the same checksum are byte-for-byte identical
    source files, regardless of source_path (e.g. the same PDF present
    at two different paths, or re-ingested on a second run). The first
    occurrence (by input list order) is kept; later duplicates are
    dropped and logged.

    Args:
        documents: List of RawDocuments, potentially containing
            checksum duplicates.

    Returns:
        A new list with duplicates removed, preserving the original
        relative order of first occurrences.
    """
    seen_checksums: set[str] = set()
    deduped: list["RawDocument"] = []
    duplicate_count = 0

    for document in documents:
        if document.checksum in seen_checksums:
            duplicate_count += 1
            logger.info(
                "Dropping duplicate document %s (checksum=%s, path=%s)",
                document.document_id,
                document.checksum,
                document.source_path,
            )
            continue
        seen_checksums.add(document.checksum)
        deduped.append(document)

    if duplicate_count:
        logger.info(
            "dedupe_by_checksum removed %d duplicate(s) out of %d input "
            "documents",
            duplicate_count,
            len(documents),
        )
    return deduped
