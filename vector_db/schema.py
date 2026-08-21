"""
Module 8 - Vector Database: data schema definitions.

Defines VectorSearchResult, the uniform result shape returned by both
VectorStore (ChromaDB, primary) and FAISSBenchmarkIndex (comparison-
only), so Module 9's hybrid retrieval controller can consume either
without caring which backend produced a given candidate.

Reference: Capstone Project Guide, Section 3E ("Vector Databases"),
Phase 4 Module 8 contract, Phase 5 file structure
(vector_db/schema.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class SchemaValidationError(Exception):
    """Raised when a VectorSearchResult is constructed with invalid field values."""


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    """
    One row of a vector similarity search result.

    Attributes:
        chunk_id: The matched EmbeddedChunk's chunk_id.
        score: Similarity score -- higher means more similar. For
            cosine-space backends (both VectorStore and
            FAISSBenchmarkIndex's default configuration), this is
            cosine similarity in roughly [-1, 1], practically [0, 1]
            for the L2-normalized embeddings Module 7 produces.
        text: The matched chunk's original text, returned alongside
            the score so callers don't need a second lookup.
        metadata: The matched chunk's metadata dict (department,
            document_type, etc.), as stored by VectorStore.add().

    Raises:
        SchemaValidationError: If chunk_id is empty or score is not a
            finite number.
    """

    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise SchemaValidationError(
                "VectorSearchResult was constructed with an empty chunk_id."
            )
        if self.score != self.score or self.score in (float("inf"), float("-inf")):
            # `x != x` is the branchless NaN check; avoids importing math
            # for a single comparison used only in validation.
            raise SchemaValidationError(
                f"VectorSearchResult {self.chunk_id!r} has a non-finite score: {self.score}."
            )
