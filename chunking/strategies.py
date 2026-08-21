"""
Module 3 - Chunking: splitting strategies.

Provides chunk_fixed_size, chunk_sentence_aware, chunk_structure_aware
per the Phase 4 module contract, plus a ChunkingStrategy abstract base
and its FixedSizeChunker / SentenceAwareChunker / StructureAwareChunker
implementations, which wrap the free functions into Chunk-producing
objects usable directly against a CleanDocument.

Word count (whitespace-delimited tokens) is used throughout as the
sizing unit, approximating the "200-500 tokens" guidance in Section
3C without requiring a model-specific subword tokenizer at the
chunking stage -- the same approximation the project guide's own
pseudocode implicitly makes.

Reference: Capstone Project Guide, Section 3C ("Chunking") and Phase
2's Technology Stack ("Custom structure-aware chunker (heading/
section-boundary regex + spaCy sentence splitter fallback)"),
Phase 4 Module 3 contract, Phase 5 file structure
(chunking/strategies.py).
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Optional

import spacy
from spacy.language import Language

from chunking.schema import Chunk
from preprocessing.schema import CleanDocument

logger = logging.getLogger(__name__)

# Matches common enterprise-document heading styles: "Section 4.2: ...",
# numbered clauses ("4.2 Maternity Leave"), and Markdown-style headers
# ("## Heading"). Callers may supply their own compiled pattern to
# chunk_structure_aware / StructureAwareChunker instead.
DEFAULT_HEADING_PATTERN: re.Pattern[str] = re.compile(
    r"^(?:section\s+\d+(?:\.\d+)*\s*[:.\-]?\s*\S.*"
    r"|\d+(?:\.\d+)+\s*[:.\-]?\s+\S.*"
    r"|#{1,6}\s+\S.*)$",
    re.IGNORECASE,
)

_sentencizer_pipeline: Optional[Language] = None


class ChunkingError(Exception):
    """Raised when a chunking operation cannot proceed for the given input."""


def _get_sentencizer() -> Language:
    """
    Lazily builds and caches a rule-based spaCy sentence-boundary
    pipeline (blank "en" + sentencizer). Deliberately avoids a
    trained/downloaded model: rule-based sentence splitting needs no
    network access and is sufficient for the well-formed enterprise
    prose this project targets.
    """
    global _sentencizer_pipeline
    if _sentencizer_pipeline is None:
        logger.debug("Building spaCy blank('en') + sentencizer pipeline")
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        _sentencizer_pipeline = nlp
    return _sentencizer_pipeline


def chunk_fixed_size(text: str, size: int = 300, overlap: int = 50) -> list[str]:
    """
    Split text into fixed-size, overlapping word-count windows.

    Args:
        text: Text to split.
        size: Target chunk size, in whitespace-delimited words.
        overlap: Number of words shared between consecutive chunks, so
            content near a boundary still appears fully in at least
            one chunk.

    Returns:
        A list of chunk strings. Returns an empty list for empty/
        whitespace-only input. If the whole text fits in one window,
        returns a single-element list.

    Raises:
        ValueError: If size <= 0, or overlap < 0, or overlap >= size.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")
    if not text or not text.strip():
        return []

    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]

    step = size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(words):
        window = words[start : start + size]
        chunks.append(" ".join(window))
        if start + size >= len(words):
            break
        start += step
    return chunks


def chunk_sentence_aware(text: str, target_size: int = 300) -> list[str]:
    """
    Split text into chunks by greedily grouping whole sentences until
    the target word-count size is reached, never splitting a sentence
    across two chunks.

    Args:
        text: Text to split.
        target_size: Target chunk size, in whitespace-delimited words.
            A single sentence longer than target_size is kept intact
            as its own (oversized) chunk rather than being truncated,
            since splitting mid-sentence is exactly what this strategy
            exists to avoid.

    Returns:
        A list of chunk strings, each containing one or more complete
        sentences. Returns an empty list for empty/whitespace-only
        input.

    Raises:
        ValueError: If target_size <= 0.
    """
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")
    if not text or not text.strip():
        return []

    nlp = _get_sentencizer()
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        if current_sentences and current_word_count + sentence_word_count > target_size:
            chunks.append(" ".join(current_sentences))
            current_sentences = []
            current_word_count = 0
        current_sentences.append(sentence)
        current_word_count += sentence_word_count

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def chunk_structure_aware(
    text: str,
    headings: Optional[re.Pattern[str]] = None,
    target_size: int = 300,
) -> list[str]:
    """
    Split text into chunks aligned to the document's own heading/
    section structure, so a clause and its exceptions/qualifiers are
    never separated (Section 3C's primary motivation for this
    strategy). Sections that individually exceed target_size fall
    back to chunk_sentence_aware for that section only, per Phase 2's
    "heading/section-boundary regex + spaCy sentence splitter
    fallback" design.

    Args:
        text: Text to split.
        headings: Compiled regex matched against each stripped line to
            detect a heading/section boundary. Defaults to
            DEFAULT_HEADING_PATTERN if not provided.
        target_size: Word-count threshold above which an individual
            section is further split via chunk_sentence_aware.

    Returns:
        A list of chunk strings, one per section (or per
        sentence-aware sub-chunk, for oversized sections). Returns an
        empty list for empty/whitespace-only input. Text with no
        detected headings is returned as chunk_sentence_aware(text)
        if it exceeds target_size, or as a single whole-text chunk
        otherwise.

    Raises:
        ValueError: If target_size <= 0.
    """
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")
    if not text or not text.strip():
        return []

    heading_pattern = headings or DEFAULT_HEADING_PATTERN

    lines = text.split("\n")
    sections: list[str] = []
    current_section_lines: list[str] = []

    for line in lines:
        if heading_pattern.match(line.strip()) and current_section_lines:
            section_text = "\n".join(current_section_lines).strip()
            if section_text:
                sections.append(section_text)
            current_section_lines = [line]
        else:
            current_section_lines.append(line)

    if current_section_lines:
        section_text = "\n".join(current_section_lines).strip()
        if section_text:
            sections.append(section_text)

    if not sections:
        return []

    chunks: list[str] = []
    for section in sections:
        word_count = len(section.split())
        if word_count <= target_size:
            chunks.append(section)
        else:
            logger.debug(
                "Section (%d words) exceeds target_size=%d; falling back to "
                "sentence-aware splitting",
                word_count,
                target_size,
            )
            chunks.extend(chunk_sentence_aware(section, target_size=target_size))

    return chunks


class ChunkingStrategy(ABC):
    """
    Abstract base for all chunking strategies. Subclasses implement
    split_text(); the chunk() template method wraps the resulting
    strings into Chunk objects with metadata inherited from a
    CleanDocument, which is what Module 3's callers (Phase 6
    orchestration) actually use.
    """

    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        """
        Split raw text into a list of chunk strings, with no
        source-document metadata attached.

        Args:
            text: Text to split.

        Returns:
            A list of chunk strings in document order.
        """
        raise NotImplementedError

    def chunk(self, document: CleanDocument) -> list[Chunk]:
        """
        Split a CleanDocument's text and wrap each piece into a Chunk,
        inheriting the document's metadata.

        Args:
            document: The CleanDocument to chunk (from Module 2).

        Returns:
            A list of Chunk objects in document order. Empty pieces
            (which can occur at strategy boundaries) are filtered out
            before Chunk construction, since Chunk itself rejects
            empty text.

        Raises:
            ChunkingError: If document is None.
        """
        if document is None:
            raise ChunkingError("chunk() received document=None, expected a CleanDocument.")

        pieces = self.split_text(document.text)
        chunks = [
            Chunk.from_clean_document(document, piece, position)
            for position, piece in enumerate(pieces)
            if piece and piece.strip()
        ]
        logger.info(
            "Chunked document %s into %d chunk(s) using %s",
            document.document_id,
            len(chunks),
            type(self).__name__,
        )
        return chunks


class FixedSizeChunker(ChunkingStrategy):
    """Fixed-size, overlapping word-count window chunking strategy."""

    def __init__(self, size: int = 300, overlap: int = 50) -> None:
        """
        Args:
            size: Target chunk size, in words. Forwarded to
                chunk_fixed_size.
            overlap: Overlap between consecutive chunks, in words.
                Forwarded to chunk_fixed_size.

        Raises:
            ValueError: If size <= 0, overlap < 0, or overlap >= size.
        """
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        if overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {overlap}")
        if overlap >= size:
            raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")
        self._size = size
        self._overlap = overlap

    def split_text(self, text: str) -> list[str]:
        return chunk_fixed_size(text, size=self._size, overlap=self._overlap)


class SentenceAwareChunker(ChunkingStrategy):
    """Sentence-boundary-respecting chunking strategy."""

    def __init__(self, target_size: int = 300) -> None:
        """
        Args:
            target_size: Target chunk size, in words. Forwarded to
                chunk_sentence_aware.

        Raises:
            ValueError: If target_size <= 0.
        """
        if target_size <= 0:
            raise ValueError(f"target_size must be positive, got {target_size}")
        self._target_size = target_size

    def split_text(self, text: str) -> list[str]:
        return chunk_sentence_aware(text, target_size=self._target_size)


class StructureAwareChunker(ChunkingStrategy):
    """
    Heading/section-boundary-aware chunking strategy, with a
    sentence-aware fallback for oversized sections. This is the
    project's chosen primary strategy (Phase 2's Technology Stack).
    """

    def __init__(
        self,
        headings: Optional[re.Pattern[str]] = None,
        target_size: int = 300,
    ) -> None:
        """
        Args:
            headings: Compiled regex for heading detection. Defaults
                to DEFAULT_HEADING_PATTERN if not provided.
            target_size: Word-count threshold above which an
                individual section falls back to sentence-aware
                splitting.

        Raises:
            ValueError: If target_size <= 0.
        """
        if target_size <= 0:
            raise ValueError(f"target_size must be positive, got {target_size}")
        self._headings = headings
        self._target_size = target_size

    def split_text(self, text: str) -> list[str]:
        return chunk_structure_aware(
            text, headings=self._headings, target_size=self._target_size
        )


_STRATEGY_REGISTRY: dict[str, type[ChunkingStrategy]] = {
    "fixed_size": FixedSizeChunker,
    "sentence_aware": SentenceAwareChunker,
    "structure_aware": StructureAwareChunker,
}


def create_chunker(strategy: str, **kwargs: object) -> ChunkingStrategy:
    """
    Factory for constructing a ChunkingStrategy by name, so the
    orchestration layer (Phase 6) can select a strategy from
    configs/chunking.yaml without importing concrete classes.

    Args:
        strategy: One of "fixed_size", "sentence_aware",
            "structure_aware".
        **kwargs: Forwarded to the selected strategy's constructor
            (e.g. size=300, overlap=50 for "fixed_size").

    Returns:
        An instantiated ChunkingStrategy.

    Raises:
        ValueError: If strategy is not a recognized strategy name.
    """
    strategy_cls = _STRATEGY_REGISTRY.get(strategy)
    if strategy_cls is None:
        raise ValueError(
            f"Unknown chunking strategy '{strategy}'. "
            f"Supported: {sorted(_STRATEGY_REGISTRY)}"
        )
    return strategy_cls(**kwargs)  # type: ignore[arg-type]
