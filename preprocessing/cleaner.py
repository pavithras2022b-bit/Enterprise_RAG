"""
Module 2 - Cleaning: text cleaning and normalization.

Provides strip_boilerplate, fix_encoding, dedupe_paragraphs per the
Phase 4 module contract, and a DocumentCleaner class that applies
them in the correct order (encoding fix -> boilerplate strip ->
whitespace normalization -> paragraph dedup).

Reference: Capstone Project Guide, Section 3B ("Data Cleaning"),
Phase 4 Module 2 contract, Phase 5 file structure
(preprocessing/cleaner.py).
"""

from __future__ import annotations

import logging
import re
from collections import Counter

import ftfy

logger = logging.getLogger(__name__)

# Lines that are pure page markers ("Page 3 of 40", "3", "- 3 -") are
# stripped unconditionally -- they carry no semantic content regardless
# of how many times they repeat.
_PAGE_NUMBER_PATTERN = re.compile(
    r"^(page\s+\d+(\s+of\s+\d+)?|[-–]?\s*\d{1,4}\s*[-–]?)$",
    re.IGNORECASE,
)


class CleaningError(Exception):
    """Raised when a cleaning step cannot be applied to the given input."""


def fix_encoding(text: str) -> str:
    """
    Repair mis-decoded characters introduced by PDF/DOCX extraction
    (smart quotes, ligatures like "fi" collapsing incorrectly, mojibake
    from encoding mismatches), using ftfy.

    Args:
        text: Raw or partially-cleaned text.

    Returns:
        Text with encoding artifacts repaired. Empty input is returned
        unchanged (not an error -- an upstream extractor may
        legitimately hand cleaner.py an empty string for a blank
        page/section).
    """
    if not text:
        return text
    return ftfy.fix_text(text)


def strip_boilerplate(
    text: str,
    min_repeats: int = 3,
    max_line_length: int = 200,
) -> str:
    """
    Remove repeated header/footer boilerplate and page-number lines.

    A line is treated as boilerplate and removed from every occurrence
    if either:
      (a) it is short (<= max_line_length characters) and its exact
          stripped text repeats at least min_repeats times across the
          document (e.g. "Confidential -- Internal Use Only" appearing
          on every page), or
      (b) it matches a pure page-number pattern ("Page 3 of 40", "12").

    Args:
        text: Text to clean (typically already encoding-fixed).
        min_repeats: Minimum number of exact repetitions before a line
            is treated as boilerplate. Set high enough to avoid
            stripping legitimately short, once-or-twice-repeated
            sentences.
        max_line_length: Only lines at or under this length are
            eligible for repeat-based stripping, since long lines
            repeating verbatim are far more likely to be a genuine
            duplicated paragraph (handled instead by
            dedupe_paragraphs) than a page header/footer.

    Returns:
        Text with boilerplate and page-number lines removed, blank
        lines preserved (so paragraph boundaries used later by
        dedupe_paragraphs are not disturbed).
    """
    if not text:
        return text

    lines = text.split("\n")
    normalized_counts = Counter(
        stripped for line in lines if (stripped := line.strip())
    )
    boilerplate_lines = {
        line
        for line, count in normalized_counts.items()
        if count >= min_repeats and len(line) <= max_line_length
    }

    kept_lines: list[str] = []
    removed_count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept_lines.append(line)
            continue
        if stripped in boilerplate_lines or _PAGE_NUMBER_PATTERN.match(stripped):
            removed_count += 1
            continue
        kept_lines.append(line)

    if removed_count:
        logger.debug("strip_boilerplate removed %d boilerplate line(s)", removed_count)

    return "\n".join(kept_lines)


def _shingles(text: str, shingle_size: int) -> set[str]:
    """Builds a set of word-level n-gram shingles for similarity comparison."""
    tokens = text.lower().split()
    if not tokens:
        return set()
    if len(tokens) < shingle_size:
        return {" ".join(tokens)}
    return {
        " ".join(tokens[i : i + shingle_size])
        for i in range(len(tokens) - shingle_size + 1)
    }


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Computes Jaccard similarity between two shingle sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def dedupe_paragraphs(
    text: str,
    similarity_threshold: float = 0.85,
    shingle_size: int = 5,
) -> str:
    """
    Remove near-duplicate paragraphs (e.g. a disclaimer block repeated
    on every page, or boilerplate too long/varied for strip_boilerplate
    to catch by exact-line matching).

    Uses word-shingling + Jaccard similarity (Section 3B's documented
    approach) rather than exact string matching, so paragraphs that
    differ only in whitespace or minor OCR noise are still recognized
    as duplicates.

    Args:
        text: Text to deduplicate, with paragraphs separated by one or
            more blank lines.
        similarity_threshold: Jaccard similarity (0-1) at or above
            which two paragraphs are considered duplicates. Higher is
            stricter (fewer false-positive removals).
        shingle_size: Number of consecutive words per shingle.

    Returns:
        Text with near-duplicate paragraphs removed, keeping the first
        occurrence of each. Paragraph order is preserved; paragraphs
        are rejoined with a blank line between them.

    Raises:
        ValueError: If similarity_threshold is not in [0, 1].
    """
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError(
            f"similarity_threshold must be in [0, 1], got {similarity_threshold}"
        )
    if not text:
        return text

    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    kept_paragraphs: list[str] = []
    kept_shingle_sets: list[set[str]] = []
    duplicate_count = 0

    for paragraph in paragraphs:
        paragraph_shingles = _shingles(paragraph, shingle_size)
        is_duplicate = any(
            _jaccard_similarity(paragraph_shingles, existing) >= similarity_threshold
            for existing in kept_shingle_sets
        )
        if is_duplicate:
            duplicate_count += 1
            continue
        kept_paragraphs.append(paragraph)
        kept_shingle_sets.append(paragraph_shingles)

    if duplicate_count:
        logger.debug("dedupe_paragraphs removed %d near-duplicate paragraph(s)", duplicate_count)

    return "\n\n".join(kept_paragraphs)


class DocumentCleaner:
    """
    Applies Module 2's full cleaning pipeline to raw text, in the
    order specified by Section 3B: encoding fix -> boilerplate strip
    -> whitespace normalization -> near-duplicate paragraph removal.

    Each stage is individually available as a free function
    (fix_encoding, strip_boilerplate, dedupe_paragraphs) for unit
    testing or standalone use; DocumentCleaner exists to apply them
    consistently, with configuration, at the orchestration layer.
    """

    def __init__(
        self,
        min_repeats: int = 3,
        max_boilerplate_line_length: int = 200,
        similarity_threshold: float = 0.85,
        shingle_size: int = 5,
    ) -> None:
        """
        Args:
            min_repeats: Passed through to strip_boilerplate.
            max_boilerplate_line_length: Passed through to
                strip_boilerplate.
            similarity_threshold: Passed through to dedupe_paragraphs.
            shingle_size: Passed through to dedupe_paragraphs.

        Raises:
            ValueError: If similarity_threshold is not in [0, 1] or
                min_repeats < 1.
        """
        if min_repeats < 1:
            raise ValueError(f"min_repeats must be >= 1, got {min_repeats}")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be in [0, 1], got {similarity_threshold}"
            )
        self._min_repeats = min_repeats
        self._max_boilerplate_line_length = max_boilerplate_line_length
        self._similarity_threshold = similarity_threshold
        self._shingle_size = shingle_size

    def clean(self, text: str) -> str:
        """
        Run the full cleaning pipeline on a single document's raw text.

        Args:
            text: Raw text, typically from Module 1's RawDocument.text.

        Returns:
            Cleaned text, stripped of leading/trailing whitespace.

        Raises:
            CleaningError: If text is None (as opposed to an empty
                string, which is valid input and returns "").
        """
        if text is None:
            raise CleaningError("DocumentCleaner.clean() received None, expected str.")
        if not text:
            logger.warning("DocumentCleaner.clean() received an empty string.")
            return ""

        text = fix_encoding(text)
        text = strip_boilerplate(
            text,
            min_repeats=self._min_repeats,
            max_line_length=self._max_boilerplate_line_length,
        )
        text = self._collapse_whitespace(text)
        text = dedupe_paragraphs(
            text,
            similarity_threshold=self._similarity_threshold,
            shingle_size=self._shingle_size,
        )
        return text.strip()

    @staticmethod
    def _collapse_whitespace(text: str) -> str:
        """Collapses runs of spaces/tabs and excess blank lines."""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text
