"""
Module 12 - LLM Response Generation: data schema definitions.

Defines GeneratedAnswer, the module's primary output type: the final
natural-language answer synthesized from top-N re-ranked evidence
(Module 11), with extracted citations and generation latency.

Reference: Capstone Project Guide, Section 3M ("LLM Generation"),
Phase 4 Module 12 contract, Phase 5 file structure (llm/schema.py).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class SchemaValidationError(Exception):
    """Raised when a GeneratedAnswer is constructed with invalid field values."""


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """
    Immutable representation of the LLM's final generated answer,
    produced by LLMClient.generate_answer().

    Attributes:
        text: The generated natural-language answer text.
        citations: Source ids (RankedCandidate.candidate_id values)
            the answer cites, extracted via extract_citations() and
            validated against the evidence actually supplied to the
            prompt -- a citation to an id that wasn't in the evidence
            never appears here (Section 13.1's "citation accuracy":
            is the citation actually correct, not just present).
        latency_ms: Wall-clock generation time in milliseconds
            (Section 13.6's cost/latency reporting).
        model_name: Identifier of the model that produced this answer
            (e.g. an Ollama model tag), for traceability across
            evaluation runs comparing different checkpoints/variants.

    Raises:
        SchemaValidationError: If text is empty, latency_ms is
            negative or non-finite, or model_name is empty.
    """

    text: str
    citations: tuple[str, ...] = field(default_factory=tuple)
    latency_ms: float = 0.0
    model_name: str = ""

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise SchemaValidationError(
                "GeneratedAnswer was constructed with empty or whitespace-only text."
            )
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise SchemaValidationError(
                f"GeneratedAnswer has invalid latency_ms ({self.latency_ms}); "
                f"must be finite and non-negative."
            )
        if not self.model_name:
            raise SchemaValidationError(
                "GeneratedAnswer was constructed with an empty model_name."
            )
        for citation in self.citations:
            if not citation or not citation.strip():
                raise SchemaValidationError(
                    "GeneratedAnswer.citations contains an empty citation id."
                )

    @property
    def has_citations(self) -> bool:
        """True if the answer cites at least one evidence source."""
        return len(self.citations) > 0
