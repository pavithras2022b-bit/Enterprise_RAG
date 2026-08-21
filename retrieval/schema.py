"""
retrieval.schema (Module 9)

Dataclasses for hybrid (vector + graph) retrieval fusion:

    - CandidateSource: which subsystem(s) surfaced a candidate.
    - GraphNodeRef: one query-mention -> graph-node link produced by
      EntityLinker.
    - GraphFact: one retrieved (subject, predicate, object) graph fact,
      rendered as both structured data and natural-language text.
    - FusedCandidate: the module's primary output -- one row of the final
      merged, scored candidate list.

All dataclasses here are frozen, slotted, and fully validated in
__post_init__, matching the pattern used throughout the codebase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ner.schema import EntityType
from relation_extraction.schema import RelationType


class SchemaValidationError(Exception):
    """Raised when a retrieval.schema dataclass is constructed with invalid data."""


class CandidateSource(str, Enum):
    VECTOR = "VECTOR"
    GRAPH = "GRAPH"
    BOTH = "BOTH"


# Fixed lookup table: RelationType -> short verb phrase, used to render a
# GraphFact's natural-language `text` field (Section 4.3 of the spec).
_PREDICATE_PHRASES: dict[RelationType, str] = {
    RelationType.BELONGS_TO: "belongs to",
    RelationType.HAS_ROLE: "has the role of",
    RelationType.FOLLOWS: "follows",
    RelationType.MANAGES: "manages",
    RelationType.ASSIGNED_TO: "is assigned to",
    RelationType.USES: "uses",
    RelationType.REQUIRES_ACCESS: "requires access to",
    RelationType.ELIGIBLE_FOR: "is eligible for",
    RelationType.GOVERNS: "governs",
    RelationType.REQUIRES_TRAINING: "requires training in",
}


def render_predicate_phrase(predicate: str) -> str:
    """
    Maps a RelationType value to its human-readable verb phrase. Falls back
    to a title-cased, underscore-stripped rendering of the raw predicate
    string for any value not in the table, rather than raising -- text
    rendering must never be the reason a valid graph fact is dropped
    (Section 4.3).
    """
    try:
        relation_type = RelationType(predicate)
    except ValueError:
        return predicate.replace("_", " ").title().lower()

    phrase = _PREDICATE_PHRASES.get(relation_type)
    if phrase is not None:
        return phrase
    return relation_type.value.replace("_", " ").title().lower()


def make_fact_id(subject: str, predicate: str, obj: str) -> str:
    """
    Deterministic fact_id, format `subject|predicate|object`, all three
    components lowercased and stripped (Section 4.3), so the same fact
    retrieved twice via different traversal paths produces the same id.
    """
    return "|".join(
        part.strip().lower() for part in (subject, predicate, obj)
    )


@dataclass(frozen=True, slots=True)
class GraphNodeRef:
    """One query-mention-to-graph-node link, produced by EntityLinker."""

    mention_text: str
    canonical_name: str
    entity_type: EntityType
    match_score: float

    def __post_init__(self) -> None:
        if not self.mention_text or not self.mention_text.strip():
            raise SchemaValidationError("mention_text must be non-empty.")
        if not self.canonical_name or not self.canonical_name.strip():
            raise SchemaValidationError("canonical_name must be non-empty.")
        if not (0.0 <= self.match_score <= 100.0):
            raise SchemaValidationError(
                f"match_score must be in [0, 100], got {self.match_score}."
            )


@dataclass(frozen=True, slots=True)
class GraphFact:
    """One retrieved graph fact, structured + natural-language rendering."""

    fact_id: str
    subject: str
    predicate: str
    object: str
    text: str
    source_chunk_id: Optional[str]
    hop_count: int
    raw_relevance: float
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.fact_id:
            raise SchemaValidationError("fact_id must be non-empty.")
        if not self.subject:
            raise SchemaValidationError("subject must be non-empty.")
        if not self.object:
            raise SchemaValidationError("object must be non-empty.")
        if not self.text:
            raise SchemaValidationError("text must be non-empty.")
        try:
            RelationType(self.predicate)
        except ValueError as exc:
            raise SchemaValidationError(
                f"predicate {self.predicate!r} is not a valid RelationType value."
            ) from exc
        if self.hop_count < 1:
            raise SchemaValidationError(
                f"hop_count must be >= 1, got {self.hop_count}."
            )
        if not (self.raw_relevance > 0.0):
            raise SchemaValidationError(
                f"raw_relevance must be > 0.0, got {self.raw_relevance}."
            )

    @classmethod
    def build(
        cls,
        subject: str,
        predicate: str,
        obj: str,
        hop_count: int,
        source_chunk_id: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> "GraphFact":
        """Convenience constructor: derives fact_id, text, and raw_relevance
        from (subject, predicate, object, hop_count) per Section 4.3's
        deterministic formulas, so callers don't duplicate that logic."""
        fact_id = make_fact_id(subject, predicate, obj)
        phrase = render_predicate_phrase(predicate)
        text = f"{subject} {phrase} {obj}."
        raw_relevance = 1.0 / hop_count if hop_count > 0 else 0.0
        return cls(
            fact_id=fact_id,
            subject=subject,
            predicate=predicate,
            object=obj,
            text=text,
            source_chunk_id=source_chunk_id,
            hop_count=hop_count,
            raw_relevance=raw_relevance,
            metadata=metadata or {},
        )


def _is_finite_unit_score(value: float) -> bool:
    return math.isfinite(value) and 0.0 <= value <= 1.0


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    """The module's primary output: one fused, scored evidence candidate."""

    candidate_id: str
    source: CandidateSource
    vector_score: Optional[float]
    graph_score: Optional[float]
    fused_score: float
    text: str
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise SchemaValidationError("candidate_id must be non-empty.")
        if not self.text:
            raise SchemaValidationError("text must be non-empty.")

        if self.vector_score is None and self.graph_score is None:
            raise SchemaValidationError(
                "At least one of vector_score/graph_score must be non-None."
            )

        for name, value in (
            ("vector_score", self.vector_score),
            ("graph_score", self.graph_score),
            ("fused_score", self.fused_score),
        ):
            if value is None:
                continue
            if not _is_finite_unit_score(value):
                raise SchemaValidationError(
                    f"{name} must be finite and within [0, 1], got {value}."
                )

        if self.source == CandidateSource.VECTOR and self.graph_score is not None:
            raise SchemaValidationError(
                "source=VECTOR requires graph_score to be None."
            )
        if self.source == CandidateSource.GRAPH and self.vector_score is not None:
            raise SchemaValidationError(
                "source=GRAPH requires vector_score to be None."
            )
        if self.source == CandidateSource.BOTH and (
            self.vector_score is None or self.graph_score is None
        ):
            raise SchemaValidationError(
                "source=BOTH requires both vector_score and graph_score to be non-None."
            )
