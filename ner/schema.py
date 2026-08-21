"""
Module 4 - NER: data schema definitions.

Defines the Entity dataclass -- a single extracted enterprise entity
span, ready to become a knowledge-graph node candidate (Module 6) once
resolved by resolver.py -- plus the EntityType enum matching Section
4.4's node-label schema, and BIO label-vocabulary helpers shared by
model.py and train.py.

Reference: Capstone Project Guide, Section 3G ("Entity Extraction"),
Section 4.4 ("How Enterprise Entities Are Identified"), Phase 4
Module 4 contract, Phase 5 file structure (ner/schema.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    """
    Enumerates the enterprise entity categories NERModel can predict,
    matching Section 4.4's node-label schema (Employee, Department,
    Policy, Project, LeaveType, Software, Role, NetworkResource).
    Values are uppercase to match standard BIO tag conventions
    (e.g. "B-EMPLOYEE").
    """

    EMPLOYEE = "EMPLOYEE"
    DEPARTMENT = "DEPARTMENT"
    POLICY = "POLICY"
    PROJECT = "PROJECT"
    SOFTWARE = "SOFTWARE"
    LEAVE_TYPE = "LEAVE_TYPE"
    ROLE = "ROLE"
    NETWORK_RESOURCE = "NETWORK_RESOURCE"


class SchemaValidationError(Exception):
    """Raised when an Entity is constructed with invalid field values."""


@dataclass(frozen=True, slots=True)
class Entity:
    """
    Immutable representation of a single extracted entity span.

    Attributes:
        text: The exact surface text span as it appeared in the
            source chunk (e.g. "Finance department").
        entity_type: The predicted enterprise entity category.
        start_offset: Character offset (inclusive) of the span's start
            within the source chunk's text.
        end_offset: Character offset (exclusive) of the span's end
            within the source chunk's text.
        source_chunk_id: The Chunk.chunk_id this entity was extracted
            from (Module 3's traceability link).
        canonical_name: The resolved, deduplicated name this entity
            maps to (e.g. "Finance department" and "Finance Dept."
            both resolving to "Finance"). None until resolver.py has
            processed it -- this is what distinguishes a raw
            prediction from a resolved entity per the Phase 4
            contract ("resolved canonical_name (post-dedup)").
        confidence: The model's average softmax confidence across the
            span's tokens, in [0, 1]. Not part of the literal Phase 4
            attribute list, but included since threshold-filtering on
            it is standard NER practice and Section 3G explicitly
            discusses extraction precision; documented here as a
            deliberate, minor addition.

    Raises:
        SchemaValidationError: If constructed with empty text, a
            negative start_offset, or end_offset <= start_offset.
    """

    text: str
    entity_type: EntityType
    start_offset: int
    end_offset: int
    source_chunk_id: str
    canonical_name: Optional[str] = None
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise SchemaValidationError(
                "Entity was constructed with empty text -- callers must "
                "filter empty spans before constructing an Entity."
            )
        if self.start_offset < 0:
            raise SchemaValidationError(
                f"Entity {self.text!r} was constructed with a negative "
                f"start_offset ({self.start_offset})."
            )
        if self.end_offset <= self.start_offset:
            raise SchemaValidationError(
                f"Entity {self.text!r} was constructed with end_offset "
                f"({self.end_offset}) <= start_offset ({self.start_offset})."
            )
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise SchemaValidationError(
                f"Entity {self.text!r} was constructed with confidence "
                f"{self.confidence} outside [0, 1]."
            )

    @property
    def span_length(self) -> int:
        """Returns the character length of the entity's span."""
        return self.end_offset - self.start_offset

    @property
    def is_resolved(self) -> bool:
        """Returns True if this entity has been through resolver.py."""
        return self.canonical_name is not None


@dataclass(frozen=True, slots=True)
class LabeledExample:
    """
    A single BIO-tagged training example for train_ner().

    Attributes:
        tokens: Pre-tokenized (whitespace/word-level) tokens.
        tags: BIO tags aligned 1:1 with tokens (e.g. "O", "B-EMPLOYEE",
            "I-EMPLOYEE"). Must be the same length as tokens.

    Raises:
        SchemaValidationError: If tokens/tags are empty, of mismatched
            length, or a tag isn't "O" or "{B,I}-{EntityType}".
    """

    tokens: tuple[str, ...]
    tags: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.tokens:
            raise SchemaValidationError("LabeledExample was constructed with no tokens.")
        if len(self.tokens) != len(self.tags):
            raise SchemaValidationError(
                f"LabeledExample has {len(self.tokens)} tokens but "
                f"{len(self.tags)} tags -- they must be the same length."
            )
        valid_type_values = {t.value for t in EntityType}
        for tag in self.tags:
            if tag == "O":
                continue
            prefix, _, type_value = tag.partition("-")
            if prefix not in ("B", "I") or type_value not in valid_type_values:
                raise SchemaValidationError(
                    f"LabeledExample has invalid BIO tag {tag!r}; expected "
                    f"'O' or '{{B,I}}-{{EntityType}}' "
                    f"(valid types: {sorted(valid_type_values)})."
                )


def build_bio_label_maps(
    entity_types: Optional[Sequence[EntityType]] = None,
) -> tuple[dict[int, str], dict[str, int]]:
    """
    Build the id2label / label2id maps for a BIO tagging scheme over
    the given entity types (or all EntityType members by default).

    Label 0 is always "O"; each entity type then contributes a
    "B-{type}" and "I-{type}" label, in EntityType declaration order.
    Both train.py (to configure the classification head) and model.py
    (to interpret its output) rely on this single source of truth so
    the two never drift apart.

    Args:
        entity_types: Entity types to include. Defaults to every
            EntityType member.

    Returns:
        A (id2label, label2id) tuple of dicts.
    """
    types = list(entity_types) if entity_types is not None else list(EntityType)
    labels = ["O"]
    for entity_type in types:
        labels.append(f"B-{entity_type.value}")
        labels.append(f"I-{entity_type.value}")
    id2label = dict(enumerate(labels))
    label2id = {label: index for index, label in id2label.items()}
    return id2label, label2id
