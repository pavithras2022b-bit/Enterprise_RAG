"""
Module 5 - Relationship Extraction: data schema definitions.

Defines the RelationType enum (the closed relationship schema from
Section 4.4) and the Relation dataclass. This file establishes the
contract Module 6 (Knowledge Graph) builds against; the extraction
model, rule-based patterns, and training entrypoint
(relation_extraction/model.py, rules.py, train.py) that populate
Relation objects are implemented when Module 5 itself is built.

Reference: Capstone Project Guide, Section 3H ("Relationship
Extraction"), Section 4.3/4.4, Phase 4 Module 5 contract, Phase 5 file
structure (relation_extraction/schema.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ner.schema import Entity

logger = logging.getLogger(__name__)


class RelationType(str, Enum):
    """
    The closed relationship schema, per Section 4.4's representative
    schema. A closed schema (vs. open information extraction) keeps
    the resulting graph clean, consistent, and directly Cypher-
    queryable -- the project's chosen tradeoff for an enterprise KG
    with well-understood entity types.
    """

    BELONGS_TO = "BELONGS_TO"          # Employee -> Department
    HAS_ROLE = "HAS_ROLE"              # Employee -> Role
    FOLLOWS = "FOLLOWS"                # Department -> Policy
    MANAGES = "MANAGES"                # Employee -> Employee (reporting line)
    ASSIGNED_TO = "ASSIGNED_TO"        # Employee -> Project
    USES = "USES"                      # Project -> Software / System
    REQUIRES_ACCESS = "REQUIRES_ACCESS"  # Role -> NetworkResource
    ELIGIBLE_FOR = "ELIGIBLE_FOR"      # Employee -> LeaveType
    GOVERNS = "GOVERNS"                # Policy -> LeaveType / Department
    REQUIRES_TRAINING = "REQUIRES_TRAINING"  # Department/Role -> Software
    REPORTS_TO = "REPORTS_TO"          # Employee -> Employee
    OWNS_SYSTEM = "OWNS_SYSTEM"        # Department/Employee -> System
    WORKS_ON = "WORKS_ON"              # Employee -> Project
    GOVERNED_BY = "GOVERNED_BY"        # Department/System -> Policy
    APPLIES_TO = "APPLIES_TO"          # Policy -> Department/Employee



class SchemaValidationError(Exception):
    """Raised when a Relation is constructed with invalid field values."""


@dataclass(frozen=True, slots=True)
class Relation:
    """
    Immutable representation of one typed, directed relationship
    between two entities, produced by Module 5 and consumed by
    Module 6 (Neo4jGraphBuilder.create_relationship).

    Attributes:
        subject_entity: The relationship's source node.
        predicate: The relationship type connecting subject to object.
        object_entity: The relationship's target node.
        source_chunk_id: The Chunk this relation was extracted from,
            preserving traceability.
        properties: Relationship-level properties (e.g. a
            "within_days" qualifier on a REQUIRES_TRAINING edge,
            per Section 4.3's worked example -- qualifiers belong on
            the relationship, not crammed onto a node).
        confidence: Extraction confidence in [0, 1].

    Raises:
        SchemaValidationError: If subject_entity and object_entity are
            identical (a self-relation, almost always an extraction
            artifact rather than a genuine fact), if source_chunk_id
            is empty, or if confidence is outside [0, 1].
    """

    subject_entity: Entity
    predicate: RelationType
    object_entity: Entity
    source_chunk_id: str
    properties: dict[str, str] = field(default_factory=dict)
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.source_chunk_id:
            raise SchemaValidationError(
                "Relation was constructed with an empty source_chunk_id."
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise SchemaValidationError(
                f"Relation has confidence {self.confidence} outside [0, 1]."
            )
        if self.subject_entity == self.object_entity:
            raise SchemaValidationError(
                f"Relation subject and object are identical "
                f"({self.subject_entity.text!r}); a self-relation almost "
                f"always indicates an extraction artifact, not a genuine "
                f"fact."
            )

    def __repr__(self) -> str:
        subject_name = (
            self.subject_entity.canonical_name
            if self.subject_entity.canonical_name is not None
            else self.subject_entity.text
        )
        object_name = (
            self.object_entity.canonical_name
            if self.object_entity.canonical_name is not None
            else self.object_entity.text
        )
        return f"Relation({subject_name!r} -[{self.predicate.value}]-> {object_name!r})"


# Label used for negative (non-relation) training examples -- most
# co-occurring entity pairs in a sentence have no relation at all, and
# a relation classifier needs explicit negative examples to learn that,
# not just the positive RelationType labels.
NO_RELATION_LABEL = "NO_RELATION"


@dataclass(frozen=True, slots=True)
class LabeledRelationExample:
    """
    One training example for Module 5's relation classifier
    (relation_extraction/train.py): a raw text window containing two
    entity mentions (by character span, not yet marker-annotated) and
    the correct label -- a RelationType value, or NO_RELATION_LABEL
    for a co-occurring pair with no relation.

    Attributes:
        text: The raw sentence/window text containing both entities.
        subject_span: (start, end) character offsets of the subject
            entity mention within `text`.
        object_span: (start, end) character offsets of the object
            entity mention within `text`.
        label: A RelationType value, or NO_RELATION_LABEL.

    Raises:
        SchemaValidationError: If either span is out of bounds,
            zero/negative-length, the two spans overlap, or label is
            neither NO_RELATION_LABEL nor a valid RelationType value.
    """

    text: str
    subject_span: tuple[int, int]
    object_span: tuple[int, int]
    label: str

    def __post_init__(self) -> None:
        for name, span in (("subject_span", self.subject_span), ("object_span", self.object_span)):
            start, end = span
            if start < 0 or end <= start or end > len(self.text):
                raise SchemaValidationError(
                    f"LabeledRelationExample.{name} {span} is invalid for "
                    f"text of length {len(self.text)}."
                )
        subject_start, subject_end = self.subject_span
        object_start, object_end = self.object_span
        if subject_start < object_end and object_start < subject_end:
            raise SchemaValidationError(
                f"LabeledRelationExample subject_span {self.subject_span} and "
                f"object_span {self.object_span} overlap."
            )
        if self.label != NO_RELATION_LABEL:
            try:
                RelationType(self.label)
            except ValueError as exc:
                raise SchemaValidationError(
                    f"LabeledRelationExample.label {self.label!r} is neither "
                    f"{NO_RELATION_LABEL!r} nor a valid RelationType value."
                ) from exc
