"""
Module 5 - Relationship Extraction: rule-based pattern baseline.

Provides apply_rule_patterns(...) and the RulePatternExtractor class
per the Phase 4 module contract: a zero-training-data complementary
baseline that matches connector phrases between two co-occurring
entities against regex patterns for common, formulaic policy-document
sentence structures (Section 3H's documented alternative to the
learned classifier in model.py).

Deviation from the literal Phase 4 signature
(apply_rule_patterns(sentence) -> list[Relation]): rule-based relation
extraction is not well-defined over a bare sentence string alone --
without knowing which spans are candidate entities, there is nothing
to relate. This module's apply_rule_patterns() therefore also takes
the entities list (Module 4's output for that text), matching the
same entities-plus-context shape used throughout Module 5 and
mirrored by model.py's predict_relation(). This mirrors the documented
deviation already taken for Module 2's tag_metadata().

Reference: Capstone Project Guide, Section 3H ("Relationship
Extraction"), Phase 4 Module 5 contract, Phase 5 file structure
(relation_extraction/rules.py).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from ner.schema import Entity, EntityType
from relation_extraction.schema import Relation, RelationType, SchemaValidationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PatternRule:
    """
    One rule-based extraction pattern: if an entity of subject_type is
    immediately followed (within max_connector_length characters) by
    an entity of object_type, and the connector text between them
    matches connector_pattern, emit a Relation of relation_type.

    Attributes:
        subject_type: Required entity type of the (textually earlier)
            subject entity.
        object_type: Required entity type of the (textually later)
            object entity.
        connector_pattern: Compiled regex searched against the text
            strictly between the two entity spans.
        relation_type: The RelationType to emit on a match.
    """

    subject_type: EntityType
    object_type: EntityType
    connector_pattern: re.Pattern[str]
    relation_type: RelationType
    reverse_direction: bool = False


def _default_rules() -> tuple[PatternRule, ...]:
    """
    Builds the default rule set, covering the closed relationship
    schema's most formulaic, reliably-phrased relation types (Section
    4.4's worked examples: "Employees are entitled to...", "The
    Finance department follows...", "...project uses...").

    The rules include explicit reporting-line language ("X reports to Y")
    as a high-precision inverse form of MANAGES.  Relations still requiring
    unstated world knowledge remain intentionally out of scope.
    """
    return (
        PatternRule(
            subject_type=EntityType.EMPLOYEE,
            object_type=EntityType.DEPARTMENT,
            connector_pattern=re.compile(r"\bbelongs?\s+to(\s+the)?\b", re.IGNORECASE),
            relation_type=RelationType.BELONGS_TO,
        ),
        PatternRule(
            # The textual order is report -> manager, while the graph edge is
            # manager -> report, hence reverse_direction.
            subject_type=EntityType.EMPLOYEE,
            object_type=EntityType.EMPLOYEE,
            connector_pattern=re.compile(r"\breports?\s+to\b", re.IGNORECASE),
            relation_type=RelationType.MANAGES,
            reverse_direction=True,
        ),
        PatternRule(
            subject_type=EntityType.DEPARTMENT,
            object_type=EntityType.POLICY,
            connector_pattern=re.compile(r"\bfollows?(\s+the)?\b", re.IGNORECASE),
            relation_type=RelationType.FOLLOWS,
        ),
        PatternRule(
            subject_type=EntityType.PROJECT,
            object_type=EntityType.SOFTWARE,
            connector_pattern=re.compile(r"\buses?\b", re.IGNORECASE),
            relation_type=RelationType.USES,
        ),
        PatternRule(
            subject_type=EntityType.POLICY,
            object_type=EntityType.LEAVE_TYPE,
            connector_pattern=re.compile(r"\bgoverns?\b", re.IGNORECASE),
            relation_type=RelationType.GOVERNS,
        ),
        PatternRule(
            subject_type=EntityType.EMPLOYEE,
            object_type=EntityType.PROJECT,
            connector_pattern=re.compile(r"\b(is\s+)?assigned\s+to(\s+the)?\b", re.IGNORECASE),
            relation_type=RelationType.ASSIGNED_TO,
        ),
        PatternRule(
            subject_type=EntityType.EMPLOYEE,
            object_type=EntityType.ROLE,
            connector_pattern=re.compile(
                r"\b(is\s+a|works?\s+as(\s+an?)?|has\s+the\s+role\s+of)\b", re.IGNORECASE
            ),
            relation_type=RelationType.HAS_ROLE,
        ),
        PatternRule(
            subject_type=EntityType.ROLE,
            object_type=EntityType.NETWORK_RESOURCE,
            connector_pattern=re.compile(r"\brequires?\s+access\s+to\b", re.IGNORECASE),
            relation_type=RelationType.REQUIRES_ACCESS,
        ),
        PatternRule(
            subject_type=EntityType.EMPLOYEE,
            object_type=EntityType.LEAVE_TYPE,
            connector_pattern=re.compile(r"\b(is\s+|are\s+)?eligible\s+for\b", re.IGNORECASE),
            relation_type=RelationType.ELIGIBLE_FOR,
        ),
        PatternRule(
            subject_type=EntityType.DEPARTMENT,
            object_type=EntityType.SOFTWARE,
            connector_pattern=re.compile(
                r"\b(must\s+complete|requires?(\s+completion\s+of)?|need(s)?\s+to\s+complete)\b",
                re.IGNORECASE,
            ),
            relation_type=RelationType.REQUIRES_TRAINING,
        ),
        PatternRule(
            subject_type=EntityType.ROLE,
            object_type=EntityType.SOFTWARE,
            connector_pattern=re.compile(
                r"\b(must\s+complete|requires?(\s+completion\s+of)?|need(s)?\s+to\s+complete)\b",
                re.IGNORECASE,
            ),
            relation_type=RelationType.REQUIRES_TRAINING,
        ),
    )


class RulePatternExtractor:
    """
    Applies a configurable set of PatternRules to extract relations
    from co-occurring entity pairs via connector-phrase regex matching
    -- fast, fully explainable, zero training data, but brittle to
    phrasing not covered by the pattern list (Section 3H/7.2a's
    documented tradeoff for rule-based baselines throughout this
    project).
    """

    def __init__(
        self,
        rules: Optional[tuple[PatternRule, ...]] = None,
        max_connector_length: int = 60,
    ) -> None:
        """
        Args:
            rules: The PatternRules to apply. Defaults to
                _default_rules() if not provided.
            max_connector_length: Maximum character length of the text
                strictly between two entities for them to be
                considered a candidate pair at all -- keeps the
                extractor from matching entities separated by
                unrelated intervening clauses.

        Raises:
            ValueError: If max_connector_length is not positive.
        """
        if max_connector_length <= 0:
            raise ValueError(
                f"max_connector_length must be positive, got {max_connector_length}"
            )
        self._rules = rules if rules is not None else _default_rules()
        self._max_connector_length = max_connector_length

    def apply_rule_patterns(
        self, text: str, entities: list[Entity], source_chunk_id: str
    ) -> list[Relation]:
        """
        Scan every textually-ordered pair of entities in `text` for a
        matching connector-phrase pattern.

        Args:
            text: The source text the entities were extracted from
                (typically a Chunk's text).
            entities: Entities to consider as candidate subjects/objects.
            source_chunk_id: Identifier attached to every produced
                Relation's source_chunk_id. Must be non-empty whenever
                entities is non-empty -- Relation itself requires this
                for traceability.

        Returns:
            A list of Relation objects, one per (pair, matching rule)
            combination -- a single pair can match more than one rule
            only if multiple rules share the same
            (subject_type, object_type), which the default rule set
            avoids, but a custom rule set could produce.
        """
        if not text or not entities:
            return []
        if not source_chunk_id:
            raise ValueError(
                "apply_rule_patterns() requires a non-empty source_chunk_id "
                "when entities is non-empty."
            )

        ordered_entities = sorted(entities, key=lambda entity: entity.start_offset)
        relations: list[Relation] = []

        for i, earlier in enumerate(ordered_entities):
            for later in ordered_entities[i + 1 :]:
                if later.start_offset < earlier.end_offset:
                    continue  # overlapping spans; not a valid pair

                connector = text[earlier.end_offset : later.start_offset]
                if len(connector) > self._max_connector_length:
                    continue

                for rule in self._rules:
                    if (
                        earlier.entity_type == rule.subject_type
                        and later.entity_type == rule.object_type
                        and rule.connector_pattern.search(connector)
                    ):
                        if rule.reverse_direction and any(
                            candidate.entity_type == EntityType.EMPLOYEE
                            for candidate in ordered_entities[i + 1 :]
                            if candidate.start_offset < later.start_offset
                        ):
                            # Apply reporting-line language only to the nearest
                            # employee pair; otherwise one phrase cross-links a
                            # whole paragraph of employees.
                            continue
                        try:
                            subject_entity = later if rule.reverse_direction else earlier
                            object_entity = earlier if rule.reverse_direction else later
                            relations.append(
                                Relation(
                                    subject_entity=subject_entity,
                                    predicate=rule.relation_type,
                                    object_entity=object_entity,
                                    source_chunk_id=source_chunk_id,
                                )
                            )
                        except SchemaValidationError as exc:
                            logger.warning(
                                "Skipping rule-matched pair (%r, %r): %s",
                                earlier.text,
                                later.text,
                                exc,
                            )

        logger.debug(
            "apply_rule_patterns found %d relation(s) via %d rule(s)",
            len(relations),
            len(self._rules),
        )
        return relations


def apply_rule_patterns(
    text: str,
    entities: list[Entity],
    source_chunk_id: str = "",
    rules: Optional[tuple[PatternRule, ...]] = None,
) -> list[Relation]:
    """
    Module-level convenience wrapper around
    RulePatternExtractor.apply_rule_patterns(), constructing a
    default-config extractor for one-off use. Callers processing many
    chunks should instantiate RulePatternExtractor directly instead,
    to avoid rebuilding the compiled rule set per call.

    Args:
        text: Source text.
        entities: Candidate entities.
        source_chunk_id: Identifier attached to every produced
            Relation. Must be non-empty whenever entities is
            non-empty. Defaults to "" only to keep this convenience
            wrapper's signature ergonomic for the (text="", empty
            entities) fast-return path; RulePatternExtractor itself
            enforces the non-empty requirement whenever it matters.
        rules: Optional custom rule set, passed through to
            RulePatternExtractor.

    Returns:
        A list of Relation objects.
    """
    return RulePatternExtractor(rules=rules).apply_rule_patterns(
        text, entities, source_chunk_id
    )
