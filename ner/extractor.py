"""
Module 4 - NER: Entity Extraction wrapper.

Provides the EntityExtractor class that wraps NERModel for use in the pipeline.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Optional

from ner.model import NERModel
from ner.schema import Entity, EntityType

if TYPE_CHECKING:
    from chunking.schema import Chunk

logger = logging.getLogger(__name__)


def map_generic_label_to_enterprise_type(
    generic_label: str, span_text: str, context: str
) -> Optional[EntityType]:
    """Heuristically map CoNLL labels into this project's graph schema.

    This compatibility baseline is not a substitute for a checkpoint trained
    with train_ner() on enterprise BIO annotations.
    """
    label = generic_label.upper()
    if label == "PER":
        return EntityType.EMPLOYEE
    if label == "LOC":
        # The graph schema has no physical-location label.
        return EntityType.NETWORK_RESOURCE

    mention = span_text.lower().strip()

    # Enterprise-specific overrides for the generic CoNLL NER checkpoint.
    if re.search(r"\b(project|initiative|programme|program)\b", mention):
        return EntityType.PROJECT

    known_software = {
        "servicenow",
        "coupa",
        "workday",
        "workday hrms",
        "sap concur",
        "salesforce",
        "microsoft 365",
        "confluence",
        "jira",
        "okta sso",
        "bitwarden enterprise",
        "greythr",
        "greythr self-service portal",
        "meridian vpn gateway",
    }

    if mention in known_software:
        return EntityType.SOFTWARE

    # Use the mention plus a local window, rather than the entire chunk. A
    # policy chunk often mentions projects, policies, and leave in separate
    # sentences; whole-chunk keyword matching would misclassify every MISC
    # mention according to whichever term happened to occur first.
    span_start = context.lower().find(span_text.lower())
    local_context = (
        context[max(0, span_start - 120) : span_start + len(span_text) + 120]
        if span_start >= 0
        else context
    )
    mention_evidence = span_text.lower()
    evidence = f"{mention_evidence} {local_context.lower()}"
    if label == "ORG":
        if re.search(r"\b(software|system|application|platform|portal|tool)\b", evidence):
            return EntityType.SOFTWARE
        return EntityType.DEPARTMENT
    if label == "MISC":
        if re.search(r"\b(leave|vacation|sabbatical|parental)\b", mention_evidence):
            return EntityType.LEAVE_TYPE
        if re.search(r"\b(policy|procedure|guideline|standard|plan)\b", mention_evidence):
            return EntityType.POLICY
        if re.search(r"\b(project|initiative|programme|program)\b", mention_evidence):
            return EntityType.PROJECT
        if re.search(r"\b(software|system|application|platform|portal|sap|coupa|jira)\b", mention_evidence):
            return EntityType.SOFTWARE
        if re.search(r"\b(manager|director|officer|engineer|analyst|counsel|partner)\b", mention_evidence):
            return EntityType.ROLE
        escaped_mention = re.escape(span_text.strip())
        if escaped_mention and re.search(
            rf"\b(uses?|maintains?|accesses?)\s+{escaped_mention}\b", local_context, re.IGNORECASE
        ):
            return EntityType.SOFTWARE
        if escaped_mention and re.search(
            rf"\b{escaped_mention}\b\s+(?:is|was)\s+(?:the\s+)?(?:project|initiative)",
            local_context,
            re.IGNORECASE,
        ):
            return EntityType.PROJECT
        return EntityType.POLICY
    return None


def _merge_adjacent_entity_fragments(entities: list[Entity], text: str) -> list[Entity]:
    """Repair adjacent same-type fragments emitted by the generic CoNLL model."""
    merged: list[Entity] = []
    for entity in entities:
        if (
            merged
            and merged[-1].entity_type == entity.entity_type
            and merged[-1].source_chunk_id == entity.source_chunk_id
            and not text[merged[-1].end_offset : entity.start_offset].strip()
        ):
            previous = merged.pop()
            total_length = previous.span_length + entity.span_length
            confidence = None
            if previous.confidence is not None and entity.confidence is not None:
                confidence = (
                    previous.confidence * previous.span_length
                    + entity.confidence * entity.span_length
                ) / total_length
            merged_text = re.sub(r"\s+", " ", text[previous.start_offset : entity.end_offset]).strip()
            merged.append(Entity(
                text=merged_text,
                entity_type=previous.entity_type,
                start_offset=previous.start_offset,
                end_offset=entity.end_offset,
                source_chunk_id=previous.source_chunk_id,
                confidence=confidence,
            ))
        else:
            merged.append(entity)
    return merged


def _filter_obvious_generic_ner_noise(entities: list[Entity]) -> list[Entity]:
    """Remove only clearly malformed fragments produced by the generic NER model."""
    filtered: list[Entity] = []

    for entity in entities:
        name = re.sub(r"\s+", " ", entity.text).strip()

        if not name:
            continue

        # Single-character fragments are never useful enterprise entities.
        if len(name) == 1:
            continue

        # Very short alphabetic fragments are usually tokenizer/NER truncations.
        # Keep common legitimate enterprise abbreviations.
        safe_short_names = {
            "HR",
            "IT",
            "AI",
            "R&D",
            "AWS",
            "SAP",
            "USA",
            "CISO",
            "HRMS",
        }

        if (
            len(name) <= 4
            and name not in safe_short_names
            and name.isalpha()
        ):
            continue

        filtered.append(entity)

    return filtered


class EntityExtractor:
    """
    Wraps an NERModel for entity extraction in the enterprise RAG pipeline.
    Provides a consistent interface for extracting entities from chunks.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cpu",
        use_generic_label_mapping: bool = False,
    ) -> None:
        """
        Initialize the EntityExtractor with a loaded NERModel.

        Args:
            model_name_or_path: A local checkpoint directory or Hugging Face Hub model id.
            device: torch device string ("cpu" or "cuda").
        """
        self.model = NERModel(model_name_or_path, device=device)
        self._entity_type_mapper = (
            map_generic_label_to_enterprise_type if use_generic_label_mapping else None
        )
        if use_generic_label_mapping:
            logger.warning(
                "Generic NER label mapping is enabled. PER/ORG/LOC/MISC are "
                "heuristically mapped; use a domain-trained train_ner() "
                "checkpoint for production-quality extraction."
            )

    def extract_entities(
        self,
        text: str,
        source_chunk_id: str = "",
        confidence_threshold: float = 0.5,
    ) -> list[Entity]:
        """
        Extract entities from text.

        Args:
            text: Text to extract entities from.
            source_chunk_id: Identifier attached to every produced Entity.
            confidence_threshold: Minimum average span confidence to keep a predicted span.

        Returns:
            A list of Entity objects.
        """
        entities = self.model.predict_entities(
            text,
            source_chunk_id=source_chunk_id,
            confidence_threshold=confidence_threshold,
            entity_type_mapper=self._entity_type_mapper,
        )
        if self._entity_type_mapper:
            entities = _merge_adjacent_entity_fragments(entities, text)
            entities = _filter_obvious_generic_ner_noise(entities)
        return entities

    def extract_entities_from_chunk(
        self,
        chunk: "Chunk",
        confidence_threshold: float = 0.5,
    ) -> list[Entity]:
        """
        Extract entities from a Module 3 Chunk.

        Args:
            chunk: A Chunk from Module 3.
            confidence_threshold: Minimum average span confidence to keep a predicted span.

        Returns:
            A list of Entity objects with source_chunk_id == chunk.chunk_id.
        """
        entities = self.model.predict_entities(
            chunk.text,
            source_chunk_id=chunk.chunk_id,
            confidence_threshold=confidence_threshold,
            entity_type_mapper=self._entity_type_mapper,
        )
        if self._entity_type_mapper:
            entities = _merge_adjacent_entity_fragments(entities, chunk.text)
            entities = _filter_obvious_generic_ner_noise(entities)
        return entities
