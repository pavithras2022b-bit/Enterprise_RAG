"""
Module 5 - Relationship Extraction: relation prediction.

Provides predict_relation(...) and the RelationClassifier class per
the Phase 4 module contract, wrapping a fine-tuned transformer
sequence-classification checkpoint that uses the entity-marker
technique (Section 3H): special tokens are inserted around the
subject and object entity spans so the model can tell which marked
span plays which role, independent of their order in the text.

Reference: Capstone Project Guide, Section 3H ("Relationship
Extraction"), Phase 4 Module 5 contract, Phase 5 file structure
(relation_extraction/model.py).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ner.schema import Entity
from relation_extraction.schema import NO_RELATION_LABEL, Relation, RelationType

if TYPE_CHECKING:
    from chunking.schema import Chunk
    from transformers import PreTrainedModel, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

# Entity-marker special tokens (Section 3H: "entity spans marked, e.g.
# via special tokens inserted around each entity"). SUBJ/OBJ markers
# (rather than generic E1/E2) let the model learn subject vs. object
# role directly from which marker wraps which span, regardless of
# their left-to-right order in the source text.
SUBJECT_START_TOKEN = "[SUBJ]"
SUBJECT_END_TOKEN = "[/SUBJ]"
OBJECT_START_TOKEN = "[OBJ]"
OBJECT_END_TOKEN = "[/OBJ]"
MARKER_TOKENS = (SUBJECT_START_TOKEN, SUBJECT_END_TOKEN, OBJECT_START_TOKEN, OBJECT_END_TOKEN)

_WHITESPACE_RUN = re.compile(r" {2,}")


class RelationModelError(Exception):
    """Raised when RelationClassifier cannot load its checkpoint or run inference."""


def build_relation_label_maps() -> tuple[dict[int, str], dict[str, int]]:
    """
    Builds the classification head's label vocabulary: NO_RELATION
    plus every RelationType value, in a fixed, deterministic order.
    Shared by RelationClassifier and train_re() so both always agree
    on the same id<->label mapping.

    Returns:
        (id2label, label2id) -- inverse mappings covering NO_RELATION
        and all closed-schema relation types.
    """
    labels = [NO_RELATION_LABEL] + [relation_type.value for relation_type in RelationType]
    id2label = {index: label for index, label in enumerate(labels)}
    label2id = {label: index for index, label in id2label.items()}
    return id2label, label2id


def insert_entity_markers(
    text: str, subject_span: tuple[int, int], object_span: tuple[int, int]
) -> str:
    """
    Builds entity-marker-annotated text from raw text plus two
    character spans, e.g. "Asha Rao belongs to Finance" with
    subject_span covering "Asha Rao" and object_span covering
    "Finance" becomes "[SUBJ] Asha Rao [/SUBJ] belongs to [OBJ]
    Finance [/OBJ]".

    Args:
        text: Raw source text.
        subject_span: (start, end) character offsets of the subject
            entity mention.
        object_span: (start, end) character offsets of the object
            entity mention.

    Returns:
        Marker-annotated text, with excess whitespace from marker
        insertion collapsed.

    Raises:
        ValueError: If either span is out of bounds for text, or the
            two spans overlap.
    """
    for name, span in (("subject_span", subject_span), ("object_span", object_span)):
        start, end = span
        if start < 0 or end <= start or end > len(text):
            raise ValueError(f"{name} {span} is invalid for text of length {len(text)}.")

    subject_start, subject_end = subject_span
    object_start, object_end = object_span
    if subject_start < object_end and object_start < subject_end:
        raise ValueError(
            f"subject_span {subject_span} and object_span {object_span} overlap."
        )

    insertions = sorted(
        [
            (subject_start, SUBJECT_START_TOKEN),
            (subject_end, SUBJECT_END_TOKEN),
            (object_start, OBJECT_START_TOKEN),
            (object_end, OBJECT_END_TOKEN),
        ],
        key=lambda item: item[0],
        reverse=True,
    )

    marked = text
    for position, token in insertions:
        marked = f"{marked[:position]} {token} {marked[position:]}"
    return _WHITESPACE_RUN.sub(" ", marked).strip()


class RelationClassifier:
    """
    Wraps a fine-tuned transformer sequence-classification checkpoint
    for enterprise relation extraction, using the entity-marker
    technique. The primary, learned counterpart to rules.py's
    zero-training-data baseline (Section 3H's documented pairing).

    Construct via the constructor for a checkpoint on disk or the Hub,
    or via from_components() to inject an already-loaded
    tokenizer/model (used by train_re() and by tests).
    """

    def __init__(self, model_name_or_path: str, device: str = "cpu") -> None:
        """
        Args:
            model_name_or_path: A local checkpoint directory (e.g. the
                output_dir of a prior train_re() run) or a Hugging
                Face Hub model id.
            device: torch device string ("cpu" or "cuda").

        Raises:
            RelationModelError: If the tokenizer/model cannot be
                loaded.
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        except Exception as exc:  # noqa: BLE001
            raise RelationModelError(
                f"Failed to load tokenizer from {model_name_or_path!r}: {exc}"
            ) from exc

        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        except Exception as exc:  # noqa: BLE001
            raise RelationModelError(
                f"Failed to load sequence-classification model from "
                f"{model_name_or_path!r}: {exc}"
            ) from exc

        self._init_from_components(tokenizer, model, device)

    @classmethod
    def from_components(
        cls,
        tokenizer: "PreTrainedTokenizerFast",
        model: "PreTrainedModel",
        device: str = "cpu",
    ) -> "RelationClassifier":
        """
        Construct a RelationClassifier from an already-loaded
        tokenizer/model pair, bypassing from_pretrained(). Used by
        train_re() and by tests using a small local checkpoint.

        Args:
            tokenizer: A tokenizer (marker tokens will be added if not
                already present).
            model: A sequence-classification model with a populated
                config.id2label.
            device: torch device string ("cpu" or "cuda").

        Returns:
            A ready-to-use RelationClassifier.
        """
        instance = cls.__new__(cls)
        instance._init_from_components(tokenizer, model, device)
        return instance

    def _init_from_components(
        self,
        tokenizer: "PreTrainedTokenizerFast",
        model: "PreTrainedModel",
        device: str,
    ) -> None:
        added = tokenizer.add_special_tokens(
            {"additional_special_tokens": list(MARKER_TOKENS)}
        )
        if added:
            model.resize_token_embeddings(len(tokenizer))
            logger.debug("Added %d entity-marker special token(s) to tokenizer vocab.", added)

        self._tokenizer = tokenizer
        self._model = model
        self._device = torch.device(device)
        self._model.to(self._device)
        self._model.eval()

    @property
    def id2label(self) -> dict[int, str]:
        """Returns the model's label-id-to-relation-label mapping."""
        return dict(self._model.config.id2label)

    def predict_relation(
        self,
        subject_entity: Entity,
        object_entity: Entity,
        context: str,
        source_chunk_id: str,
        confidence_threshold: float = 0.5,
        max_length: int = 256,
    ) -> Optional[Relation]:
        """
        Predict whether -- and what type of -- relationship holds
        between two entities co-occurring in the same context.

        Args:
            subject_entity: Candidate subject entity. Its
                start_offset/end_offset must be valid character
                offsets into `context`.
            object_entity: Candidate object entity, likewise offset
                into `context`.
            context: The sentence/window text both entities were
                extracted from (typically a Chunk's text).
            source_chunk_id: Identifier attached to the produced
                Relation's source_chunk_id, if one is returned. Must
                be non-empty -- Relation itself requires this for
                traceability, so an empty value is rejected here with
                a clear error rather than surfacing as a confusing
                SchemaValidationError from deep inside Relation's
                construction.
            confidence_threshold: Minimum softmax probability required
                to return a relation; below this (or if the predicted
                label is NO_RELATION), returns None.
            max_length: Maximum token sequence length for the
                marker-annotated input.

        Returns:
            A Relation if the classifier predicts a relation type
            above confidence_threshold, otherwise None.

        Raises:
            ValueError: If confidence_threshold is not in [0, 1], or
                source_chunk_id is empty, or either entity's offsets
                are invalid for context, or the two entities' spans
                overlap.
        """
        if not source_chunk_id:
            raise ValueError("predict_relation() requires a non-empty source_chunk_id.")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {confidence_threshold}"
            )

        marked_text = insert_entity_markers(
            context,
            (subject_entity.start_offset, subject_entity.end_offset),
            (object_entity.start_offset, object_entity.end_offset),
        )

        encoding = self._tokenizer(
            marked_text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoding = {key: value.to(self._device) for key, value in encoding.items()}

        with torch.no_grad():
            outputs = self._model(**encoding)

        probabilities = torch.softmax(outputs.logits[0], dim=-1)
        confidence, predicted_id = torch.max(probabilities, dim=-1)
        predicted_label = self.id2label.get(int(predicted_id.item()), NO_RELATION_LABEL)
        confidence_value = float(confidence.item())

        if predicted_label == NO_RELATION_LABEL or confidence_value < confidence_threshold:
            return None

        try:
            predicate = RelationType(predicted_label)
        except ValueError:
            logger.warning(
                "Model predicted unknown relation label %r; treating as no relation.",
                predicted_label,
            )
            return None

        return Relation(
            subject_entity=subject_entity,
            predicate=predicate,
            object_entity=object_entity,
            source_chunk_id=source_chunk_id,
            confidence=confidence_value,
        )


def predict_relation(
    model: RelationClassifier,
    subject_entity: Entity,
    object_entity: Entity,
    context: str,
    source_chunk_id: str,
    confidence_threshold: float = 0.5,
) -> Optional[Relation]:
    """
    Module-level convenience wrapper around
    RelationClassifier.predict_relation(), matching the Phase 4 module
    contract's function signature (predict_relation(entity1, entity2,
    context) -> Relation | None). The model is threaded through
    explicitly, mirroring Module 4's predict_entities() convention:
    callers processing many pairs should call
    model.predict_relation() directly in a loop instead.

    Args:
        model: A loaded RelationClassifier.
        subject_entity: Candidate subject entity.
        object_entity: Candidate object entity.
        context: Shared source text both entities are offset into.
        source_chunk_id: Identifier attached to the produced Relation.
            Must be non-empty.
        confidence_threshold: Minimum confidence to return a relation.

    Returns:
        A Relation, or None if no relation was predicted with
        sufficient confidence.
    """
    return model.predict_relation(
        subject_entity,
        object_entity,
        context,
        source_chunk_id,
        confidence_threshold=confidence_threshold,
    )


def predict_relations_for_chunk(
    model: RelationClassifier,
    chunk: "Chunk",
    entities: list[Entity],
    confidence_threshold: float = 0.5,
    max_pair_distance: Optional[int] = 200,
) -> list[Relation]:
    """
    Run relation extraction over every ordered pair of entities
    co-occurring within a single Chunk, using its chunk_id as
    source_chunk_id automatically -- the integration point Phase 6's
    orchestration layer will actually call.

    Both orderings of every unordered pair are tried (entity A as
    subject with B as object, and vice versa), since a relation's
    direction cannot be assumed from text position alone (e.g. "SAP FI
    is used by the Payments project" has the object appearing first).

    Args:
        model: A loaded RelationClassifier.
        chunk: A Chunk from Module 3, providing the shared context
            text.
        entities: Entities extracted from this chunk (Module 4's
            output, filtered to this chunk's chunk_id by the caller).
        confidence_threshold: Minimum confidence to keep a predicted
            relation.
        max_pair_distance: If set, entity pairs whose spans are more
            than this many characters apart are skipped, both as a
            precision guard (distant entities are rarely directly
            related) and an efficiency guard (avoids O(n^2) inference
            calls on chunks with many entities). None disables the
            limit.

    Returns:
        A list of Relation objects with source_chunk_id ==
        chunk.chunk_id.
    """
    relations: list[Relation] = []
    for i, entity_a in enumerate(entities):
        for entity_b in entities[i + 1 :]:
            if entity_a.start_offset <= entity_b.start_offset:
                first, second = entity_a, entity_b
            else:
                first, second = entity_b, entity_a

            if max_pair_distance is not None:
                distance = second.start_offset - first.end_offset
                if distance > max_pair_distance:
                    continue

            for subject_entity, object_entity in ((first, second), (second, first)):
                relation = model.predict_relation(
                    subject_entity,
                    object_entity,
                    chunk.text,
                    source_chunk_id=chunk.chunk_id,
                    confidence_threshold=confidence_threshold,
                )
                if relation is not None:
                    relations.append(relation)

    logger.debug(
        "predict_relations_for_chunk found %d relation(s) in chunk %s",
        len(relations),
        chunk.chunk_id,
    )
    return relations
