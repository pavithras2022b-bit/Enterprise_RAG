"""
Module 4 - NER: entity prediction.

Provides predict_entities(...) and the NERModel class per the Phase 4
module contract, wrapping a fine-tuned transformer token-classification
checkpoint (Section 12.3: bert-base-cased or a spaCy transformer,
fine-tuned per Section 8.4) and decoding its BIO-tagged output into
character-offset Entity spans.

Reference: Capstone Project Guide, Section 3G ("Entity Extraction"),
Phase 4 Module 4 contract, Phase 5 file structure (ner/model.py).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from ner.schema import Entity, EntityType

if TYPE_CHECKING:
    from chunking.schema import Chunk
    from transformers import PreTrainedModel, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class NERModelError(Exception):
    """Raised when NERModel cannot load its checkpoint or run inference."""


def _decode_bio_spans(
    text: str,
    labels: list[str],
    offsets: list[tuple[int, int]],
    confidences: list[float],
    source_chunk_id: str,
    confidence_threshold: float,
    entity_type_mapper: Optional[Callable[[str, str, str], Optional[EntityType]]] = None,
) -> list[Entity]:
    """
    Decode per-token BIO labels + character offsets into merged Entity
    spans, per the BIO scheme described in Section 3G ("B-eginning,
    I-nside, O-utside").

    A "B-{type}" token starts a new span. A following "I-{type}" token
    (matching type) extends the current span. Any other transition
    (a different type, an "O", or an "I-" with no matching open span)
    closes the current span first. Special tokens are identified by a
    zero-width offset (start == end) and skipped, since PreTrainedTokenizer
    fast tokenizers report ([CLS], [SEP], padding) offsets as (0, 0).

    Args:
        text: The original text the labels/offsets were predicted over.
        labels: Predicted BIO label string per token, aligned with
            offsets/confidences.
        offsets: (start, end) character offset per token.
        confidences: Model confidence (max softmax probability) per
            token.
        source_chunk_id: Chunk.chunk_id to attach to every produced
            Entity.
        confidence_threshold: Minimum average span confidence required
            to keep a span; spans below this are dropped, not raised
            as an error, since low-confidence predictions are expected
            and routine model output.

    Returns:
        A list of Entity objects, one per closed, kept span, in
        left-to-right order.
    """
    entities: list[Entity] = []
    current_type: Optional[str] = None
    current_start: Optional[int] = None
    current_end: Optional[int] = None
    current_confidences: list[float] = []

    def flush() -> None:
        nonlocal current_type, current_start, current_end, current_confidences
        if current_type is not None and current_start is not None and current_end is not None:
            average_confidence = (
                sum(current_confidences) / len(current_confidences)
                if current_confidences
                else 0.0
            )
            if average_confidence >= confidence_threshold:
                span_text = text[current_start:current_end]
                try:
                    entity_type = EntityType(current_type)
                except ValueError:
                    entity_type = (
                        entity_type_mapper(current_type, span_text, text)
                        if entity_type_mapper is not None
                        else None
                    )
                    if entity_type is None:
                        logger.warning(
                            "Decoded BIO label references unknown entity type "
                            "%r; skipping span.", current_type,
                        )
                if entity_type is not None:
                    if span_text.strip():
                        entities.append(
                            Entity(
                                text=span_text,
                                entity_type=entity_type,
                                start_offset=current_start,
                                end_offset=current_end,
                                source_chunk_id=source_chunk_id,
                                confidence=average_confidence,
                            )
                        )
        current_type = None
        current_start = None
        current_end = None
        current_confidences = []

    for label, (start, end), confidence in zip(labels, offsets, confidences):
        if start == end:
            continue  # special token (CLS/SEP/PAD) -- has no text span
        if label == "O":
            flush()
            continue

        prefix, _, entity_type_str = label.partition("-")
        if prefix == "B" or (prefix == "I" and entity_type_str != current_type):
            # A "B-" always starts fresh; a stray "I-" with no matching
            # open span (or a type change) is treated as starting a new
            # span too, for robustness against imperfect model output.
            flush()
            current_type = entity_type_str
            current_start = start
            current_end = end
            current_confidences = [confidence]
        else:  # prefix == "I" and entity_type_str == current_type
            current_end = end
            current_confidences.append(confidence)

    flush()
    return entities


class NERModel:
    """
    Wraps a fine-tuned transformer token-classification checkpoint
    (bert-base-cased-family, per Section 12.3) for enterprise entity
    extraction, and decodes its output into character-offset Entity
    spans.

    Construct via the constructor for a checkpoint on disk or the Hub
    (requires network access to the Hub for non-local paths), or via
    from_components() to inject an already-loaded tokenizer/model
    (used for testing, or when the caller manages loading itself).
    """

    def __init__(self, model_name_or_path: str, device: str = "cpu") -> None:
        """
        Args:
            model_name_or_path: A local checkpoint directory (e.g. the
                output_dir of a prior train_ner() run) or a Hugging
                Face Hub model id.
            device: torch device string ("cpu" or "cuda").

        Raises:
            NERModelError: If the tokenizer/model cannot be loaded, or
                the loaded tokenizer is not a "fast" tokenizer (fast
                tokenizers are required for offset-mapping support,
                which BIO-span decoding depends on).
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        except Exception as exc:  # noqa: BLE001
            raise NERModelError(
                f"Failed to load tokenizer from {model_name_or_path!r}: {exc}"
            ) from exc

        try:
            model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
        except Exception as exc:  # noqa: BLE001
            raise NERModelError(
                f"Failed to load token-classification model from "
                f"{model_name_or_path!r}: {exc}"
            ) from exc

        self._init_from_components(tokenizer, model, device)

    @classmethod
    def from_components(
        cls,
        tokenizer: "PreTrainedTokenizerFast",
        model: "PreTrainedModel",
        device: str = "cpu",
    ) -> "NERModel":
        """
        Construct an NERModel from an already-loaded tokenizer/model
        pair, bypassing from_pretrained(). Used by train_ner() (which
        already holds the freshly fine-tuned objects in memory) and by
        tests that build a small local checkpoint without network
        access.

        Args:
            tokenizer: A fast tokenizer supporting offset mapping.
            model: A token-classification model with a populated
                config.id2label.
            device: torch device string ("cpu" or "cuda").

        Returns:
            A ready-to-use NERModel.

        Raises:
            NERModelError: If tokenizer is not a fast tokenizer.
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
        if not getattr(tokenizer, "is_fast", False):
            raise NERModelError(
                "NERModel requires a fast tokenizer for offset-mapping "
                "support; got a non-fast tokenizer."
            )
        self._tokenizer = tokenizer
        self._model = model
        self._device = torch.device(device)
        self._model.to(self._device)
        self._model.eval()

    @property
    def id2label(self) -> dict[int, str]:
        """Returns the model's label-id-to-BIO-tag-string mapping."""
        return dict(self._model.config.id2label)

    def predict_entities(
        self,
        text: str,
        source_chunk_id: str = "",
        confidence_threshold: float = 0.5,
        max_length: int = 512,
        entity_type_mapper: Optional[Callable[[str, str, str], Optional[EntityType]]] = None,
    ) -> list[Entity]:
        """
        Run entity extraction over a single piece of text.

        Args:
            text: Text to extract entities from (typically a Chunk's
                text, per Module 3's output).
            source_chunk_id: Identifier attached to every produced
                Entity's source_chunk_id field.
            confidence_threshold: Minimum average span confidence to
                keep a predicted span.
            max_length: Maximum token sequence length; longer inputs
                are truncated (a chunk from Module 3 should already be
                well under this, per its 200-500-token sizing target).

        Returns:
            A list of Entity objects with resolved character offsets
            into text, in left-to-right order. Empty/whitespace-only
            text returns an empty list without running the model.

        Raises:
            ValueError: If confidence_threshold is not in [0, 1].
        """
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {confidence_threshold}"
            )
        if not text or not text.strip():
            return []

        encoding = self._tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        offset_mapping = encoding.pop("offset_mapping")[0].tolist()
        encoding = {key: value.to(self._device) for key, value in encoding.items()}

        with torch.no_grad():
            outputs = self._model(**encoding)

        logits = outputs.logits[0]
        probabilities = torch.softmax(logits, dim=-1)
        token_confidences, predicted_ids = torch.max(probabilities, dim=-1)

        id2label = self.id2label
        predicted_labels = [id2label.get(i, "O") for i in predicted_ids.tolist()]
        offsets: list[tuple[int, int]] = [tuple(pair) for pair in offset_mapping]

        entities = _decode_bio_spans(
            text=text,
            labels=predicted_labels,
            offsets=offsets,
            confidences=token_confidences.tolist(),
            source_chunk_id=source_chunk_id,
            confidence_threshold=confidence_threshold,
            entity_type_mapper=entity_type_mapper,
        )
        logger.debug(
            "predict_entities found %d entit(y/ies) in chunk %s",
            len(entities),
            source_chunk_id,
        )
        return entities


def predict_entities(
    model: NERModel,
    text: str,
    source_chunk_id: str = "",
    confidence_threshold: float = 0.5,
) -> list[Entity]:
    """
    Module-level convenience wrapper around NERModel.predict_entities(),
    matching the Phase 4 module contract's function signature
    (predict_entities(text) -> list[Entity]). NERModel is threaded
    through explicitly since (unlike a stateless function) entity
    prediction requires a loaded model; callers processing many chunks
    should call model.predict_entities() directly in a loop instead of
    this wrapper, to make that dependency explicit.

    Args:
        model: A loaded NERModel.
        text: Text to extract entities from.
        source_chunk_id: Identifier attached to every produced Entity.
        confidence_threshold: Minimum average span confidence to keep
            a predicted span.

    Returns:
        A list of Entity objects.
    """
    return model.predict_entities(
        text, source_chunk_id=source_chunk_id, confidence_threshold=confidence_threshold
    )


def predict_entities_for_chunk(
    model: NERModel,
    chunk: "Chunk",
    confidence_threshold: float = 0.5,
) -> list[Entity]:
    """
    Run entity extraction over a Module 3 Chunk, using its chunk_id as
    source_chunk_id automatically -- the integration point Phase 6's
    orchestration layer will actually call.

    Args:
        model: A loaded NERModel.
        chunk: A Chunk from Module 3.
        confidence_threshold: Minimum average span confidence to keep
            a predicted span.

    Returns:
        A list of Entity objects with source_chunk_id == chunk.chunk_id.
    """
    return model.predict_entities(
        chunk.text,
        source_chunk_id=chunk.chunk_id,
        confidence_threshold=confidence_threshold,
    )
