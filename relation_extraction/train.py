"""
Module 5 - Relationship Extraction: fine-tuning entrypoint.

Provides train_re(...) per the Phase 4 module contract: fine-tunes a
pretrained transformer checkpoint's sequence-classification head on
entity-marker-annotated, labeled relation examples.

Reference: Capstone Project Guide, Section 3H ("Relationship
Extraction"), Section 8 (transfer-learning treatment applied
throughout the pipeline), Phase 4 Module 5 contract, Phase 5 file
structure (relation_extraction/train.py).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from relation_extraction.model import (
    MARKER_TOKENS,
    RelationClassifier,
    build_relation_label_maps,
    insert_entity_markers,
)
from relation_extraction.schema import LabeledRelationExample

logger = logging.getLogger(__name__)


class RelationTrainingError(Exception):
    """Raised when relation-classifier fine-tuning cannot proceed for the given input."""


class _SequenceClassificationDataset(torch.utils.data.Dataset):
    """
    Wraps a list of LabeledRelationExample into a torch Dataset of
    tokenizer-encoded, entity-marker-annotated examples with a single
    classification label each.
    """

    def __init__(
        self,
        examples: list[LabeledRelationExample],
        tokenizer: "AutoTokenizer",
        label2id: dict[str, int],
        max_length: int = 256,
    ) -> None:
        self._encodings: list[dict[str, torch.Tensor]] = []
        self._labels: list[int] = []

        for example in examples:
            marked_text = insert_entity_markers(
                example.text, example.subject_span, example.object_span
            )
            encoding = tokenizer(
                marked_text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            self._encodings.append(
                {key: torch.tensor(value) for key, value in encoding.items()}
            )
            self._labels.append(label2id[example.label])

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = dict(self._encodings[index])
        item["labels"] = torch.tensor(self._labels[index])
        return item


def _compute_sequence_classification_metrics(eval_prediction: EvalPrediction) -> dict[str, float]:
    """Computes simple example-level accuracy as a training-time sanity check."""
    predictions = np.argmax(eval_prediction.predictions, axis=-1)
    labels = eval_prediction.label_ids
    accuracy = float((predictions == labels).mean()) if len(labels) else 0.0
    return {"accuracy": accuracy}


def train_re(
    labeled_data: list[LabeledRelationExample],
    base_checkpoint: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    validation_split: float = 0.2,
    seed: int = 42,
) -> RelationClassifier:
    """
    Fine-tune a pretrained transformer checkpoint's sequence-
    classification head on entity-marker-annotated relation examples,
    per Section 8's transfer-learning treatment.

    Args:
        labeled_data: Labeled (text, subject_span, object_span, label)
            examples, including NO_RELATION negatives (Section 3H/7's
            "hundreds to low-thousands of examples" scale).
        base_checkpoint: A local checkpoint directory or Hugging Face
            Hub model id to fine-tune from.
        output_dir: Directory to save the fine-tuned tokenizer/model
            to; also used as Trainer's checkpoint directory.
        num_epochs: Number of training epochs.
        batch_size: Per-device train/eval batch size.
        learning_rate: AdamW learning rate.
        validation_split: Fraction of labeled_data held out for
            per-epoch evaluation (0 disables validation).
        seed: Random seed for the train/validation split and Trainer.

    Returns:
        A RelationClassifier wrapping the fine-tuned tokenizer/model,
        ready for predict_relation() calls. The fine-tuned checkpoint
        is also persisted to output_dir.

    Raises:
        RelationTrainingError: If labeled_data is empty or
            base_checkpoint cannot be loaded.
        ValueError: If validation_split is not in [0, 1), or
            num_epochs/batch_size are not positive.
    """
    if not labeled_data:
        raise RelationTrainingError("train_re() received an empty labeled_data list.")
    if not 0.0 <= validation_split < 1.0:
        raise ValueError(f"validation_split must be in [0, 1), got {validation_split}")
    if num_epochs <= 0:
        raise ValueError(f"num_epochs must be positive, got {num_epochs}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    id2label, label2id = build_relation_label_maps()

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_checkpoint, use_fast=True)
    except Exception as exc:  # noqa: BLE001
        raise RelationTrainingError(
            f"Failed to load tokenizer from base_checkpoint {base_checkpoint!r}: {exc}"
        ) from exc

    added = tokenizer.add_special_tokens({"additional_special_tokens": list(MARKER_TOKENS)})

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            base_checkpoint,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise RelationTrainingError(
            f"Failed to load sequence-classification model from base_checkpoint "
            f"{base_checkpoint!r}: {exc}"
        ) from exc

    if added:
        model.resize_token_embeddings(len(tokenizer))

    shuffled_data = list(labeled_data)
    random.Random(seed).shuffle(shuffled_data)
    split_index = int(len(shuffled_data) * (1.0 - validation_split))
    train_examples = shuffled_data[:split_index] or shuffled_data
    validation_examples = shuffled_data[split_index:] if validation_split > 0 else []

    train_dataset = _SequenceClassificationDataset(train_examples, tokenizer, label2id)
    validation_dataset = (
        _SequenceClassificationDataset(validation_examples, tokenizer, label2id)
        if validation_examples
        else None
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch" if validation_dataset is not None else "no",
        save_strategy="no",
        logging_strategy="epoch",
        seed=seed,
        report_to=[],
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=(
            _compute_sequence_classification_metrics if validation_dataset is not None else None
        ),
    )

    logger.info(
        "Starting relation-classifier fine-tuning: %d train example(s), %d "
        "validation example(s), %d epoch(s), base_checkpoint=%r",
        len(train_examples),
        len(validation_examples),
        num_epochs,
        base_checkpoint,
    )
    trainer.train()

    if validation_dataset is not None:
        metrics = trainer.evaluate()
        logger.info("Final validation metrics: %s", metrics)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Saved fine-tuned relation-classifier checkpoint to %s", output_dir)

    return RelationClassifier.from_components(tokenizer=tokenizer, model=model)
