"""
Module 4 - NER: fine-tuning entrypoint.

Provides train_ner(...) per the Phase 4 module contract: fine-tunes a
pretrained transformer checkpoint's token-classification head on
BIO-tagged enterprise sentences, per Section 8.4's transfer-learning
treatment of NER.

Reference: Capstone Project Guide, Section 3G ("Entity Extraction"),
Section 8.4 ("Transfer Learning Applied to NER"), Phase 4 Module 4
contract, Phase 5 file structure (ner/train.py).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from ner.model import NERModel
from ner.schema import LabeledExample, build_bio_label_maps

logger = logging.getLogger(__name__)


class NERTrainingError(Exception):
    """Raised when NER fine-tuning cannot proceed for the given input."""


class _TokenClassificationDataset(torch.utils.data.Dataset):
    """
    Wraps a list of LabeledExample into a torch Dataset of
    tokenizer-encoded, label-aligned tensors, using the standard
    Hugging Face "label first subtoken only" alignment: a word split
    into multiple subword tokens has its BIO tag on the first subtoken
    and -100 (ignored by the loss) on continuation subtokens and
    special tokens.
    """

    def __init__(
        self,
        examples: list[LabeledExample],
        tokenizer: "AutoTokenizer",
        label2id: dict[str, int],
        max_length: int = 128,
    ) -> None:
        self._encodings: list[dict[str, torch.Tensor]] = []
        self._labels: list[list[int]] = []

        for example in examples:
            encoding = tokenizer(
                list(example.tokens),
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            word_ids = encoding.word_ids()
            label_ids: list[int] = []
            previous_word_id: Optional[int] = None
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != previous_word_id:
                    label_ids.append(label2id[example.tags[word_id]])
                else:
                    label_ids.append(-100)
                previous_word_id = word_id

            self._encodings.append(
                {key: torch.tensor(value) for key, value in encoding.items()}
            )
            self._labels.append(label_ids)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = dict(self._encodings[index])
        item["labels"] = torch.tensor(self._labels[index])
        return item


def _compute_token_classification_metrics(eval_prediction: EvalPrediction) -> dict[str, float]:
    """
    Computes token-level accuracy over non-ignored (label != -100)
    positions, per Section 3G's "evaluation via token-level ... F1
    score" guidance (accuracy is reported here as the lightweight,
    dependency-free metric; entity-level F1 is Module 13's job with
    the full evaluation harness, not this training-time sanity check).
    """
    predictions = np.argmax(eval_prediction.predictions, axis=-1)
    labels = eval_prediction.label_ids

    correct = 0
    total = 0
    for prediction_row, label_row in zip(predictions, labels):
        for predicted_id, true_id in zip(prediction_row, label_row):
            if true_id == -100:
                continue
            total += 1
            if predicted_id == true_id:
                correct += 1

    return {"token_accuracy": correct / total if total else 0.0}


def train_ner(
    labeled_data: list[LabeledExample],
    base_checkpoint: str,
    output_dir: str,
    id2label: Optional[dict[int, str]] = None,
    label2id: Optional[dict[str, int]] = None,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    validation_split: float = 0.2,
    seed: int = 42,
) -> NERModel:
    """
    Fine-tune a pretrained transformer checkpoint's token-classification
    head on BIO-tagged labeled_data, per Section 8.4's transfer-learning
    treatment of NER (freeze/lightly-tune general language understanding,
    specialize the classification decision on top).

    Args:
        labeled_data: BIO-tagged training examples (Section 3G/7's
            "hundreds to low-thousands of examples" scale).
        base_checkpoint: A local checkpoint directory or Hugging Face
            Hub model id to fine-tune from (e.g. "bert-base-cased" per
            Section 12.3).
        output_dir: Directory to save the fine-tuned tokenizer/model
            to; also used as Trainer's checkpoint directory.
        id2label: Label-id-to-BIO-tag map for the classification head.
            Defaults to build_bio_label_maps()'s full EntityType
            vocabulary if not provided.
        label2id: Inverse of id2label. Must be consistent with
            id2label if both are provided. Defaults alongside id2label.
        num_epochs: Number of training epochs.
        batch_size: Per-device train/eval batch size.
        learning_rate: AdamW learning rate.
        validation_split: Fraction of labeled_data held out for
            per-epoch evaluation (0 disables validation).
        seed: Random seed for the train/validation split and Trainer.

    Returns:
        An NERModel wrapping the fine-tuned tokenizer/model, ready for
        predict_entities() calls. The fine-tuned checkpoint is also
        persisted to output_dir.

    Raises:
        NERTrainingError: If labeled_data is empty, base_checkpoint
            cannot be loaded, or the loaded tokenizer is not a fast
            tokenizer.
        ValueError: If validation_split is not in [0, 1), or
            num_epochs/batch_size are not positive.
    """
    if not labeled_data:
        raise NERTrainingError("train_ner() received an empty labeled_data list.")
    if not 0.0 <= validation_split < 1.0:
        raise ValueError(f"validation_split must be in [0, 1), got {validation_split}")
    if num_epochs <= 0:
        raise ValueError(f"num_epochs must be positive, got {num_epochs}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    resolved_id2label, resolved_label2id = id2label, label2id
    if resolved_id2label is None or resolved_label2id is None:
        resolved_id2label, resolved_label2id = build_bio_label_maps()
        logger.info(
            "train_ner() defaulting to the full EntityType BIO label "
            "vocabulary (%d labels)",
            len(resolved_id2label),
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_checkpoint, use_fast=True)
    except Exception as exc:  # noqa: BLE001
        raise NERTrainingError(
            f"Failed to load tokenizer from base_checkpoint {base_checkpoint!r}: {exc}"
        ) from exc
    if not tokenizer.is_fast:
        raise NERTrainingError(
            f"train_ner() requires a fast tokenizer for word-to-subtoken "
            f"label alignment; {base_checkpoint!r} did not provide one."
        )

    try:
        model = AutoModelForTokenClassification.from_pretrained(
            base_checkpoint,
            num_labels=len(resolved_id2label),
            id2label=resolved_id2label,
            label2id=resolved_label2id,
            ignore_mismatched_sizes=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise NERTrainingError(
            f"Failed to load token-classification model from base_checkpoint "
            f"{base_checkpoint!r}: {exc}"
        ) from exc

    shuffled_data = list(labeled_data)
    random.Random(seed).shuffle(shuffled_data)
    split_index = int(len(shuffled_data) * (1.0 - validation_split))
    train_examples = shuffled_data[:split_index] or shuffled_data
    validation_examples = shuffled_data[split_index:] if validation_split > 0 else []

    train_dataset = _TokenClassificationDataset(train_examples, tokenizer, resolved_label2id)
    validation_dataset = (
        _TokenClassificationDataset(validation_examples, tokenizer, resolved_label2id)
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
            _compute_token_classification_metrics if validation_dataset is not None else None
        ),
    )

    logger.info(
        "Starting NER fine-tuning: %d train example(s), %d validation "
        "example(s), %d epoch(s), base_checkpoint=%r",
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
    logger.info("Saved fine-tuned NER checkpoint to %s", output_dir)

    return NERModel.from_components(tokenizer=tokenizer, model=model)
