"""
Module 11 - Cross-Encoder Re-ranking: fine-tuning entrypoint.

Provides fine_tune_cross_encoder(...) per the Phase 4 module contract:
fine-tunes a cross-encoder checkpoint on enterprise (query,
relevant-passage, irrelevant-passage) relevance triples, per Section
8.5's transfer-learning treatment of the re-ranker.

Reference: Capstone Project Guide, Section 8.5 ("Transfer Learning
Applied to the Re-ranker (Cross-Encoder)"), Phase 4 Module 11
contract, Phase 5 file structure (reranking/train.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

from reranking.model import CrossEncoderReranker

logger = logging.getLogger(__name__)


class SchemaValidationError(Exception):
    """Raised when a RelevanceTriple is constructed with invalid values."""


class RerankerTrainingError(Exception):
    """Raised when cross-encoder fine-tuning cannot proceed for the given input."""


@dataclass(frozen=True, slots=True)
class RelevanceTriple:
    """
    One (query, relevant-passage, irrelevant-passage) training example
    (Section 8.5), decomposed by fine_tune_cross_encoder() into two
    labeled (query, passage) pairs -- label 1.0 for the relevant
    passage, 0.0 for the irrelevant one -- since CrossEncoder.fit()'s
    standard training loop consumes labeled pairs, not triples
    directly.

    Attributes:
        query: The query text.
        relevant_passage: A passage that correctly answers/is relevant
            to query.
        irrelevant_passage: A passage that is not relevant to query
            (e.g. topically similar but not actually responsive --
            Section 8.5's example: a passage merely mentioning "leave"
            in an unrelated exit-policy context, for a query about
            leave balance).

    Raises:
        SchemaValidationError: If any field is empty/whitespace-only.
    """

    query: str
    relevant_passage: str
    irrelevant_passage: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("query", self.query),
            ("relevant_passage", self.relevant_passage),
            ("irrelevant_passage", self.irrelevant_passage),
        ):
            if not value or not value.strip():
                raise SchemaValidationError(f"RelevanceTriple.{field_name} must not be empty.")


def fine_tune_cross_encoder(
    triples: list[RelevanceTriple],
    base_model: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 16,
    warmup_steps: int = 100,
    seed: int = 42,
) -> CrossEncoderReranker:
    """
    Fine-tune a cross-encoder checkpoint on enterprise relevance
    triples. Each triple is decomposed into two labeled InputExamples
    (relevant -> 1.0, irrelevant -> 0.0) and trained with
    CrossEncoder.fit()'s standard binary relevance objective.

    Args:
        triples: Relevance triples.
        base_model: A local checkpoint directory or Hugging Face Hub
            model id to fine-tune from (e.g.
            "cross-encoder/ms-marco-MiniLM-L-6-v2" per Section 12.3 --
            already pretrained on passage relevance, so this is a
            "fine-tune a fine-tuned model further" case per Section
            8.5).
        output_dir: Directory to save the fine-tuned model to.
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        warmup_steps: Learning-rate warmup steps.
        seed: Random seed for reproducibility.

    Returns:
        A CrossEncoderReranker wrapping the fine-tuned model, ready
        for score_pair()/rerank() calls. The fine-tuned checkpoint is
        also persisted to output_dir.

    Raises:
        RerankerTrainingError: If triples is empty or base_model
            cannot be loaded.
        ValueError: If num_epochs/batch_size are not positive.
    """
    if not triples:
        raise RerankerTrainingError("fine_tune_cross_encoder() received an empty triples list.")
    if num_epochs <= 0:
        raise ValueError(f"num_epochs must be positive, got {num_epochs}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    torch.manual_seed(seed)

    try:
        model = CrossEncoder(base_model)
    except Exception as exc:  # noqa: BLE001
        raise RerankerTrainingError(
            f"Failed to load CrossEncoder from base_model {base_model!r}: {exc}"
        ) from exc

    train_examples: list[InputExample] = []
    for triple in triples:
        train_examples.append(InputExample(texts=[triple.query, triple.relevant_passage], label=1.0))
        train_examples.append(InputExample(texts=[triple.query, triple.irrelevant_passage], label=0.0))

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting cross-encoder fine-tuning: %d triple(s) (%d labeled pairs), "
        "%d epoch(s), base_model=%r",
        len(triples),
        len(train_examples),
        num_epochs,
        base_model,
    )
    model.fit(
        train_dataloader=train_dataloader,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=False,
    )
    # fit()'s output_path/save_best_model mechanism only saves a
    # checkpoint when an evaluator triggers it (Section 8.5's simple
    # triple-based fine-tuning doesn't use one) -- save explicitly here
    # so a checkpoint is always produced regardless.
    model.save(output_dir)
    logger.info("Saved fine-tuned cross-encoder checkpoint to %s", output_dir)

    return CrossEncoderReranker.from_cross_encoder(model)
