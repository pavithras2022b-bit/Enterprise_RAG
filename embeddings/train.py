"""
Module 7 - Embedding Generation: contrastive fine-tuning entrypoint.

Provides fine_tune_embeddings(...) per the Phase 4 module contract:
contrastively fine-tunes a Sentence-BERT base checkpoint on enterprise
(query, positive_passage) pairs using Multiple Negatives Ranking Loss,
per Section 8.3/8.7's transfer-learning treatment of embeddings.

Reference: Capstone Project Guide, Section 8.3 ("Transfer Learning
Applied to Embeddings"), Section 8.7 ("Contrastive Learning"), Phase 4
Module 7 contract, Phase 5 file structure (embeddings/train.py).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from embeddings.model import EmbeddingModel
from embeddings.schema import ContrastivePair

logger = logging.getLogger(__name__)


class EmbeddingTrainingError(Exception):
    """Raised when embedding fine-tuning cannot proceed for the given input."""


def fine_tune_embeddings(
    pairs: list[ContrastivePair],
    base_model: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 16,
    warmup_ratio: float = 0.1,
    seed: int = 42,
) -> EmbeddingModel:
    """
    Contrastively fine-tune a Sentence-BERT base checkpoint on
    enterprise (query, positive_passage) pairs, using Multiple
    Negatives Ranking Loss: for each pair in a training batch, every
    other passage in the same batch serves as an implicit negative for
    that query, so no explicit negative examples need to be
    hand-curated (Section 8.7).

    Args:
        pairs: Contrastive training pairs.
        base_model: A local checkpoint directory or Hugging Face Hub
            model id to fine-tune from (e.g. "all-MiniLM-L6-v2" per
            Section 12.3).
        output_dir: Directory to save the fine-tuned model to.
        num_epochs: Number of training epochs.
        batch_size: Training batch size. Multiple Negatives Ranking
            Loss's implicit-negative strategy benefits from a larger
            batch (more in-batch negatives per query); very small
            batches (e.g. 1-2) weaken its training signal but still
            run correctly.
        warmup_ratio: Fraction of total training steps used for
            learning-rate warmup.
        seed: Random seed for reproducibility.

    Returns:
        An EmbeddingModel wrapping the fine-tuned SentenceTransformer,
        ready for embed()/embed_batch() calls. The fine-tuned
        checkpoint is also persisted to output_dir.

    Raises:
        EmbeddingTrainingError: If pairs is empty or base_model cannot
            be loaded.
        ValueError: If num_epochs/batch_size are not positive, or
            warmup_ratio is not in [0, 1].
    """
    if not pairs:
        raise EmbeddingTrainingError("fine_tune_embeddings() received an empty pairs list.")
    if num_epochs <= 0:
        raise ValueError(f"num_epochs must be positive, got {num_epochs}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError(f"warmup_ratio must be in [0, 1], got {warmup_ratio}")

    torch.manual_seed(seed)

    try:
        model = SentenceTransformer(base_model)
    except Exception as exc:  # noqa: BLE001
        raise EmbeddingTrainingError(
            f"Failed to load SentenceTransformer from base_model {base_model!r}: {exc}"
        ) from exc

    train_examples = [
        InputExample(texts=[pair.query, pair.positive_passage]) for pair in pairs
    ]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    steps_per_epoch = len(train_dataloader)
    warmup_steps = int(steps_per_epoch * num_epochs * warmup_ratio)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting embedding fine-tuning: %d pair(s), %d epoch(s), batch_size=%d, "
        "base_model=%r",
        len(pairs),
        num_epochs,
        batch_size,
        base_model,
    )
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        show_progress_bar=False,
    )
    logger.info("Saved fine-tuned embedding checkpoint to %s", output_dir)

    return EmbeddingModel.from_sentence_transformer(model)
