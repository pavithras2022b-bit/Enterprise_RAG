"""
Module 7 - Embedding Generation: embedding computation.

Provides embed(...), embed_batch(...), and the EmbeddingModel class
per the Phase 4 module contract, wrapping a (optionally domain-
fine-tuned) Sentence-BERT model per Section 3D/5.1-5.2.

Reference: Capstone Project Guide, Section 3D ("Embeddings"), Section
5.1-5.2 ("What Embeddings Are" / "How Sentence-BERT Generates
Embeddings"), Phase 4 Module 7 contract, Phase 5 file structure
(embeddings/model.py).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer

from chunking.schema import Chunk
from embeddings.schema import EmbeddedChunk

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class EmbeddingModelError(Exception):
    """Raised when EmbeddingModel cannot load its checkpoint or compute an embedding."""


class EmbeddingModel:
    """
    Wraps a Sentence-BERT-family model (Section 12.3: base
    all-MiniLM-L6-v2, contrastively fine-tuned per Section 8.3) for
    computing chunk/query embeddings.

    Embeddings are L2-normalized by default, so cosine similarity and
    dot-product similarity coincide (Section 5.2's normalization
    step) -- this is what Module 9's hybrid retrieval assumes when it
    treats vector scores as already living in a comparable [0, 1]-ish
    range alongside graph traversal scores.

    Construct via the constructor for a checkpoint on disk or the Hub,
    or via from_sentence_transformer() to inject an already-loaded
    SentenceTransformer (used by fine_tune_embeddings() and tests).
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cpu",
        normalize_embeddings: bool = True,
    ) -> None:
        """
        Args:
            model_name_or_path: A local checkpoint directory (e.g. the
                output_dir of a prior fine_tune_embeddings() run) or a
                Hugging Face Hub model id (e.g. "all-MiniLM-L6-v2").
            device: torch device string ("cpu" or "cuda").
            normalize_embeddings: Whether to L2-normalize output
                vectors.

        Raises:
            EmbeddingModelError: If the model cannot be loaded.
        """
        try:
            model = SentenceTransformer(model_name_or_path, device=device)
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingModelError(
                f"Failed to load SentenceTransformer from {model_name_or_path!r}: {exc}"
            ) from exc
        self._init_from_model(model, normalize_embeddings)

    @classmethod
    def from_sentence_transformer(
        cls,
        model: SentenceTransformer,
        normalize_embeddings: bool = True,
    ) -> "EmbeddingModel":
        """
        Construct an EmbeddingModel from an already-loaded
        SentenceTransformer, bypassing from_pretrained()-style
        loading. Used by fine_tune_embeddings() (which already holds
        the freshly fine-tuned model in memory) and by tests using a
        small local checkpoint.

        Args:
            model: A loaded SentenceTransformer.
            normalize_embeddings: Whether to L2-normalize output
                vectors.

        Returns:
            A ready-to-use EmbeddingModel.
        """
        instance = cls.__new__(cls)
        instance._init_from_model(model, normalize_embeddings)
        return instance

    def _init_from_model(self, model: SentenceTransformer, normalize_embeddings: bool) -> None:
        self._model = model
        self._normalize = normalize_embeddings

    @property
    def dimension(self) -> int:
        """Returns the model's output embedding dimensionality."""
        return int(self._model.get_embedding_dimension())

    def embed(self, text: str) -> np.ndarray:
        """
        Compute the embedding vector for a single piece of text.

        Args:
            text: Text to embed (a chunk's text, or a user query).

        Returns:
            A 1-D float32 array of length self.dimension.

        Raises:
            EmbeddingModelError: If text is empty or whitespace-only.
        """
        if not text or not text.strip():
            raise EmbeddingModelError("embed() received empty or whitespace-only text.")
        vector = self._model.encode(
            text,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vector.astype(np.float32)

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Compute embedding vectors for a batch of texts in one forward
        pass per batch, far more efficient than calling embed() in a
        loop for anything beyond a handful of texts.

        Args:
            texts: Texts to embed. Must be non-empty and contain no
                empty/whitespace-only entries.
            batch_size: Encoding batch size.

        Returns:
            A 2-D float32 array of shape (len(texts), self.dimension).
            Returns an empty (0, dimension) array if texts is empty.

        Raises:
            EmbeddingModelError: If any entry in texts is empty or
                whitespace-only.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        if any(not text or not text.strip() for text in texts):
            raise EmbeddingModelError("embed_batch() received an empty text among texts.")

        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    def embed_chunk(self, chunk: Chunk) -> EmbeddedChunk:
        """
        Embed a single Chunk and wrap the result into an EmbeddedChunk.

        Args:
            chunk: A Chunk from Module 3.

        Returns:
            An EmbeddedChunk carrying chunk.text's embedding plus
            inherited metadata.
        """
        vector = self.embed(chunk.text)
        return EmbeddedChunk.from_chunk(chunk, vector)

    def embed_chunks(self, chunks: list[Chunk], batch_size: int = 32) -> list[EmbeddedChunk]:
        """
        Embed a batch of Chunks in one encoding pass.

        Args:
            chunks: Chunks from Module 3.
            batch_size: Encoding batch size.

        Returns:
            A list of EmbeddedChunk, one per input chunk, in the same
            order. Empty input returns an empty list.
        """
        if not chunks:
            return []
        vectors = self.embed_batch([chunk.text for chunk in chunks], batch_size=batch_size)
        return [
            EmbeddedChunk.from_chunk(chunk, vector) for chunk, vector in zip(chunks, vectors)
        ]


def embed(model: EmbeddingModel, text: str) -> np.ndarray:
    """
    Module-level convenience wrapper around EmbeddingModel.embed(),
    matching the Phase 4 module contract's function signature
    (embed(text) -> np.ndarray). The model is threaded through
    explicitly, mirroring Modules 4/5's convention: callers embedding
    many texts should call model.embed_batch() directly instead of
    looping this wrapper, for the batching efficiency gain.

    Args:
        model: A loaded EmbeddingModel.
        text: Text to embed.

    Returns:
        A 1-D embedding vector.
    """
    return model.embed(text)


def embed_batch(model: EmbeddingModel, texts: list[str], batch_size: int = 32) -> np.ndarray:
    """
    Module-level convenience wrapper around EmbeddingModel.embed_batch(),
    matching the Phase 4 module contract's function signature
    (embed_batch(texts) -> np.ndarray).

    Args:
        model: A loaded EmbeddingModel.
        texts: Texts to embed.
        batch_size: Encoding batch size.

    Returns:
        A 2-D array of embedding vectors, one row per input text.
    """
    return model.embed_batch(texts, batch_size=batch_size)
