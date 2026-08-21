"""
Module 8 - Vector Database: FAISS benchmark-only comparison index.

Provides FAISSBenchmarkIndex per the Phase 4 module contract --
explicitly comparison-only (Section 5.4): ChromaDB is the project's
primary vector store, chosen for its built-in persistence and metadata
filtering; FAISS is retained purely as a raw-ANN-speed baseline for
Section 13's evaluation. Unlike VectorStore, this class has no
persistence and no metadata filtering -- exactly the tradeoff Section
5.4 documents as FAISS being "a library, not a full database."

Reference: Capstone Project Guide, Section 3E/5.4 ("FAISS"), Phase 4
Module 8 contract, Phase 5 file structure (vector_db/faiss_index.py).
"""

from __future__ import annotations

import logging
from typing import Literal

import faiss
import numpy as np

from embeddings.schema import EmbeddedChunk
from vector_db.schema import VectorSearchResult

logger = logging.getLogger(__name__)

IndexType = Literal["flat", "hnsw"]


class FAISSIndexError(Exception):
    """Raised when FAISSBenchmarkIndex cannot construct or query its index."""


class FAISSBenchmarkIndex:
    """
    A minimal FAISS wrapper for benchmark comparison against
    VectorStore (Section 13's evaluation design).

    Uses inner-product search (IndexFlatIP / HNSW with inner-product
    metric), which is equivalent to cosine similarity for the
    L2-normalized embeddings Module 7 produces -- kept consistent with
    VectorStore's cosine-similarity scoring so the two backends'
    scores are directly comparable in an ablation.

    No persistence: this index lives only in memory for the duration
    of a benchmark run. No delete: FAISS's flat/HNSW indexes don't
    support efficient single-vector removal; rebuilding the index is
    the standard approach and is out of scope for a comparison-only
    benchmark class. Both are documented limitations, not oversights
    (Section 5.4).
    """

    def __init__(self, dimension: int, index_type: IndexType = "flat") -> None:
        """
        Args:
            dimension: Embedding dimensionality every added vector
                must match.
            index_type: "flat" for exact brute-force search
                (IndexFlatIP), or "hnsw" for approximate search
                (IndexHNSWFlat) -- Section 5.4's two documented index
                families.

        Raises:
            ValueError: If dimension is not positive or index_type is
                unrecognized.
        """
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")

        if index_type == "flat":
            index = faiss.IndexFlatIP(dimension)
        elif index_type == "hnsw":
            # The metric MUST be passed at construction time -- setting
            # index.metric_type after construction does not reconfigure
            # the already-built HNSW graph and silently produces empty
            # search results (verified empirically while building this
            # module).
            index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unknown index_type {index_type!r}; expected 'flat' or 'hnsw'.")

        self._index = index
        self._dimension = dimension
        self._index_type = index_type
        self._chunk_ids: list[str] = []
        self._texts: list[str] = []

    @property
    def size(self) -> int:
        """Returns the number of vectors currently in the index."""
        return self._index.ntotal

    def add(self, chunks: list[EmbeddedChunk]) -> None:
        """
        Add a batch of EmbeddedChunks to the index.

        Args:
            chunks: EmbeddedChunks from Module 7. No-op if empty.

        Raises:
            FAISSIndexError: If any chunk's vector dimensionality
                doesn't match self._dimension.
        """
        if not chunks:
            return

        vectors = np.stack([chunk.vector for chunk in chunks]).astype(np.float32)
        if vectors.shape[1] != self._dimension:
            raise FAISSIndexError(
                f"Vector dimension mismatch: index expects {self._dimension}, "
                f"got {vectors.shape[1]}."
            )

        self._index.add(vectors)
        self._chunk_ids.extend(chunk.chunk_id for chunk in chunks)
        self._texts.extend(chunk.text for chunk in chunks)
        logger.debug("Added %d chunk(s) to FAISS index (total=%d).", len(chunks), self.size)

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[VectorSearchResult]:
        """
        Run a k-nearest-neighbor inner-product search.

        Args:
            query_vector: A 1-D query embedding.
            k: Number of nearest neighbors to return. Silently capped
                to the index's current size if fewer vectors are
                stored.

        Returns:
            A list of VectorSearchResult, ordered by descending
            similarity. Empty metadata dict for every result, since
            FAISS stores no metadata (Section 5.4's documented
            tradeoff vs. ChromaDB). Empty list if the index is empty.

        Raises:
            ValueError: If k is not positive or query_vector is not
                1-D or doesn't match the index dimensionality.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if query_vector.ndim != 1:
            raise ValueError(f"query_vector must be 1-D, got shape {query_vector.shape}")
        if query_vector.shape[0] != self._dimension:
            raise ValueError(
                f"query_vector dimension {query_vector.shape[0]} does not match "
                f"index dimension {self._dimension}."
            )
        if self.size == 0:
            return []

        effective_k = min(k, self.size)
        query = query_vector.astype(np.float32).reshape(1, -1)
        scores, indices = self._index.search(query, effective_k)

        results = [
            VectorSearchResult(
                chunk_id=self._chunk_ids[index], score=float(score), text=self._texts[index]
            )
            for score, index in zip(scores[0], indices[0])
            if index != -1  # FAISS pads with -1 when fewer than k matches exist
        ]
        logger.debug("search() returned %d result(s) for k=%d", len(results), k)
        return results
