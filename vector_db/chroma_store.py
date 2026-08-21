"""
Module 8 - Vector Database: ChromaDB-backed primary vector store.

Provides add(...), search(...), delete(...) and the VectorStore class
per the Phase 4 module contract, wrapping ChromaDB (Section 5.5,
12.4): the project's primary vector store, chosen specifically for
its built-in persistence and metadata filtering, which raw FAISS
lacks out of the box.

Reference: Capstone Project Guide, Section 3E/5.5 ("ChromaDB"), Phase
4 Module 8 contract, Phase 5 file structure (vector_db/chroma_store.py).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import chromadb
import numpy as np
from chromadb.config import Settings

from embeddings.schema import EmbeddedChunk
from vector_db.schema import VectorSearchResult

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Raised when VectorStore cannot connect, write, or query."""


class VectorStore:
    """
    Wraps a ChromaDB collection for persisting EmbeddedChunks and
    running metadata-filtered cosine-similarity search.

    The collection is configured with hnsw:space="cosine", so
    ChromaDB's internal distance ("how far apart") is converted to a
    similarity score ("how close", higher is better) via
    similarity = 1 - distance before being wrapped into
    VectorSearchResult -- verified empirically: an identical vector
    query returns distance 0.0 (similarity 1.0); an orthogonal vector
    returns distance 1.0 (similarity 0.0), matching cosine distance's
    definition of 1 - cosine_similarity.
    """

    def __init__(
        self,
        collection_name: str,
        persist_directory: Optional[str] = None,
        client: Optional["chromadb.ClientAPI"] = None,
    ) -> None:
        """
        Args:
            collection_name: Name of the ChromaDB collection to use
                (created if it doesn't already exist).
            persist_directory: Directory for on-disk persistence. If
                None and no client is supplied, an in-memory
                (non-persistent) client is used instead -- appropriate
                for tests, not for a real indexing run.
            client: An already-constructed ChromaDB client, injectable
                for testing. If provided, persist_directory is
                ignored.

        Raises:
            VectorStoreError: If the client or collection cannot be
                constructed.
        """
        if client is not None:
            self._client = client
        else:
            settings = Settings(anonymized_telemetry=False)
            try:
                if persist_directory:
                    self._client = chromadb.PersistentClient(
                        path=persist_directory, settings=settings
                    )
                else:
                    self._client = chromadb.EphemeralClient(settings=settings)
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreError(f"Failed to construct ChromaDB client: {exc}") from exc

        try:
            self._collection = self._client.get_or_create_collection(
                collection_name, metadata={"hnsw:space": "cosine"}
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(
                f"Failed to get_or_create_collection {collection_name!r}: {exc}"
            ) from exc

    def add(self, chunks: list[EmbeddedChunk]) -> None:
        """
        Persist a batch of EmbeddedChunks, upserting by chunk_id (a
        re-added chunk with the same chunk_id replaces the existing
        entry rather than duplicating it -- important for re-running
        indexing after a document update).

        Args:
            chunks: EmbeddedChunks from Module 7. No-op if empty.

        Raises:
            VectorStoreError: If the underlying write fails.
        """
        if not chunks:
            return

        ids = [chunk.chunk_id for chunk in chunks]
        embeddings = [chunk.vector.tolist() for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.to_metadata_dict() for chunk in chunks]

        try:
            self._collection.upsert(
                ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to add {len(chunks)} chunk(s): {exc}") from exc

        logger.debug("Added/updated %d chunk(s) in collection.", len(chunks))

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        where: Optional[dict[str, Any]] = None,
    ) -> list[VectorSearchResult]:
        """
        Run a k-nearest-neighbor cosine-similarity search, optionally
        scoped by a metadata filter (e.g. {"department": "Finance"}),
        combining vector similarity and structured filtering in one
        query -- the project's stated reason for choosing ChromaDB
        over raw FAISS (Section 5.5).

        Args:
            query_vector: A 1-D query embedding (same dimensionality
                as the stored vectors).
            k: Number of nearest neighbors to return.
            where: Optional ChromaDB metadata filter, e.g.
                {"department": "Finance"} or
                {"$and": [{"department": "Finance"}, {"document_type": "policy"}]}.

        Returns:
            A list of VectorSearchResult, ordered by descending
            similarity (nearest first). Empty if the collection has
            no matching entries.

        Raises:
            ValueError: If k is not positive or query_vector is not 1-D.
            VectorStoreError: If the underlying query fails.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if query_vector.ndim != 1:
            raise ValueError(f"query_vector must be 1-D, got shape {query_vector.shape}")

        try:
            raw_results = self._collection.query(
                query_embeddings=[query_vector.tolist()], n_results=k, where=where
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"search() failed: {exc}") from exc

        ids = raw_results.get("ids", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]

        results = [
            VectorSearchResult(
                chunk_id=chunk_id,
                score=1.0 - distance,
                text=document or "",
                metadata=dict(metadata) if metadata else {},
            )
            for chunk_id, distance, document, metadata in zip(ids, distances, documents, metadatas)
        ]
        logger.debug("search() returned %d result(s) for k=%d", len(results), k)
        return results

    def delete(self, chunk_id: str) -> None:
        """
        Remove a single chunk from the collection by id.

        Args:
            chunk_id: The chunk_id to remove.

        Raises:
            ValueError: If chunk_id is empty.
            VectorStoreError: If the underlying delete fails.
        """
        if not chunk_id:
            raise ValueError("delete() requires a non-empty chunk_id.")
        try:
            self._collection.delete(ids=[chunk_id])
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to delete chunk_id {chunk_id!r}: {exc}") from exc
        logger.debug("Deleted chunk_id %s", chunk_id)

    def count(self) -> int:
        """Returns the number of chunks currently stored in the collection."""
        return self._collection.count()
