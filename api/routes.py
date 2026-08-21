"""
Module 14 - API & Service Layer: FastAPI Routes.

Exposes REST API endpoints for document ingestion, query execution, and health status.
"""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, HTTPException, status

from api.schema import HealthResponse, IngestRequest, IngestResponse, QueryRequest, QueryResponse

router = APIRouter()

# Global pipeline instance set during application startup
_pipeline: Any = None


def set_pipeline(pipeline_instance: Any) -> None:
    """Sets the global RAG pipeline instance used by API endpoints."""
    global _pipeline
    _pipeline = pipeline_instance


def get_pipeline() -> Any:
    """Returns the active global pipeline instance or raises HTTP 503."""
    if _pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_533_SERVICE_UNAVAILABLE if hasattr(status, 'HTTP_533_SERVICE_UNAVAILABLE') else 503,
            detail="RAG pipeline is not initialized.",
        )
    return _pipeline


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Returns system status."""
    return HealthResponse(status="healthy", version="1.0.0")


@router.post("/api/v1/ingest", response_model=IngestResponse)
def ingest_documents(payload: IngestRequest) -> IngestResponse:
    """Ingests a list of document file paths into the RAG pipeline."""
    pipeline = get_pipeline()
    try:
        return pipeline.ingest_files(payload.file_paths)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        ) from exc


@router.post("/api/v1/query", response_model=QueryResponse)
def execute_query(payload: QueryRequest) -> QueryResponse:
    """Executes end-to-end hybrid retrieval and generation for a user query."""
    pipeline = get_pipeline()
    try:
        return pipeline.answer_query(
            query=payload.query,
            alpha=payload.alpha,
            top_k=payload.top_k,
            use_adaptive=payload.use_adaptive,
            use_reranking=payload.use_reranking,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {exc}",
        ) from exc
