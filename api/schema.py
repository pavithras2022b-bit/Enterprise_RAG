"""
Module 14 - API & Service Layer: Pydantic Request/Response Schemas.

Defines schemas for API request validation and response serialization per Section 10.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Payload for POST /api/v1/query."""

    query: str = Field(..., min_length=1, description="User search or question text")
    alpha: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Override fusion weight (0.0=graph, 1.0=vector)"
    )
    top_k: int = Field(5, ge=1, le=100, description="Number of top results/citations to return")
    use_adaptive: bool = Field(True, description="Whether to use adaptive alpha routing if alpha is None")
    use_reranking: bool = Field(True, description="Whether to apply cross-encoder reranking")


class CandidateDTO(BaseModel):
    """DTO representing a retrieved/fused candidate passage or fact."""

    candidate_id: str
    source: str
    fused_score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """Response returned by POST /api/v1/query."""

    query: str
    answer: str
    citations: list[str]
    alpha_used: float
    latency_ms: float
    source_candidates: list[CandidateDTO] = Field(default_factory=list)


class IngestRequest(BaseModel):
    """Payload for POST /api/v1/ingest."""

    file_paths: list[str] = Field(..., min_length=1, description="Local document file paths to ingest")


class IngestResponse(BaseModel):
    """Response returned by POST /api/v1/ingest."""

    documents_processed: int
    chunks_created: int
    entities_extracted: int
    relations_extracted: int
    status: str = "success"


class HealthResponse(BaseModel):
    """Response returned by GET /health."""

    status: str = "healthy"
    version: str = "1.0.0"
