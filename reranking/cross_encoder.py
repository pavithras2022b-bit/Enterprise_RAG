"""
Module 11 - Cross-Encoder Re-ranking: cross_encoder alias module.

Provides direct re-exports of CrossEncoderReranker, score_pair, and rerank
from reranking.model for contract conformance.
"""

from reranking.model import (
    CrossEncoderReranker,
    CrossEncoderRerankerError,
    rerank,
    score_pair,
)

__all__ = [
    "CrossEncoderReranker",
    "CrossEncoderRerankerError",
    "score_pair",
    "rerank",
]
