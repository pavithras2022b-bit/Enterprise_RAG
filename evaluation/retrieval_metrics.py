"""
Module 13 - Evaluation: Retrieval Metrics.

Provides standard evaluation metrics for information retrieval in RAG systems:
- Hit Rate (Recall@K)
- Mean Reciprocal Rank (MRR@K)
- Normalized Discounted Cumulative Gain (NDCG@K)
"""

from __future__ import annotations

import math
from typing import Sequence


def calculate_hit_rate(
    retrieved_chunk_ids: Sequence[str],
    ground_truth_chunk_ids: Sequence[str],
    k: int = 10,
) -> float:
    """
    Calculates Hit Rate@K (Recall@K flag): returns 1.0 if at least one
    ground-truth chunk is present in top-k retrieved chunks, else 0.0.
    """
    if not retrieved_chunk_ids or not ground_truth_chunk_ids or k <= 0:
        return 0.0
    top_k = set(retrieved_chunk_ids[:k])
    gt_set = set(ground_truth_chunk_ids)
    return 1.0 if len(top_k.intersection(gt_set)) > 0 else 0.0


def calculate_mrr(
    retrieved_chunk_ids: Sequence[str],
    ground_truth_chunk_ids: Sequence[str],
    k: int = 10,
) -> float:
    """
    Calculates Reciprocal Rank@K (MRR element): 1 / rank of the first
    relevant chunk found in top-k, or 0.0 if none found.
    """
    if not retrieved_chunk_ids or not ground_truth_chunk_ids or k <= 0:
        return 0.0
    gt_set = set(ground_truth_chunk_ids)
    for rank, chunk_id in enumerate(retrieved_chunk_ids[:k], start=1):
        if chunk_id in gt_set:
            return 1.0 / rank
    return 0.0


def calculate_ndcg(
    retrieved_chunk_ids: Sequence[str],
    ground_truth_chunk_ids: Sequence[str],
    k: int = 10,
) -> float:
    """
    Calculates NDCG@K for binary relevance.
    """
    if not retrieved_chunk_ids or not ground_truth_chunk_ids or k <= 0:
        return 0.0

    gt_set = set(ground_truth_chunk_ids)
    dcg = 0.0
    for i, chunk_id in enumerate(retrieved_chunk_ids[:k], start=1):
        if chunk_id in gt_set:
            dcg += 1.0 / math.log2(i + 1)

    idcg = 0.0
    n_rel = min(len(gt_set), k)
    for i in range(1, n_rel + 1):
        idcg += 1.0 / math.log2(i + 1)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate_retrieval_batch(
    test_cases: Sequence[dict[str, list[str]]],
    k: int = 5,
) -> dict[str, float]:
    """
    Evaluates a list of test cases, each containing 'retrieved' and 'ground_truth' lists.
    Returns average hit_rate, mrr, and ndcg across all test cases.
    """
    if not test_cases:
        return {"hit_rate": 0.0, "mrr": 0.0, "ndcg": 0.0}

    total_hit = 0.0
    total_mrr = 0.0
    total_ndcg = 0.0

    for item in test_cases:
        retrieved = item.get("retrieved", [])
        ground_truth = item.get("ground_truth", [])
        total_hit += calculate_hit_rate(retrieved, ground_truth, k=k)
        total_mrr += calculate_mrr(retrieved, ground_truth, k=k)
        total_ndcg += calculate_ndcg(retrieved, ground_truth, k=k)

    n = len(test_cases)
    return {
        f"hit_rate@{k}": total_hit / n,
        f"mrr@{k}": total_mrr / n,
        f"ndcg@{k}": total_ndcg / n,
    }
