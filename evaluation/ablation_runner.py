"""
Module 13 - Evaluation: Ablation Runner.

Runs systematic comparative ablation experiments across retrieval strategies:
1. Vector-only retrieval (alpha = 1.0)
2. Graph-only retrieval (alpha = 0.0)
3. Static Hybrid retrieval (alpha = 0.5)
4. Adaptive Hybrid retrieval (learned or rule-based alpha)

Per Capstone Guide Section 3M ("Evaluation & Ablation Studies").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from evaluation.retrieval_metrics import (
    calculate_hit_rate,
    calculate_mrr,
    calculate_ndcg,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvaluationExample:
    """
    One query item for ablation evaluation.
    """

    query: str
    ground_truth_chunk_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AblationStrategyResult:
    """
    Evaluation metrics for a single retrieval strategy.
    """

    strategy_name: str
    hit_rate: float
    mrr: float
    ndcg: float
    avg_latency_ms: float


class AblationRunner:
    """
    Orchestrates comparative ablation testing over an evaluation dataset.
    """

    def __init__(self, hybrid_controller: Any, adaptive_engine: Any = None) -> None:
        self.hybrid_controller = hybrid_controller
        self.adaptive_engine = adaptive_engine

    def evaluate_strategy(
        self,
        strategy_name: str,
        examples: Sequence[EvaluationExample],
        alpha: float | None = None,
        use_adaptive: bool = False,
        k: int = 5,
    ) -> AblationStrategyResult:
        """
        Runs evaluation for a specific strategy configuration over all examples.
        """
        if not examples:
            return AblationStrategyResult(strategy_name, 0.0, 0.0, 0.0, 0.0)

        hits = []
        mrrs = []
        ndcgs = []

        for item in examples:
            if use_adaptive and self.adaptive_engine is not None:
                _, candidates = self.adaptive_engine.retrieve(query=item.query, k_vector=k, k_graph=k)
            else:
                effective_alpha = 0.5 if alpha is None else alpha
                candidates = self.hybrid_controller.retrieve(
                    query=item.query, alpha=effective_alpha, k_vector=k, k_graph=k
                )

            retrieved_ids = [c.candidate_id for c in candidates]
            hits.append(calculate_hit_rate(retrieved_ids, item.ground_truth_chunk_ids, k=k))
            mrrs.append(calculate_mrr(retrieved_ids, item.ground_truth_chunk_ids, k=k))
            ndcgs.append(calculate_ndcg(retrieved_ids, item.ground_truth_chunk_ids, k=k))

        n = len(examples)
        return AblationStrategyResult(
            strategy_name=strategy_name,
            hit_rate=sum(hits) / n,
            mrr=sum(mrrs) / n,
            ndcg=sum(ndcgs) / n,
            avg_latency_ms=0.0,
        )

    def run_all(
        self,
        examples: Sequence[EvaluationExample],
        k: int = 5,
    ) -> dict[str, AblationStrategyResult]:
        """
        Runs all four core ablation strategies:
        - Vector-Only (alpha=1.0)
        - Graph-Only (alpha=0.0)
        - Static Hybrid (alpha=0.5)
        - Adaptive Hybrid (if adaptive engine present)
        """
        results = {}
        results["vector_only"] = self.evaluate_strategy("vector_only", examples, alpha=1.0, k=k)
        results["graph_only"] = self.evaluate_strategy("graph_only", examples, alpha=0.0, k=k)
        results["static_hybrid"] = self.evaluate_strategy("static_hybrid", examples, alpha=0.5, k=k)

        if self.adaptive_engine is not None:
            results["adaptive_hybrid"] = self.evaluate_strategy(
                "adaptive_hybrid", examples, use_adaptive=True, k=k
            )

        return results
