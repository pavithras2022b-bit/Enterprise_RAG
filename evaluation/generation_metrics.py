"""
Module 13 - Evaluation: Generation Metrics.

Provides metrics for validating LLM generated output quality in RAG:
- Faithfulness (groundedness in retrieved evidence)
- Citation Precision (validity of cited chunk/fact IDs)
- Answer Relevance (overlap/alignment with user query)
"""

from __future__ import annotations

import re
from typing import Sequence


def calculate_citation_precision(
    citations: Sequence[str],
    valid_source_ids: Sequence[str],
) -> float:
    """
    Calculates the proportion of extracted citations that match valid source IDs.
    Returns 1.0 if no citations were produced and valid sources is also empty,
    or ratio of valid citations to total citations.
    """
    if not citations:
        return 1.0 if not valid_source_ids else 0.0

    valid_set = set(valid_source_ids)
    correct_count = sum(1 for c in citations if c in valid_set)
    return correct_count / len(citations)


def calculate_faithfulness(
    answer: str,
    evidence_texts: Sequence[str],
) -> float:
    """
    Calculates word-overlap based faithfulness proxy of generated answer against
    provided evidence texts. Returns score in [0.0, 1.0].
    """
    if not answer:
        return 0.0
    if not evidence_texts:
        return 0.0

    answer_words = set(re.findall(r"\w+", answer.lower()))
    if not answer_words:
        return 0.0

    combined_evidence = " ".join(evidence_texts).lower()
    evidence_words = set(re.findall(r"\w+", combined_evidence))

    overlap = answer_words.intersection(evidence_words)
    return len(overlap) / len(answer_words)


def calculate_answer_relevance(
    answer: str,
    query: str,
) -> float:
    """
    Calculates word-overlap relevance between query and answer.
    """
    if not answer or not query:
        return 0.0

    query_words = set(re.findall(r"\w+", query.lower()))
    if not query_words:
        return 0.0

    answer_words = set(re.findall(r"\w+", answer.lower()))
    overlap = query_words.intersection(answer_words)
    return len(overlap) / len(query_words)


def evaluate_generation_batch(
    test_cases: Sequence[dict],
) -> dict[str, float]:
    """
    Evaluates a list of generation test dicts with keys:
    'answer', 'query', 'evidence_texts', 'citations', 'valid_source_ids'.
    Returns average metrics across test cases.
    """
    if not test_cases:
        return {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "citation_precision": 0.0,
        }

    tot_faith = 0.0
    tot_rel = 0.0
    tot_prec = 0.0

    for item in test_cases:
        answer = item.get("answer", "")
        query = item.get("query", "")
        evidence = item.get("evidence_texts", [])
        citations = item.get("citations", [])
        valid_sources = item.get("valid_source_ids", [])

        tot_faith += calculate_faithfulness(answer, evidence)
        tot_rel += calculate_answer_relevance(answer, query)
        tot_prec += calculate_citation_precision(citations, valid_sources)

    n = len(test_cases)
    return {
        "faithfulness": tot_faith / n,
        "answer_relevance": tot_rel / n,
        "citation_precision": tot_prec / n,
    }
