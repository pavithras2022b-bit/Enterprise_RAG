from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_loader import create_embedding_model
from evaluation.generation_metrics import (
    calculate_answer_relevance,
    calculate_citation_precision,
    calculate_faithfulness,
)
from llm.client import LLMClient
from reranking.schema import RankedCandidate


V2_EVIDENCE_FILE = (
    PROJECT_ROOT / "outputs" / "vector_only_v2_evidence.json"
)

VAL_FILE = (
    PROJECT_ROOT / "data" / "labeled" / "lora_val.jsonl"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "outputs" / "vector_only_v2_generation_evaluation.json"
)

NUM_QUERIES = 15

BASE_URL = "http://localhost:11434"
TIMEOUT = 180.0
TEMPERATURE = 0.2

MODELS = [
    {
        "model_name": "llama3.2:3b",
        "label": "Llama 3.2-3B",
    },
    {
        "model_name": "qwen2.5-3b-lora-ragqa",
        "label": "Qwen2.5-3B + LoRA",
    },
]


def parse_validation_examples():
    examples = []

    with open(VAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)

            prompt = item["prompt"]
            completion = item["completion"].strip()

            question_match = re.search(
                r"Question:\s*(.*?)\s*Answer:\s*$",
                prompt,
                flags=re.DOTALL,
            )

            if not question_match:
                continue

            query = question_match.group(1).strip()

            examples.append(
                {
                    "query": query,
                    "reference_answer": completion,
                }
            )

    return examples


def load_v2_evidence():
    with open(V2_EVIDENCE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["per_query_results"]


def remove_citations(text: str) -> str:
    return re.sub(
        r"\[[^\[\]]+\]",
        "",
        text,
    ).strip()


def normalize_answer(text: str) -> str:
    text = remove_citations(text)
    text = text.lower()

    # Remove punctuation.
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove English articles.
    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text,
    )

    # Normalize whitespace.
    text = " ".join(text.split())

    return text


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)

    return (
        2 * precision * recall
        / (precision + recall)
    )


def exact_match(prediction: str, reference: str) -> float:
    return (
        1.0
        if normalize_answer(prediction)
        == normalize_answer(reference)
        else 0.0
    )


def cosine_similarity(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b)
        / denominator
    )


def semantic_similarity(
    embedding_model,
    prediction: str,
    reference: str,
) -> float:

    pred_text = remove_citations(prediction)
    ref_text = remove_citations(reference)

    pred_embedding = embedding_model.embed(pred_text)
    ref_embedding = embedding_model.embed(ref_text)

    return cosine_similarity(
        pred_embedding,
        ref_embedding,
    )


def build_ranked_candidates(item):
    return [
        RankedCandidate(
            candidate_id=evidence["candidate_id"],
            text=evidence["text"],
            rerank_score=float(
                evidence["rerank_score"]
            ),
            original_candidate=None,
            rank=int(evidence["rank"]),
        )
        for evidence in item["v2_evidence"]
    ]


def main():
    print("=" * 80)
    print("VECTOR-ONLY ABLATION V2 — LLM GENERATION EVALUATION")
    print("=" * 80)

    print("[1/3] Loading frozen V2 evidence...")
    v2_items = load_v2_evidence()

    if len(v2_items) != NUM_QUERIES:
        raise RuntimeError(
            f"Expected {NUM_QUERIES} V2 queries, "
            f"found {len(v2_items)}."
        )

    print("[2/3] Loading validation references...")
    validation_examples = parse_validation_examples()

    references_by_query = {
        item["query"]: item["reference_answer"]
        for item in validation_examples
    }

    print("[3/3] Loading MiniLM for semantic similarity...")
    embedding_model = create_embedding_model()

    all_results = []
    model_summaries = {}

    for model_info in MODELS:

        model_name = model_info["model_name"]
        model_label = model_info["label"]

        print()
        print("#" * 80)
        print(f"MODEL: {model_label}")
        print("#" * 80)

        client = LLMClient(
            base_url=BASE_URL,
            model_name=model_name,
            timeout=TIMEOUT,
            temperature=TEMPERATURE,
        )

        model_results = []

        for item in v2_items:

            index = item["query_index"]
            query = item["query"]

            if query not in references_by_query:
                raise RuntimeError(
                    f"No reference answer found for: {query}"
                )

            reference = references_by_query[query]

            evidence = build_ranked_candidates(item)

            valid_ids = [
                candidate.candidate_id
                for candidate in evidence
            ]

            print()
            print("=" * 80)
            print(
                f"{model_label} — "
                f"QUERY {index:02d}/{NUM_QUERIES}"
            )
            print(query)
            print("-" * 80)

            try:
                generated = client.generate_answer(
                    query=query,
                    evidence=evidence,
                )

                answer = generated.text
                citations = list(
                    generated.citations
                )
                latency_ms = float(
                    generated.latency_ms
                )
                runtime_failure = False

            except Exception as exc:
                print(
                    f"GENERATION FAILURE: {exc}"
                )

                answer = ""
                citations = []
                latency_ms = 0.0
                runtime_failure = True

            evidence_texts = [
                candidate.text
                for candidate in evidence
            ]

            if runtime_failure:
                f1 = 0.0
                em = 0.0
                sem = 0.0
                faithfulness = 0.0
                relevance = 0.0
                citation_precision = 0.0
                citation_compliance = 0.0

            else:
                f1 = token_f1(
                    answer,
                    reference,
                )

                em = exact_match(
                    answer,
                    reference,
                )

                sem = semantic_similarity(
                    embedding_model,
                    answer,
                    reference,
                )

                faithfulness = (
                    calculate_faithfulness(
                        answer,
                        evidence_texts,
                    )
                )

                relevance = (
                    calculate_answer_relevance(
                        answer,
                        query,
                    )
                )

                citation_precision = (
                    calculate_citation_precision(
                        citations,
                        valid_ids,
                    )
                )

                citation_compliance = (
                    1.0
                    if citations
                    else 0.0
                )

            print(f"Answer: {answer}")
            print(f"Reference: {reference}")
            print(f"Citations: {citations}")

            print(
                f"Token F1: "
                f"{f1:.4f}"
            )

            print(
                f"Semantic similarity: "
                f"{sem:.4f}"
            )

            print(
                f"Exact match: "
                f"{em:.0f}"
            )

            print(
                f"Citation compliance: "
                f"{citation_compliance:.0f}"
            )

            print(
                f"Latency: "
                f"{latency_ms / 1000:.2f} s"
            )

            result = {
                "model": model_name,
                "model_label": model_label,

                "query_index": index,
                "query": query,

                "reference_answer": reference,
                "answer": answer,

                "citations": citations,
                "valid_source_ids": valid_ids,

                "token_f1": f1,
                "semantic_similarity": sem,
                "exact_match": em,

                "faithfulness": faithfulness,
                "answer_relevance": relevance,

                "citation_precision": (
                    citation_precision
                ),

                "citation_compliance": (
                    citation_compliance
                ),

                "latency_ms": latency_ms,
                "runtime_failure": runtime_failure,
            }

            model_results.append(result)
            all_results.append(result)

        valid = [
            item
            for item in model_results
            if not item["runtime_failure"]
        ]

        if not valid:
            raise RuntimeError(
                f"All generations failed for "
                f"{model_label}."
            )

        summary = {
            "model": model_name,
            "model_label": model_label,

            "queries_attempted": NUM_QUERIES,
            "valid_generations": len(valid),
            "runtime_failures": (
                NUM_QUERIES - len(valid)
            ),

            "token_f1": statistics.mean(
                x["token_f1"]
                for x in valid
            ),

            "semantic_similarity": statistics.mean(
                x["semantic_similarity"]
                for x in valid
            ),

            "exact_match": statistics.mean(
                x["exact_match"]
                for x in valid
            ),

            "faithfulness": statistics.mean(
                x["faithfulness"]
                for x in valid
            ),

            "answer_relevance": statistics.mean(
                x["answer_relevance"]
                for x in valid
            ),

            "citation_precision": statistics.mean(
                x["citation_precision"]
                for x in valid
            ),

            "citation_compliance": statistics.mean(
                x["citation_compliance"]
                for x in valid
            ),

            "average_latency_ms": statistics.mean(
                x["latency_ms"]
                for x in valid
            ),
        }

        model_summaries[model_name] = summary

        print()
        print("=" * 80)
        print(
            f"{model_label} — V2 SUMMARY"
        )
        print("=" * 80)

        print(
            f"Token F1:              "
            f"{summary['token_f1']:.6f}"
        )

        print(
            f"Semantic similarity:   "
            f"{summary['semantic_similarity']:.6f}"
        )

        print(
            f"Exact Match:           "
            f"{summary['exact_match'] * 100:.2f}%"
        )

        print(
            f"Citation Compliance:   "
            f"{summary['citation_compliance'] * 100:.2f}%"
        )

        print(
            f"Faithfulness proxy:    "
            f"{summary['faithfulness']:.6f}"
        )

        print(
            f"Query-overlap proxy:   "
            f"{summary['answer_relevance']:.6f}"
        )

        print(
            f"Average CPU latency:   "
            f"{summary['average_latency_ms'] / 1000:.2f} s"
        )

    output = {
        "evaluation_type": (
            "vector_only_v2_generation_model_comparison"
        ),

        "architecture": (
            "vector-only retrieval + "
            "cross-encoder reranking + "
            "grounded LLM generation"
        ),

        "experimental_change_from_v1": (
            "Knowledge-graph retrieval and learned "
            "adaptive hybrid fusion are removed. "
            "Vector retrieval, cross-encoder reranking, "
            "Top-5 evidence size, PromptBuilder, "
            "temperature=0.2, citation validation, "
            "query set, reference answers, and scoring "
            "methodology remain controlled."
        ),

        "evidence_file": str(
            V2_EVIDENCE_FILE
        ),

        "models": model_summaries,

        "per_query_results": all_results,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("#" * 80)
    print("VECTOR-ONLY V2 — FINAL MODEL COMPARISON")
    print("#" * 80)

    llama = model_summaries[
        "llama3.2:3b"
    ]

    qwen = model_summaries[
        "qwen2.5-3b-lora-ragqa"
    ]

    print()
    print(
        f"{'Metric':<28}"
        f"{'Llama V2':>15}"
        f"{'Qwen V2':>15}"
    )

    print("-" * 58)

    print(
        f"{'Token F1':<28}"
        f"{llama['token_f1']:>15.4f}"
        f"{qwen['token_f1']:>15.4f}"
    )

    print(
        f"{'Semantic Similarity':<28}"
        f"{llama['semantic_similarity']:>15.4f}"
        f"{qwen['semantic_similarity']:>15.4f}"
    )

    print(
        f"{'Exact Match':<28}"
        f"{llama['exact_match'] * 100:>14.2f}%"
        f"{qwen['exact_match'] * 100:>14.2f}%"
    )

    print(
        f"{'Citation Compliance':<28}"
        f"{llama['citation_compliance'] * 100:>14.2f}%"
        f"{qwen['citation_compliance'] * 100:>14.2f}%"
    )

    print(
        f"{'Faithfulness Proxy':<28}"
        f"{llama['faithfulness']:>15.4f}"
        f"{qwen['faithfulness']:>15.4f}"
    )

    print(
        f"{'Query-Overlap Proxy':<28}"
        f"{llama['answer_relevance']:>15.4f}"
        f"{qwen['answer_relevance']:>15.4f}"
    )

    print(
        f"{'CPU Latency (s)':<28}"
        f"{llama['average_latency_ms'] / 1000:>15.2f}"
        f"{qwen['average_latency_ms'] / 1000:>15.2f}"
    )

    print()
    print(f"Saved: {OUTPUT_FILE}")
    print("#" * 80)


if __name__ == "__main__":
    main()
