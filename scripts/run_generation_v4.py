from __future__ import annotations

import json
import re
import chromadb
from chromadb.config import Settings
from retrieval.schema import render_predicate_phrase
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_loader import create_llm_client
from evaluation.generation_metrics import (
    calculate_answer_relevance,
    calculate_citation_precision,
    calculate_faithfulness,
)
from reranking.schema import RankedCandidate


VAL_FILE = PROJECT_ROOT / "data" / "labeled" / "lora_val.jsonl"
V1_FILE = PROJECT_ROOT / "outputs" / "llm_generation_evaluation.json"

OUTPUT_FILE = (
    PROJECT_ROOT / "outputs" / "llm_generation_evaluation_v4.json"
)

MODEL_NAME = "qwen2.5-3b-lora-ragqa"
MODEL_LABEL = "Qwen2.5 3B + LoRA — Prompt V2"
NUM_QUERIES = 15


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

            context_match = re.search(
                r"Context:\s*(.*?)\s*Question:",
                prompt,
                flags=re.DOTALL,
            )

            if not context_match:
                continue

            context = context_match.group(1)

            evidence = []

            matches = list(
                re.finditer(
                    r"\[([^\[\]\n]+)\]\s*",
                    context,
                )
            )

            for i, match in enumerate(matches):
                candidate_id = match.group(1).strip()

                start = match.end()
                end = (
                    matches[i + 1].start()
                    if i + 1 < len(matches)
                    else len(context)
                )

                text = context[start:end].strip()

                evidence.append(
                    {
                        "candidate_id": candidate_id,
                        "text": text,
                    }
                )

            examples.append(
                {
                    "query": query,
                    "reference_answer": completion,
                    "evidence": evidence,
                }
            )

    return examples


def load_v1_qwen_results():
    with open(V1_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = [
        item
        for item in data["per_query_results"]
        if item["model"] == MODEL_NAME
    ]

    results.sort(key=lambda x: x["query_index"])

    return results


def remove_citations(text: str) -> str:
    return re.sub(r"\[[^\[\]]+\]", "", text).strip()


def main():
    examples = parse_validation_examples()
    v1_results = load_v1_qwen_results()

    if len(v1_results) < NUM_QUERIES:
        raise RuntimeError(
            f"Expected at least {NUM_QUERIES} frozen Qwen V1 results, "
            f"found {len(v1_results)}."
        )

    # Match validation examples by exact query instead of assuming
    # that the first 15 JSONL rows are the evaluation subset.
    examples_by_query = {
        item["query"]: item
        for item in examples
    }

    llm_client = create_llm_client()
    chroma_client = chromadb.PersistentClient(
    path="data/chroma_store",
    settings=Settings(anonymized_telemetry=False),
)
    chroma_collection = chroma_client.get_collection("enterprise_docs")
    per_query_results = []

    for index, v1 in enumerate(v1_results[:NUM_QUERIES], start=1):
        query = v1["query"]

        if query not in examples_by_query:
            raise RuntimeError(
                f"Could not find validation example for query: {query}"
            )

        example = examples_by_query[query]

        frozen_ids = v1["valid_source_ids"]


        chunk_ids = [cid for cid in frozen_ids if "|" not in cid]


        batch = chroma_collection.get(
            ids=chunk_ids,
            include=["documents"],
        )

        evidence_by_id = {
            cid: text or ""
            for cid, text in zip(
                batch.get("ids", []),
                batch.get("documents", []),
            )
        }

        for cid in frozen_ids:
            if "|" in cid:
                subject, predicate, obj = cid.split("|", 2)
                evidence_by_id[cid] = (
                    f"{subject} {render_predicate_phrase(predicate)} {obj}."
                )

        missing_ids = [
            cid for cid in frozen_ids
            if cid not in evidence_by_id
        ]

        if missing_ids:
            raise RuntimeError(
                f"Query {index}: frozen V1 evidence could not be reconstructed: "
                f"{missing_ids}"
            )
        ranked_candidates = [
            RankedCandidate(
                candidate_id=candidate_id,
                text=evidence_by_id[candidate_id],
                rerank_score=float(len(frozen_ids) - rank),
                original_candidate=None,
                rank=rank + 1,
            )
            for rank, candidate_id in enumerate(frozen_ids)
        ]

        print()
        print("=" * 70)
        print(f"V4 QUERY {index}/{NUM_QUERIES}")
        print(query)
        print("=" * 70)

        try:
            generated = llm_client.generate_answer(
                query=query,
                evidence=ranked_candidates,
            )

            answer = generated.text
            citations = list(generated.citations)
            latency_ms = generated.latency_ms
            runtime_failure = False

        except Exception as exc:
            print(f"GENERATION FAILURE: {exc}")

            answer = ""
            citations = []
            latency_ms = 0.0
            runtime_failure = True

        evidence_texts = [
            candidate.text
            for candidate in ranked_candidates
        ]

        faithfulness = calculate_faithfulness(
            answer,
            evidence_texts,
        )

        answer_relevance = calculate_answer_relevance(
            answer,
            query,
        )

        citation_precision = calculate_citation_precision(
            citations,
            frozen_ids,
        )

        citation_compliance = 1.0 if citations else 0.0

        print(f"Answer: {answer}")
        print(f"Citations: {citations}")
        print(f"Latency: {latency_ms:.2f} ms")
        print(f"Faithfulness proxy: {faithfulness:.4f}")
        print(f"Query-overlap proxy: {answer_relevance:.4f}")

        per_query_results.append(
            {
                "model": MODEL_NAME,
                "model_label": MODEL_LABEL,
                "query_index": index,
                "query": query,
                "answer": answer,
                "reference_answer": example["reference_answer"],
                "citations": citations,
                "valid_source_ids": frozen_ids,
                "retrieved_evidence_count": len(ranked_candidates),
                "alpha_v1": v1.get("alpha"),
                "latency_ms": latency_ms,
                "faithfulness": faithfulness,
                "answer_relevance": answer_relevance,
                "citation_precision": citation_precision,
                "citation_compliance": citation_compliance,
                "runtime_failure": runtime_failure,
            }
        )

    valid = [
        item
        for item in per_query_results
        if not item["runtime_failure"]
    ]

    if not valid:
        raise RuntimeError("All V4 generations failed.")

    summary = {
        "model": MODEL_NAME,
        "model_label": MODEL_LABEL,
        "queries_attempted": NUM_QUERIES,
        "valid_generations": len(valid),
        "runtime_failures": NUM_QUERIES - len(valid),
        "faithfulness": statistics.mean(
            item["faithfulness"] for item in valid
        ),
        "answer_relevance": statistics.mean(
            item["answer_relevance"] for item in valid
        ),
        "citation_precision": statistics.mean(
            item["citation_precision"] for item in valid
        ),
        "citation_compliance": statistics.mean(
            item["citation_compliance"] for item in valid
        ),
        "average_latency_ms": statistics.mean(
            item["latency_ms"] for item in valid
        ),
    }

    output = {
        "evaluation_type": "controlled_qwen_v4_temperature_zero_generation",
        "validation_file": str(VAL_FILE),
        "queries_evaluated": NUM_QUERIES,
        "experimental_change": (
            "V4 decoding ablation: pre-V3 prompt with temperature=0.0; "
            "retrieved evidence IDs frozen from Qwen V1."
        ),
        "models": {
            MODEL_NAME: summary,
        },
        "per_query_results": per_query_results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 70)
    print("GENERATION V4 SUMMARY")
    print("=" * 70)
    print(f"Valid generations:       {len(valid)}/{NUM_QUERIES}")
    print(f"Faithfulness proxy:      {summary['faithfulness']:.6f}")
    print(f"Query-overlap proxy:     {summary['answer_relevance']:.6f}")
    print(f"Citation precision:      {summary['citation_precision']:.6f}")
    print(f"Citation compliance:     {summary['citation_compliance']:.6f}")
    print(
        f"Average generation time: "
        f"{summary['average_latency_ms'] / 1000:.2f} s"
    )
    print()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
