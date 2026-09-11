"""
Production Pipeline Runner & System Evaluation Benchmark
Enterprise Hybrid RAG System - Meridian Dynamics Inc.

Author: Pavithra S. (Integrated M.Tech Software Engineering, VIT Vellore)
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("EnterpriseRAG.Production")

from api.pipeline import EnterpriseRAGPipeline
from config_loader import (
    create_embedding_model,
    create_vector_store,
    create_neo4j_connection,
    create_entity_extractor,
    create_relation_classifier,
    create_cross_encoder_reranker,
    create_llm_client,
)
from evaluation.retrieval_metrics import (
    calculate_hit_rate,
    calculate_mrr,
    calculate_ndcg,
    evaluate_retrieval_batch,
)
from evaluation.generation_metrics import (
    calculate_faithfulness,
    calculate_citation_precision,
    calculate_answer_relevance,
    evaluate_generation_batch,
)
from evaluation.ablation_runner import AblationRunner, EvaluationExample
from adaptive_retrieval.fusion import AdaptiveFusionEngine
from adaptive_retrieval.rules import RuleBasedAlphaClassifier


def run_corpus_ingestion(pipeline: EnterpriseRAGPipeline) -> dict:
    """Ingest 140 enterprise documents from data/raw/documents/."""
    raw_dir = PROJECT_ROOT / "data" / "raw" / "documents"
    if not raw_dir.exists():
        logger.error("Raw documents directory %s does not exist!", raw_dir)
        return {"status": "error", "error": f"Directory not found: {raw_dir}"}

    file_paths = [str(p) for p in sorted(raw_dir.glob("*")) if p.is_file()]
    logger.info("Found %d enterprise documents for ingestion in %s", len(file_paths), raw_dir)

    start = time.perf_counter()
    response = pipeline.ingest_files(file_paths)
    elapsed = time.perf_counter() - start

    logger.info("=" * 60)
    logger.info("INGESTION COMPLETED in %.2f seconds", elapsed)
    logger.info("  Documents processed: %d", response.documents_processed)
    logger.info("  Chunks created:     %d", response.chunks_created)
    logger.info("  Entities extracted: %d", response.entities_extracted)
    logger.info("  Relations extracted:%d", response.relations_extracted)
    logger.info("  Vector store count: %d", pipeline.vector_store.count())
    logger.info("=" * 60)

    return {
        "documents_processed": response.documents_processed,
        "chunks_created": response.chunks_created,
        "entities_extracted": response.entities_extracted,
        "relations_extracted": response.relations_extracted,
        "vector_store_count": pipeline.vector_store.count(),
        "elapsed_seconds": elapsed,
    }


def parse_val_examples(val_file: Path) -> list[EvaluationExample]:
    """Parse validation dataset supporting both prompt/completion and structured query/chunk_ids format."""
    examples: list[EvaluationExample] = []
    with open(val_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            query = None
            gold_chunk_ids = []

            if "prompt" in item:
                prompt_text = item["prompt"]
                q_match = re.search(r"Question:\s*(.+?)(?:\nAnswer:|\Z)", prompt_text, re.DOTALL)
                if q_match:
                    query = q_match.group(1).strip()
            elif "query" in item or "question" in item:
                query = item.get("query") or item.get("question")

            if "completion" in item:
                gold_chunk_ids = re.findall(r"\[([a-zA-Z0-9_-]+_chunk_\d+)\]", item["completion"])
            elif "chunk_ids" in item:
                gold_chunk_ids = item["chunk_ids"]
            elif "source_id" in item:
                gold_chunk_ids = [item["source_id"]]

            if query and gold_chunk_ids:
                examples.append(EvaluationExample(query=query, ground_truth_chunk_ids=tuple(gold_chunk_ids)))

    return examples


def run_ablation_evaluation(pipeline: EnterpriseRAGPipeline) -> dict:
    """Run comparative retrieval ablation over validation QA pairs."""
    val_file = PROJECT_ROOT / "data" / "labeled" / "lora_val.jsonl"
    if not val_file.exists():
        logger.warning("Validation file %s not found; skipping ablation benchmark.", val_file)
        return {}

    examples = parse_val_examples(val_file)
    logger.info("Loaded %d parsed validation examples for retrieval ablation.", len(examples))
    if not examples:
        logger.warning("No valid examples extracted from %s", val_file)
        return {}
    # Learned adaptive engine already belongs to the production pipeline.
    learned_runner = AblationRunner(
        hybrid_controller=pipeline.hybrid_controller,
        adaptive_engine=pipeline.adaptive_engine,
    )

    learned_results = learned_runner.run_all(examples, k=5)

    # Evaluation-only rule-based adaptive engine.
    rule_based_engine = AdaptiveFusionEngine(
        controller=pipeline.hybrid_controller,
        embedding_model=pipeline.embedding_model,
        entity_linker=pipeline.entity_linker,
        classifier=RuleBasedAlphaClassifier(),
    )

    rule_based_runner = AblationRunner(
        hybrid_controller=pipeline.hybrid_controller,
        adaptive_engine=rule_based_engine,
    )

    rule_based_result = rule_based_runner.evaluate_strategy(
        strategy_name="rule_based_adaptive",
        examples=examples,
        use_adaptive=True,
        k=5,
    )

    results = {
        "vector_only": learned_results["vector_only"],
        "graph_only": learned_results["graph_only"],
        "static_hybrid": learned_results["static_hybrid"],
        "rule_based_adaptive": rule_based_result,
        "learned_adaptive": learned_results["adaptive_hybrid"],
    }
    logger.info("=" * 65)
    logger.info("RETRIEVAL ABLATION BENCHMARK RESULTS (Top-5)")
    logger.info("=" * 65)
    logger.info("%-22s | %-12s | %-10s | %-10s", "Strategy", "HitRate@5", "MRR@5", "NDCG@5")
    logger.info("-" * 65)
    for name, res in results.items():
        logger.info(
            "%-22s | %12.4f | %10.4f | %10.4f",
            name,
            res.hit_rate,
            res.mrr,
            res.ndcg,
        )
    logger.info("=" * 65)

    return {
        name: {
            "hit_rate": res.hit_rate,
            "mrr": res.mrr,
            "ndcg": res.ndcg,
        }
        for name, res in results.items()
    }


def run_negative_query_evaluation(pipeline: EnterpriseRAGPipeline) -> dict:
    """
    Evaluate grounding robustness using hard-negative question/evidence
    pairs from negatives.jsonl.

    Each record contains a question paired with deliberately irrelevant
    evidence. The model should refuse rather than answer from unsupported
    context.
    """
    from reranking.schema import RankedCandidate

    neg_file = PROJECT_ROOT / "data" / "labeled" / "negatives.jsonl"

    logger.info("Testing hard-negative question/evidence pairs from %s", neg_file)

    negative_examples = []

    if neg_file.exists():
        with neg_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                item = json.loads(line)

                question = item.get("question") or item.get("query")
                candidate_id = item.get("candidate_id")
                text = item.get("text")

                if question and candidate_id and text:
                    negative_examples.append(
                        (question, candidate_id, text)
                    )

    tested = 0
    refusals = 0
    unsupported_citations = 0

    refusal_keywords = [
        "insufficient",
        "not found",
        "does not contain",
        "cannot determine",
        "no relevant",
        "cannot find",
        "no information",
        "not specified",
        "not enough information",
        "i don't have enough information",
        "there is no mention",
        "no mention of",
        "not enough context",
        "does not provide enough information",
        "does not mention",
        "don't have enough context",
        "the context only",
        "cannot answer",
        "unable to determine",
    ]

    for question, candidate_id, text in negative_examples[:15]:
        tested += 1

        negative_evidence = RankedCandidate(
            candidate_id=candidate_id,
            text=text,
            rerank_score=0.0,
            original_candidate=None,
            rank=1,
        )

        response = pipeline.llm_client.generate_answer(
            query=question,
            evidence=[negative_evidence],
        )

        answer_lower = response.text.lower()
        is_refusal = any(
            phrase in answer_lower
            for phrase in refusal_keywords
        )

        logger.info("-" * 60)
        logger.info("NEGATIVE TEST %d", tested)
        logger.info("Question: %s", question)
        logger.info("Evidence ID: %s", candidate_id)
        logger.info("LLM Answer: %s", response.text)
        logger.info("Detected Refusal: %s", is_refusal)
        logger.info("Citations: %s", list(response.citations))
        logger.info("-" * 60)

        if is_refusal:
            refusals += 1
        else:
            unsupported_citations += len(response.citations)

    refusal_rate = refusals / tested if tested > 0 else 1.0

    logger.info("=" * 60)
    logger.info("HARD-NEGATIVE GROUNDING ROBUSTNESS")
    logger.info("=" * 60)
    logger.info("  Queries Tested:              %d", tested)
    logger.info(
        "  Correct Refusals/Flags:     %d (%.1f%%)",
        refusals,
        refusal_rate * 100.0,
    )
    logger.info(
        "  Citations on Non-Refusals:  %d",
        unsupported_citations,
    )
    logger.info("=" * 60)

    return {
        "tested": tested,
        "refusals": refusals,
        "refusal_rate": refusal_rate,
        "unsupported_citations": unsupported_citations,
    }


def run_corpus_negative_evaluation(
    pipeline: EnterpriseRAGPipeline,
    start_at: int = 1,
) -> dict:
    """
    Evaluate full-RAG refusal behavior on genuinely unanswerable
    corpus-level questions from negatives.jsonl.

    Unlike the hard-negative evaluator, this intentionally ignores the
    supplied candidate_id/text distractor and sends each question through
    the complete retrieval -> reranking -> generation pipeline.
    """

    neg_file = PROJECT_ROOT / "data" / "labeled" / "negatives.jsonl"

    logger.info(
        "Testing corpus-level unanswerable questions through full RAG from %s",
        neg_file,
    )

    questions = []

    if neg_file.exists():
        with neg_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                item = json.loads(line)
                question = item.get("question") or item.get("query")

                if question:
                    questions.append(question)

    refusal_keywords = [
        "insufficient",
        "not found",
        "does not contain",
        "cannot determine",
        "no relevant",
        "cannot find",
        "no information",
        "not specified",
        "not enough information",
        "i don't have enough information",
        "there is no mention",
        "no mention of",
        "not enough context",
        "does not provide enough information",
        "does not mention",
        "don't have enough context",
        "the context only",
        "cannot answer",
        "unable to determine",
    ]

    tested = 0
    refusals = 0
    non_refusals = 0
    runtime_failures = 0
    citations_on_non_refusals = 0

    selected_questions = questions[max(start_at - 1, 0):]

    for absolute_index, question in enumerate(
        selected_questions,
        start=max(start_at, 1),
    ):
        tested += 1

        response = pipeline.answer_query(
            question,
            top_k=5,
            use_adaptive=True,
        )

        answer_lower = response.answer.lower()

        is_runtime_failure = (
            response.answer.startswith(
                "Based on the retrieved policy documents:"
            )
            or response.answer
            == "No relevant policy documents found for this query."
        )

        is_refusal = (
            not is_runtime_failure
            and any(
                phrase in answer_lower
                for phrase in refusal_keywords
            )
        )

        logger.info("-" * 70)
        logger.info("CORPUS NEGATIVE TEST %d", absolute_index)
        logger.info("Question: %s", question)
        logger.info("Alpha: %.4f", response.alpha_used)
        logger.info("Answer: %s", response.answer)
        logger.info("Detected Refusal: %s", is_refusal)
        logger.info("Runtime Failure: %s", is_runtime_failure)
        logger.info("Citations: %s", response.citations)

        if response.source_candidates:
            logger.info(
                "Retrieved Sources: %s",
                [
                    (
                        candidate.source,
                        candidate.candidate_id,
                        round(candidate.fused_score, 4),
                    )
                    for candidate in response.source_candidates
                ],
            )

        logger.info("-" * 70)

        if is_runtime_failure:
            runtime_failures += 1
        elif is_refusal:
            refusals += 1
        else:
            non_refusals += 1
            citations_on_non_refusals += len(response.citations)

    valid_tests = tested - runtime_failures
    refusal_rate = (
        refusals / valid_tests
        if valid_tests > 0
        else 0.0
    )

    logger.info("=" * 70)
    logger.info("CORPUS-LEVEL UNANSWERABLE QUERY EVALUATION")
    logger.info("=" * 70)
    logger.info("  Queries Tested:             %d", tested)
    logger.info(
        "  Correct Refusals:           %d (%.1f%%)",
        refusals,
        refusal_rate * 100.0,
    )
    logger.info("  Non-Refusals:               %d", non_refusals)
    logger.info("  Runtime Failures:            %d", runtime_failures)
    logger.info(
        "  Citations on Non-Refusals: %d",
        citations_on_non_refusals,
    )
    logger.info("=" * 70)

    return {
        "tested": tested,
        "refusals": refusals,
        "refusal_rate": refusal_rate,
        "non_refusals": non_refusals,
        "runtime_failures": runtime_failures,
        "citations_on_non_refusals": citations_on_non_refusals,
    }


def main():
    logger.info("Initializing Enterprise RAG Pipeline for production execution...")
    embed_model = create_embedding_model()
    vector_store = create_vector_store()
    neo4j_conn = create_neo4j_connection()
    entity_extractor = create_entity_extractor()
    relation_classifier = create_relation_classifier()
    reranker = create_cross_encoder_reranker()
    llm_client = create_llm_client()

    pipeline = EnterpriseRAGPipeline(
        neo4j_connection=neo4j_conn,
        vector_store=vector_store,
        embedding_model=embed_model,
        entity_extractor=entity_extractor,
        relation_classifier=relation_classifier,
        reranker=reranker,
        llm_client=llm_client,
    )

    # 1. Ingestion
    ingestion_results = run_corpus_ingestion(pipeline)

    # 2. Retrieval Ablation
    ablation_results = run_ablation_evaluation(pipeline)

    # 3. Negative Queries
    negative_results = run_negative_query_evaluation(pipeline)

    # 4. Sample End-to-End Grounded Queries
    sample_queries = [
        "What is the annual leave policy for full-time employees at Meridian Dynamics?",
        "Which IT systems are governed by the Enterprise Security Policy?",
        "Who manages the Cloud Infrastructure engineering team?",
    ]

    logger.info("Executing sample production queries:")
    sample_outputs = []
    for q in sample_queries:
        res = pipeline.answer_query(q, top_k=3, use_adaptive=True)
        sample_outputs.append({
            "query": q,
            "alpha_used": res.alpha_used,
            "latency_ms": res.latency_ms,
            "citations": res.citations,
            "answer_preview": res.answer[:200] + "..." if len(res.answer) > 200 else res.answer,
        })
        logger.info("\n[QUERY]: %s", q)
        logger.info("[ALPHA]: %.3f | [LATENCY]: %.1f ms", res.alpha_used, res.latency_ms)
        logger.info("[CITATIONS]: %s", res.citations)
        logger.info("[ANSWER]: %s", res.answer)

    # Save summary report
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "production_evaluation_summary.json"

    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ingestion": ingestion_results,
        "ablation_retrieval": ablation_results,
        "negative_evaluation": negative_results,
        "sample_queries": sample_outputs,
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    logger.info("\nProduction evaluation summary successfully saved to %s", report_file)


if __name__ == "__main__":
    main()
