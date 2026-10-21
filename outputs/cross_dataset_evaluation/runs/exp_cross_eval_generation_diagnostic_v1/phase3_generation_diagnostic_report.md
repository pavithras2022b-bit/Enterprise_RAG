# Phase 3 — LLM Answer Generation Diagnostic Report

**Status:** Diagnostic outputs generated; this report does not declare a project phase gate passed.

> **D0 limitation:** D0 is a **Validation-Derived Diagnostic Baseline**. Its 20 pilot questions originate from `lora_val.jsonl` and their answers appear in validation completions. It is not an independent test set; do not make unseen-test generalization claims from D0.

## Execution summary

Generated 75 per-query rows: 60 answerable questions with retrieved-reranked (Pass A) and gold-context (Pass B) generations, plus 15 refusal-candidate queries with a retrieved-reranked generation only. Pass A used Phase 4 top-five reranked contexts. Pass B used exact adjudicated indexed passages for the benchmark gold chunk IDs. No ChromaDB, Neo4j, source benchmark, model weights, or project system code was modified by this diagnostic.

The configured Ollama model tag was `qwen2.5-3b-lora-ragqa` (Qwen2 3.1B, Q4_K_M). The configured adapter files were present. Ollama served a GGUF model; its model metadata did not independently confirm whether the PEFT adapter was merged into the served artifact. The report therefore identifies the configured LoRA-named tag and records this verification limitation.

One generation request failed: `d0_prop_q1` Pass A returned HTTP 500 because the local llama-server could not allocate a ~1.33 GB CPU_REPACK buffer. The answer and affected Pass A metrics are blank in the finalized CSV, and the per-query error is retained. Side-by-side metrics and paired comparisons exclude this item for both regimes; therefore D0 n=19 and D2/D3 n=20. No retry was performed.

## Side-by-side metrics

Macro means over successful paired answerable questions (D0 n=19; D2/D3 n=20). The evidence-token score is a lexical overlap proxy, not entailment or factual grounding. Citation compliance means at least one cited ID exactly matched a supplied context ID; citation grounding rate is valid citation markers divided by all markers, with zero where no citation was emitted.

| Dataset | Regime | n | Token F1 | Semantic similarity | Citation compliance | Citation grounding | Evidence-token proxy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| D0 | Pass A — retrieved reranked | 19 | 0.7194 | 0.7715 | 1.0000 | 1.0000 | 0.9925 |
| D0 | Pass B — gold context | 19 | 0.6772 | 0.7611 | 1.0000 | 1.0000 | 0.9925 |
| D0 | Delta (B − A) | 19 | -0.0422 | -0.0104 | 0.0000 | 0.0000 | 0.0000 |
| D2 | Pass A — retrieved reranked | 20 | 0.4420 | 0.6375 | 0.5000 | 0.5000 | 0.9770 |
| D2 | Pass B — gold context | 20 | 0.4555 | 0.6290 | 1.0000 | 1.0000 | 0.9367 |
| D2 | Delta (B − A) | 20 | 0.0135 | -0.0085 | 0.5000 | 0.5000 | -0.0402 |
| D3 | Pass A — retrieved reranked | 20 | 0.4982 | 0.6989 | 0.6000 | 0.6000 | 0.9435 |
| D3 | Pass B — gold context | 20 | 0.5652 | 0.7057 | 0.9000 | 0.9000 | 0.9950 |
| D3 | Delta (B − A) | 20 | 0.0670 | 0.0068 | 0.3000 | 0.3000 | 0.0515 |

## Refusal-candidate results

The model produced a phrase-detector-positive refusal on 5/5 candidate queries in each dataset (15/15 overall; 1.000 candidate refusal accuracy). This is a keyword/phrase heuristic, not a semantic judgment. The 15 items were approved refusal-test candidates, but corpus-wide answer absence was not independently established; this must not be presented as verified out-of-domain refusal accuracy.

| Dataset | Candidates | Heuristic refusal detected | Candidate refusal accuracy |
|---|---:|---:|---:|
| D0 | 5 | 5 | 1.0000 |
| D2 | 5 | 5 | 1.0000 |
| D3 | 5 | 5 | 1.0000 |
| Overall | 15 | 15 | 1.0000 |

## Per-query improvement / degradation classifications

Classification uses successful paired rows only and paired token-F1 change (Pass B − Pass A): improvement > 0.001, degradation < −0.001, otherwise tie. Semantic-similarity changes are shown alongside. The failed `d0_prop_q1` Pass A generation is excluded from all paired comparisons and marked not scored below.

| Dataset | Paired n | Pass B improved | Pass A improved | Tie | Mean Δ F1 | Mean Δ similarity |
|---|---:|---:|---:|---:|---:|---:|
| D0 | 19 | 3 | 3 | 13 | -0.0422 | -0.0104 |
| D2 | 20 | 8 | 9 | 3 | 0.0135 | -0.0085 |
| D3 | 20 | 11 | 5 | 4 | 0.0670 | 0.0068 |

### Paired item detail

| Dataset | Question ID | Δ F1 (B−A) | Δ similarity (B−A) | Classification |
|---|---|---:|---:|---|
| D0 | d0_prop_q1 | n/a | n/a | Not scored — generation failed |
| D0 | d0_prop_q2 | +0.0000 | -0.0215 | Tie |
| D0 | d0_prop_q3 | +0.0000 | +0.0000 | Tie |
| D0 | d0_prop_q4 | +0.0420 | -0.0158 | Pass B improved |
| D0 | d0_prop_q5 | +0.0000 | +0.0000 | Tie |
| D0 | d0_prop_q6 | +0.0000 | +0.0000 | Tie |
| D0 | d0_prop_q7 | +0.0000 | +0.0000 | Tie |
| D0 | d0_prop_q8 | +0.0000 | +0.0077 | Tie |
| D0 | d0_prop_q9 | -0.5185 | -0.1737 | Pass A improved |
| D0 | d0_prop_q10 | +0.0000 | +0.0000 | Tie |
| D0 | d0_prop_q11 | +0.0000 | +0.0000 | Tie |
| D0 | d0_prop_q12 | +0.0000 | +0.0000 | Tie |
| D0 | d0_prop_q13 | -0.2222 | -0.2179 | Pass A improved |
| D0 | d0_prop_q14 | +0.0000 | +0.0000 | Tie |
| D0 | d0_prop_q15 | +0.0000 | +0.0000 | Tie |
| D0 | d0_prop_q16 | +0.3667 | +0.4154 | Pass B improved |
| D0 | d0_prop_q17 | +0.0000 | -0.0098 | Tie |
| D0 | d0_prop_q18 | +0.4190 | +0.4366 | Pass B improved |
| D0 | d0_prop_q19 | -0.8889 | -0.6186 | Pass A improved |
| D0 | d0_prop_q20 | +0.0000 | +0.0000 | Tie |
| D2 | d2_prop_q1 | +0.0758 | -0.0683 | Pass B improved |
| D2 | d2_prop_q2 | -0.0036 | -0.1135 | Pass A improved |
| D2 | d2_prop_q3 | -0.4565 | -0.4443 | Pass A improved |
| D2 | d2_prop_q4 | -0.0840 | -0.0842 | Pass A improved |
| D2 | d2_prop_q5 | -0.0065 | -0.1311 | Pass A improved |
| D2 | d2_prop_q6 | +0.1627 | -0.1644 | Pass B improved |
| D2 | d2_prop_q7 | +0.0000 | +0.0000 | Tie |
| D2 | d2_prop_q8 | +0.1785 | +0.0318 | Pass B improved |
| D2 | d2_prop_q9 | +0.0000 | +0.0000 | Tie |
| D2 | d2_prop_q10 | -0.1600 | -0.1072 | Pass A improved |
| D2 | d2_prop_q11 | +0.2759 | +0.3137 | Pass B improved |
| D2 | d2_prop_q12 | -0.0057 | +0.0364 | Pass A improved |
| D2 | d2_prop_q13 | +0.0862 | +0.0640 | Pass B improved |
| D2 | d2_prop_q14 | -0.1318 | -0.1080 | Pass A improved |
| D2 | d2_prop_q15 | +0.2181 | +0.4703 | Pass B improved |
| D2 | d2_prop_q16 | -0.0727 | -0.0115 | Pass A improved |
| D2 | d2_prop_q17 | +0.1595 | +0.0526 | Pass B improved |
| D2 | d2_prop_q18 | +0.0584 | +0.1323 | Pass B improved |
| D2 | d2_prop_q19 | -0.0244 | -0.0386 | Pass A improved |
| D2 | d2_prop_q20 | +0.0000 | +0.0000 | Tie |
| D3 | d3_prop_q1 | -0.1609 | +0.0240 | Pass A improved |
| D3 | d3_prop_q2 | +0.0317 | -0.1240 | Pass B improved |
| D3 | d3_prop_q3 | +0.0000 | +0.0000 | Tie |
| D3 | d3_prop_q4 | +0.0000 | +0.0000 | Tie |
| D3 | d3_prop_q5 | -0.0794 | -0.4967 | Pass A improved |
| D3 | d3_prop_q6 | -0.0572 | +0.0021 | Pass A improved |
| D3 | d3_prop_q7 | +0.0000 | +0.0000 | Tie |
| D3 | d3_prop_q8 | +0.0296 | -0.0904 | Pass B improved |
| D3 | d3_prop_q9 | +0.0455 | +0.0564 | Pass B improved |
| D3 | d3_prop_q10 | +0.0584 | +0.0480 | Pass B improved |
| D3 | d3_prop_q11 | +0.0000 | +0.0000 | Tie |
| D3 | d3_prop_q12 | +0.1696 | +0.0907 | Pass B improved |
| D3 | d3_prop_q13 | +0.1635 | +0.0892 | Pass B improved |
| D3 | d3_prop_q14 | +0.1477 | +0.0694 | Pass B improved |
| D3 | d3_prop_q15 | +0.1948 | +0.0394 | Pass B improved |
| D3 | d3_prop_q16 | +0.1523 | -0.0313 | Pass B improved |
| D3 | d3_prop_q17 | +0.1630 | +0.0672 | Pass B improved |
| D3 | d3_prop_q18 | +0.6789 | +0.4543 | Pass B improved |
| D3 | d3_prop_q19 | -0.0692 | -0.0539 | Pass A improved |
| D3 | d3_prop_q20 | -0.1286 | -0.0073 | Pass A improved |

## Error and interpretation

- Pass B vs Pass A is a diagnostic contrast between the supplied retrieved context and supplied gold context; it does not isolate every retrieval or generation factor and is not a causal estimate.
- In D2, gold context increased mean token F1 slightly (+0.0135), while mean semantic similarity decreased slightly (−0.0085); citation compliance increased from 0.50 to 1.00. This mixed signal does not support a simple retrieval-only explanation.
- In D3, gold context improved mean token F1 (+0.0670), semantic similarity (+0.0069), and citation compliance (+0.30), consistent with context selection contributing to some answer-quality loss. It does not establish that retrieval alone caused the loss.
- D0 has a stronger reference-overlap baseline than D2/D3, but its validation-derived provenance prevents independent generalization claims. D0 metrics in the table use the 19 successful paired items only.
- Low lexical overlap, embedding similarity, and citation metrics are imperfect proxies. Human review is needed to assess factual correctness, entailment, completeness, and whether citations support claims.

## Reproducibility and artifacts

The run manifest records configuration, input hashes, model metadata and limits, adapter file hashes, metric definitions, and the D0 claim restriction. Per-query raw outputs and citations are in `per_query_generation_diagnostic.csv`. The runner and report generator are preserved alongside this report.

No generation was performed for a gold-context refusal condition because no oracle gold context exists for those refusal candidates.
