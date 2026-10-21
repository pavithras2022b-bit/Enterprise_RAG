# Phase 6 — Full Adaptive-Hybrid Cross-Dataset Diagnostic

> **D0 claim restriction:** D0 is a **Validation-Derived Diagnostic Baseline**. All 20 pilot questions match questions from `lora_val.jsonl` and their answers appear in validation completions. It is not an independent or unseen test set. Results support diagnostic statements only.

**Execution status:** Completed 75/75 query rows (60 answerable and 15 refusal-test candidates); zero recorded generation errors. This report does not independently certify a phase gate.

## Phase 5 graph verification

D2/D3 graph populations were already present and agreed with recorded ingestion summaries. Phase 5 performed read-only count/provenance verification; it did not re-extract entities or write/modify graph data. Counts and limitations are documented in [phase5_graph_ingestion_summary.md](phase5_graph_ingestion_summary.md). Existing edge coverage is sparse, so graph existence must not be interpreted as complete knowledge coverage.

## Matched retrieval comparison

Macro averages over 20 answerable questions per dataset. Phase 2 used dense vector top-5; Phase 4 reranked the Phase 2 top-20 dense candidate pool; Phase 6 fused top-20 vector and graph candidates using the configured learned alpha controller, reranked the merged pool, and evaluated the final five. Delta columns are arithmetic differences between adjacent phases. Precision@5 for Phase 6 is gold chunks in the five evaluated evidence slots divided by five; unresolved graph facts without a source chunk are non-matches.

| Dataset | Metric | P2 Vector @5 | P4 Reranked @5 | P6 Adaptive-Hybrid @5 | Δ P4−P2 | Δ P6−P4 |
|---|---|---|---|---|---|---|
| D0 | Hit Rate @ 5 | 0.9000 | 1.0000 | 1.0000 | 0.1000 | 0.0000 |
| D0 | MRR @ 5 | 0.8250 | 0.9167 | 0.9167 | 0.0917 | 0.0000 |
| D0 | NDCG @ 5 | 0.8446 | 0.9381 | 0.9381 | 0.0935 | 0.0000 |
| D0 | Precision @ 5 | 0.1800 | 0.2000 | 0.2000 | 0.0200 | 0.0000 |
| D2 | Hit Rate @ 5 | 0.6500 | 0.9000 | 0.9000 | 0.2500 | 0.0000 |
| D2 | MRR @ 5 | 0.4750 | 0.8167 | 0.8167 | 0.3417 | 0.0000 |
| D2 | NDCG @ 5 | 0.5193 | 0.8381 | 0.8381 | 0.3188 | 0.0000 |
| D2 | Precision @ 5 | 0.1300 | 0.1800 | 0.1800 | 0.0500 | 0.0000 |
| D3 | Hit Rate @ 5 | 0.5000 | 0.7000 | 0.7000 | 0.2000 | 0.0000 |
| D3 | MRR @ 5 | 0.3542 | 0.6167 | 0.6167 | 0.2625 | 0.0000 |
| D3 | NDCG @ 5 | 0.3912 | 0.6381 | 0.6381 | 0.2469 | 0.0000 |
| D3 | Precision @ 5 | 0.1000 | 0.1400 | 0.1400 | 0.0400 | 0.0000 |

## Generation and citation metrics

Phase 6 generation scores are macro means over answerable queries. Citation compliance is at least one emitted citation validated against supplied reranked evidence; citation validity is query-level: all emitted citation IDs validated (0 when citations were emitted but invalid, and 0 if none were emitted). These are operational checks, not a human factual-grounding judgment. The Phase 3 retrieved-context Pass A comparison follows.

| Dataset | Phase 6 measure | n | Mean |
|---|---|---|---|
| D0 | token_f1 | 20 | 0.7235 |
| D0 | semantic_similarity | 20 | 0.7597 |
| D0 | citation_compliance | 20 | 0.9500 |
| D0 | citation_validity | 20 | 0.9500 |
| D2 | token_f1 | 20 | 0.4282 |
| D2 | semantic_similarity | 20 | 0.6271 |
| D2 | citation_compliance | 20 | 0.6000 |
| D2 | citation_validity | 20 | 0.6000 |
| D3 | token_f1 | 20 | 0.4818 |
| D3 | semantic_similarity | 20 | 0.6912 |
| D3 | citation_compliance | 20 | 0.6000 |
| D3 | citation_validity | 20 | 0.6000 |

### Phase 3 Pass A versus Phase 6 answer generation

Paired answerable query comparison; improvement/degradation uses token F1 delta thresholds ±0.001.

| Dataset | P3 Pass A F1 | P6 F1 | Δ F1 | P3 Pass A similarity | P6 similarity | Δ similarity | P6 compliance |
|---|---|---|---|---|---|---|---|
| D0 | 0.7194 | 0.7235 | 0.0041 | 0.7715 | 0.7597 | -0.0118 | 0.9500 |
| D2 | 0.4420 | 0.4282 | -0.0138 | 0.6375 | 0.6271 | -0.0104 | 0.6000 |
| D3 | 0.4982 | 0.4818 | -0.0164 | 0.6989 | 0.6912 | -0.0077 | 0.6000 |

| Dataset | F1 improved | F1 degraded | Tie | Not comparable |
|---|---|---|---|---|
| D0 | 2 | 0 | 17 | 1 |
| D2 | 3 | 2 | 15 | 0 |
| D3 | 4 | 2 | 14 | 0 |

## Alpha(q), graph contribution, and retrieval failures

The configured controller interprets larger alpha as vector-oriented and smaller alpha as graph-oriented. The last two columns count alpha ≥0.5 and <0.5, respectively.

| Dataset | n (answerable + candidates) | Mean α | SD | Min | Median | Max | α ≥0.5 | α <0.5 |
|---|---|---|---|---|---|---|---|---|
| D0 | 25 | 0.7894 | 0.1087 | 0.4077 | 0.8212 | 0.9272 | 24 | 1 |
| D2 | 25 | 0.8009 | 0.0909 | 0.5263 | 0.8167 | 0.9240 | 25 | 0 |
| D3 | 25 | 0.8118 | 0.0599 | 0.6550 | 0.8139 | 0.9240 | 25 | 0 |

Graph retrieval was sparse in answerable query runs; query count, queries returning any graph fact, mean fact count, and queries with a graph source in the final fused candidate pool are shown below.

| Dataset | Queries | Any graph facts | Mean facts/query | Graph source in fused pool |
|---|---|---|---|---|
| D0 | 20 | 2 | 0.3500 | 2 |
| D2 | 20 | 0 | 0.0000 | 0 |
| D3 | 20 | 0 | 0.0000 | 0 |

| Dataset | Answerable n | Hit queries | Missed top-5 | Missed query IDs |
|---|---|---|---|---|
| D0 | 20 | 20 | 0 | — |
| D2 | 20 | 18 | 2 | d2_prop_q10, d2_prop_q15 |
| D3 | 20 | 14 | 6 | d3_prop_q5, d3_prop_q10, d3_prop_q12, d3_prop_q14, d3_prop_q15, d3_prop_q18 |

## Refusal-candidate diagnostic

The 15 items are approved refusal-test candidates, but corpus-wide answer absence was not independently established. The detector is a phrase-pattern heuristic; these values are **not verified out-of-domain refusal accuracy** and should not be reported as such.

| Dataset | Candidates | Heuristic refusals | Heuristic fraction | Interpretation |
|---|---|---|---|---|
| D0 | 5 | 5 | 1.0000 | phrase-pattern heuristic only |
| D2 | 5 | 5 | 1.0000 | phrase-pattern heuristic only |
| D3 | 5 | 5 | 1.0000 | phrase-pattern heuristic only |

## Runtime, corpus scale, and reproducibility

- Embedding: `all-MiniLM-L6-v2`; candidate reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (max length 512); generator tag: `qwen2.5-3b-lora-ragqa`, configured temperature 0.2; alpha classifier loaded as `LearnedAlphaClassifier`.
- Top-k controls: vector 20, graph 20, reranked 5. Query scope was own-dataset collection and graph. D0 was constrained to legacy graph nodes/edges with no dataset ID.
- Indexed collection sizes differ materially: D0 149 chunks, D2 1,164, D3 6,119. Procedure and top-k are matched; corpus sizes are not. The graph populations are D0 483 nodes/106 relationships, D2 5,671/10, D3 11,145/11.
- Vector retrieval used the isolated Phase 4 Chroma snapshot. The Phase 4 snapshot manifest records its integrity comparison. Phase 2 previously noted that Chroma may internally touch its SQLite metadata file during opens/queries; because a pre-run database hash was not captured, strict no-modification of that metadata file cannot be certified.
- Ollama served the configured Qwen model tag; the tag name does not independently verify whether the LoRA adapter was merged into the served artifact.
- Exact benchmark, Phase 2/3/4 input, and Phase 6 output hashes and collection identities are in `phase6_run_manifest.json`.

## Generation errors

None recorded in the Phase 6 run manifest.

## Artifacts and interpretation

- `per_query_adaptive_hybrid_eval.csv`: one row per each of the 75 queries, with alpha, linked entities, raw vector and graph retrievals, fusion candidates, reranked top five, gold flags, retrieval metrics, answer/citations, generation metrics, and refusal heuristic fields.
- `phase5_graph_verification.json` and `phase5_graph_ingestion_summary.md`: read-only graph counts and provenance.
- `phase6_run_manifest.json`: controls, model tags, benchmark/input hashes, graph/collection counts, run status and output hash.

The result is a diagnostic comparison. Sparse graph edges, corpus-scale differences, the D0 validation overlap, heuristic refusal detection, model-adapter provenance uncertainty, and the previously recorded Chroma SQLite integrity caveat limit claims. No benchmark, source index, model weights, project configuration, or system code was modified by the Phase 6 runner.
