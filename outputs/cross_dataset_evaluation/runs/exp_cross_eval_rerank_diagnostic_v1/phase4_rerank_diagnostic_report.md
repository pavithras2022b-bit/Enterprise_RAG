# Phase 4 — Cross-Encoder Reranking Diagnostic

**Status: completed. Pure reranking diagnostic; no text generation, LLM inference, fine-tuning, or system-code changes.**

## Matched retrieval and reranking setup

For each of 60 answerable items, the configured `all-MiniLM-L6-v2` embedding model retrieved 20 candidates from the item’s own cosine Chroma collection. The configured `cross-encoder/ms-marco-MiniLM-L-6-v2` scored the 20 `(query, chunk_text)` pairs on CPU (max length 512); candidates were sorted by cross-encoder score and the first five evaluated. The same pipeline was also applied to the 15 approved refusal-test candidates, which are logged without relevance metrics because they have no verified gold IDs.

The database was copied to `chroma_snapshot/` and queried there, leaving the original store unopened by Chroma in this phase. `chroma_snapshot_manifest.json` records source and snapshot hashes. For all 60 answerable queries, the isolated snapshot’s initial dense-vector top-5 IDs exactly matched the Phase 2 log. This validates the baseline comparison.

## Vector baseline versus reranked results

| Dataset | Metric | Phase 2 vector @5 | Phase 4 reranked @5 | Delta (reranked − vector) |
|---|---|---:|---:|---:|
| D0 | Hit Rate @ 5 | 0.9000 | 1.0000 | +0.1000 |
| D0 | MRR @ 5 | 0.8250 | 0.9167 | +0.0917 |
| D0 | NDCG @ 5 | 0.8446 | 0.9381 | +0.0935 |
| D0 | Precision @ 5 | 0.1800 | 0.2000 | +0.0200 |
| D2 | Hit Rate @ 5 | 0.6500 | 0.9000 | +0.2500 |
| D2 | MRR @ 5 | 0.4750 | 0.8167 | +0.3417 |
| D2 | NDCG @ 5 | 0.5193 | 0.8381 | +0.3188 |
| D2 | Precision @ 5 | 0.1300 | 0.1800 | +0.0500 |
| D3 | Hit Rate @ 5 | 0.5000 | 0.7000 | +0.2000 |
| D3 | MRR @ 5 | 0.3542 | 0.6167 | +0.2625 |
| D3 | NDCG @ 5 | 0.3912 | 0.6381 | +0.2469 |
| D3 | Precision @ 5 | 0.1000 | 0.1400 | +0.0400 |

Metrics are macro averages over 20 answerable queries per dataset. Hit Rate is binary gold presence in top five; MRR uses the first gold rank; NDCG uses binary relevance; Precision@5 uses gold IDs divided by five.

## Rank-shift and per-query classification

Per-query classification uses **NDCG@5 change**: improved (>0), unchanged (=0), degraded (<0), comparing the Phase 4 top five to the Phase 2 top five. Gold-rank shift compares the first gold rank in the initial vector top-20 with its rank after cross-encoder sorting within those same 20 candidates; positive mean shift means movement earlier.

### D0

- NDCG improved: 4/20 (`d0_prop_q8`, `d0_prop_q13`, `d0_prop_q18`, `d0_prop_q20`).
- NDCG unchanged: 13/20.
- NDCG degraded: 3/20 (`d0_prop_q7`, `d0_prop_q14`, `d0_prop_q17`).
- Hit Rate transitions: gained 2 (`d0_prop_q8`, `d0_prop_q13`); lost 0 (none).
- Among 20 queries with a gold ID in initial top-20, mean first-gold rank movement was +1.10 positions (earlier is positive); earlier=4, unchanged=13, later=3. Gold absent from initial top-20: 0/20. Gold moved from vector ranks 6–20 into reranked top five: 2.

### D2

- NDCG improved: 10/20 (`d2_prop_q1`, `d2_prop_q2`, `d2_prop_q6`, `d2_prop_q9`, `d2_prop_q11`, `d2_prop_q12`, `d2_prop_q13`, `d2_prop_q17`, `d2_prop_q18`, `d2_prop_q19`).
- NDCG unchanged: 10/20.
- NDCG degraded: 0/20 (none).
- Hit Rate transitions: gained 5 (`d2_prop_q2`, `d2_prop_q6`, `d2_prop_q11`, `d2_prop_q12`, `d2_prop_q18`); lost 0 (none).
- Among 18 queries with a gold ID in initial top-20, mean first-gold rank movement was +3.00 positions (earlier is positive); earlier=10, unchanged=8, later=0. Gold absent from initial top-20: 2/20. Gold moved from vector ranks 6–20 into reranked top five: 5.

### D3

- NDCG improved: 8/20 (`d3_prop_q1`, `d3_prop_q4`, `d3_prop_q6`, `d3_prop_q7`, `d3_prop_q13`, `d3_prop_q16`, `d3_prop_q17`, `d3_prop_q20`).
- NDCG unchanged: 10/20.
- NDCG degraded: 2/20 (`d3_prop_q2`, `d3_prop_q19`).
- Hit Rate transitions: gained 4 (`d3_prop_q1`, `d3_prop_q4`, `d3_prop_q6`, `d3_prop_q7`); lost 0 (none).
- Among 14 queries with a gold ID in initial top-20, mean first-gold rank movement was +2.50 positions (earlier is positive); earlier=8, unchanged=4, later=2. Gold absent from initial top-20: 6/20. Gold moved from vector ranks 6–20 into reranked top five: 4.

The per-query CSV contains the complete candidate ordering, initial vector rank, vector cosine similarity, cross-encoder score, gold flag, both top-five metric sets, deltas, and classification.

## Controls and interpretation

Search space was not corpus-size normalized: collection sizes were D0 149, D2 1,164, and D3 6,119 chunks. Each dataset used its own collection, same embedding model, cosine metric, initial top-20, cross-encoder, and final top-five. The results measure ranking changes conditional on the top-20 vector candidate pool; reranking cannot recover gold evidence absent from that pool.

**D0 is a Validation-Derived Diagnostic Baseline**, not an independent test set: its 20 questions match LoRA validation questions and all 20 answers appear in validation completions. Restrict D0 conclusions accordingly. Refusal candidates are approved for testing but not independently verified as corpus-wide unanswerable.

## Integrity and reproducibility notes

The reranking phase queried an isolated, hash-verified Chroma snapshot. The prior Phase 2 run used a persistent Chroma client against the source store: its `chroma.sqlite3` last-write time advanced from 2026-10-13 13:23:25 to 2026-10-14 10:38:50, local filesystem time; no pre-run SQLite hash/backup exists, so Phase 2 metadata-store immutability cannot be certified. The Phase 2 runtime also reported ChromaDB 1.5.9 while the repository specifies `<1.0`. These pre-existing Phase 2 caveats are retained; Phase 4 did not open its source store with Chroma.

Artifacts: `per_query_rerank_retrieval.csv`, `phase4_run_manifest.json`, `chroma_snapshot_manifest.json`, and `run_phase4_reranking.py`.
