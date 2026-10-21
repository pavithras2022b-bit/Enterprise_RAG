# Phase 2 — Vector-Only Retrieval Diagnostic

**Status: completed.** This was a retrieval-only diagnostic with no generation, reranking, graph, LLM inference, or fine-tuning.

## Controlled setup

Each approved answerable query and refusal candidate was embedded with `all-MiniLM-L6-v2` on CPU, normalized, 384 dimensions, and searched only against its own dataset collection. Chroma collection metadata reported cosine distance; similarity is `1 - distance`. The loaded runtime reported ChromaDB 1.5.9, while `requirements.txt` specifies `chromadb>=0.5,<1.0`; this environment mismatch is recorded in `run_manifest.json` and limits reproducibility. All searches used top_k=5, the same embedding model, metric, and ranking procedure. Chroma was accessed for collection metadata/counts, gold-ID preflight, and queries only.

| Dataset | Collection | Indexed chunks | Answerable queries | Refusal candidates | Hit Rate @ 5 | MRR @ 5 | NDCG @ 5 | Precision @ 5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| D0 | `enterprise_docs` | 149 | 20 | 5 | 0.9000 | 0.8250 | 0.8446 | 0.1800 |
| D2 | `nasa_tech_d2` | 1,164 | 20 | 5 | 0.6500 | 0.4750 | 0.5193 | 0.1300 |
| D3 | `goverment_policy_d3` | 6,119 | 20 | 5 | 0.5000 | 0.3542 | 0.3912 | 0.1000 |

Metrics use the 20 answerable queries per dataset. Hit Rate is the fraction with at least one verified gold ID in top five; MRR uses the first gold rank; NDCG uses binary gold relevance; Precision@5 divides returned gold IDs by five. Full precision and per-query rankings are in `per_query_vector_retrieval.csv`.

## Search-space scale controls

The evaluation was matched on top_k=5, embedding model, normalization, cosine similarity, and own-corpus-only scope. Corpus sizes differ substantially (149, 1,164, and 6,119 chunks), so this controls the retrieval procedure but not corpus scale. No subsampling or cross-corpus union was applied. Collection IDs and metadata are in `run_manifest.json`.

## Per-query failure classification

Answerable failures are classified as `gold_not_retrieved_top5`; hits are split between rank 1 and ranks 2–5.

- **D0:** 2/20 missed all gold IDs in top five: `d0_prop_q8`, `d0_prop_q13`.
- **D2:** 7/20 missed all gold IDs in top five: `d2_prop_q2`, `d2_prop_q6`, `d2_prop_q10`, `d2_prop_q11`, `d2_prop_q12`, `d2_prop_q15`, `d2_prop_q18`.
- **D3:** 10/20 missed all gold IDs in top five: `d3_prop_q1`, `d3_prop_q4`, `d3_prop_q5`, `d3_prop_q6`, `d3_prop_q7`, `d3_prop_q10`, `d3_prop_q12`, `d3_prop_q14`, `d3_prop_q15`, `d3_prop_q18`.

For refusal candidates, the run logged top-five retrieval but did not compute answerability or semantic refusal success because there are no verified relevant IDs and no generation was run. Returned chunks are not automatically false positives, and an empty result would not prove corpus-wide absence. Each candidate is labeled for human review in the per-query CSV.

## D0 claim restriction

**D0 is a Validation-Derived Diagnostic Baseline.** All 20 pilot questions match `lora_val.jsonl` questions and all 20 answers appear in validation completions. Do not claim independent/unseen test performance or independent generalization from D0. This warning is included in both the clean benchmark manifest and the experiment run manifest.

## Reproducibility and outputs

- `per_query_vector_retrieval.csv`: 375 ranked rows (top five for 75 queries), including query, rank, chunk ID, cosine similarity, retrieved text, gold-match flag, metrics, and failure classification.
- `run_manifest.json`: corpus collection IDs/counts, controls, exact metrics, software versions, and clean benchmark hashes.
- `../benchmarks/clean_v1/clean_v1_manifest.json`: frozen benchmark and source hashes.

No Chroma add/update/delete calls, Neo4j graph changes, model-weight changes, configuration changes, or system-code changes were made. The persistent Chroma client did advance `chroma.sqlite3` last-write time during open/query (observed 2026-10-13 13:23:25 before and 2026-10-14 10:38:50 after, local filesystem time). The target HNSW vector-segment files retained earlier timestamps (D0 2026-08-31; D2 2026-10-01; D3 2026-10-02). The SQLite metadata store may have been touched internally; because no pre-run database hash/backup was captured, the strict no-modification constraint cannot be fully certified. Results are retained, and this integrity caveat must accompany them. No Phase 3+ work was started.


\n