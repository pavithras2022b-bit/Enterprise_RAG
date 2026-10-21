# Ablation Study 2: Cross-Dataset Generalization under Fixed RAG Architecture

## Scope and data provenance

This analysis uses completed Phase 6 per-query records, Phase 2 vector-only and Phase 4 reranked-vector retrieval baselines. Retrieved-versus-gold-context citation values use Phase 3 Pass B because Phase 6 has no gold-context generation arm. No underlying benchmark or evaluation CSV was modified.

**D0 limitation:** D0 is a **Validation-Derived Diagnostic Baseline**. Its 20 pilot questions originate from `lora_val.jsonl` and answers appear in validation completions. It is not an independent test set; do not present its measurements as unseen-test generalization.

## Metric summary

### Retrieval trajectory

| Dataset | P2 HR@5 | P4 reranked HR@5 | P6 hybrid HR@5 | P2 MRR@5 | P6 MRR@5 | P2 NDCG@5 | P6 NDCG@5 |
|---|---|---|---|---|---|---|---|
| D0 | 0.9000 | 1.0000 | 1.0000 | 0.8250 | 0.9167 | 0.8446 | 0.9381 |
| D2 | 0.6500 | 0.9000 | 0.9000 | 0.4750 | 0.8167 | 0.5193 | 0.8381 |
| D3 | 0.5000 | 0.7000 | 0.7000 | 0.3542 | 0.6167 | 0.3912 | 0.6381 |

### Phase 6 generation and citation checks

| Dataset | n | Token F1 | Semantic similarity | Citation compliance |
|---|---|---|---|---|
| D0 | 20 | 0.7235 | 0.7597 | 95.0% |
| D2 | 20 | 0.4282 | 0.6271 | 60.0% |
| D3 | 20 | 0.4818 | 0.6912 | 60.0% |

### Retrieved versus gold-context citation compliance

| Dataset | Phase 6 retrieved context | Phase 3 Pass B gold context | Phase 3 n |
|---|---|---|---|
| D0 | 95.0% | 100.0% | 19 |
| D2 | 60.0% | 100.0% | 20 |
| D3 | 60.0% | 90.0% | 20 |

### Verified graph population and refusal phrase triggers

| Dataset | Graph nodes | Relationships |
|---|---|---|
| D0 | 483 | 106 |
| D2 | 5,671 | 10 |
| D3 | 11,145 | 11 |

The phrase detector triggered on 15/15 candidates (100.0%) overall: D0 5/5, D2 5/5, D3 5/5. Treat this as a string-pattern trigger rate only; it is not verified OOD refusal accuracy.

## Results and discussion

### 1. Retrieval generalization and search-space impact

Dense vector Hit Rate@5 decreases across these dataset/corpus combinations from 90.0% on D0 (149 indexed chunks) to 65.0% on D2 (1,164 chunks) and 50.0% on D3 (6,119 chunks). D3 has 41.1× the indexed chunks of D0. Phase 4 reranking raises Hit Rate@5 to 90.0% on D2 and 70.0% on D3 (D0: 100.0%); Phase 6 produces the same retrieval metrics as Phase 4 for all datasets. This is consistent with improved ordering of the dense candidate pool helping transfer-domain retrieval. It does not prove corpus size alone caused the decline: domain, document content, and corpus construction also differ. The full hybrid should not be credited with gains beyond Phase 4 reranking, since no additional retrieval-metric gain is observed here.

### 2. Domain-specific generation and citation dynamics

Phase 6 answerable-query token F1 is 0.7235 on D0, 0.4282 on D2, and 0.4818 on D3. In the Phase 3 gold-context arm, citation compliance is 100% for D2 (n=20) and 90% for D3 (n=20); D0 is 100% on the paired n=19 subset. Phase 6 retrieved-context citation compliance is 95%, 60%, and 60%, respectively. Higher D2/D3 compliance under gold context is compatible with evidence selection/context relevance contributing to citation failures. Since this is a cross-phase comparison, it does not establish that context noise rather than generator behavior alone causes the differences.

Even with gold passages, D2 token F1 remains approximately 0.46 (Phase 3 Pass B mean 0.4555). Evidence availability alone therefore does not eliminate the reference-overlap gap. Specialized aerospace vocabulary, answer phrasing, and lexical mismatch are plausible contributors, but aggregate scores cannot identify their causal shares. D2 Phase 3 Pass B semantic similarity was 0.6290, so token F1 alone is not a complete semantic-quality measure.

### 3. Knowledge-graph sparsity and adaptive fallback

The active Neo4j populations contain 10 D2 relationships and 11 D3 relationships despite 5,671 and 11,145 nodes. In Phase 6 answerable runs, no D2 or D3 query returned graph facts; D0 returned graph facts for two of 20 answerable questions. Mean learned alpha was 0.7894, 0.8009, and 0.8118 for D0, D2, and D3, respectively, over 25 answerable/refusal queries per dataset. These observations are consistent with vector-dominant fallback under sparse graph evidence, with retrieval metrics not falling below the Phase 4 reranker baseline. They do not prove robust fallback under other query distributions or denser graph conditions. Graph retrieval did not improve measured Hit Rate, MRR, or NDCG beyond Phase 4 here.

## Reporting limitations

- Dataset identity and corpus size are confounded; the trend is descriptive, not a causal corpus-size experiment.
- Phase 6 equals Phase 4 reranked-vector retrieval metrics; gains over Phase 2 cannot be credited as an incremental hybrid contribution.
- Phase 3 gold-context values are a cross-phase descriptive comparison, not a single-run randomized context ablation.
- Refusal values are phrase-detector triggers; corpus-wide answer absence was not independently established.
- D0 is validation-derived. Consult the Phase 6 report and manifest for model-adapter provenance uncertainty and the previously recorded Chroma SQLite metadata integrity caveat.

## Figure captions for paper

**Figure 1.** Hit Rate@5, MRR@5, and NDCG@5 across datasets for Phase 2 dense vector retrieval and Phase 6 adaptive hybrid retrieval. Phase 6 matched Phase 4 reranked-vector scores; its unchanged delta should not be attributed to graph fusion.
**Figure 2.** Phase 6 token-level reference F1, semantic similarity, and citation compliance. Citation compliance is an operational check against supplied evidence IDs, not a human factual-grounding assessment.
**Figure 3.** Phase 2 and Phase 6 Hit Rate@5 versus corpus size on a logarithmic x-axis. D3 contains approximately 41.1 times as many chunks as D0. Dataset domain and corpus construction vary, so the relationship is descriptive rather than causal. Phase 4 reranking-only hit rates equal Phase 6.
**Figure 4.** Citation compliance for Phase 6 retrieved/reranked context and Phase 3 Pass B gold context. The cross-phase pattern is consistent with evidence selection affecting citation behavior but does not show context noise is the sole cause.
**Figure 5.** Fraction of refusal-test candidates triggering the configured phrase detector. The 15/15 trigger count is a heuristic, not verified out-of-domain refusal accuracy.

Phase 6 input status: `completed`; 75 rows.
