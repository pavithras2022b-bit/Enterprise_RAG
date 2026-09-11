# Final Frozen Evaluation Results

**Project:** Domain-Adaptive Transfer Learning with Knowledge Graph and Hybrid Vector Retrieval for Enterprise Retrieval-Augmented Generation
**Status:** Finalized and Frozen

> This document records the finalized experimental results of the Enterprise RAG system. Frozen baseline results are preserved without modification. Controlled experimental variants are reported separately and are not substituted for the original baseline results.

> **Version terminology:**
> **V1** = Proposed Adaptive-Hybrid RAG (canonical system)
> **V2** = Vector-Only Architectural Ablation
> **Guarded Adaptive Diagnostic** = alpha-floor validation experiment
> **P2 / P3 / D1** = prompt/decoding ablations

---

## 1. Corpus and Knowledge Graph Statistics

| Statistic | Final Value |
|---|---:|
| Documents | **140** |
| Chunks | **149** |
| Entity Spans | **2,432** |
| Relations | **91** |
| Vector Store Chunks | **149** |

> **Note:** Entity and relation counts above are the corrected final ingestion statistics.

---

## 2. Retrieval Performance

### 2.1 Frozen Retrieval Benchmark

| Retrieval Method | HitRate@5 | MRR@5 | NDCG@5 |
|---|---:|---:|---:|
| Vector Only | **0.9091** | **0.7836** | **0.8148** |
| Graph Only | **0.9091** | 0.7124 | 0.7613 |
| Static Hybrid | **0.9091** | **0.7836** | **0.8148** |
| Rule-Based Adaptive | **0.9091** | 0.7579 | 0.7954 |
| Learned Adaptive V1 | **0.9091** | 0.7715 | 0.8057 |

The learned adaptive method improves over the rule-based adaptive baseline, but vector-only and static-hybrid retrieval remain slightly stronger on aggregate MRR and NDCG in the frozen benchmark.

### 2.2 Guarded Adaptive Diagnostic

The learned adaptive retrieval system computes a query-dependent vector fusion weight:

\[
\alpha(q)=\sigma(w^T e_q+b)
\]

where \(e_q\) is the 384-dimensional semantic query embedding.

Error analysis identified one regression relative to vector-only retrieval among the 55 validation examples. The regression occurred when graph evidence received excessive influence despite incomplete graph information.

A guarded adaptive diagnostic was therefore evaluated:

\[
\alpha_{\text{guard}}(q)=\max(\alpha(q),0.50)
\]

| Metric | Learned Adaptive V1 | Guarded Adaptive Diagnostic | Absolute Change |
|---|---:|---:|---:|
| MRR@5 | 0.7715 | **0.7836** | **+0.0121** |
| NDCG@5 | 0.8057 | **0.8148** | **+0.0091** |

The guarded diagnostic eliminated the only observed adaptive regression on the 55-example validation benchmark and matched the strongest vector-only/static-hybrid ranking results.

> **Important:** The Guarded Adaptive Diagnostic is a validation-benchmark result. It is **not** presented as an independent held-out test result and does **not** replace the frozen V1 production retrieval configuration.

---

## 3. Adaptive Alpha Classifier

| Metric | Result |
|---|---:|
| Held-out Test Accuracy | **93.18% (41/44)** |
| Test Samples | **44** |
| Correct Predictions | **41** |
| Mean \(\alpha(q)\) — Graph-Oriented Queries | **0.2514** |
| Mean \(\alpha(q)\) — Vector-Oriented Queries | **0.8203** |
| Query Embedding Dimension | **384** |
| Relation-Group Leakage | **0** |

The classifier shows clear separation between graph-oriented and vector-oriented queries. Graph-oriented queries receive a lower vector fusion weight on average, while vector-oriented queries receive a substantially higher vector fusion weight.

> **93.18% is held-out test accuracy, not training accuracy.**

---

## 4. V1 — Proposed Adaptive-Hybrid RAG

### 4.1 V1 Architecture

V1 is the **canonical proposed system**.

```text
Vector Retrieval + Knowledge-Graph Retrieval
                    |
                    v
          Learned α(q) Fusion
                    |
                    v
          Cross-Encoder Reranking
                    |
                    v
              Top-5 Evidence
                    |
                    v
              LLM Generation
```

The final accepted generation configuration uses:

- **Primary generation model:** Qwen2.5-3B + LoRA
- **Baseline generation model:** Llama 3.2-3B
- **Serving:** Ollama
- **Quantization:** Q4_K_M GGUF
- **Temperature:** 0.2
- **Generation configuration:** V1 accepted/frozen

### 4.2 V1 Generation Model Comparison

Both models below receive evidence from the **same V1 Adaptive-Hybrid retrieval architecture**.

Therefore, this table answers:

> **Given the same V1 retrieval pipeline, how do Llama 3.2-3B and Qwen2.5-3B + LoRA compare as generation models?**

| Metric | Llama 3.2-3B | Qwen2.5-3B + LoRA |
|---|---:|---:|
| Reference Token F1 | 0.6658 | **0.6930** |
| Reference Semantic Similarity | 0.7953 | **0.8416** |
| Exact Match | 13.33% | **33.33%** |
| Citation Compliance | 73.33% | **93.33%** |
| Faithfulness — Lexical Proxy | **0.9111** | 0.8418 |
| Query-Overlap — Lexical Proxy | **0.7519** | 0.4286 |
| Average CPU Generation Latency | **21.33 s** | 24.10 s |
| Runtime Failures | **0** | **0** |

### 4.3 V1 Model-Comparison Interpretation

Under the same V1 Adaptive-Hybrid retrieval pipeline, **Qwen2.5-3B + LoRA achieved stronger primary reference-based generation metrics** than Llama 3.2-3B:

- Token F1: **0.6658 → 0.6930**
- Semantic Similarity: **0.7953 → 0.8416**
- Exact Match: **13.33% → 33.33%**
- Citation Compliance: **73.33% → 93.33%**

Llama retained:

- lower CPU latency;
- higher Faithfulness — Lexical Proxy;
- higher Query-Overlap — Lexical Proxy.

Because the two systems use different base model architectures, these results are **not interpreted as evidence that LoRA alone caused the improvement**.

---

## 5. Generation Metric Interpretation

### 5.1 Primary Reference-Based Metrics

- **Reference Token F1:** normalized token overlap between generated and reference answers.
- **Reference Semantic Similarity:** cosine similarity between generated and reference answers using `all-MiniLM-L6-v2`.
- **Exact Match:** strict normalized exact-answer agreement.
- **Citation Compliance:** whether the generated answer satisfies the citation requirement.

### 5.2 Diagnostic Lexical Proxies

- **Faithfulness — Lexical Proxy:** lexical overlap between the generated answer and retrieved evidence.
- **Query-Overlap — Lexical Proxy:** fraction of unique query words appearing literally in the generated answer.

> The Qwen V1 Query-Overlap value of **0.4286 must not be interpreted as 42.86% answer accuracy**. It is a lexical diagnostic and can be low even when an answer is semantically correct but does not repeat the wording of the question.

---

## 6. Robustness Evaluation

Two robustness evaluations are reported separately.

| Evaluation | Final Result |
|---|---:|
| Genuine Out-of-Corpus Queries | **25/25 correct refusals (100%)** |
| Hard-Negative / Direct-Evidence Evaluation | **15/15 correct refusals/flags (100%)** |
| Accepted Runtime Failures | **0** |

The 25-query out-of-corpus evaluation and the 15-query hard-negative evaluation are separate experiments and must **not** be combined into a single 40-query metric.

---

## 7. Controlled Prompt / Decoding Ablations

After freezing the accepted V1 generation result, controlled prompt/decoding variants were evaluated using the same 15-query benchmark and frozen retrieved evidence.

| Experiment | Change | Token F1 | Exact Match | Citation Compliance | Decision |
|---|---|---:|---:|---:|---|
| **V1** | Accepted baseline configuration | **0.6930** | **33.33%** | **93.33%** | **RETAINED** |
| P2 | Concise answer-type prompt | 0.6305 | 26.67% | 80.00% | Rejected |
| P3 | Evidence-span-oriented prompt | 0.5942 | 26.67% | 26.67% | Rejected |
| D1 | Temperature 0.2 → 0.0 | 0.6120 | 26.67% | 80.00% | Rejected |

None of these prompt/decoding variants improved the primary reference-based metrics over the accepted V1 generation configuration.

Therefore, **V1 remains the final accepted generation configuration**.

---

## 8. V2 — Vector-Only Architectural Ablation

### 8.1 What V2 Changes

V2 is a **controlled architectural ablation**, not a replacement for V1.

It removes:

- knowledge-graph retrieval;
- learned adaptive \(\alpha(q)\) fusion.

It retains:

- the same vector index;
- the same initial vector retrieval depth;
- the same cross-encoder reranker;
- the same Top-5 evidence size;
- the same generation models;
- the same accepted prompt structure;
- temperature = 0.2;
- the same citation validation;
- the same 15 evaluation queries;
- the same reference answers;
- the same scoring methodology.

### 8.2 V2 Architecture

```text
Vector Retrieval Only
        |
        v
Cross-Encoder Reranking
        |
        v
    Top-5 Evidence
        |
        v
    LLM Generation
```

### 8.3 V1 vs V2 Evidence-Level Difference

| Evidence Comparison | Result |
|---|---:|
| Queries Evaluated | **15** |
| Top-5 Evidence Changed | **3/15** |
| Identical Top-5 Order | **12/15** |
| Average Top-5 Set Overlap | **94.67%** |

Although V1 and V2 are architecturally different, their final Top-5 evidence sets remained highly overlapping on this benchmark.

### 8.4 V2 Generation Model Comparison

Both models below receive evidence from the **same Vector-Only V2 architecture**.

Therefore, this table answers:

> **Given the same Vector-Only V2 retrieval pipeline, how do Llama 3.2-3B and Qwen2.5-3B + LoRA compare as generation models?**

| Metric | Llama 3.2-3B | Qwen2.5-3B + LoRA |
|---|---:|---:|
| Reference Token F1 | 0.5382 | **0.6268** |
| Reference Semantic Similarity | 0.6582 | **0.7404** |
| Exact Match | 6.67% | **26.67%** |
| Citation Compliance | 66.67% | **73.33%** |
| Faithfulness — Lexical Proxy | 0.8425 | **0.8615** |
| Query-Overlap — Lexical Proxy | 0.2293 | **0.2357** |
| Average CPU Generation Latency | **20.78 s** | 21.98 s |

### 8.5 V2 Model-Comparison Interpretation

Under Vector-Only V2, **Qwen2.5-3B + LoRA still outperformed Llama 3.2-3B on the primary reference-based metrics**.

Llama remained slightly faster on CPU.

---

## 9. V1 vs V2 — Architecture Comparison

This section answers a **different question** from the model-comparison tables above.

The V1 and V2 model tables compare **Llama vs Qwen while architecture is held constant**.

This section instead asks:

> **What happens to each generation model when the retrieval architecture changes from Adaptive-Hybrid V1 to Vector-Only V2?**

### 9.1 Architecture Comparison — Llama 3.2-3B

| Metric | V1 Adaptive-Hybrid | V2 Vector-Only | Change |
|---|---:|---:|---:|
| Reference Token F1 | **0.6658** | 0.5382 | **-0.1276** |
| Reference Semantic Similarity | **0.7953** | 0.6582 | **-0.1371** |
| Exact Match | **13.33%** | 6.67% | **-6.66 pp** |
| Citation Compliance | **73.33%** | 66.67% | **-6.66 pp** |
| Faithfulness — Lexical Proxy | **0.9111** | 0.8425 | -0.0686 |
| Query-Overlap — Lexical Proxy | **0.7519** | 0.2293 | -0.5226 |
| Average CPU Generation Latency | 21.33 s | **20.78 s** | **-0.55 s** |

### 9.2 Architecture Comparison — Qwen2.5-3B + LoRA

| Metric | V1 Adaptive-Hybrid | V2 Vector-Only | Change |
|---|---:|---:|---:|
| Reference Token F1 | **0.6930** | 0.6268 | **-0.0662** |
| Reference Semantic Similarity | **0.8416** | 0.7404 | **-0.1012** |
| Exact Match | **33.33%** | 26.67% | **-6.66 pp** |
| Citation Compliance | **93.33%** | 73.33% | **-20.00 pp** |
| Faithfulness — Lexical Proxy | 0.8418 | **0.8615** | **+0.0197** |
| Query-Overlap — Lexical Proxy | **0.4286** | 0.2357 | -0.1929 |
| Average CPU Generation Latency | 24.10 s | **21.98 s** | **-2.12 s** |

### 9.3 Architecture-Comparison Interpretation

Changing retrieval from V1 Adaptive-Hybrid to V2 Vector-Only reduced the following primary reference-based metrics for **both** generation models:

- Reference Token F1;
- Reference Semantic Similarity;
- Exact Match;
- Citation Compliance.

At the same time:

- Vector-Only V2 reduced CPU latency for both models;
- Qwen V2 slightly improved its Faithfulness — Lexical Proxy from **0.8418 to 0.8615**.

These improvements and regressions are both retained to avoid cherry-picking.

> **Conclusion:** On this controlled 15-query benchmark, **V1 — Proposed Adaptive-Hybrid RAG produced stronger primary downstream generation quality than V2 — Vector-Only**. This is benchmark-specific ablation evidence and is not presented as universal superiority of hybrid retrieval.

---

## 10. Final Metric Register

### 10.1 Retrieval and Adaptive Fusion

| Component | Metric | Final Result |
|---|---|---:|
| Retrieval | HitRate@5 | **90.91%** |
| Retrieval | Best MRR@5 | **0.7836** |
| Retrieval | Best NDCG@5 | **0.8148** |
| Learned Adaptive V1 | MRR@5 | **0.7715** |
| Learned Adaptive V1 | NDCG@5 | **0.8057** |
| Guarded Adaptive Diagnostic | MRR@5 | **0.7836** |
| Guarded Adaptive Diagnostic | NDCG@5 | **0.8148** |
| Adaptive Classifier | Held-out Test Accuracy | **93.18%** |
| Adaptive Classifier | Mean \(\alpha(q)\) — Graph Queries | **0.2514** |
| Adaptive Classifier | Mean \(\alpha(q)\) — Vector Queries | **0.8203** |

### 10.2 V1 Generation

| Model | Metric | Final Result |
|---|---|---:|
| Llama 3.2-3B | Token F1 | **0.6658** |
| Llama 3.2-3B | Semantic Similarity | **0.7953** |
| Llama 3.2-3B | Exact Match | **13.33%** |
| Llama 3.2-3B | Citation Compliance | **73.33%** |
| Llama 3.2-3B | Average CPU Latency | **21.33 s/query** |
| Qwen2.5-3B + LoRA | Token F1 | **0.6930** |
| Qwen2.5-3B + LoRA | Semantic Similarity | **0.8416** |
| Qwen2.5-3B + LoRA | Exact Match | **33.33%** |
| Qwen2.5-3B + LoRA | Citation Compliance | **93.33%** |
| Qwen2.5-3B + LoRA | Average CPU Latency | **24.10 s/query** |

### 10.3 V2 Generation

| Model | Metric | Final Result |
|---|---|---:|
| Llama 3.2-3B | Token F1 | **0.5382** |
| Llama 3.2-3B | Semantic Similarity | **0.6582** |
| Llama 3.2-3B | Exact Match | **6.67%** |
| Llama 3.2-3B | Citation Compliance | **66.67%** |
| Llama 3.2-3B | Average CPU Latency | **20.78 s/query** |
| Qwen2.5-3B + LoRA | Token F1 | **0.6268** |
| Qwen2.5-3B + LoRA | Semantic Similarity | **0.7404** |
| Qwen2.5-3B + LoRA | Exact Match | **26.67%** |
| Qwen2.5-3B + LoRA | Citation Compliance | **73.33%** |
| Qwen2.5-3B + LoRA | Average CPU Latency | **21.98 s/query** |

### 10.4 Robustness and Ablation

| Component | Metric | Final Result |
|---|---|---:|
| Architectural Ablation | V1/V2 Average Top-5 Evidence Overlap | **94.67%** |
| Robustness | Out-of-Corpus Refusal | **25/25 (100%)** |
| Robustness | Hard-Negative / Direct-Evidence | **15/15 (100%)** |
| Runtime | Accepted Evaluation Failures | **0** |

---

## 11. Final Configuration Status

### 11.1 Frozen Canonical V1

- **Production Retrieval V1:** Frozen
- **Generation V1:** Frozen and retained
- **Llama 3.2-3B baseline:** Frozen
- **Qwen2.5-3B + LoRA:** Final accepted generation model

### 11.2 Guarded Adaptive Validation Diagnostic

- Minimum vector fusion weight: **0.50**
- Validation MRR@5: **0.7836**
- Validation NDCG@5: **0.8148**
- Status: **Frozen validation diagnostic**
- Does **not** replace V1 production retrieval

### 11.3 Vector-Only Architectural Ablation

- **V2 — Vector-Only Ablation:** Completed and frozen
- Knowledge-graph retrieval: removed
- Learned adaptive fusion: removed
- Cross-encoder reranking: retained
- Top-5 evidence size: retained
- Llama 3.2-3B and Qwen2.5-3B + LoRA: both evaluated
- Status: **Controlled architectural ablation**
- Does **not** replace V1

### 11.4 Rejected Prompt / Decoding Experiments

- **P2:** Concise answer-type prompt — Rejected
- **P3:** Evidence-span-oriented prompt — Rejected
- **D1:** Temperature `0.2 → 0.0` — Rejected

Rejected prompt/decoding experiments remain preserved for reproducibility and ablation evidence.

---

## 12. Result Provenance

Primary machine-generated evaluation artifacts are preserved under `outputs/`.

### 12.1 Frozen V1 Artifacts

- `outputs/production_evaluation_summary.json`
- `outputs/llm_generation_evaluation.json`
- `outputs/llm_reference_answer_evaluation.json`

### 12.2 Prompt / Decoding Ablation Artifacts

- `outputs/llm_generation_evaluation_v2.json`
- `outputs/llm_generation_evaluation_v3.json`
- `outputs/llm_generation_evaluation_v4.json`

### 12.3 Vector-Only V2 Artifacts

- `outputs/vector_only_v2_evidence.json`
- `outputs/vector_only_v2_generation_evaluation.json`

### 12.4 Reproduction Scripts

Guarded adaptive diagnostic:

- `scripts/diagnose_adaptive_retrieval.py`
- `scripts/sweep_alpha_floor.py`

Vector-Only V2 architectural ablation:

- `scripts/run_vector_only_v2_evidence.py`
- `scripts/run_vector_only_v2_generation.py`

### 12.5 Frozen V1 Artifact Integrity

`outputs/llm_generation_evaluation.json`

```text
SHA-256:
011CC0B47F9A7BF78A72A830C6F71DFA3AF1A3EB5229417898EA20D6086582DE
```

`outputs/llm_reference_answer_evaluation.json`

```text
SHA-256:
3740B4D214E4A97B79547112266B93D3F2A093B5217420DC9CAD1BAD1F3C1271
```

These hashes identify the frozen V1 generation evaluation artifacts.

---

## 13. Evaluation Freeze

The metric-optimization and planned architectural-ablation phase are considered complete.

**V1 remains the canonical proposed system.**
**Vector-Only V2 remains a separately frozen architectural ablation.**

No frozen baseline artifact should be overwritten. Any future experiment, if required, must:

1. preserve the existing frozen results;
2. use a new experiment/version identifier;
3. document the exact methodological change;
4. use the same evaluation methodology where comparison is claimed;
5. report both improvements and regressions without cherry-picking.

> **Current status:** V1 metrics finalized and frozen; Vector-Only V2 architectural ablation completed and frozen.
