**\# Enterprise Retrieval-Augmented Generation**

**\## Domain-Adaptive Transfer Learning with Knowledge Graph and Hybrid
Vector Retrieval for Enterprise RAG**

An end-to-end Enterprise Retrieval-Augmented Generation system
combining:

\- semantic vector retrieval,

\- Neo4j knowledge-graph retrieval,

\- learned query-adaptive hybrid fusion,

\- cross-encoder reranking,

\- LoRA-based domain adaptation,

\- grounded answer generation,

\- strict citation validation,

\- unsupported-query handling.

The system dynamically determines how much vector and graph retrieval
should contribute to each query and uses a domain-adapted Qwen2.5-3B
model for final grounded generation.

---

**\## 1. System Architecture**

\`\`\`text

Enterprise Documents

        \|

        v

PDF / DOCX / HTML Parsing

        \|

        v

Structure-Aware Chunking

        \|

        +-------------------------+

        \|                         \|

        v                         v

Sentence Embeddings        NER + Relation Extraction

        \|                         \|

        v                         v

     ChromaDB                Neo4j Graph

        \|                         \|

        +------------+------------+

                     \|

                  User Query

                     \|

                     v

              Query Embedding

                     \|

                     v

          Learned Adaptive α(q)

                     \|

          +----------+----------+

          \|                     \|

          v                     v

   Vector Retrieval       Graph Retrieval

          \|                     \|

          +----------+----------+

                     \|

                     v

           Adaptive Hybrid Fusion

                     \|

                     v

          Cross-Encoder Reranking

                     \|

                     v

            Top-k Evidence

                     \|

                     v

         Qwen2.5-3B + LoRA

                     \|

                     v

       Citation Extraction/Validation

                     \|

                     v

             Grounded Answer

\`\`\`

The learned fusion weight is:

\\\[

`\alpha`{=tex}(q)=`\sigma`{=tex}(w\^Te_q+b)

\\\]

and the hybrid retrieval score is:

\\\[

S(q,d)=`\alpha`{=tex}(q)S_v(q,d)+(1-`\alpha`{=tex}(q))S_g(q,d)

\\\]

A high \`alpha\` favors vector retrieval, while a low \`alpha\` gives
greater influence to knowledge-graph retrieval.

---

**\## 2. Major Components**

**\### Document Ingestion**

The ingestion pipeline supports:

\- PDF

\- DOCX

\- HTML

The final experimental corpus contains:

\- **\*\*140 enterprise documents\*\***

\- **\*\*149 indexed chunks\*\***

Source documents are stored under:

\`\`\`text

data/raw/documents/

\`\`\`

**\### Semantic Vector Retrieval**

Embedding model:

\`\`\`text

sentence-transformers/all-MiniLM-L6-v2

\`\`\`

Embedding dimensionality:

\`\`\`text

384

\`\`\`

Primary vector database:

\`\`\`text

ChromaDB

\`\`\`

**\### Knowledge Graph**

Graph database:

\`\`\`text

Neo4j 5

\`\`\`

Named Entity Recognition:

\`\`\`text

dslim/bert-base-NER

\`\`\`

Representative relations include:

\`\`\`text

MANAGES

USES

REQUIRES_TRAINING

ASSIGNED_TO

\`\`\`

**\### Learned Adaptive Retrieval**

A logistic-regression classifier predicts the query-dependent fusion
weight \`alpha(q)\`.

Training dataset:

\`\`\`text

220 examples

110 graph-oriented

110 vector-oriented

\`\`\`

Held-out evaluation:

\`\`\`text

Accuracy: 0.9318

Graph-query average α(q): 0.2514

Vector-query average α(q): 0.8203

Relation-group leakage: 0

\`\`\`

The trained classifier is stored at:

\`\`\`text

models/alpha_classifier/model.joblib

\`\`\`

**\### Cross-Encoder Reranking**

Reranking model:

\`\`\`text

cross-encoder/ms-marco-MiniLM-L-6-v2

\`\`\`

Retrieved hybrid candidates are reranked before evidence is passed to
the LLM.

**\### Domain-Adapted Generation**

Base model:

\`\`\`text

Qwen/Qwen2.5-3B-Instruct

\`\`\`

LoRA configuration:

\`\`\`text

rank: 16

alpha: 32

dropout: 0.05

target modules:

q_proj

k_proj

v_proj

o_proj

\`\`\`

The trained adapter is available under:

\`\`\`text

outputs/qwen2.5-3b-lora-ragqa/final_adapter/

\`\`\`

For local deployment, the adapter was merged with the base model,
converted to GGUF, and quantized using \`Q4_K_M\`.

Ollama model name:

\`\`\`text

qwen2.5-3b-lora-ragqa

\`\`\`

The final GGUF model is approximately 1.8 GB.

The GGUF deployment artifact is intentionally stored outside this
repository because of its size.

---

**\## 3. Project Structure**

\`\`\`text

Enterprise-Retrieval-Augmented-Generation/

|

\|-- adaptive_retrieval/

\|   \|-- classifier.py

\|   \|-- fusion.py

\|   \|-- rules.py

\|   \`-- train.py

|

\|-- api/

\|   \|-- main.py

\|   \|-- pipeline.py

\|   \|-- routes.py

\|   \`-- schema.py

|

\|-- chunking/

\|-- embeddings/

\|-- evaluation/

\|-- ingestion/

\|-- knowledge_graph/

\|-- llm/

\|-- ner/

\|-- preprocessing/

\|-- relation_extraction/

\|-- reranking/

\|-- retrieval/

\|-- vector_db/

|

\|-- configs/

\|   \|-- adaptive_retrieval.yaml

\|   \|-- chunking.yaml

\|   \|-- embeddings.yaml

\|   \|-- ingestion.yaml

\|   \|-- knowledge_graph.yaml

\|   \|-- llm.yaml

\|   \|-- ner.yaml

\|   \|-- preprocessing.yaml

\|   \|-- relation_extraction.yaml

\|   \|-- reranking.yaml

\|   \|-- retrieval.yaml

\|   \`-- vector_db.yaml

|

\|-- data/

\|   \|-- labeled/

\|   \|-- raw/

\|   \`-- schema/

|

\|-- models/

\|   \`-- alpha_classifier/

|

\|-- outputs/

\|   \|-- llm_generation_evaluation.json

\|   \|-- llm_generation_evaluation.csv

\|   \|-- llm_reference_answer_evaluation.json

\|   \`-- production_evaluation_summary.json

|

\|-- scripts/

\|   \|-- lora_dataset_prep/

\|   \|-- build_alpha_balanced_training.py

\|   \|-- evaluate_alpha_classifier_holdout.py

\|   \|-- ingest_raw_corpus.py

\|   \|-- run_all_tests.py

\|   \|-- run_production_pipeline.py

\|   \`-- train_alpha_classifier.py

|

\|-- tests/

|

\|-- config_loader.py

\|-- rag_demo.html

\|-- pytest.ini

\|-- requirements.txt

\`-- README.md

\`\`\`

---

**\## 4. Environment Setup**

**\### Requirements**

Recommended:

\`\`\`text

Python 3.11+

Neo4j 5

Ollama

\`\`\`

Create a virtual environment:

\`\`\`powershell

python -m venv .venv

\`\`\`

Activate it on Windows PowerShell:

\`\`\`powershell

.\\.venv`\Scripts`{=tex}`\Activate`{=tex}.ps1

\`\`\`

Install dependencies:

\`\`\`powershell

python -m pip install -r requirements.txt

\`\`\`

---

**\## 5. Neo4j Setup**

The knowledge-graph retrieval component requires a running Neo4j
instance.

The project uses the standard local Bolt endpoint:

\`\`\`text

bolt://localhost:7687

\`\`\`

Neo4j Browser is normally available at:

\`\`\`text

http://localhost:7474

\`\`\`

Environment-specific Neo4j credentials should be stored in \`.env\` and
must not be committed to source control.

Verify that Neo4j is running before starting the RAG API.

---

**\## 6. Ollama and Qwen-LoRA Setup**

The final generation model is served through Ollama.

Verify Ollama:

\`\`\`powershell

ollama list

\`\`\`

The expected adapted model name is:

\`\`\`text

qwen2.5-3b-lora-ragqa:latest

\`\`\`

Test the model directly:

\`\`\`powershell

ollama run qwen2.5-3b-lora-ragqa "Reply with exactly: QWEN LORA READY"

\`\`\`

Expected response:

\`\`\`text

QWEN LORA READY

\`\`\`

The active LLM is configured in:

\`\`\`text

configs/llm.yaml

\`\`\`

with:

\`\`\`yaml

llm:

  client:

    base_url: http://localhost:11434

    model_name: qwen2.5-3b-lora-ragqa

    timeout_seconds: 180.0

    temperature: 0.2

\`\`\`

---

**\## 7. Running the API**

Ensure both Neo4j and Ollama are running.

Then start FastAPI from the project root:

\`\`\`powershell

python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

\`\`\`

The API will be available locally on port \`8000\`.

FastAPI interactive documentation is available through the standard
Swagger UI route:

\`\`\`text

/docs

\`\`\`

---

**\## 8. API Endpoints**

**\### Health Check**

\`\`\`http

GET /health

\`\`\`

Expected response:

\`\`\`json

{

  "status": "healthy",

  "version": "1.0.0"

}

\`\`\`

**\### Query**

\`\`\`http

POST /api/v1/query

\`\`\`

The endpoint supports:

\`\`\`text

query

alpha

top_k

use_adaptive

use_reranking

\`\`\`

For the final proposed configuration:

\`\`\`text

top_k = 5

use_adaptive = true

use_reranking = true

\`\`\`

The response includes the generated answer together with
retrieval/generation information such as citations and the adaptive α(q)
value.

**\### Ingestion**

\`\`\`http

POST /api/v1/ingest

\`\`\`

The endpoint accepts document file paths and processes them through the
ingestion pipeline.

The final corpus has already been ingested for the reported experiments.
Re-ingestion is therefore not required for reproducing the frozen
demo/evaluation state.

---

**\## 9. Running Tests**

Run the complete test suite:

\`\`\`powershell

python -m pytest tests -v

\`\`\`

A convenience runner is also available:

\`\`\`powershell

python .`\scripts`{=tex}`\run`{=tex}\_all_tests.py

\`\`\`

---

**\## 10. Evaluation**

**\### Retrieval Ablation**

The final retrieval benchmark compares:

\- Vector Only

\- Graph Only

\- Static Hybrid

\- Rule-Based Adaptive

\- Learned Adaptive

Final frozen results:

\| **\*\*Retrieval Method\*\*** \| **\*\*Hit Rate@5\*\*** \|
 **\*\*MRR@5\*\*** \| **\*\*NDCG@5\*\*** \|

\| -------------------- \| -------------: \| ---------: \| ---------: \|

\| Vector Only          \|         0.9091 \| **\*\*0.7836\*\*** \|
**\*\*0.8148\*\*** \|

\| Graph Only           \|         0.9091 \|     0.7124 \|     0.7613 \|

\| Static Hybrid        \|         0.9091 \| **\*\*0.7836\*\*** \|
**\*\*0.8148\*\*** \|

\| Rule-Based Adaptive  \|         0.9091 \|     0.7579 \|     0.7954 \|

\| **\*\*Learned Adaptive\*\*** \|     **\*\*0.9091\*\*** \|
**\*\*0.7715\*\*** \| **\*\*0.8057\*\*** \|

Learned adaptive retrieval improves over rule-based adaptive retrieval
but does not outperform the strongest vector/static baselines on
aggregate MRR/NDCG.

### Guarded Adaptive Fusion --- Validation Diagnostic

Error analysis of the learned adaptive retrieval results identified one
validation query for which incomplete graph evidence displaced stronger
semantic-vector evidence.

A guarded adaptive variant was therefore evaluated:

\[
`\alpha`{=tex}\_{`\text{guard}`{=tex}}(q)=`\max`{=tex}(`\alpha`{=tex}(q),0.50)
\]

The threshold `0.50` was the smallest tested value that achieved the
maximum observed validation result.

  Retrieval Variant            MRR@5       NDCG@5
  --------------------- ------------ ------------
  Learned Adaptive V1         0.7715       0.8057
  Guarded Adaptive        **0.7836**   **0.8148**

On the 55-example validation benchmark, the guarded variant eliminated
the only observed regression relative to vector-only retrieval. This is
reported as a validation diagnostic and does **not** replace the frozen
V1 production configuration.

---

**\## 11. V1 --- Proposed Adaptive-Hybrid RAG Generation Evaluation**

The final controlled generation comparison evaluates:

\`\`\`text

Llama 3.2-3B

vs.

Qwen2.5-3B + LoRA

\`\`\`

Both models use the same:

\- corpus,

\- vector index,

\- knowledge graph,

\- adaptive fusion classifier,

\- retrieval configuration,

\- cross-encoder reranker,

\- prompt structure,

\- citation validator,

\- evaluation questions.

Final 15-query results:

\| **\*\*Evaluation Metric\*\***         \| **\*\*Llama 3.2-3B\*\*** \|
**\*\*Qwen2.5-3B + LoRA\*\*** \|

\| ----------------------------- \| ---------------: \|
--------------------: \|

\| Reference Token F1            \|           0.6658 \|          
 **\*\*0.6930\*\*** \|

\| Reference Semantic Similarity \|           0.7953 \|          
 **\*\*0.8416\*\*** \|

\| Exact Match Rate              \|            13.3% \|            
**\*\*33.3%\*\*** \|

\| Citation Compliance           \|            73.3% \|            
**\*\*93.3%\*\*** \|

\| Faithfulness --- Lexical Proxy  \|       **\*\*0.9111\*\*** \|      
         0.8418 \|

\| Query-Overlap --- Lexical Proxy \|       **\*\*0.7519\*\*** \|      
         0.4286 \|

\| Average CPU Latency           \|      **\*\*21.33 s\*\*** \|        
      24.10 s \|

\| Runtime Failures              \|            **\*\*0\*\*** \|        
        **\*\*0\*\*** \|

Qwen2.5-3B + LoRA improves reference-answer alignment and citation
compliance, while Llama retains lower CPU latency and higher
lexical-overlap proxy scores.

The faithfulness and query-overlap values above are lexical proxy
metrics and should not be interpreted as semantic entailment or semantic
answer correctness.

---

**\## 12. Robustness Evaluation**

Two distinct negative-query evaluations were performed.

**\### Genuine Corpus-Negative Queries**

\`\`\`text

25 / 25 correctly refused

100% on the evaluated set

\`\`\`

**\### Hard-Negative / Distractor Cases**

\`\`\`text

15 / 15 correctly handled

100% on the evaluated set

\`\`\`

These experiments are reported separately because they evaluate
different failure conditions.

The results indicate strong refusal behavior on the evaluated negative
sets and should not be interpreted as a guarantee for every possible
unsupported query.

---

**\## 13. Evaluation Artifacts**

Final evaluation outputs are stored under:

\`\`\`text

outputs/llm_generation_evaluation.json

outputs/llm_generation_evaluation.csv

outputs/llm_reference_answer_evaluation.json

outputs/production_evaluation_summary.json

\`\`\`

The original V1 artifacts remain frozen. Vector-Only V2 artifacts are
stored separately and represent a controlled architectural ablation
rather than a replacement production configuration.

---

**\## 14. Research Findings**

The project addresses four principal research questions.

**\### RQ1 --- Does learned query-adaptive fusion improve hybrid
retrieval?**

Yes relative to rule-based adaptive retrieval.

The learned classifier also demonstrates strong separation between
graph-oriented and vector-oriented queries.

However, vector-only and static-hybrid retrieval remain slightly
stronger on aggregate MRR and NDCG in the current benchmark.

**\### RQ2 --- Does domain-adapted generation improve enterprise RAG
responses?**

Under the controlled comparison, Qwen2.5-3B + LoRA improves:

\- Reference Token F1

\- Reference Semantic Similarity

\- Exact Match

\- Citation Compliance

compared with the Llama 3.2-3B baseline.

The adapted model incurs moderately higher CPU inference latency.

**\### RQ3 --- Can the system avoid unsupported answers?**

On the evaluated negative sets, the system correctly handled:

\`\`\`text

25/25 genuine corpus-negative queries

15/15 hard-negative cases

\`\`\`

---

**\## 15. Key Contributions**

**\### 1. Query-Adaptive Hybrid Retrieval**

Instead of using a constant graph/vector fusion weight, the system
learns:

\\\[

`\alpha`{=tex}(q)=`\sigma`{=tex}(w\^Te_q+b)

\\\]

from the incoming query.

\*\*### 2. Guarded Adaptive Fusion Diagnostic

Error analysis motivated a guarded adaptive formulation that preserves
learned query-dependent weighting while enforcing a minimum vector
contribution when graph evidence may be incomplete:

\[
`\alpha`{=tex}\_{`\text{guard}`{=tex}}(q)=`\max`{=tex}(`\alpha`{=tex}(q),0.50)
\]

This diagnostic improved validation MRR@5 from `0.7715` to `0.7836` and
NDCG@5 from `0.8057` to `0.8148`.

### 3. Domain-Adaptive Generation\*\*

Qwen2.5-3B-Instruct is adapted to enterprise RAG question answering
using parameter-efficient LoRA.

**\### 4. Evidence-Aware Generation**

Retrieved evidence retains exact source identifiers throughout the
pipeline.

Generated citations are validated against the evidence actually supplied
to the LLM.

**\### 5. Unsupported-Query Handling**

The system explicitly supports refusal behavior when available
enterprise evidence is insufficient.

---

**\## 16. Limitations**

The current evaluation uses a relatively small synthetic enterprise
corpus and a 15-query controlled generation benchmark.

The Vector-Only V2 generation ablation uses the same 15-query controlled
benchmark; therefore, its observed advantage or disadvantage should be
interpreted as benchmark-specific evidence rather than universal
superiority of one retrieval architecture.

The learned adaptive retrieval method improves over rule-based
adaptation but does not exceed vector/static retrieval on every
aggregate ranking metric.

The implemented faithfulness and query-relevance metrics are lexical
proxies rather than semantic entailment metrics.

Citation validation confirms that cited IDs correspond to supplied
evidence, but a valid citation ID alone does not prove that every
generated statement is semantically entailed by the cited passage.

Final LLM inference was evaluated locally on CPU using quantized models,
so latency results are specific to that environment.

---

**\## 17. Future Work**

Potential extensions include:

\- direct optimization of adaptive fusion against retrieval ranking
objectives,

\- learned domain-specific relation extraction,

\- claim-level citation attribution,

\- semantic entailment-based faithfulness evaluation,

\- temporal and version-aware enterprise retrieval,

\- authorization-aware RAG,

\- evaluation on larger real-world enterprise corpora,

\- GPU/server deployment and throughput benchmarking.

---

**\## 18. Evaluation Results**

The finalized and frozen experimental results---including retrieval,

adaptive fusion, generation, robustness, and evaluation statistics---are

documented in \[FINAL_RESULTS.md\](FINAL_RESULTS.md).

The corresponding machine-generated evaluation artifacts are preserved

under \`outputs/\` for reproducibility.

**\## Final Summary**

The project demonstrates an end-to-end enterprise RAG architecture
combining:

\`\`\`text

Knowledge Graph Retrieval

        +

Semantic Vector Retrieval

        +

Learned Query-Adaptive Fusion

        +

Cross-Encoder Reranking

        +

LoRA Domain Adaptation

        +

Citation Validation

        +

Unsupported-Query Handling

\`\`\`

The experimental results show that learned query-adaptive fusion
improves over manually defined adaptive routing, while the
domain-adapted Qwen2.5-3B + LoRA configuration improves reference-answer
alignment and citation compliance relative to the Llama 3.2-3B baseline
under the evaluated conditions.

A controlled Vector-Only Ablation (V2) further evaluated the
contribution of the retrieval architecture. Although V1 and V2 retained
**94.67% average Top-5 evidence overlap**, removing knowledge-graph
retrieval and learned adaptive fusion reduced the primary
reference-based generation metrics for both evaluated LLMs. For
Qwen2.5-3B + LoRA, Token F1 decreased from `0.6930` to `0.6268`,
semantic similarity from `0.8416` to `0.7404`, Exact Match from `33.33%`
to `26.67%`, and citation compliance from `93.33%` to `73.33%`.

Accordingly, **V1 --- Proposed Adaptive-Hybrid RAG** remains the
canonical system, while **V2 --- Vector-Only Ablation** is retained as a
controlled architectural ablation for analyzing the contribution and
trade-offs of hybrid enterprise retrieval.
