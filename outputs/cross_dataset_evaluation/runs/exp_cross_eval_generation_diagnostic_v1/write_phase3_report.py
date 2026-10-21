import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

out = Path(__file__).resolve().parent
rows = list(csv.DictReader((out / 'per_query_generation_diagnostic.csv').open(encoding='utf-8-sig')))
answerable = [r for r in rows if r['query_type'] == 'answerable']
paired_success = [r for r in answerable if not r.get('generation_error')]
refusal = [r for r in rows if r['query_type'] == 'refusal_candidate']
metrics = [
    ('Token-level reference F1', 'token_f1'),
    ('Semantic similarity (MiniLM cosine)', 'semantic_similarity'),
    ('Citation compliance', 'citation_compliance'),
    ('Citation grounding rate', 'citation_grounding_rate'),
    ('Evidence-token grounding proxy', 'evidence_token_grounding_rate'),
]
def val(r, key):
    try: return float(r[key])
    except (TypeError, ValueError): return None
def avg(rs, key):
    xs = [val(r, key) for r in rs]
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None
def fmt(x): return 'n/a' if x is None else f'{x:.4f}'

lines = [
    '# Phase 3 — LLM Answer Generation Diagnostic Report', '',
    '**Status:** Diagnostic outputs generated; this report does not declare a project phase gate passed.', '',
    '> **D0 limitation:** D0 is a **Validation-Derived Diagnostic Baseline**. Its 20 pilot questions originate from `lora_val.jsonl` and their answers appear in validation completions. It is not an independent test set; do not make unseen-test generalization claims from D0.', '',
    '## Execution summary', '',
    'Generated 75 per-query rows: 60 answerable questions with retrieved-reranked (Pass A) and gold-context (Pass B) generations, plus 15 refusal-candidate queries with a retrieved-reranked generation only. Pass A used Phase 4 top-five reranked contexts. Pass B used exact adjudicated indexed passages for the benchmark gold chunk IDs. No ChromaDB, Neo4j, source benchmark, model weights, or project system code was modified by this diagnostic.', '',
    'The configured Ollama model tag was `qwen2.5-3b-lora-ragqa` (Qwen2 3.1B, Q4_K_M). The configured adapter files were present. Ollama served a GGUF model; its model metadata did not independently confirm whether the PEFT adapter was merged into the served artifact. The report therefore identifies the configured LoRA-named tag and records this verification limitation.', '',
    'One generation request failed: `d0_prop_q1` Pass A returned HTTP 500 because the local llama-server could not allocate a ~1.33 GB CPU_REPACK buffer. The answer and affected Pass A metrics are blank in the finalized CSV, and the per-query error is retained. Side-by-side metrics and paired comparisons exclude this item for both regimes; therefore D0 n=19 and D2/D3 n=20. No retry was performed.', '',
    '## Side-by-side metrics', '',
    'Macro means over successful paired answerable questions (D0 n=19; D2/D3 n=20). The evidence-token score is a lexical overlap proxy, not entailment or factual grounding. Citation compliance means at least one cited ID exactly matched a supplied context ID; citation grounding rate is valid citation markers divided by all markers, with zero where no citation was emitted.', '',
    '| Dataset | Regime | n | Token F1 | Semantic similarity | Citation compliance | Citation grounding | Evidence-token proxy |',
    '|---|---|---:|---:|---:|---:|---:|---:|---:|'
]
for ds in ['D0','D2','D3']:
    rs=[r for r in paired_success if r['dataset']==ds]
    for regime in ['a','b']:
        lines.append(f"| {ds} | {'Pass A — retrieved reranked' if regime=='a' else 'Pass B — gold context'} | {len(rs)} | " + ' | '.join(fmt(avg(rs, f'pass_{regime}_{m}')) for _,m in metrics) + ' |')
    lines.append(f"| {ds} | Delta (B − A) | {len(rs)} | " + ' | '.join(fmt(avg(rs, f'pass_b_{m}')-avg(rs, f'pass_a_{m}') if avg(rs, f'pass_b_{m}') is not None and avg(rs, f'pass_a_{m}') is not None else None) for _,m in metrics) + ' |')
lines += ['', '## Refusal-candidate results', '', 'The model produced a phrase-detector-positive refusal on 5/5 candidate queries in each dataset (15/15 overall; 1.000 candidate refusal accuracy). This is a keyword/phrase heuristic, not a semantic judgment. The 15 items were approved refusal-test candidates, but corpus-wide answer absence was not independently established; this must not be presented as verified out-of-domain refusal accuracy.', '', '| Dataset | Candidates | Heuristic refusal detected | Candidate refusal accuracy |', '|---|---:|---:|---:|']
for ds in ['D0','D2','D3']:
    rs=[r for r in refusal if r['dataset']==ds]
    lines.append(f"| {ds} | {len(rs)} | {sum(r['ood_refusal_detected']=='true' for r in rs)} | {fmt(avg(rs,'ood_refusal_accuracy'))} |")
lines.append(f"| Overall | {len(refusal)} | {sum(r['ood_refusal_detected']=='true' for r in refusal)} | {fmt(avg(refusal,'ood_refusal_accuracy'))} |")
lines += ['', '## Per-query improvement / degradation classifications', '', 'Classification uses successful paired rows only and paired token-F1 change (Pass B − Pass A): improvement > 0.001, degradation < −0.001, otherwise tie. Semantic-similarity changes are shown alongside. The failed `d0_prop_q1` Pass A generation is excluded from all paired comparisons and marked not scored below.', '', '| Dataset | Paired n | Pass B improved | Pass A improved | Tie | Mean Δ F1 | Mean Δ similarity |', '|---|---:|---:|---:|---:|---:|---:|']
for ds in ['D0','D2','D3']:
    rs=[r for r in paired_success if r['dataset']==ds]
    diffs=[(val(r,'pass_b_token_f1') or 0)-(val(r,'pass_a_token_f1') or 0) for r in rs]
    sims=[(val(r,'pass_b_semantic_similarity') or 0)-(val(r,'pass_a_semantic_similarity') or 0) for r in rs]
    lines.append(f'| {ds} | {len(rs)} | {sum(x>0.001 for x in diffs)} | {sum(x<-.001 for x in diffs)} | {sum(abs(x)<=.001 for x in diffs)} | {fmt(mean(diffs))} | {fmt(mean(sims))} |')
lines += ['', '### Paired item detail', '', '| Dataset | Question ID | Δ F1 (B−A) | Δ similarity (B−A) | Classification |', '|---|---|---:|---:|---|']
for r in answerable:
    if r.get('generation_error'):
        lines.append(f"| {r['dataset']} | {r['query_id']} | n/a | n/a | Not scored — generation failed |")
        continue
    a=val(r,'pass_a_token_f1') or 0; b=val(r,'pass_b_token_f1') or 0
    sa=val(r,'pass_a_semantic_similarity') or 0; sb=val(r,'pass_b_semantic_similarity') or 0
    delta=b-a; cls='Pass B improved' if delta>.001 else ('Pass A improved' if delta<-.001 else 'Tie')
    lines.append(f"| {r['dataset']} | {r['query_id']} | {delta:+.4f} | {sb-sa:+.4f} | {cls} |")
lines += ['', '## Error and interpretation', '', '- Pass B vs Pass A is a diagnostic contrast between the supplied retrieved context and supplied gold context; it does not isolate every retrieval or generation factor and is not a causal estimate.', '- In D2, gold context increased mean token F1 slightly (+0.0135), while mean semantic similarity decreased slightly (−0.0085); citation compliance increased from 0.50 to 1.00. This mixed signal does not support a simple retrieval-only explanation.', '- In D3, gold context improved mean token F1 (+0.0670), semantic similarity (+0.0069), and citation compliance (+0.30), consistent with context selection contributing to some answer-quality loss. It does not establish that retrieval alone caused the loss.', '- D0 has a stronger reference-overlap baseline than D2/D3, but its validation-derived provenance prevents independent generalization claims. D0 metrics in the table use the 19 successful paired items only.', '- Low lexical overlap, embedding similarity, and citation metrics are imperfect proxies. Human review is needed to assess factual correctness, entailment, completeness, and whether citations support claims.', '', '## Reproducibility and artifacts', '', 'The run manifest records configuration, input hashes, model metadata and limits, adapter file hashes, metric definitions, and the D0 claim restriction. Per-query raw outputs and citations are in `per_query_generation_diagnostic.csv`. The runner and report generator are preserved alongside this report.', '', 'No generation was performed for a gold-context refusal condition because no oracle gold context exists for those refusal candidates.', '']
(out/'phase3_generation_diagnostic_report.md').write_text('\n'.join(lines),encoding='utf-8')
print(out/'phase3_generation_diagnostic_report.md')
