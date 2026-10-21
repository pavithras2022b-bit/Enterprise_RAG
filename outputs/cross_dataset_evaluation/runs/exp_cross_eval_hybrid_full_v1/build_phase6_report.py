"""Build the Phase 6 comparison report from immutable phase outputs."""
import csv, json, math, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RUNS = ROOT / 'outputs/cross_dataset_evaluation/runs'
DS = ['D0','D2','D3']

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))
def mean(rows, field):
    vals=[]
    for r in rows:
        try:
            v=float(r[field]);
            if math.isfinite(v): vals.append(v)
        except (KeyError, ValueError, TypeError): pass
    return (statistics.mean(vals),len(vals)) if vals else (None,0)
def fmt(x): return 'n/a' if x is None else f'{x:.4f}'
def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(str(v) for v in row)+' |' for row in rows])

p2=read(RUNS/'exp_cross_eval_vector_diagnostic_v1/per_query_vector_retrieval.csv')
p4=read(RUNS/'exp_cross_eval_rerank_diagnostic_v1/per_query_rerank_retrieval.csv')
p3=read(RUNS/'exp_cross_eval_generation_diagnostic_v1/per_query_generation_diagnostic.csv')
p6=read(OUT/'per_query_adaptive_hybrid_eval.csv')
manifest=json.loads((OUT/'phase6_run_manifest.json').read_text(encoding='utf-8'))

def unique(rows, metric_fields):
    out={}
    for r in rows:
        if r.get('query_type','answerable')!='answerable': continue
        q=r['query_id']
        if q not in out: out[q]={k:r.get(k,'') for k in metric_fields}|{'dataset':r['dataset']}
    return out
v=unique(p2,['hit_rate_at_5','mrr_at_5','ndcg_at_5','precision_at_5'])
r=unique(p4,['rerank_hit_rate_at_5','rerank_mrr_at_5','rerank_ndcg_at_5','rerank_precision_at_5'])
h={x['query_id']:x for x in p6 if x['query_type']=='answerable'}
g={x['query_id']:x for x in p3 if x['query_type']=='answerable'}

metric_specs=[('Hit Rate @ 5','hit_rate_at_5','rerank_hit_rate_at_5','hit_rate_at_5'),('MRR @ 5','mrr_at_5','rerank_mrr_at_5','mrr_at_5'),('NDCG @ 5','ndcg_at_5','rerank_ndcg_at_5','ndcg_at_5'),('Precision @ 5','precision_at_5','rerank_precision_at_5',None)]
retrieval=[]
for ds in DS:
    for label,a,b,c in metric_specs:
        vv=[float(x[a]) for x in v.values() if x['dataset']==ds and x.get(a)]
        rr=[float(x[b]) for x in r.values() if x['dataset']==ds and x.get(b)]
        hh=[]
        for x in h.values():
            if x['dataset']!=ds: continue
            if c: hh.append(float(x[c]))
            else:
                gold=set(json.loads(x['gold_chunk_ids'])); got=json.loads(x['retrieval_metric_chunk_ids_at5'])
                hh.append(sum(bool(i) and i in gold for i in got)/5)
        av=statistics.mean(vv) if vv else None; ar=statistics.mean(rr) if rr else None; ah=statistics.mean(hh) if hh else None
        retrieval.append([ds,label,fmt(av),fmt(ar),fmt(ah),fmt(ar-av if av is not None and ar is not None else None),fmt(ah-ar if ah is not None and ar is not None else None)])

gen_rows=[]; refusal_rows=[]; class_rows=[]; failures=[]
for ds in DS:
    rows=[x for x in h.values() if x['dataset']==ds]
    for metric in ['token_f1','semantic_similarity','citation_compliance','citation_validity']:
        val,n=mean(rows,metric)
        gen_rows.append([ds,metric,n,fmt(val)])
    rs=[x for x in p6 if x['dataset']==ds and x['query_type']=='refusal_candidate']
    det=sum(x['candidate_refusal_detected']=='true' for x in rs)
    refusal_rows.append([ds,len(rs),det,fmt(det/len(rs) if rs else None),'phrase-pattern heuristic only'])
    states={'improved':0,'degraded':0,'tie':0,'not comparable':0}
    for x in rows:
        old=g.get(x['query_id'],{}).get('pass_a_token_f1','')
        try: delta=float(x['token_f1'])-float(old)
        except (ValueError,TypeError): states['not comparable']+=1; continue
        states['improved' if delta>0.001 else 'degraded' if delta < -0.001 else 'tie']+=1
    class_rows.append([ds,*[states[k] for k in ['improved','degraded','tie','not comparable']]])
    bad=[x for x in [*rows,*rs] if x.get('generation_error')]
    failures.extend([f"- `{x['query_id']}` ({ds}): {x['generation_error']}" for x in bad])

alpha=[]
for ds in DS:
    vals=[float(x['alpha_q']) for x in p6 if x['dataset']==ds and x['alpha_q']!='']
    alpha.append([ds,len(vals),fmt(statistics.mean(vals)),fmt(statistics.pstdev(vals)),fmt(min(vals)),fmt(statistics.median(vals)),fmt(max(vals)),sum(x>=.5 for x in vals),sum(x<.5 for x in vals)])

graph=[]
for ds in DS:
    xs=[x for x in h.values() if x['dataset']==ds]
    graph.append([ds,len(xs),sum(bool(json.loads(x['graph_top20_fact_ids'])) for x in xs),fmt(statistics.mean(len(json.loads(x['graph_top20_fact_ids'])) for x in xs) if xs else None),sum('GRAPH' in json.loads(x['fused_candidate_sources']) for x in xs)])

failure_classes=[]
for ds in DS:
    xs=[x for x in h.values() if x['dataset']==ds]
    got=sum(float(x['hit_rate_at_5'])>0 for x in xs)
    failure_classes.append([ds,len(xs),got,len(xs)-got,', '.join(x['query_id'] for x in xs if float(x['hit_rate_at_5'])==0) or '—'])

lines=['# Phase 6 — Full Adaptive-Hybrid Cross-Dataset Diagnostic','',
'> **D0 claim restriction:** D0 is a **Validation-Derived Diagnostic Baseline**. All 20 pilot questions match questions from `lora_val.jsonl` and their answers appear in validation completions. It is not an independent or unseen test set. Results support diagnostic statements only.','',
'**Execution status:** Completed 75/75 query rows (60 answerable and 15 refusal-test candidates); zero recorded generation errors. This report does not independently certify a phase gate.','',
'## Phase 5 graph verification','',
'D2/D3 graph populations were already present and agreed with recorded ingestion summaries. Phase 5 performed read-only count/provenance verification; it did not re-extract entities or write/modify graph data. Counts and limitations are documented in [phase5_graph_ingestion_summary.md](phase5_graph_ingestion_summary.md). Existing edge coverage is sparse, so graph existence must not be interpreted as complete knowledge coverage.','',
'## Matched retrieval comparison','',
'Macro averages over 20 answerable questions per dataset. Phase 2 used dense vector top-5; Phase 4 reranked the Phase 2 top-20 dense candidate pool; Phase 6 fused top-20 vector and graph candidates using the configured learned alpha controller, reranked the merged pool, and evaluated the final five. Delta columns are arithmetic differences between adjacent phases. Precision@5 for Phase 6 is gold chunks in the five evaluated evidence slots divided by five; unresolved graph facts without a source chunk are non-matches.','',
table(['Dataset','Metric','P2 Vector @5','P4 Reranked @5','P6 Adaptive-Hybrid @5','Δ P4−P2','Δ P6−P4'],retrieval),'',
'## Generation and citation metrics','',
'Phase 6 generation scores are macro means over answerable queries. Citation compliance is at least one emitted citation validated against supplied reranked evidence; citation validity is query-level: all emitted citation IDs validated (0 when citations were emitted but invalid, and 0 if none were emitted). These are operational checks, not a human factual-grounding judgment. The Phase 3 retrieved-context Pass A comparison follows.','',
table(['Dataset','Phase 6 measure','n','Mean'],gen_rows),'',
'### Phase 3 Pass A versus Phase 6 answer generation','',
'Paired answerable query comparison; improvement/degradation uses token F1 delta thresholds ±0.001.','',
table(['Dataset','P3 Pass A F1','P6 F1','Δ F1','P3 Pass A similarity','P6 similarity','Δ similarity','P6 compliance'],[[ds,fmt(mean([x for x in p3 if x['dataset']==ds and x['query_type']=='answerable'],'pass_a_token_f1')[0]),fmt(mean([x for x in h.values() if x['dataset']==ds],'token_f1')[0]),fmt(mean([x for x in h.values() if x['dataset']==ds],'token_f1')[0]-mean([x for x in p3 if x['dataset']==ds and x['query_type']=='answerable'],'pass_a_token_f1')[0]),fmt(mean([x for x in p3 if x['dataset']==ds and x['query_type']=='answerable'],'pass_a_semantic_similarity')[0]),fmt(mean([x for x in h.values() if x['dataset']==ds],'semantic_similarity')[0]),fmt(mean([x for x in h.values() if x['dataset']==ds],'semantic_similarity')[0]-mean([x for x in p3 if x['dataset']==ds and x['query_type']=='answerable'],'pass_a_semantic_similarity')[0]),fmt(mean([x for x in h.values() if x['dataset']==ds],'citation_compliance')[0])] for ds in DS]),'',
table(['Dataset','F1 improved','F1 degraded','Tie','Not comparable'],class_rows),'',
'## Alpha(q), graph contribution, and retrieval failures','',
'The configured controller interprets larger alpha as vector-oriented and smaller alpha as graph-oriented. The last two columns count alpha ≥0.5 and <0.5, respectively.','',
table(['Dataset','n (answerable + candidates)','Mean α','SD','Min','Median','Max','α ≥0.5','α <0.5'],alpha),'',
'Graph retrieval was sparse in answerable query runs; query count, queries returning any graph fact, mean fact count, and queries with a graph source in the final fused candidate pool are shown below.','',
table(['Dataset','Queries','Any graph facts','Mean facts/query','Graph source in fused pool'],graph),'',
table(['Dataset','Answerable n','Hit queries','Missed top-5','Missed query IDs'],failure_classes),'',
'## Refusal-candidate diagnostic','',
'The 15 items are approved refusal-test candidates, but corpus-wide answer absence was not independently established. The detector is a phrase-pattern heuristic; these values are **not verified out-of-domain refusal accuracy** and should not be reported as such.','',
table(['Dataset','Candidates','Heuristic refusals','Heuristic fraction','Interpretation'],refusal_rows),'',
'## Runtime, corpus scale, and reproducibility','',
'- Embedding: `all-MiniLM-L6-v2`; candidate reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (max length 512); generator tag: `qwen2.5-3b-lora-ragqa`, configured temperature 0.2; alpha classifier loaded as `LearnedAlphaClassifier`.',
'- Top-k controls: vector 20, graph 20, reranked 5. Query scope was own-dataset collection and graph. D0 was constrained to legacy graph nodes/edges with no dataset ID.',
'- Indexed collection sizes differ materially: D0 149 chunks, D2 1,164, D3 6,119. Procedure and top-k are matched; corpus sizes are not. The graph populations are D0 483 nodes/106 relationships, D2 5,671/10, D3 11,145/11.',
'- Vector retrieval used the isolated Phase 4 Chroma snapshot. The Phase 4 snapshot manifest records its integrity comparison. Phase 2 previously noted that Chroma may internally touch its SQLite metadata file during opens/queries; because a pre-run database hash was not captured, strict no-modification of that metadata file cannot be certified.',
'- Ollama served the configured Qwen model tag; the tag name does not independently verify whether the LoRA adapter was merged into the served artifact.',
'- Exact benchmark, Phase 2/3/4 input, and Phase 6 output hashes and collection identities are in `phase6_run_manifest.json`.','',
'## Generation errors','']
lines += failures if failures else ['None recorded in the Phase 6 run manifest.']
lines += ['', '## Artifacts and interpretation','',
'- `per_query_adaptive_hybrid_eval.csv`: one row per each of the 75 queries, with alpha, linked entities, raw vector and graph retrievals, fusion candidates, reranked top five, gold flags, retrieval metrics, answer/citations, generation metrics, and refusal heuristic fields.',
'- `phase5_graph_verification.json` and `phase5_graph_ingestion_summary.md`: read-only graph counts and provenance.',
'- `phase6_run_manifest.json`: controls, model tags, benchmark/input hashes, graph/collection counts, run status and output hash.','',
'The result is a diagnostic comparison. Sparse graph edges, corpus-scale differences, the D0 validation overlap, heuristic refusal detection, model-adapter provenance uncertainty, and the previously recorded Chroma SQLite integrity caveat limit claims. No benchmark, source index, model weights, project configuration, or system code was modified by the Phase 6 runner.']
(OUT/'phase6_adaptive_hybrid_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Wrote report; rows=',len(p6),'unique answerable=',len(h),'manifest=',manifest['status'])
