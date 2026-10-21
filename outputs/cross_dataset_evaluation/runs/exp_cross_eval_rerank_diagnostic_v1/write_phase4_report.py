import csv,json,statistics
from pathlib import Path
OUT=Path(__file__).resolve().parent
m=json.loads((OUT/'phase4_run_manifest.json').read_text(encoding='utf-8'))
with (OUT/'per_query_rerank_retrieval.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
groups={}
for r in rows: groups.setdefault((r['dataset'],r['query_id']),[]).append(r)
stats={}
for ds in ('D0','D2','D3'):
    queries=[rs for (d,_),rs in groups.items() if d==ds and rs[0]['query_type']=='answerable']
    shift=[]; rescued=0; initial_top20_miss=0; rank_better=rank_same=rank_worse=0
    improved=[]; degraded=[]; unchanged=[]; hit_gained=[]; hit_lost=[]
    for rs in queries:
        r=rs[0]; cl=r['per_query_ndcg_change_classification']; qid=r['query_id']
        {'improved':improved,'degraded':degraded,'unchanged':unchanged}[cl].append(qid)
        vh=float(r['vector_hit_rate_at_5']); rh=float(r['rerank_hit_rate_at_5'])
        if vh==0 and rh==1: hit_gained.append(qid)
        if vh==1 and rh==0: hit_lost.append(qid)
        a=r['initial_first_gold_rank_top20']; b=r['reranked_first_gold_rank_top20']
        if not a:
            initial_top20_miss+=1
        else:
            a=int(a); b=int(b); shift.append(a-b)
            if a>5 and b<=5: rescued+=1
            if b<a: rank_better+=1
            elif b==a: rank_same+=1
            else: rank_worse+=1
    stats[ds]={'improved':improved,'degraded':degraded,'unchanged':unchanged,'hit_gained':hit_gained,'hit_lost':hit_lost,
      'rank_shift_mean':statistics.mean(shift) if shift else 0,'rank_shift_n':len(shift),'gold_promoted_to_top5':rescued,
      'gold_absent_from_initial_top20':initial_top20_miss,'rank_better':rank_better,'rank_same':rank_same,'rank_worse':rank_worse}

with (OUT/'phase4_rerank_diagnostic_report.md').open('w',encoding='utf-8') as f:
    f.write('# Phase 4 — Cross-Encoder Reranking Diagnostic\n\n**Status: completed. Pure reranking diagnostic; no text generation, LLM inference, fine-tuning, or system-code changes.**\n\n')
    f.write('## Matched retrieval and reranking setup\n\n')
    f.write('For each of 60 answerable items, the configured `all-MiniLM-L6-v2` embedding model retrieved 20 candidates from the item’s own cosine Chroma collection. The configured `cross-encoder/ms-marco-MiniLM-L-6-v2` scored the 20 `(query, chunk_text)` pairs on CPU (max length 512); candidates were sorted by cross-encoder score and the first five evaluated. The same pipeline was also applied to the 15 approved refusal-test candidates, which are logged without relevance metrics because they have no verified gold IDs.\n\n')
    f.write('The database was copied to `chroma_snapshot/` and queried there, leaving the original store unopened by Chroma in this phase. `chroma_snapshot_manifest.json` records source and snapshot hashes. For all 60 answerable queries, the isolated snapshot’s initial dense-vector top-5 IDs exactly matched the Phase 2 log. This validates the baseline comparison.\n\n')
    f.write('## Vector baseline versus reranked results\n\n| Dataset | Metric | Phase 2 vector @5 | Phase 4 reranked @5 | Delta (reranked − vector) |\n|---|---|---:|---:|---:|\n')
    for ds in ('D0','D2','D3'):
        x=m['answerable_metrics'][ds]
        for label,key in [('Hit Rate @ 5','hit_rate_at_5'),('MRR @ 5','mrr_at_5'),('NDCG @ 5','ndcg_at_5'),('Precision @ 5','precision_at_5')]:
            f.write(f"| {ds} | {label} | {x['vector_'+key]:.4f} | {x['rerank_'+key]:.4f} | {x['delta_'+key]:+.4f} |\n")
    f.write('\nMetrics are macro averages over 20 answerable queries per dataset. Hit Rate is binary gold presence in top five; MRR uses the first gold rank; NDCG uses binary relevance; Precision@5 uses gold IDs divided by five.\n\n')
    f.write('## Rank-shift and per-query classification\n\nPer-query classification uses **NDCG@5 change**: improved (>0), unchanged (=0), degraded (<0), comparing the Phase 4 top five to the Phase 2 top five. Gold-rank shift compares the first gold rank in the initial vector top-20 with its rank after cross-encoder sorting within those same 20 candidates; positive mean shift means movement earlier.\n\n')
    for ds in ('D0','D2','D3'):
        s=stats[ds]
        f.write(f"### {ds}\n\n")
        f.write(f"- NDCG improved: {len(s['improved'])}/20 ({', '.join('`'+q+'`' for q in s['improved']) or 'none'}).\n")
        f.write(f"- NDCG unchanged: {len(s['unchanged'])}/20.\n- NDCG degraded: {len(s['degraded'])}/20 ({', '.join('`'+q+'`' for q in s['degraded']) or 'none'}).\n")
        f.write(f"- Hit Rate transitions: gained {len(s['hit_gained'])} ({', '.join('`'+q+'`' for q in s['hit_gained']) or 'none'}); lost {len(s['hit_lost'])} ({', '.join('`'+q+'`' for q in s['hit_lost']) or 'none'}).\n")
        f.write(f"- Among {s['rank_shift_n']} queries with a gold ID in initial top-20, mean first-gold rank movement was {s['rank_shift_mean']:+.2f} positions (earlier is positive); earlier={s['rank_better']}, unchanged={s['rank_same']}, later={s['rank_worse']}. Gold absent from initial top-20: {s['gold_absent_from_initial_top20']}/20. Gold moved from vector ranks 6–20 into reranked top five: {s['gold_promoted_to_top5']}.\n\n")
    f.write('The per-query CSV contains the complete candidate ordering, initial vector rank, vector cosine similarity, cross-encoder score, gold flag, both top-five metric sets, deltas, and classification.\n\n')
    f.write('## Controls and interpretation\n\n')
    f.write('Search space was not corpus-size normalized: collection sizes were D0 149, D2 1,164, and D3 6,119 chunks. Each dataset used its own collection, same embedding model, cosine metric, initial top-20, cross-encoder, and final top-five. The results measure ranking changes conditional on the top-20 vector candidate pool; reranking cannot recover gold evidence absent from that pool.\n\n')
    f.write('**D0 is a Validation-Derived Diagnostic Baseline**, not an independent test set: its 20 questions match LoRA validation questions and all 20 answers appear in validation completions. Restrict D0 conclusions accordingly. Refusal candidates are approved for testing but not independently verified as corpus-wide unanswerable.\n\n')
    f.write('## Integrity and reproducibility notes\n\n')
    f.write('The reranking phase queried an isolated, hash-verified Chroma snapshot. The prior Phase 2 run used a persistent Chroma client against the source store: its `chroma.sqlite3` last-write time advanced from 2026-10-13 13:23:25 to 2026-10-14 10:38:50, local filesystem time; no pre-run SQLite hash/backup exists, so Phase 2 metadata-store immutability cannot be certified. The Phase 2 runtime also reported ChromaDB 1.5.9 while the repository specifies `<1.0`. These pre-existing Phase 2 caveats are retained; Phase 4 did not open its source store with Chroma.\n\n')
    f.write('Artifacts: `per_query_rerank_retrieval.csv`, `phase4_run_manifest.json`, `chroma_snapshot_manifest.json`, and `run_phase4_reranking.py`.\n')
print('Report written. Query classifications:',json.dumps({ds:{'improved':len(s['improved']),'unchanged':len(s['unchanged']),'degraded':len(s['degraded']),'mean_gold_rank_shift':s['rank_shift_mean']} for ds,s in stats.items()}))
