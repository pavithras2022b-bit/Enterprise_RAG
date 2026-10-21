"""Phase 4 reranking diagnostic using an isolated Chroma snapshot."""
import csv, hashlib, json, math, shutil
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.config import Settings
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
BENCH=ROOT/'outputs/cross_dataset_evaluation/benchmarks/clean_v1'
SOURCE_CHROMA=ROOT/'data/chroma_store'
SNAPSHOT=OUT/'chroma_snapshot'
PHASE2=ROOT/'outputs/cross_dataset_evaluation/runs/exp_cross_eval_vector_diagnostic_v1'
DATASETS=[('D0','enterprise_docs','d0_meridian_clean_v1.json'),
          ('D2','nasa_tech_d2','d2_nasa_clean_v1.json'),
          ('D3','goverment_policy_d3','d3_government_clean_v1.json')]
EMBEDDING_ID='all-MiniLM-L6-v2'
RERANKER_ID='cross-encoder/ms-marco-MiniLM-L-6-v2'
INITIAL_K=20
FINAL_K=5

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def tree_hashes(root): return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
def metrics(ids,gold,k=5):
    rel=[int(i in gold) for i in ids[:k]]
    hit=float(any(rel)); rr=next((1/(i+1) for i,x in enumerate(rel) if x),0.0)
    dcg=sum(x/math.log2(i+2) for i,x in enumerate(rel))
    ideal=sum(1/math.log2(i+2) for i in range(min(k,len(gold))))
    return hit,rr,dcg/ideal if ideal else 0.0,sum(rel)/k

if not SNAPSHOT.exists():
    print('Copying Chroma store to isolated run snapshot; original store is not opened by Chroma.',flush=True)
    shutil.copytree(SOURCE_CHROMA,SNAPSHOT)
source_hashes=tree_hashes(SOURCE_CHROMA)
snapshot_hashes=tree_hashes(SNAPSHOT)
if source_hashes!=snapshot_hashes: raise RuntimeError('Isolated Chroma snapshot differs from source file hashes.')
(OUT/'chroma_snapshot_manifest.json').write_text(json.dumps({'source':'data/chroma_store','source_file_sha256':source_hashes,'snapshot_verified_identical':True},indent=2)+'\n',encoding='utf-8')

print('Opening isolated Chroma snapshot...',flush=True)
client=chromadb.PersistentClient(path=str(SNAPSHOT),settings=Settings(anonymized_telemetry=False))
collections={}; corpus_info={}
for ds,name,_ in DATASETS:
    c=client.get_collection(name=name); collections[ds]=c
    if (c.metadata or {}).get('hnsw:space')!='cosine': raise RuntimeError(f'{name} is not cosine indexed')
    corpus_info[ds]={'collection_name':name,'collection_id':str(c.id),'count':c.count(),'metadata':c.metadata or {}}

print('Loading configured embedding and cross-encoder models from local cache...',flush=True)
embedding=SentenceTransformer(EMBEDDING_ID,device='cpu',local_files_only=True)
model_root=Path.home()/'.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2/snapshots'
snapshots=sorted(p for p in model_root.iterdir() if p.is_dir())
if not snapshots: raise RuntimeError(f'Configured cross-encoder is not cached locally: {RERANKER_ID}')
reranker_path=snapshots[-1]
reranker=CrossEncoder(str(reranker_path),device='cpu',max_length=512)
if embedding.get_embedding_dimension()!=384: raise RuntimeError('Unexpected embedding dimension')

phase2_rows={}
with (PHASE2/'per_query_vector_retrieval.csv').open(encoding='utf-8-sig',newline='') as f:
    for r in csv.DictReader(f):
        if r['query_type']=='answerable': phase2_rows.setdefault(r['query_id'],[]).append(r)
answerable={}; refusal={ds:[] for ds,_,_ in DATASETS}
for ds,_,filename in DATASETS:
    obj=json.loads((BENCH/filename).read_text(encoding='utf-8'))
    answerable[ds]=obj['eval_queries']
refusal_obj=json.loads((BENCH/'cross_dataset_refusal_clean_v1.json').read_text(encoding='utf-8'))
for q in refusal_obj['queries']: refusal[q['dataset_id']].append(q)

rows=[]; summary={}; improvement={}
for ds,collection_name,_ in DATASETS:
    c=collections[ds]; perq=[]; improvements={'improved':0,'unchanged':0,'degraded':0}
    items=[(x['query_id'],x['query'],set(x['ground_truth_chunk_ids']),'answerable') for x in answerable[ds]]
    items += [(x['query_id'],x['query'],set(),'refusal_candidate') for x in refusal[ds]]
    for qid,query,gold,qtype in items:
        qvec=embedding.encode(query,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
        got=c.query(query_embeddings=[qvec.tolist()],n_results=INITIAL_K,include=['documents','distances'])
        cand_ids=got['ids'][0]; docs=got['documents'][0]; dists=got['distances'][0]
        vector_scores=[1.0-float(d) for d in dists]
        if len(cand_ids)<INITIAL_K: raise RuntimeError(f'{ds}/{qid}: expected 20 candidates, got {len(cand_ids)}')
        cross_scores=reranker.predict([(query,doc or '') for doc in docs],show_progress_bar=False,batch_size=16)
        order=sorted(range(len(cand_ids)),key=lambda i:(-float(cross_scores[i]),i))
        reranked_ids=[cand_ids[i] for i in order]
        if qtype=='answerable':
            rh,rm,rn,rp=metrics(reranked_ids,gold,FINAL_K)
            vrows=phase2_rows.get(qid,[])
            if len(vrows)!=5: raise RuntimeError(f'Missing Phase 2 baseline rows for {qid}')
            vector_ids=[r['retrieved_chunk_id'] for r in sorted(vrows,key=lambda x:int(x['rank']))]
            vh,vm,vn,vp=metrics(vector_ids,gold,FINAL_K)
            # Confirm the cloned collection's dense-vector top five matches the Phase 2 baseline.
            if cand_ids[:5]!=vector_ids: raise RuntimeError(f'Phase 2 top-5 vector baseline mismatch for {qid}')
            dndcg=rn-vn
            cls='improved' if dndcg>1e-12 else ('degraded' if dndcg < -1e-12 else 'unchanged')
            improvements[cls]+=1
            first=next((j+1 for j,idx in enumerate(order) if cand_ids[idx] in gold),None)
            initial_gold=next((i+1 for i,x in enumerate(cand_ids) if x in gold),None)
            perq.append({'query_id':qid,'vector_hit_rate_at_5':vh,'rerank_hit_rate_at_5':rh,'vector_mrr_at_5':vm,'rerank_mrr_at_5':rm,'vector_ndcg_at_5':vn,'rerank_ndcg_at_5':rn,'vector_precision_at_5':vp,'rerank_precision_at_5':rp})
            failure='hit_at_1' if first==1 else ('hit_at_2_to_5' if first else ('gold_in_top20_but_reranked_below_5' if initial_gold else 'gold_not_in_initial_top20'))
        else:
            vh=vm=vn=vp=rh=rm=rn=rp=None; first=None; initial_gold=None
            cls='not_scored_no_verified_gold'; failure='reranked_candidates_for_human_review'
        qbase={'dataset':ds,'collection_name':collection_name,'query_id':qid,'query_type':qtype,'query':query,
          'expected_behavior':'abstain_or_refuse' if qtype=='refusal_candidate' else 'answer_from_gold_evidence',
          'gold_chunk_ids':'|'.join(sorted(gold)),'vector_metric_scope':'top-5 metrics; initial vector candidate pool top-20',
          'vector_hit_rate_at_5':vh,'rerank_hit_rate_at_5':rh,'delta_hit_rate_at_5':None if vh is None else rh-vh,
          'vector_mrr_at_5':vm,'rerank_mrr_at_5':rm,'delta_mrr_at_5':None if vm is None else rm-vm,
          'vector_ndcg_at_5':vn,'rerank_ndcg_at_5':rn,'delta_ndcg_at_5':None if vn is None else rn-vn,
          'vector_precision_at_5':vp,'rerank_precision_at_5':rp,'delta_precision_at_5':None if vp is None else rp-vp,
          'initial_first_gold_rank_top20':initial_gold or '', 'reranked_first_gold_rank_top20':first or '',
          'per_query_ndcg_change_classification':cls,'failure_classification':failure}
        for rank,idx in enumerate(order,1):
            row=dict(qbase); row.update({'reranked_rank':rank,'initial_vector_rank':idx+1,'retrieved_chunk_id':cand_ids[idx],
              'vector_cosine_similarity':f'{vector_scores[idx]:.9f}','cross_encoder_score':f'{float(cross_scores[idx]):.9f}',
              'gold_match':str(cand_ids[idx] in gold).lower(),'in_reranked_top5':str(rank<=FINAL_K).lower(),'retrieved_chunk_text':docs[idx] or ''})
            rows.append(row)
    improvement[ds]=improvements
    keys=['hit_rate_at_5','mrr_at_5','ndcg_at_5','precision_at_5']
    summary[ds]={}
    for k in keys:
        v='vector_'+k; r='rerank_'+k
        summary[ds][v]=sum(x[v] for x in perq)/len(perq)
        summary[ds][r]=sum(x[r] for x in perq)/len(perq)
        summary[ds]['delta_'+k]=summary[ds][r]-summary[ds][v]
    summary[ds]['answerable_query_count']=len(perq)
    summary[ds]['per_query_ndcg_change_classification']=improvements

with (OUT/'per_query_rerank_retrieval.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
manifest={'experiment_id':'exp_cross_eval_rerank_diagnostic_v1','completed_utc':datetime.now(timezone.utc).isoformat(),
 'mode':'dense vector top-20 plus cross-encoder reranking to top-5; no generation or fine-tuning',
 'embedding_model':EMBEDDING_ID,'reranker_model':RERANKER_ID,'reranker_snapshot':reranker_path.name,
 'device':'cpu','embedding_dimension':384,'normalized_embeddings':True,'initial_top_k':INITIAL_K,'reranked_top_k':FINAL_K,
 'cross_encoder_max_length':512,'similarity':'cosine; score=1-distance','corpora':corpus_info,
 'phase2_top5_baseline_match':'All 60 answerable queries had identical Phase 2 vector top-5 IDs when queried against the isolated snapshot.',
 'answerable_metrics':summary,'d0_claim_restriction':'Validation-Derived Diagnostic Baseline only; not an independent test set.',
 'refusal_candidates':'15 candidates were also reranked and logged; no relevance metrics computed because no verified gold IDs.',
 'artifacts':{'per_query_rerank_retrieval.csv_sha256':sha(OUT/'per_query_rerank_retrieval.csv'),
              'chroma_snapshot_manifest.json_sha256':sha(OUT/'chroma_snapshot_manifest.json')}}
(OUT/'phase4_run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'summary':summary,'ndcg_classifications':improvement,'ranked_candidate_rows':len(rows)},indent=2))
