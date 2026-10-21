"""Read-only dense vector retrieval diagnostic; does not invoke generation."""
import csv, hashlib, json, math, sys
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.config import Settings
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
BENCH=ROOT/'outputs/cross_dataset_evaluation/benchmarks/clean_v1'
CHROMA=ROOT/'data/chroma_store'
CONFIG=ROOT/'configs/embeddings.yaml'
DATASETS=[('D0','enterprise_docs','d0_meridian_clean_v1.json'),
          ('D2','nasa_tech_d2','d2_nasa_clean_v1.json'),
          ('D3','goverment_policy_d3','d3_government_clean_v1.json')]
MODEL_NAME='all-MiniLM-L6-v2'
TOP_K=5

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def metrics(ids,gold,k=5):
    rel=[1 if x in gold else 0 for x in ids[:k]]
    hit=float(any(rel))
    rr=next((1.0/(i+1) for i,x in enumerate(rel) if x),0.0)
    dcg=sum(x/math.log2(i+2) for i,x in enumerate(rel))
    ideal=sum(1.0/math.log2(i+2) for i in range(min(k,len(gold))))
    return hit,rr,(dcg/ideal if ideal else 0.0),sum(rel)/k

print('Opening Chroma client (read/query only)...',flush=True)
client=chromadb.PersistentClient(path=str(CHROMA),settings=Settings(anonymized_telemetry=False))
collections={}
collection_meta={}
for ds,name,_ in DATASETS:
    print(f'Opening existing collection {name}...',flush=True)
    coll=client.get_collection(name=name)
    collections[ds]=coll
    collection_meta[ds]={'collection_name':name,'collection_id':str(coll.id),'count':coll.count(),'metadata':coll.metadata or {}}
    space=(coll.metadata or {}).get('hnsw:space')
    if space!='cosine': raise RuntimeError(f'{name}: expected cosine collection, got {space!r}')

print(f'Loading embedding model {MODEL_NAME} on CPU...',flush=True)
model=SentenceTransformer(MODEL_NAME,device='cpu',local_files_only=True)
dim=int(model.get_embedding_dimension())
if dim!=384: raise RuntimeError(f'Expected all-MiniLM-L6-v2 dimension 384, got {dim}')

answerable={}; refusal={ds:[] for ds,_,_ in DATASETS}
for ds,_,filename in DATASETS:
    obj=json.loads((BENCH/filename).read_text(encoding='utf-8'))
    answerable[ds]=obj['eval_queries']
refusal_obj=json.loads((BENCH/'cross_dataset_refusal_clean_v1.json').read_text(encoding='utf-8'))
for q in refusal_obj['queries']:
    refusal[q['dataset_id']].append(q)

# Fail closed if any approved gold id is absent before computing retrieval metrics.
for ds,items in answerable.items():
    print(f'Checking gold IDs in {ds}...',flush=True)
    ids=sorted({gid for item in items for gid in item['ground_truth_chunk_ids']})
    present=set(collections[ds].get(ids=ids,include=['metadatas'])['ids'])
    missing=sorted(set(ids)-present)
    if missing: raise RuntimeError(f'{ds}: missing gold IDs: {missing}')

rows=[]
aggregates={}
query_counts={}
for ds,coll_name,_ in DATASETS:
    print(f'Retrieving {ds} queries...',flush=True)
    coll=collections[ds]
    batch=[]
    for item in answerable[ds]:
        batch.append((item['query_id'],item['query'],set(item['ground_truth_chunk_ids']),'answerable'))
    for item in refusal[ds]:
        batch.append((item['query_id'],item['query'],set(),'refusal_candidate'))
    for qid,qtext,gold,qtype in batch:
        vec=model.encode(qtext,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype(np.float32)
        res=coll.query(query_embeddings=[vec.tolist()],n_results=TOP_K,include=['distances','documents'])
        ids=res['ids'][0]; distances=res['distances'][0]; docs=res['documents'][0]
        scores=[1.0-float(d) for d in distances]
        if len(ids)>TOP_K: raise RuntimeError('Chroma returned more than top_k')
        if qtype=='answerable':
            h,m,n,p=metrics(ids,gold,TOP_K)
            first=next((i+1 for i,x in enumerate(ids) if x in gold),None)
            failure='hit_at_1' if first==1 else ('hit_at_2_to_5' if first else 'gold_not_retrieved_top5')
            for key,val in [('hit_rate_at_5',h),('mrr_at_5',m),('ndcg_at_5',n),('precision_at_5',p)]:
                aggregates.setdefault(ds,{}).setdefault(key,[]).append(val)
        else:
            h=m=n=p=None; first=None
            failure='candidate_chunks_returned_for_human_review' if ids else 'no_candidates_returned_not_proof_of_absence'
        query_counts.setdefault(ds,{'answerable':0,'refusal_candidates':0})[('answerable' if qtype=='answerable' else 'refusal_candidates')]+=1
        if not ids:
            rows.append({'dataset':ds,'collection_name':coll_name,'query_id':qid,'query_type':qtype,'query':qtext,
             'expected_behavior':'abstain_or_refuse' if qtype=='refusal_candidate' else 'answer_from_gold_evidence',
             'answerability_evidence':'human-approved candidate; corpus-wide absence not independently established' if qtype=='refusal_candidate' else 'human-approved answerable benchmark',
             'gold_chunk_ids':'|'.join(sorted(gold)),'rank':'','retrieved_chunk_id':'','similarity_cosine':'','gold_match':'',
             'retrieved_chunk_text':'','hit_rate_at_5':h,'mrr_at_5':m,'ndcg_at_5':n,'precision_at_5':p,
             'first_gold_rank':'','failure_classification':failure,'top_k':TOP_K})
        for rank,(rid,score,doc) in enumerate(zip(ids,scores,docs),1):
            rows.append({'dataset':ds,'collection_name':coll_name,'query_id':qid,'query_type':qtype,'query':qtext,
             'expected_behavior':'abstain_or_refuse' if qtype=='refusal_candidate' else 'answer_from_gold_evidence',
             'answerability_evidence':'human-approved candidate; corpus-wide absence not independently established' if qtype=='refusal_candidate' else 'human-approved answerable benchmark',
             'gold_chunk_ids':'|'.join(sorted(gold)),'rank':rank,'retrieved_chunk_id':rid,'similarity_cosine':f'{score:.9f}',
             'gold_match':str(rid in gold).lower(),'retrieved_chunk_text':doc or '',
             'hit_rate_at_5':h,'mrr_at_5':m,'ndcg_at_5':n,'precision_at_5':p,
             'first_gold_rank':first if first is not None else '', 'failure_classification':failure,'top_k':TOP_K})

fields=list(rows[0])
with (OUT/'per_query_vector_retrieval.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

summary={}
for ds,_,_ in DATASETS:
    summary[ds]={k:sum(v)/len(v) for k,v in aggregates[ds].items()}
    summary[ds]['answerable_query_count']=query_counts[ds]['answerable']
    summary[ds]['refusal_candidate_query_count']=query_counts[ds]['refusal_candidates']

manifest={'experiment_id':'exp_cross_eval_vector_diagnostic_v1','started_completed_utc':datetime.now(timezone.utc).isoformat(),
 'retrieval_mode':'dense vector only; Chroma query; no generation, reranking, graph, or LLM inference',
 'model':MODEL_NAME,'device':'cpu','embedding_dimension':dim,'normalized_embeddings':True,'top_k':TOP_K,
 'similarity':'cosine (Chroma stored distances converted as 1-distance)','search_space_control':'each dataset evaluated only against its own named Chroma collection; no cross-collection mixing',
 'corpora':collection_meta,'query_counts':query_counts,'answerable_metrics':summary,
 'dataset_specific_claim_restrictions':{'D0':'Validation-Derived Diagnostic Baseline only. All 20 pilot questions match lora_val.jsonl questions and answers appear in validation completions; do not claim unseen independent test performance.'},
 'input_benchmark_sha256':{p.name:sha(p) for p in sorted(BENCH.glob('*.json'))},
 'metric_definitions':{'hit_rate_at_5':'fraction of answerable queries with >=1 adjudicated gold ID in top 5',
 'mrr_at_5':'mean reciprocal rank of first adjudicated gold ID, truncated at 5',
 'ndcg_at_5':'binary relevance DCG/log2(rank+1), normalized by ideal ranking for number of gold IDs, truncated at 5',
 'precision_at_5':'number of gold IDs in top 5 divided by 5; fixed denominator even if collection returns fewer'},
 'refusal_evaluation':'retrieval candidates logged only; no gold relevant IDs or semantic refusal outcome asserted'}
(OUT/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'aggregates':summary,'corpora':collection_meta,'query_counts':query_counts,'rows':len(rows)},indent=2))
