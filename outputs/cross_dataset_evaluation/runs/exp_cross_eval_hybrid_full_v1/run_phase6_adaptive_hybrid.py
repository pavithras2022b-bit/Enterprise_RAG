"""Phase 6 matched adaptive-hybrid evaluation; reads frozen indexes and graphs."""
from __future__ import annotations
import csv, hashlib, json, math, re, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import replace

import chromadb
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
BENCH = ROOT / 'outputs/cross_dataset_evaluation/benchmarks/clean_v1'
SNAPSHOT = ROOT / 'outputs/cross_dataset_evaluation/runs/exp_cross_eval_rerank_diagnostic_v1/chroma_snapshot'
P2 = ROOT / 'outputs/cross_dataset_evaluation/runs/exp_cross_eval_vector_diagnostic_v1/per_query_vector_retrieval.csv'
P4 = ROOT / 'outputs/cross_dataset_evaluation/runs/exp_cross_eval_rerank_diagnostic_v1/per_query_rerank_retrieval.csv'
P3 = ROOT / 'outputs/cross_dataset_evaluation/runs/exp_cross_eval_generation_diagnostic_v1/per_query_generation_diagnostic.csv'
DATASETS = [('D0','enterprise_docs','d0_meridian_clean_v1.json'),('D2','nasa_tech_d2','d2_nasa_clean_v1.json'),('D3','goverment_policy_d3','d3_government_clean_v1.json')]

from config_loader import (create_embedding_model, create_cross_encoder_reranker,
                           create_alpha_classifier, create_llm_client,
                           create_neo4j_connection, get_adaptive_retrieval_config,
                           get_reranking_config, get_llm_config)
from adaptive_retrieval.fusion import AdaptiveFusionEngine
from knowledge_graph.schema import GraphSchema
from retrieval.entity_linker import EntityLinker
from retrieval.hybrid_controller import (HybridRetrievalController, RetrievalError,
                                         _SCOPED_TRAVERSAL_QUERY_TEMPLATE)
from retrieval.schema import GraphFact, CandidateSource
from vector_db.schema import VectorSearchResult
from evaluation.retrieval_metrics import calculate_hit_rate, calculate_mrr, calculate_ndcg
from reranking.schema import RankedCandidate

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def token_f1(pred, ref):
    p=re.findall(r'\w+', (pred or '').lower()); t=re.findall(r'\w+', (ref or '').lower())
    if not p or not t: return 0.0
    common=sum((Counter(p)&Counter(t)).values()); pr=common/len(p); rc=common/len(t)
    return (2*pr*rc/(pr+rc)) if pr+rc else 0.0
def detect_refusal(answer):
    return bool(re.search(r"\b(not specified|not in (?:the )?(?:provided )?context|does not (?:contain|provide|specify)|do(?:es)? not have enough information|no (?:relevant )?information|insufficient (?:information|evidence)|cannot answer|can't answer|unable to answer|cannot determine|can't determine|not enough information|information is unavailable)\b", answer or '', re.I))
def metric_ids(ranked, source_by_fact):
    out=[]; seen=set()
    for r in ranked[:5]:
        cand=r.candidate_id
        src=source_by_fact.get(cand) if '|' in cand and cand not in source_by_fact else None
        # Fused graph-only facts carry source_chunk_id in diagnostic metadata when available.
        src = r.original_candidate.metadata.get('phase6_source_chunk_id', src)
        doc_id=src or (cand if r.original_candidate.source != CandidateSource.GRAPH else '')
        if doc_id and doc_id in seen:
            out.append(f'__duplicate__{doc_id}')
        else:
            out.append(doc_id)
            if doc_id: seen.add(doc_id)
    return out

class ReadOnlySnapshotVectorStore:
    """Search-only adapter over Phase 4's isolated Chroma snapshot."""
    def __init__(self, collection): self.collection=collection
    def search(self, query_vector, k=20, where=None):
        if k<=0: raise ValueError('k must be positive')
        result=self.collection.query(query_embeddings=[query_vector.tolist()],n_results=k,where=where)
        ids=result.get('ids',[[]])[0]; distances=result.get('distances',[[]])[0]
        texts=result.get('documents',[[]])[0]; metas=result.get('metadatas',[[]])[0]
        return [VectorSearchResult(chunk_id=i,score=1-float(d),text=t or '',metadata=dict(m or {}))
                for i,d,t,m in zip(ids,distances,texts,metas)]

class LegacyD0EntityLinker(EntityLinker):
    """Entity linker index restricted to original, untagged D0 nodes."""
    def refresh_index(self):
        rows=self._connection.execute_read("MATCH (n) WHERE n.dataset_id IS NULL RETURN labels(n)[0] AS label,n.canonical_name AS canonical_name")
        self._index=[(r['label'],r['canonical_name']) for r in rows if r.get('label') and r.get('canonical_name')]

class LegacyD0Controller(HybridRetrievalController):
    """Project controller with D0-only predicates for its legacy untagged graph."""
    def run_graph_retrieval(self, entities, max_hops=None, k=20):
        hops=self._max_hops if max_hops is None else max_hops
        if hops<=0 or k<=0: raise ValueError('max_hops and k must be positive')
        if not entities: return []
        query=f"""MATCH path=(start {{canonical_name:$start_name}})-[*1..{int(hops)}]-(end)
WHERE start.dataset_id IS NULL AND all(n IN nodes(path) WHERE n.dataset_id IS NULL)
  AND all(r IN relationships(path) WHERE r.dataset_id IS NULL)
RETURN [n IN nodes(path)|n.canonical_name] AS node_names,
       [r IN relationships(path)|type(r)] AS rel_types,
       [r IN relationships(path)|r.source_chunk_id] AS source_chunk_ids,
       [r IN relationships(path)|startNode(r).canonical_name] AS rel_start_names,
       length(path) AS hops"""
        best={}
        for e in entities:
            rows=self._graph_connection.execute_read(query,{'start_name':e.canonical_name})
            for row in rows:
                names=row.get('node_names') or []; rels=row.get('rel_types') or []
                srcs=row.get('source_chunk_ids') or []; starts=row.get('rel_start_names') or []
                for i, rel in enumerate(rels):
                    if rel not in self._graph_schema.allowed_relationship_types:
                        raise RetrievalError(f'Out-of-schema graph relationship: {rel}')
                    a,b=names[i],names[i+1]
                    if i<len(starts) and starts[i]!=a: a,b=b,a
                    fact=GraphFact.build(a,rel,b,i+1,srcs[i] if i<len(srcs) else None)
                    if fact.fact_id not in best or fact.hop_count<best[fact.fact_id].hop_count: best[fact.fact_id]=fact
        return sorted(best.values(),key=lambda f:f.raw_relevance,reverse=True)[:k]

def read_baseline(path, phase):
    values={}
    with path.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            if row.get('query_type')!='answerable': continue
            qid=row['query_id']
            if qid in values: continue
            if phase==2:
                values[qid]={k:float(row[k]) for k in ['hit_rate_at_5','mrr_at_5','ndcg_at_5']}
            else:
                values[qid]={k:float(row[k]) for k in ['rerank_hit_rate_at_5','rerank_mrr_at_5','rerank_ndcg_at_5']}
    return values

def write_manifest(data):
    (OUT/'phase6_run_manifest.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def main():
    csv_path=OUT/'per_query_adaptive_hybrid_eval.csv'
    manifest_path=OUT/'phase6_run_manifest.json'
    resuming=csv_path.exists()
    if resuming:
        with csv_path.open(encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
        done={r['query_id'] for r in rows}
        manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
        manifest['status']='resuming'
        errors=[{'query_id':r['query_id'],'dataset':r['dataset'],'error':r['generation_error']} for r in rows if r.get('generation_error')]
        print(f'Resuming from {len(done)} saved query rows.',flush=True)
    else:
        rows=[]; done=set(); errors=[]
        manifest={'experiment_id':'exp_cross_eval_hybrid_full_v1','status':'initializing',
          'started_utc':datetime.now(timezone.utc).isoformat(),
          'd0_claim_restriction':'D0 is a Validation-Derived Diagnostic Baseline; the 20 pilot questions originate from lora_val.jsonl and answers appear in validation completions; not an independent test set.',
          'benchmark_files':{},'clean_v1_manifest_sha256':sha(BENCH/'clean_v1_manifest.json'),'phase2_csv_sha256':sha(P2),'phase4_csv_sha256':sha(P4),'phase3_csv_sha256':sha(P3),
          'chroma_source':'Phase 4 isolated read-only snapshot; no Chroma add/update/delete API used.',
          'execution_controls':{'vector_top_k':20,'graph_top_k':20,'reranked_top_k':5,'query_scope':'each dataset searches only its named collection and matching dataset-scoped graph; D0 uses legacy untagged graph only','temperature':None,'model_tag':None},
          'completion':{'answerable_queries_expected':60,'refusal_candidates_expected':15,'rows_written':0,'generation_errors':[]}}
    write_manifest(manifest)
    print('Loading frozen configs and local models...',flush=True)
    emb=create_embedding_model(); reranker=create_cross_encoder_reranker(); classifier=create_alpha_classifier(); llm=create_llm_client()
    alpha_cfg=get_adaptive_retrieval_config(); rerank_cfg=get_reranking_config(); llm_cfg=get_llm_config()
    client=chromadb.PersistentClient(path=str(SNAPSHOT),settings=chromadb.config.Settings(anonymized_telemetry=False))
    p2=read_baseline(P2,2); p4=read_baseline(P4,4)
    p3={}
    with P3.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['query_type']=='answerable': p3[r['query_id']]=r
    clean_manifest=json.loads((BENCH/'clean_v1_manifest.json').read_text(encoding='utf-8'))
    conn=create_neo4j_connection(); conn.verify_connectivity()
    collections={}; stores={}; counts={}
    for ds,name,_ in DATASETS:
        col=client.get_collection(name=name)
        stores[ds]=ReadOnlySnapshotVectorStore(col)
        collections[ds]={'name':name,'id':str(col.id),'count':col.count(),'metadata':col.metadata or {}}
        expected={'D0':149,'D2':1164,'D3':6119}[ds]
        if col.count()!=expected: raise RuntimeError(f'{ds} snapshot count {col.count()} != verified Phase 2/4 count {expected}')
    graph_counts=conn.execute_read("""MATCH (n) WITH CASE WHEN n.dataset_id IS NULL THEN 'D0' ELSE n.dataset_id END AS ds,count(n) AS n RETURN ds,n""")
    graph_edges=conn.execute_read("""MATCH ()-[r]->() WITH CASE WHEN r.dataset_id IS NULL THEN 'D0' ELSE r.dataset_id END AS ds,count(r) AS n RETURN ds,n""")
    counts={r['ds']:{'nodes':r['n']} for r in graph_counts}
    for r in graph_edges: counts.setdefault(r['ds'],{})['edges']=r['n']
    linkers={}; controllers={}; engines={}
    for ds,_,_ in DATASETS:
        if ds=='D0':
            linker=LegacyD0EntityLinker(conn,refresh_on_init=False)
            linker.refresh_index()
            controller=LegacyD0Controller(stores[ds],conn,emb,linker,dataset_id=None)
        else:
            linker=EntityLinker(conn,refresh_on_init=False,dataset_id=ds)
            linker.refresh_index()
            controller=HybridRetrievalController(stores[ds],conn,emb,linker,dataset_id=ds)
        linkers[ds]=linker; controllers[ds]=controller
        engines[ds]=AdaptiveFusionEngine(controller,emb,linker,classifier)
    alpha_name=type(classifier).__name__
    manifest.update({'status':'running','model_configuration':{'embedding_model':'all-MiniLM-L6-v2','reranker_model':rerank_cfg.get('model',{}).get('checkpoint'),'reranker_max_length':rerank_cfg.get('model',{}).get('max_length'),'alpha_classifier_config':alpha_cfg.get('classifier_type'),'alpha_classifier_loaded':alpha_name,'generator_model_tag':llm._model_name,'generator_endpoint':llm_cfg.get('client',{}).get('base_url'),'generator_temperature':llm_cfg.get('client',{}).get('temperature'),'generation_method':'configured LLMClient.generate_answer; citation extraction/validation enabled'},
      'execution_controls':{'vector_top_k':20,'graph_top_k':20,'reranked_top_k':5,'query_scope':'each dataset searches only its named collection and matching dataset-scoped graph; D0 uses legacy untagged graph only','temperature':llm_cfg.get('client',{}).get('temperature'),'model_tag':llm._model_name},
      'collections':collections,'graph_counts':counts,'benchmarks':{},'alpha_summary':{},'software_versions':{'chromadb':chromadb.__version__}})
    fieldnames=['query_id','dataset','query_type','query','reference_answer','gold_chunk_ids','alpha_q','alpha_classifier','query_entity_links','vector_top20_ids','vector_top20_scores','vector_top20_texts','graph_top20_fact_ids','graph_top20_source_chunk_ids','graph_top20_texts','fused_candidate_ids','fused_candidate_sources','fused_candidate_scores','reranked_top5_candidate_ids','reranked_top5_source_chunk_ids','reranked_top5_sources','reranked_top5_scores','reranked_top5_gold_flags','retrieval_metric_chunk_ids_at5','hit_rate_at_5','mrr_at_5','ndcg_at_5','generated_answer','validated_citations','citation_compliance','citation_validity','token_f1','semantic_similarity','generation_latency_seconds','generation_error','candidate_refusal_expected','candidate_refusal_detected','candidate_refusal_accuracy','refusal_evidence_caveat']
    out_csv=csv_path
    fcsv=out_csv.open('a' if resuming else 'w',encoding='utf-8-sig',newline=''); writer=csv.DictWriter(fcsv,fieldnames=fieldnames)
    if not resuming: writer.writeheader()
    fcsv.flush()
    def eval_one(ds,qid,query,ref,gold,query_type):
        start=time.perf_counter(); error=''; alpha=None; vec=[]; facts=[]; fused=[]; ranked=[]; answer=''; citations=[]
        try:
            alpha=engines[ds].compute_alpha(query)
            print(f'{qid}: alpha ready',flush=True)
            linked=linkers[ds].link(query)
            print(f'{qid}: linked {len(linked)} entities',flush=True)
            vec=controllers[ds].run_vector_retrieval(query,k=20)
            print(f'{qid}: vector {len(vec)}',flush=True)
            facts=controllers[ds].run_graph_retrieval(linked,k=20)
            print(f'{qid}: graph {len(facts)}',flush=True)
            fused=controllers[ds].static_fuse(vec,facts,alpha=alpha)
            print(f'{qid}: fused {len(fused)}',flush=True)
            fact_src={f.fact_id:f.source_chunk_id for f in facts}
            expanded=[]
            for c in fused:
                if c.source==CandidateSource.GRAPH:
                    source_id=fact_src.get(c.candidate_id)
                    md=dict(c.metadata)
                    if source_id: md['phase6_source_chunk_id']=source_id
                    c=replace(c,metadata=md)
                expanded.append(c)
            fused=expanded
            ranked=reranker.rerank(query,fused,top_n=5)
            print(f'{qid}: reranked {len(ranked)}',flush=True)
            if not ranked: raise RuntimeError('No candidates reached generation stage.')
            gen=llm.generate_answer(query=query,evidence=ranked)
            answer=gen.text; citations=list(gen.citations)
            print(f'{qid}: generation returned',flush=True)
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}'
            errors.append({'query_id':qid,'dataset':ds,'error':error})
        latency=time.perf_counter()-start
        fact_src={f.fact_id:f.source_chunk_id for f in facts}
        metric_chunks=metric_ids(ranked,fact_src)
        is_answerable=query_type=='answerable'
        hit=calculate_hit_rate(metric_chunks,gold,k=5) if is_answerable else ''
        mrr=calculate_mrr(metric_chunks,gold,k=5) if is_answerable else ''
        ndcg=calculate_ndcg(metric_chunks,gold,k=5) if is_answerable else ''
        if is_answerable and not ref:
            f1=''; sim=''
        elif is_answerable and error:
            f1=''; sim=''
        elif is_answerable:
            f1=token_f1(answer,ref); sim=float(np.dot(emb.embed(answer),emb.embed(ref)))
        else:
            f1=''; sim=''
        valid_ids={r.candidate_id for r in ranked}
        citation_ok=bool(citations) and all(c in valid_ids for c in citations)
        refusal=detect_refusal(answer) if not error and query_type=='refusal_candidate' else False
        out={
          'query_id':qid,'dataset':ds,'query_type':query_type,'query':query,'reference_answer':ref or '',
          'gold_chunk_ids':json.dumps(gold,ensure_ascii=False),'alpha_q':alpha if alpha is not None else '',
          'alpha_classifier':alpha_name,'query_entity_links':json.dumps([{'canonical_name':e.canonical_name,'entity_type':e.entity_type.value,'match_score':e.match_score} for e in linked] if 'linked' in locals() else [],ensure_ascii=False),
          'vector_top20_ids':json.dumps([x.chunk_id for x in vec],ensure_ascii=False),'vector_top20_scores':json.dumps([float(x.score) for x in vec]),'vector_top20_texts':json.dumps([x.text for x in vec],ensure_ascii=False),
          'graph_top20_fact_ids':json.dumps([x.fact_id for x in facts],ensure_ascii=False),'graph_top20_source_chunk_ids':json.dumps([x.source_chunk_id for x in facts],ensure_ascii=False),'graph_top20_texts':json.dumps([x.text for x in facts],ensure_ascii=False),
          'fused_candidate_ids':json.dumps([x.candidate_id for x in fused],ensure_ascii=False),'fused_candidate_sources':json.dumps([x.source.value for x in fused]),'fused_candidate_scores':json.dumps([float(x.fused_score) for x in fused]),
          'reranked_top5_candidate_ids':json.dumps([x.candidate_id for x in ranked],ensure_ascii=False),'reranked_top5_source_chunk_ids':json.dumps([x.original_candidate.metadata.get('phase6_source_chunk_id') or (x.candidate_id if x.original_candidate.source in (CandidateSource.VECTOR,CandidateSource.BOTH) else None) for x in ranked],ensure_ascii=False),
          'reranked_top5_sources':json.dumps([x.original_candidate.source.value for x in ranked]),'reranked_top5_scores':json.dumps([float(x.rerank_score) for x in ranked]),
          'reranked_top5_gold_flags':json.dumps([bool((x.original_candidate.metadata.get('phase6_source_chunk_id') or (x.candidate_id if x.original_candidate.source in (CandidateSource.VECTOR,CandidateSource.BOTH) else None)) in gold) for x in ranked]),
          'retrieval_metric_chunk_ids_at5':json.dumps(metric_chunks,ensure_ascii=False),'hit_rate_at_5':hit,'mrr_at_5':mrr,'ndcg_at_5':ndcg,
          'generated_answer':answer,'validated_citations':json.dumps(citations,ensure_ascii=False),'citation_compliance':float(citation_ok) if not error else '',
          'citation_validity':float(citation_ok) if citations else (0.0 if not error and ranked else ''),'token_f1':f1,'semantic_similarity':sim,'generation_latency_seconds':round(latency,3),
          'generation_error':error,'candidate_refusal_expected':'candidate_refusal' if query_type=='refusal_candidate' else '',
          'candidate_refusal_detected':str(refusal).lower() if query_type=='refusal_candidate' else '',
          'candidate_refusal_accuracy':float(refusal) if query_type=='refusal_candidate' and not error else '',
          'refusal_evidence_caveat':'Approved refusal-test candidate only; corpus-wide answer absence is not independently established.' if query_type=='refusal_candidate' else ''}
        writer.writerow(out); fcsv.flush(); rows.append(out)
        return out
    try:
      for ds,_,filename in DATASETS:
        obj=json.loads((BENCH/filename).read_text(encoding='utf-8'))
        manifest['benchmarks'][ds]={'file':filename,'sha256':sha(BENCH/filename),'answerable_queries':len(obj['eval_queries'])}
        write_manifest(manifest)
        for i,item in enumerate(obj['eval_queries'],1):
            if item['query_id'] in done:
                print(f"{ds} answerable {i}/{len(obj['eval_queries'])} already checkpointed",flush=True)
                continue
            eval_one(ds,item['query_id'],item['query'],item['reference_answer'],item['ground_truth_chunk_ids'],'answerable')
            done.add(item['query_id'])
            manifest['completion'].update({'rows_written':len(rows),'last_query_id':item['query_id'],'generation_errors':len(errors)})
            write_manifest(manifest)
            print(f"{ds} answerable {i}/{len(obj['eval_queries'])} (rows {len(rows)})",flush=True)
      refusal_obj=json.loads((BENCH/'cross_dataset_refusal_clean_v1.json').read_text(encoding='utf-8'))
      manifest['benchmarks']['refusal']={'file':'cross_dataset_refusal_clean_v1.json','sha256':sha(BENCH/'cross_dataset_refusal_clean_v1.json'),'candidate_queries':len(refusal_obj['queries'])}
      for i,item in enumerate(refusal_obj['queries'],1):
          if item['query_id'] in done:
              print(f"Refusal candidate {i}/{len(refusal_obj['queries'])} already checkpointed",flush=True)
              continue
          eval_one(item['dataset_id'],item['query_id'],item['query'],'',[],'refusal_candidate')
          done.add(item['query_id'])
          manifest['completion'].update({'rows_written':len(rows),'last_query_id':item['query_id'],'generation_errors':len(errors)})
          write_manifest(manifest)
          print(f"Refusal candidate {i}/{len(refusal_obj['queries'])} (rows {len(rows)})",flush=True)
    finally:
      fcsv.close(); conn.close()
    alpha_summary={}
    for ds,_,_ in DATASETS:
        qvals=[float(r['alpha_q']) for r in rows if r['dataset']==ds and r['alpha_q']!='']
        if not qvals: continue
        alpha_summary[ds]={'n':len(qvals),'mean':float(np.mean(qvals)),'std':float(np.std(qvals)),'min':float(np.min(qvals)),'p25':float(np.quantile(qvals,.25)),'median':float(np.median(qvals)),'p75':float(np.quantile(qvals,.75)),'max':float(np.max(qvals)),'vector_oriented_alpha_ge_0_5':sum(x>=.5 for x in qvals),'graph_oriented_alpha_lt_0_5':sum(x<.5 for x in qvals)}
    manifest.update({'status':'completed','completed_utc':datetime.now(timezone.utc).isoformat(),'completion':{'rows_written':len(rows),'answerable_generation_errors':sum(bool(r['generation_error']) for r in rows if r['query_type']=='answerable'),'refusal_generation_errors':sum(bool(r['generation_error']) for r in rows if r['query_type']=='refusal_candidate'),'generation_errors':errors},'alpha_summary':alpha_summary,'output_sha256':sha(out_csv)})
    write_manifest(manifest)
    print(f"Completed {len(rows)} rows; errors={len(errors)}",flush=True)

if __name__=='__main__':
    main()
