"""Phase 3 generation diagnostic using frozen benchmarks and Phase 4 contexts."""
import csv, hashlib, json, re, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent
BENCH=ROOT/'outputs/cross_dataset_evaluation/benchmarks/clean_v1'
ADJ=ROOT/'outputs/cross_dataset_evaluation/runs/exp_cross_eval_adjudication_v1/answerable_item_adjudication.csv'
PHASE4=ROOT/'outputs/cross_dataset_evaluation/runs/exp_cross_eval_rerank_diagnostic_v1/per_query_rerank_retrieval.csv'
DATASETS=[('D0','d0_meridian_clean_v1.json'),('D2','d2_nasa_clean_v1.json'),('D3','d3_government_clean_v1.json')]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def token_f1(pred,target):
    p=re.findall(r'\w+',pred.lower()); t=re.findall(r'\w+',target.lower())
    if not p or not t:return 0.0
    common=sum((Counter(p)&Counter(t)).values()); precision=common/len(p); recall=common/len(t)
    return 2*precision*recall/(precision+recall) if precision+recall else 0.0
def citation_stats(answer,valid_ids):
    markers=[x.strip() for x in re.findall(r'\[([^\[\]]+)\]',answer)]
    valid=[x for x in markers if x in valid_ids]
    return float(bool(valid)), (len(valid)/len(markers) if markers else 0.0), markers, valid
def lexical_grounding(answer,evidence):
    clean=re.sub(r'\[[^\[\]]+\]',' ',answer)
    words=set(re.findall(r'\w+',clean.lower()))
    if not words:return 0.0
    context=set(re.findall(r'\w+',' '.join(evidence).lower()))
    return len(words&context)/len(words)
def is_refusal(answer):
    pat=re.compile(r"\b(not specified|not in (?:the )?(?:provided )?context|does not (?:contain|provide|specify)|do(?:es)? not have enough information|no (?:relevant )?information|insufficient (?:information|evidence)|cannot answer|can't answer|unable to answer|cannot determine|can't determine|not enough information|information is unavailable)\b",re.I)
    return bool(pat.search(answer or ''))

# Import project-configured prompt/client wrappers without touching retrievers or other subsystems.
from config_loader import get_llm_config
from llm.client import LLMClient
from llm.prompt_builder import PromptBuilder

config=get_llm_config(); client_cfg=config.get('client',{})
llm=LLMClient(base_url=client_cfg.get('base_url','http://localhost:11434'),
 model_name=client_cfg.get('model_name','qwen2.5-3b-lora-ragqa'),
 timeout=client_cfg.get('timeout_seconds',180.0),temperature=client_cfg.get('temperature',0.2))
builder=PromptBuilder(system_instruction=config.get('prompt_builder',{}).get('system_instruction'))
print(f"Configured generation endpoint {client_cfg.get('base_url')} model={client_cfg.get('model_name')}; temperature={client_cfg.get('temperature',0.2)}",flush=True)

print('Loading configured semantic-similarity embedding model from local cache...',flush=True)
sim_model=SentenceTransformer('all-MiniLM-L6-v2',device='cpu',local_files_only=True)
adjudications={}
with ADJ.open(encoding='utf-8-sig',newline='') as f:
    for r in csv.DictReader(f): adjudications[r['question_id']]=r
phase4={}
with PHASE4.open(encoding='utf-8-sig',newline='') as f:
    for r in csv.DictReader(f):
        if r['query_type']=='answerable' and int(r['reranked_rank'])<=5:
            phase4.setdefault(r['query_id'],[]).append(r)
        elif r['query_type']=='refusal_candidate' and int(r['reranked_rank'])<=5:
            phase4.setdefault(r['query_id'],[]).append(r)

records=[]; generated_texts=[]; ref_texts=[]; semantic_targets=[]
def run_one(qid,ds,query,reference,context_type,items,is_refusal_query=False):
    evidence=[{'candidate_id':item['retrieved_chunk_id'],'text':item['retrieved_chunk_text']} for item in items]
    valid_ids={e['candidate_id'] for e in evidence}
    prompt=builder.build_prompt(query,[SimpleNamespace(**e) for e in evidence])
    start=time.perf_counter(); raw=''; error=''
    try: raw=llm.generate(prompt)
    except Exception as exc: error=f'{type(exc).__name__}: {exc}'
    latency=time.perf_counter()-start
    if is_refusal_query:
        refused=is_refusal(raw) if not error else False
        return {'query_id':qid,'dataset':ds,'query_type':'refusal_candidate','query':query,'reference_answer':'',
          'retrieved_pass_a_context_ids':json.dumps([e['candidate_id'] for e in evidence]),'oracle_gold_context_ids':'',
          'pass_a_generated_answer':raw,'pass_a_citations':json.dumps(re.findall(r'\[([^\[\]]+)\]',raw)),
          'pass_a_valid_citations':json.dumps(citation_stats(raw,valid_ids)[3]),'pass_a_citation_compliance':citation_stats(raw,valid_ids)[0],
          'pass_a_citation_grounding_rate':citation_stats(raw,valid_ids)[1],'pass_a_evidence_token_grounding_rate':lexical_grounding(raw,[e['text'] for e in evidence]),
          'pass_a_token_f1':'','pass_a_semantic_similarity':'','pass_a_latency_seconds':round(latency,3),
          'pass_b_generated_answer':'','pass_b_citations':'','pass_b_valid_citations':'','pass_b_citation_compliance':'',
          'pass_b_citation_grounding_rate':'','pass_b_evidence_token_grounding_rate':'','pass_b_token_f1':'','pass_b_semantic_similarity':'','pass_b_latency_seconds':'',
          'ood_refusal_expected':'yes_candidate_only','ood_refusal_detected':str(refused).lower(),'ood_refusal_accuracy':float(refused) if not error else '',
          'generation_error':error,'refusal_label_caveat':'Approved refusal-test candidate; corpus-wide answer absence not independently established.'}
    citations_a=citation_stats(raw,valid_ids)
    rec={'query_id':qid,'dataset':ds,'query_type':'answerable','query':query,'reference_answer':reference,
      'retrieved_pass_a_context_ids':json.dumps([e['candidate_id'] for e in evidence]),'oracle_gold_context_ids':'',
      'pass_a_generated_answer':raw,'pass_a_citations':json.dumps(citations_a[2]),'pass_a_valid_citations':json.dumps(citations_a[3]),
      'pass_a_citation_compliance':citations_a[0],'pass_a_citation_grounding_rate':citations_a[1],
      'pass_a_evidence_token_grounding_rate':lexical_grounding(raw,[e['text'] for e in evidence]),
      'pass_a_token_f1':token_f1(raw,reference),'pass_a_semantic_similarity':'','pass_a_latency_seconds':round(latency,3),
      'pass_b_generated_answer':'','pass_b_citations':'','pass_b_valid_citations':'','pass_b_citation_compliance':'',
      'pass_b_citation_grounding_rate':'','pass_b_evidence_token_grounding_rate':'','pass_b_token_f1':'','pass_b_semantic_similarity':'','pass_b_latency_seconds':'',
      'ood_refusal_expected':'','ood_refusal_detected':'','ood_refusal_accuracy':'','generation_error':error,'refusal_label_caveat':''}
    generated_texts.append(raw); ref_texts.append(reference); semantic_targets.append((rec,'A'))
    return rec

for ds,filename in DATASETS:
    obj=json.loads((BENCH/filename).read_text(encoding='utf-8'))
    for ix,item in enumerate(obj['eval_queries'],1):
        qid=item['query_id']; q=query=item['query']; ref=item['reference_answer']
        retrieved=sorted(phase4.get(qid,[]),key=lambda x:int(x['reranked_rank']))
        if len(retrieved)!=5: raise RuntimeError(f'{qid}: expected 5 Phase 4 retrieved contexts, got {len(retrieved)}')
        row=run_one(qid,ds,q,ref,'retrieved',retrieved)
        row['retrieved_pass_a_context_ids']=json.dumps([x['retrieved_chunk_id'] for x in retrieved])
        gold_ids=item['ground_truth_chunk_ids']; gold_evidence=[]
        for gid in gold_ids:
            adj=adjudications.get(qid)
            if not adj: raise RuntimeError(f'Missing evidence audit row {qid}')
            audit_ids=adj['gold_chunk_ids'].split('|')
            if gid not in audit_ids: raise RuntimeError(f'{qid}: clean gold id {gid} differs from adjudication record')
            gold_evidence.append({'candidate_id':gid,'text':adj['exact_indexed_gold_passage']})
        prompt=builder.build_prompt(q,[SimpleNamespace(**e) for e in gold_evidence]); start=time.perf_counter(); raw=''; error_b=''
        try: raw=llm.generate(prompt)
        except Exception as exc: error_b=f'{type(exc).__name__}: {exc}'
        latency=time.perf_counter()-start
        stats=citation_stats(raw,{x['candidate_id'] for x in gold_evidence})
        row.update({'oracle_gold_context_ids':json.dumps([x['candidate_id'] for x in gold_evidence]),
          'pass_b_generated_answer':raw,'pass_b_citations':json.dumps(stats[2]),'pass_b_valid_citations':json.dumps(stats[3]),
          'pass_b_citation_compliance':stats[0],'pass_b_citation_grounding_rate':stats[1],
          'pass_b_evidence_token_grounding_rate':lexical_grounding(raw,[x['text'] for x in gold_evidence]),
          'pass_b_token_f1':token_f1(raw,ref),'pass_b_latency_seconds':round(latency,3),
          'generation_error':'; '.join(x for x in [row['generation_error'],error_b] if x)})
        generated_texts.append(raw); ref_texts.append(ref); semantic_targets.append((row,'B'))
        records.append(row)
    print(f'{ds}: completed answerable contexts {len(obj["eval_queries"])}/{len(obj["eval_queries"])}',flush=True)

# Refusal testing uses only the retrieved/reranked context regime; no oracle exists.
refobj=json.loads((BENCH/'cross_dataset_refusal_clean_v1.json').read_text(encoding='utf-8'))
for ix,item in enumerate(refobj['queries'],1):
    retrieved=sorted(phase4.get(item['query_id'],[]),key=lambda x:int(x['reranked_rank']))
    if len(retrieved)!=5: raise RuntimeError(f"{item['query_id']}: expected 5 refusal contexts, got {len(retrieved)}")
    records.append(run_one(item['query_id'],item['dataset_id'],item['query'],'','retrieved',retrieved,True))
    if ix%5==0: print(f'Refusal candidates completed {ix}/15',flush=True)

# Encode generated answers and references once, then cosine similarity per paired output.
print(f'Computing semantic similarities for {len(generated_texts)} answer outputs...',flush=True)
vectors=sim_model.encode(generated_texts+ref_texts,normalize_embeddings=True,convert_to_numpy=True,batch_size=32,show_progress_bar=False)
n=len(generated_texts)
for i,(row,regime) in enumerate(semantic_targets):
    sim=float(np.dot(vectors[i],vectors[n+i]))
    row[f'pass_{regime.lower()}_semantic_similarity']=sim

outpath=OUT/'per_query_generation_diagnostic.csv'
with outpath.open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
manifest={'experiment_id':'exp_cross_eval_generation_diagnostic_v1','completed_utc':datetime.now(timezone.utc).isoformat(),
 'model_endpoint':client_cfg.get('base_url'),'configured_model_tag':llm._model_name,'temperature':client_cfg.get('temperature',0.2),
 'timeout_seconds':client_cfg.get('timeout_seconds',180),'served_model_metadata':{'family':'qwen2','parameter_size':'3.1B','quantization':'Q4_K_M','context_length_reported':32768,'Ollama_model_tag_verified':True},
 'adapter_configuration':{'configured_active_adapter':'qwen2.5-3b-lora-ragqa','configured_adapter_path':'outputs/qwen2.5-3b-lora-ragqa/final_adapter','adapter_files_present':True,'adapter_model_file_sha256':sha(ROOT/'outputs/qwen2.5-3b-lora-ragqa/final_adapter/adapter_model.safetensors'),'Ollama_show_explicit_adapter_directive':False,'note':'Served model tag is the configured LoRA-named tag; API metadata does not independently prove whether PEFT weights are merged into the served GGUF.'},
 'context_regimes':{'pass_a':'Top five reranked Phase 4 contexts, cross-encoder rank order.','pass_b':'Exact adjudicated indexed text for clean benchmark ground_truth_chunk_ids.'},
 'generation_counts':{'answerable_items':60,'answerable_passes':120,'refusal_candidates':15,'refusal_passes':15,'total_requests_expected':135},
 'metrics':{'token_f1':r'Multiset token overlap F1 with \w+ lowercase tokens.','semantic_similarity':'Cosine similarity of normalized all-MiniLM-L6-v2 answer/reference embeddings.',
  'citation_compliance':'Binary: at least one citation marker exactly matches a chunk ID in the supplied context.',
  'citation_grounding_rate':'Valid supplied-context citation markers divided by all citation markers; zero if no markers.',
  'evidence_token_grounding_rate':'Distinct generated answer tokens present in supplied evidence; lexical proxy, not entailment.'},
 'refusal_scoring':'Keyword/phrase detector; label denotes approved refusal-test candidate only, not independently verified corpus-wide absence.',
 'd0_claim_restriction':'D0 is a Validation-Derived Diagnostic Baseline, not an independent test set.',
 'input_sha256':{p.name:sha(p) for p in sorted(BENCH.glob('*.json'))},
 'phase4_retrieval_csv_sha256':sha(PHASE4),'adapter_config_sha256':sha(ROOT/'outputs/qwen2.5-3b-lora-ragqa/final_adapter/adapter_config.json')}
(OUT/'generation_run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(f'Wrote {len(records)} per-query rows to {outpath}',flush=True)
