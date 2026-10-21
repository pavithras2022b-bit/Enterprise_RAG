"""Freeze approved Phase 1 adjudications into a new clean_v1 copy."""
import csv, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
PROPOSED=ROOT/'data/benchmarks/proposed'
ADJ=ROOT/'outputs/cross_dataset_evaluation/runs/exp_cross_eval_adjudication_v1/answerable_item_adjudication.csv'
UNANS=ROOT/'outputs/cross_dataset_evaluation/runs/exp_cross_eval_adjudication_v1/unanswerable_item_adjudication.csv'
OUT=ROOT/'outputs/cross_dataset_evaluation/benchmarks/clean_v1'
RUN=ROOT/'outputs/cross_dataset_evaluation/runs/exp_cross_eval_vector_diagnostic_v1'
OUT.mkdir(parents=True,exist_ok=False)
RUN.mkdir(parents=True,exist_ok=True)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
with ADJ.open(encoding='utf-8-sig',newline='') as f: adjudications=list(csv.DictReader(f))
approved={r['question_id']:r for r in adjudications}
if len(adjudications)!=60: raise RuntimeError(f'Expected 60 answerable adjudications, got {len(adjudications)}')
mapping=[('D0','d0_benchmark_v2.json','d0_meridian_clean_v1.json'),('D2','d2_benchmark_v2.json','d2_nasa_clean_v1.json'),('D3','d3_benchmark_v2.json','d3_government_clean_v1.json')]
source_hashes={}
clean={}
for dataset,src_name,dst_name in mapping:
    src=PROPOSED/src_name; source_hashes[src_name]=sha(src)
    obj=json.loads(src.read_text(encoding='utf-8'))
    for item in obj['eval_queries']:
        row=approved.get(item['query_id'])
        if row is None or row['dataset']!=dataset: raise RuntimeError(f"Missing/mismatched adjudication for {item['query_id']}")
        item['query']=row['proposed_question'] or row['question']
        item['reference_answer']=row['proposed_reference_answer'] or row['reference_answer']
        item['ground_truth_chunk_ids']=[x for x in row['gold_chunk_ids'].split('|') if x]
        item['validation_status']='human_approved_clean_v1'
        item['adjudication_status']='approved_by_user_instruction_2026-10-14'
    if len(obj['eval_queries'])!=20: raise RuntimeError(f'{dataset} expected 20 eval queries')
    clean[dataset]=obj
    (OUT/dst_name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

with UNANS.open(encoding='utf-8-sig',newline='') as f: unans=list(csv.DictReader(f))
if len(unans)!=15: raise RuntimeError(f'Expected 15 refusal candidates, got {len(unans)}')
refusal=[]
for r in unans:
    refusal.append({'query_id':r['question_id'],'dataset_id':r['dataset'],'query':r['question'],
      'target_collection':{'D0':'enterprise_docs','D2':'nasa_tech_d2','D3':'goverment_policy_d3'}[r['dataset']],
      'expected_behavior':'abstain_or_refuse','answerability_status':'human-approved candidate for out-of-domain refusal testing; corpus-wide absence not independently established',
      'corpus_boundary':r['corpus_boundary'],'approval_status':'approved_by_user_instruction_2026-10-14'})
refusal_obj={'benchmark_id':'cross_dataset_refusal_clean_v1','dataset_ids':['D0','D2','D3'],'query_count':len(refusal),'queries':refusal}
(OUT/'cross_dataset_refusal_clean_v1.json').write_text(json.dumps(refusal_obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
source_hashes['answerable_item_adjudication.csv']=sha(ADJ); source_hashes['unanswerable_item_adjudication.csv']=sha(UNANS)
manifest={'benchmark_version':'clean_v1','decision_basis':'Explicit human approvals supplied in user instruction dated 2026-10-14; no automated recommendation is represented as human approval.',
 'answerable_counts':{'D0':20,'D2':20,'D3':20,'total':60},'approved_correction_count':9,
 'refusal_candidate_count':15,'d0_provenance_warning':'D0 is a Validation-Derived Diagnostic Baseline: all 20 pilot questions match lora_val.jsonl questions and answers appear in validation completions. It is not an independent test set.',
 'gold_ids':'Preserved from the adjudicated evidence rows; all 60 were reported present in their intended collections during Phase 1B.',
 'source_sha256':source_hashes,'frozen_files':{p.name:sha(p) for p in OUT.glob('*.json')}}
(OUT/'clean_v1_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

\n