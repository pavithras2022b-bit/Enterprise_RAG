import csv, hashlib, json
from pathlib import Path

root=Path(__file__).resolve().parents[4]
bench=root/'outputs/cross_dataset_evaluation/benchmarks/clean_v1'
adj=root/'outputs/cross_dataset_evaluation/runs/exp_cross_eval_adjudication_v1/answerable_item_adjudication.csv'
with adj.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
byid={r['question_id']:r for r in rows}
manifest=json.loads((bench/'clean_v1_manifest.json').read_text(encoding='utf-8'))
for ds,name in [('D0','d0_meridian_clean_v1.json'),('D2','d2_nasa_clean_v1.json'),('D3','d3_government_clean_v1.json')]:
    obj=json.loads((bench/name).read_text(encoding='utf-8'))
    assert len(obj['eval_queries'])==20
    for q in obj['eval_queries']:
        r=byid[q['query_id']]
        assert q['query']==(r['proposed_question'] or r['question'])
        assert q['reference_answer']==(r['proposed_reference_answer'] or r['reference_answer'])
        assert q['ground_truth_chunk_ids']==r['gold_chunk_ids'].split('|')
refusal=json.loads((bench/'cross_dataset_refusal_clean_v1.json').read_text(encoding='utf-8'))
assert len(refusal['queries'])==15
for name,digest in manifest['frozen_files'].items():
    assert hashlib.sha256((bench/name).read_bytes()).hexdigest()==digest
print('Artifact consistency verified: 60 answerable records, 9 correction records in clean copies, 15 refusal candidates, frozen hashes match.')
