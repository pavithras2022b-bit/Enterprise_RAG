import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parent
path = root / 'per_query_generation_diagnostic.csv'
with path.open(encoding='utf-8-sig', newline='') as f:
    rows = list(csv.DictReader(f))
fields = list(rows[0])
failed = []
for row in rows:
    if row['query_type'] == 'answerable' and row['generation_error']:
        failed.append(row['query_id'])
        for key in ('pass_a_token_f1', 'pass_a_semantic_similarity', 'pass_a_citation_compliance',
                    'pass_a_citation_grounding_rate', 'pass_a_evidence_token_grounding_rate'):
            row[key] = ''
with path.open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

manifest_path = root / 'generation_run_manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
manifest['generation_counts'].update({
    'answerable_rows_saved': sum(r['query_type'] == 'answerable' for r in rows),
    'refusal_candidate_rows_saved': sum(r['query_type'] == 'refusal_candidate' for r in rows),
    'answerable_generation_errors': len(failed),
    'answerable_generation_error_ids': failed,
    'successful_paired_answerable_items': sum(r['query_type'] == 'answerable' and not r['generation_error'] for r in rows),
})
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(f'Rows={len(rows)}; failed answerable generations={failed}')
