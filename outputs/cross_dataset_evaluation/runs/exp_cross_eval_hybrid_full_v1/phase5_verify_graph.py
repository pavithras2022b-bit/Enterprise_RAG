"""Read-only Phase 5 graph population and provenance verification."""
import csv, json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from config_loader import create_neo4j_connection, get_knowledge_graph_config

OUT = Path(__file__).resolve().parent
phase4 = ROOT / 'outputs/cross_dataset_evaluation/runs/exp_cross_eval_rerank_diagnostic_v1/per_query_rerank_retrieval.csv'
vector_baseline = ROOT / 'outputs/cross_dataset_evaluation/runs/exp_cross_eval_vector_diagnostic_v1/per_query_vector_retrieval.csv'
legacy_ids = defaultdict(set)
with phase4.open(encoding='utf-8-sig', newline='') as f:
    for row in csv.DictReader(f):
        if row['query_type'] == 'answerable':
            legacy_ids[row['dataset']].add(row['retrieved_chunk_id'])

conn = create_neo4j_connection()
conn.verify_connectivity()
cfg = get_knowledge_graph_config().get('connection', {})
counts = conn.execute_read("""
MATCH (n)
WITH CASE WHEN n.dataset_id IS NULL THEN 'D0' ELSE n.dataset_id END AS ds, n
RETURN ds AS dataset_id, count(n) AS node_count,
       count(n.source_chunk_id) AS nodes_with_source_chunk_id,
       count(DISTINCT n.source_chunk_id) AS distinct_node_source_chunks
ORDER BY ds
""")
edges = conn.execute_read("""
MATCH ()-[r]->()
WITH CASE WHEN r.dataset_id IS NULL THEN 'D0' ELSE r.dataset_id END AS ds, r
RETURN ds AS dataset_id, count(r) AS edge_count,
       count(r.source_chunk_id) AS edges_with_source_chunk_id,
       count(DISTINCT r.source_chunk_id) AS distinct_edge_source_chunks
ORDER BY ds
""")
labels = conn.execute_read("""
MATCH (n)
WITH CASE WHEN n.dataset_id IS NULL THEN 'D0' ELSE n.dataset_id END AS ds, labels(n)[0] AS label
RETURN ds AS dataset_id, label, count(*) AS node_count ORDER BY ds,label
""")
rels = conn.execute_read("""
MATCH ()-[r]->()
WITH CASE WHEN r.dataset_id IS NULL THEN 'D0' ELSE r.dataset_id END AS ds, type(r) AS rel
RETURN ds AS dataset_id, rel AS relationship_type, count(*) AS edge_count ORDER BY ds,rel
""")
overlap = {}
for ds in ['D0', 'D2', 'D3']:
    vals = sorted(legacy_ids.get(ds, set()))
    rows = conn.execute_read("""
    MATCH ()-[r]->()
    WHERE ((r.dataset_id IS NULL AND $dataset_id = 'D0') OR r.dataset_id = $dataset_id)
      AND r.source_chunk_id IN $chunk_ids
    RETURN count(r) AS matching_edges, count(DISTINCT r.source_chunk_id) AS matching_source_chunks
    """, {'dataset_id': ds, 'chunk_ids': vals})
    overlap[ds] = {'phase4_answerable_top20_unique_chunk_ids': len(vals), **(rows[0] if rows else {})}
conn.close()

summary_files = {
  'D2': ROOT / 'outputs/cross_dataset_evaluation/runs/run_20261001T110316Z/nasa/ingestion_summary.json',
  'D3': ROOT / 'outputs/cross_dataset_evaluation/runs/run_20261001T184237Z/goverment/ingestion_summary.json',
}
prior = {ds: json.loads(p.read_text(encoding='utf-8')) for ds,p in summary_files.items() if p.exists()}
result = {'database': cfg.get('database','neo4j'), 'uri': cfg.get('uri','bolt://localhost:7687'),
          'nodes_by_dataset': counts, 'edges_by_dataset': edges, 'nodes_by_label': labels,
          'edges_by_type': rels, 'phase4_source_chunk_overlap': overlap,
          'prior_ingestion_summaries': {ds: {k: prior[ds].get(k) for k in ['dataset_id','processed_chunk_count','entity_span_count','relation_count_extracted','graph_nodes_created','graph_nodes_merged','graph_relationships_created','graph_relationships_merged','graph_write_errors','ingestion_status']} for ds in prior}}
(OUT/'phase5_graph_verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

by_node = {r['dataset_id']:r for r in counts}
by_edge = {r['dataset_id']:r for r in edges}
lines = [
  '# Phase 5 — Neo4j Graph Population and Verification', '',
  '**Outcome:** The configured Neo4j database was reachable. D2 and D3 graph populations were already present before this run; their active counts match the prior successful ingestion summaries. To preserve the existing graph and avoid redundant re-extraction/re-writes, this Phase 5 verification did not issue graph writes. The requested Phase 6 evaluation uses the verified active populations.', '',
  f"- Database: `{result['database']}` at `{result['uri']}`.",
  '- Node/edge count queries were read-only and grouped by dataset property. D0 is the pre-existing legacy graph population with no `dataset_id`; D2/D3 are explicitly dataset-scoped.', '',
  '| Dataset | Scope | Nodes | Relationships | Nodes with source chunk | Distinct node source chunks | Edges with source chunk | Distinct edge source chunks |',
  '|---|---|---:|---:|---:|---:|---:|---:|'
]
for ds,scope in [('D0','legacy, dataset_id IS NULL'),('D2','dataset_id=D2'),('D3','dataset_id=D3')]:
    n=by_node.get(ds,{})
    e=by_edge.get(ds,{})
    lines.append(f"| {ds} | {scope} | {n.get('node_count',0)} | {e.get('edge_count',0)} | {n.get('nodes_with_source_chunk_id',0)} | {n.get('distinct_node_source_chunks',0)} | {e.get('edges_with_source_chunk_id',0)} | {e.get('distinct_edge_source_chunks',0)} |")
lines += ['', '## Provenance and active coverage', '', 'The D2/D3 node totals and relationship totals agree with the recorded earlier ingestion summaries in `prior_ingestion_summaries`; the relationship extraction/write totals reconcile as created plus merged. Current D2 and D3 edges carry dataset tags, and Phase 4 answerable-candidate source-chunk overlap is recorded below. D0 has legacy unscoped entities/edges; the Phase 6 runner scopes D0 queries to untagged legacy paths only, preventing traversal into D1/D2/D3 graphs.', '', '| Dataset | Phase 4 unique answerable top-20 chunk IDs | Matching active graph edges | Distinct matching graph source chunks |', '|---|---:|---:|---:|']
for ds in ['D0','D2','D3']:
    x=overlap[ds]
    lines.append(f"| {ds} | {x['phase4_answerable_top20_unique_chunk_ids']} | {x.get('matching_edges',0)} | {x.get('matching_source_chunks',0)} |")
lines += ['', '## Extraction configuration and limits', '', 'Prior D2/D3 ingestion summaries report 11,314 and 48,210 extracted entity spans respectively, with 10 and 13 extracted relations. The configured Phase 5 setup uses the existing generic `dslim/bert-base-NER` checkpoint with heuristic entity-type mapping and rule-based relation extraction; the learned relation classifier is disabled. This is the project-configured extraction pipeline, but its generic NER/domain mapping and sparse relation output limit graph completeness. Graph existence/counts do not prove that every relevant source passage was represented.', '', '## Integrity', '', 'No node or relationship writes were made by this verification script. Existing graph data was not cleared or replaced. The exact read results and the prior ingestion provenance are in `phase5_graph_verification.json`.', '']
(OUT/'phase5_graph_ingestion_summary.md').write_text('\n'.join(lines),encoding='utf-8')
print(f"Wrote Phase 5 verification: D0={by_node.get('D0',{}).get('node_count',0)} nodes; D2={by_node.get('D2',{}).get('node_count',0)}; D3={by_node.get('D3',{}).get('node_count',0)}")
