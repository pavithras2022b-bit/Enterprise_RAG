# Phase 5 — Neo4j Graph Population and Verification

**Outcome:** The configured Neo4j database was reachable. D2 and D3 graph populations were already present before this run; their active counts match the prior successful ingestion summaries. To preserve the existing graph and avoid redundant re-extraction/re-writes, this Phase 5 verification did not issue graph writes. The requested Phase 6 evaluation uses the verified active populations.

- Database: `neo4j` at `bolt://localhost:7687`.
- Node/edge count queries were read-only and grouped by dataset property. D0 is the pre-existing legacy graph population with no `dataset_id`; D2/D3 are explicitly dataset-scoped.

| Dataset | Scope | Nodes | Relationships | Nodes with source chunk | Distinct node source chunks | Edges with source chunk | Distinct edge source chunks |
|---|---|---:|---:|---:|---:|---:|---:|
| D0 | legacy, dataset_id IS NULL | 483 | 106 | 483 | 113 | 0 | 0 |
| D2 | dataset_id=D2 | 5671 | 10 | 5671 | 977 | 10 | 9 |
| D3 | dataset_id=D3 | 11145 | 11 | 11145 | 2917 | 11 | 7 |

## Provenance and active coverage

The D2/D3 node totals and relationship totals agree with the recorded earlier ingestion summaries in `prior_ingestion_summaries`; the relationship extraction/write totals reconcile as created plus merged. Current D2 and D3 edges carry dataset tags, and Phase 4 answerable-candidate source-chunk overlap is recorded below. D0 has legacy unscoped entities/edges; the Phase 6 runner scopes D0 queries to untagged legacy paths only, preventing traversal into D1/D2/D3 graphs.

| Dataset | Phase 4 unique answerable top-20 chunk IDs | Matching active graph edges | Distinct matching graph source chunks |
|---|---:|---:|---:|
| D0 | 131 | 0 | 0 |
| D2 | 239 | 1 | 1 |
| D3 | 349 | 2 | 2 |

## Extraction configuration and limits

Prior D2/D3 ingestion summaries report 11,314 and 48,210 extracted entity spans respectively, with 10 and 13 extracted relations. The configured Phase 5 setup uses the existing generic `dslim/bert-base-NER` checkpoint with heuristic entity-type mapping and rule-based relation extraction; the learned relation classifier is disabled. This is the project-configured extraction pipeline, but its generic NER/domain mapping and sparse relation output limit graph completeness. Graph existence/counts do not prove that every relevant source passage was represented.

## Integrity

No node or relationship writes were made by this verification script. Existing graph data was not cleared or replaced. The exact read results and the prior ingestion provenance are in `phase5_graph_verification.json`.
