"""Render publication-style plots and a data-derived insights report."""
from __future__ import annotations
import csv, json, math, statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
RUNS=ROOT/'outputs/cross_dataset_evaluation/runs'
PH6=RUNS/'exp_cross_eval_hybrid_full_v1'
PH2=RUNS/'exp_cross_eval_vector_diagnostic_v1'
PH3=RUNS/'exp_cross_eval_generation_diagnostic_v1'
PH4=RUNS/'exp_cross_eval_rerank_diagnostic_v1'
DATASETS=['D0','D2','D3']
LABELS={'D0':'D0\nMeridian baseline','D2':'D2\nNASA aerospace','D3':'D3\nGovernment policy'}
COLORS={'D0':'#4477AA','D2':'#EE7733','D3':'#228833'}

def read_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def avg(rows,key):
    vals=[]
    for row in rows:
        try:
            v=float(row[key])
            if math.isfinite(v): vals.append(v)
        except (KeyError,ValueError,TypeError): pass
    return (statistics.mean(vals) if vals else float('nan'),len(vals))
def fmt(v): return '—' if v is None or not math.isfinite(v) else f'{v:.4f}'
def pct(v): return '—' if v is None or not math.isfinite(v) else f'{v*100:.1f}%'
def mdtable(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])

p2=read_csv(PH2/'per_query_vector_retrieval.csv')
p3=read_csv(PH3/'per_query_generation_diagnostic.csv')
p4=read_csv(PH4/'per_query_rerank_retrieval.csv')
p6=read_csv(PH6/'per_query_adaptive_hybrid_eval.csv')
manifest=json.loads((PH6/'phase6_run_manifest.json').read_text(encoding='utf-8'))
assert manifest.get('status')=='completed'
assert len(p6)==75, f'Expected 75 phase 6 records, found {len(p6)}'

def unique_answerable(rows):
    out={}
    for x in rows:
        if x.get('query_type','answerable')=='answerable': out.setdefault(x['query_id'],x)
    return out
v=unique_answerable(p2); r=unique_answerable(p4)
h={x['query_id']:x for x in p6 if x['query_type']=='answerable'}

metrics=[('Hit Rate @ 5','hit_rate_at_5','rerank_hit_rate_at_5','hit_rate_at_5'),('MRR @ 5','mrr_at_5','rerank_mrr_at_5','mrr_at_5'),('NDCG @ 5','ndcg_at_5','rerank_ndcg_at_5','ndcg_at_5')]
retrieval={}
for d in DATASETS:
    retrieval[d]={}
    for label,a,b,c in metrics:
        retrieval[d][label]=(avg([x for x in v.values() if x['dataset']==d],a)[0],avg([x for x in r.values() if x['dataset']==d],b)[0],avg([x for x in h.values() if x['dataset']==d],c)[0])

plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'figure.dpi':120,'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.8,'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42})
def save(fig,stem):
    fig.savefig(OUT/f'{stem}.png',dpi=300,bbox_inches='tight',facecolor='white')
    svg=OUT/f'{stem}.svg'
    fig.savefig(svg,bbox_inches='tight',facecolor='white')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)

# 1. Retrieval trajectory: three metrics, two phases, three datasets.
fig,axes=plt.subplots(1,3,figsize=(7.4,3.2),sharey=True)
x=np.arange(3); width=.34
for ax,(label,_,_,_) in zip(axes,metrics):
    vals2=[retrieval[d][label][0] for d in DATASETS]; vals6=[retrieval[d][label][2] for d in DATASETS]
    b1=ax.bar(x-width/2,vals2,width,label='Phase 2 · Vector',color='#999999',edgecolor='white')
    b2=ax.bar(x+width/2,vals6,width,label='Phase 6 · Hybrid',color='#4477AA',edgecolor='white')
    ax.set_title(label); ax.set_xticks(x,DATASETS); ax.set_ylim(0,1.12); ax.grid(axis='y',alpha=.2,linewidth=.5); ax.set_axisbelow(True)
    ax.bar_label(b1,fmt='%.2f',padding=2,fontsize=7); ax.bar_label(b2,fmt='%.2f',padding=2,fontsize=7)
axes[0].set_ylabel('Score (higher is better)')
handles,leg=axes[0].get_legend_handles_labels(); fig.legend(handles,leg,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,1.04))
fig.suptitle('Cross-dataset retrieval progression',y=1.16,fontweight='bold',fontsize=11)
fig.text(.5,-.02,'Phase 6 includes adaptive fusion and reranking; its metrics match Phase 4 reranked-vector results.',ha='center',fontsize=7.5)
fig.tight_layout(rect=[0,.04,1,.93]); save(fig,'fig1_cross_dataset_retrieval_progression')

# 2. Generation quality and citation checks.
gm=[('Token F1','token_f1'),('Semantic similarity','semantic_similarity'),('Citation compliance','citation_compliance')]
fig,ax=plt.subplots(figsize=(7.1,3.8)); x=np.arange(3); width=.23
for j,(lab,key) in enumerate(gm):
    vals=[avg([q for q in h.values() if q['dataset']==d],key)[0] for d in DATASETS]
    bars=ax.bar(x+(j-1)*width,vals,width,label=lab,color=['#4477AA','#EE7733','#228833'][j],edgecolor='white')
    ax.bar_label(bars,fmt='%.2f',padding=2,fontsize=7)
ax.set_xticks(x,[LABELS[d] for d in DATASETS]); ax.set_ylim(0,1.12); ax.set_ylabel('Score / compliance rate'); ax.set_title('Generation quality and citation checks · Phase 6',fontweight='bold')
ax.grid(axis='y',alpha=.2,linewidth=.5); ax.set_axisbelow(True); ax.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.14))
fig.text(.5,-.24,'Citation compliance requires at least one emitted citation ID to validate against supplied evidence.',ha='center',fontsize=7.5)
fig.tight_layout(rect=[0,.08,1,1]); save(fig,'fig2_generation_and_grounding_comparison')

# 3. Corpus scale and retrieval hit rate (logarithmic x scale).
chunks=np.array([149,1164,6119]); ph2=np.array([retrieval[d]['Hit Rate @ 5'][0] for d in DATASETS]); ph6=np.array([retrieval[d]['Hit Rate @ 5'][2] for d in DATASETS])
fig,ax=plt.subplots(figsize=(6.8,4.0)); ax.set_xscale('log')
for i,d in enumerate(DATASETS):
    ax.plot(chunks[i],ph2[i],marker='o',markersize=7,color='#888888',markeredgecolor='white',label='Phase 2 · Vector' if i==0 else None)
    ax.plot(chunks[i],ph6[i],marker='s',markersize=7,color=COLORS[d],markeredgecolor='white',label='Phase 6 · Hybrid' if i==0 else None)
    ax.annotate(d,(chunks[i],ph2[i]),xytext=(0,-15),textcoords='offset points',ha='center',fontsize=8,color='#555555')
    ax.annotate(f'{ph2[i]:.0%} → {ph6[i]:.0%}',(chunks[i],ph6[i]),xytext=(0,10),textcoords='offset points',ha='center',fontsize=8,color=COLORS[d],fontweight='bold')
ax.set_xticks(chunks,[f'{n:,}' for n in chunks]); ax.xaxis.set_minor_formatter(NullFormatter()); ax.set_xlim(120,8000); ax.set_xlabel('Indexed corpus chunks (log scale)'); ax.set_ylabel('Hit Rate @ 5'); ax.set_ylim(.35,1.1); ax.grid(alpha=.22,linewidth=.5); ax.set_axisbelow(True); ax.legend(frameon=False,loc='lower left')
ax.set_title('Retrieval quality across corpus scale',fontweight='bold')
ax.text(.99,.03,'D3 corpus is 41.1× D0',transform=ax.transAxes,ha='right',va='bottom',fontsize=8,bbox={'boxstyle':'round,pad=.3','fc':'white','ec':'#cccccc','lw':.6})
fig.text(.5,-.02,'Phase 6 combines graph fusion and cross-encoder reranking; Phase 4 isolates reranking and matches Phase 6 hit rates.',ha='center',fontsize=7.5)
fig.tight_layout(rect=[0,.04,1,1]); save(fig,'fig3_search_space_scale_vs_retrieval_decay')

# 4. Phase 6 retrieved context and Phase 3 gold-context citation compliance.
retrieved=[avg([q for q in h.values() if q['dataset']==d],'citation_compliance')[0] for d in DATASETS]
oracle=[avg([q for q in p3 if q['dataset']==d and q['query_type']=='answerable' and not q.get('generation_error')],'pass_b_citation_compliance')[0] for d in DATASETS]
oracle_n=[avg([q for q in p3 if q['dataset']==d and q['query_type']=='answerable' and not q.get('generation_error')],'pass_b_citation_compliance')[1] for d in DATASETS]
fig,ax=plt.subplots(figsize=(6.8,3.8)); x=np.arange(3); width=.34
b1=ax.bar(x-width/2,retrieved,width,label='Retrieved reranked context · Phase 6',color='#4477AA',edgecolor='white')
b2=ax.bar(x+width/2,oracle,width,label='Gold/oracle context · Phase 3 Pass B',color='#228833',edgecolor='white')
ax.bar_label(b1,labels=[f'{z:.0%}' for z in retrieved],padding=3,fontsize=8); ax.bar_label(b2,labels=[f'{z:.0%}' for z in oracle],padding=3,fontsize=8)
ax.set_xticks(x,DATASETS); ax.set_ylim(0,1.16); ax.set_ylabel('Citation compliance'); ax.set_title('Citation checks under retrieved vs. gold context',fontweight='bold'); ax.grid(axis='y',alpha=.2,linewidth=.5); ax.set_axisbelow(True); ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.12))
fig.text(.5,-.22,'Phase 3 Pass B: D0 n=19 (one Pass A generation failed), D2/D3 n=20. Cross-phase comparison is descriptive, not causal.',ha='center',fontsize=7.5)
fig.tight_layout(rect=[0,.08,1,1]); save(fig,'fig4_citation_compliance_retrieved_vs_oracle')

# 5. Phrase-detector triggers; deliberately avoid calling these validated accuracy.
ref=[q for q in p6 if q['query_type']=='refusal_candidate']; triggers=[]; totals=[]
for d in DATASETS:
    xs=[q for q in ref if q['dataset']==d]; totals.append(len(xs)); triggers.append(sum(q['candidate_refusal_detected'].lower()=='true' for q in xs))
fig,ax=plt.subplots(figsize=(6.5,3.8)); bars=ax.bar(np.arange(3),np.array(triggers)/np.maximum(totals,1),color=[COLORS[d] for d in DATASETS],edgecolor='white',width=.55)
ax.bar_label(bars,labels=[f'{a}/{b} · {a/b:.0%}' if b else 'n=0' for a,b in zip(triggers,totals)],padding=4,fontsize=9)
ax.set_xticks(np.arange(3),DATASETS); ax.set_ylim(0,1.22); ax.set_ylabel('Phrase-detector trigger fraction'); ax.set_title('Refusal phrase triggers on candidate queries',fontweight='bold'); ax.grid(axis='y',alpha=.2,linewidth=.5); ax.set_axisbelow(True)
ax.text(.5,.9,'15/15 overall',transform=ax.transAxes,ha='center',va='center',fontweight='bold',fontsize=10,color='#333333')
fig.text(.5,-.03,'Heuristic only: not verified out-of-domain refusal accuracy; corpus-wide answer absence was not established.',ha='center',fontsize=7.5)
fig.tight_layout(rect=[0,.06,1,1]); save(fig,'fig5_out_of_domain_refusal_accuracy')

# Manuscript-ready tables and captions.
retr_rows=[]
for d in DATASETS:
    cols=[('Hit Rate @ 5',0),('Hit Rate @ 5',1),('Hit Rate @ 5',2),('MRR @ 5',0),('MRR @ 5',2),('NDCG @ 5',0),('NDCG @ 5',2)]
    retr_rows.append([d]+[f'{retrieval[d][m][phase]:.4f}' for m,phase in cols])
gen_rows=[]; cite_rows=[]
for d in DATASETS:
    xs=[q for q in h.values() if q['dataset']==d]
    gen_rows.append([d,len(xs),fmt(avg(xs,'token_f1')[0]),fmt(avg(xs,'semantic_similarity')[0]),pct(avg(xs,'citation_compliance')[0])])
    ys=[q for q in p3 if q['dataset']==d and q['query_type']=='answerable' and not q.get('generation_error')]
    cite_rows.append([d,pct(avg(xs,'citation_compliance')[0]),pct(avg(ys,'pass_b_citation_compliance')[0]),avg(ys,'pass_b_citation_compliance')[1]])
graph_rows=[['D0','483','106'],['D2','5,671','10'],['D3','11,145','11']]
captions=[
'**Figure 1.** Hit Rate@5, MRR@5, and NDCG@5 across datasets for Phase 2 dense vector retrieval and Phase 6 adaptive hybrid retrieval. Phase 6 matched Phase 4 reranked-vector scores; its unchanged delta should not be attributed to graph fusion.',
'**Figure 2.** Phase 6 token-level reference F1, semantic similarity, and citation compliance. Citation compliance is an operational check against supplied evidence IDs, not a human factual-grounding assessment.',
'**Figure 3.** Phase 2 and Phase 6 Hit Rate@5 versus corpus size on a logarithmic x-axis. D3 contains approximately 41.1 times as many chunks as D0. Dataset domain and corpus construction vary, so the relationship is descriptive rather than causal. Phase 4 reranking-only hit rates equal Phase 6.',
'**Figure 4.** Citation compliance for Phase 6 retrieved/reranked context and Phase 3 Pass B gold context. The cross-phase pattern is consistent with evidence selection affecting citation behavior but does not show context noise is the sole cause.',
'**Figure 5.** Fraction of refusal-test candidates triggering the configured phrase detector. The 15/15 trigger count is a heuristic, not verified out-of-domain refusal accuracy.'
]
report=f'''# Ablation Study 2: Cross-Dataset Generalization under Fixed RAG Architecture

## Scope and data provenance

This analysis uses completed Phase 6 per-query records, Phase 2 vector-only and Phase 4 reranked-vector retrieval baselines. Retrieved-versus-gold-context citation values use Phase 3 Pass B because Phase 6 has no gold-context generation arm. No underlying benchmark or evaluation CSV was modified.

**D0 limitation:** D0 is a **Validation-Derived Diagnostic Baseline**. Its 20 pilot questions originate from `lora_val.jsonl` and answers appear in validation completions. It is not an independent test set; do not present its measurements as unseen-test generalization.

## Metric summary

### Retrieval trajectory

{mdtable(['Dataset','P2 HR@5','P4 reranked HR@5','P6 hybrid HR@5','P2 MRR@5','P6 MRR@5','P2 NDCG@5','P6 NDCG@5'],retr_rows)}

### Phase 6 generation and citation checks

{mdtable(['Dataset','n','Token F1','Semantic similarity','Citation compliance'],gen_rows)}

### Retrieved versus gold-context citation compliance

{mdtable(['Dataset','Phase 6 retrieved context','Phase 3 Pass B gold context','Phase 3 n'],cite_rows)}

### Verified graph population and refusal phrase triggers

{mdtable(['Dataset','Graph nodes','Relationships'],graph_rows)}

The phrase detector triggered on {sum(triggers)}/{sum(totals)} candidates ({sum(triggers)/sum(totals):.1%}) overall: {', '.join(f'{d} {triggers[i]}/{totals[i]}' for i,d in enumerate(DATASETS))}. Treat this as a string-pattern trigger rate only; it is not verified OOD refusal accuracy.

## Results and discussion

### 1. Retrieval generalization and search-space impact

Dense vector Hit Rate@5 decreases across these dataset/corpus combinations from 90.0% on D0 (149 indexed chunks) to 65.0% on D2 (1,164 chunks) and 50.0% on D3 (6,119 chunks). D3 has 41.1× the indexed chunks of D0. Phase 4 reranking raises Hit Rate@5 to 90.0% on D2 and 70.0% on D3 (D0: 100.0%); Phase 6 produces the same retrieval metrics as Phase 4 for all datasets. This is consistent with improved ordering of the dense candidate pool helping transfer-domain retrieval. It does not prove corpus size alone caused the decline: domain, document content, and corpus construction also differ. The full hybrid should not be credited with gains beyond Phase 4 reranking, since no additional retrieval-metric gain is observed here.

### 2. Domain-specific generation and citation dynamics

Phase 6 answerable-query token F1 is 0.7235 on D0, 0.4282 on D2, and 0.4818 on D3. In the Phase 3 gold-context arm, citation compliance is 100% for D2 (n=20) and 90% for D3 (n=20); D0 is 100% on the paired n=19 subset. Phase 6 retrieved-context citation compliance is 95%, 60%, and 60%, respectively. Higher D2/D3 compliance under gold context is compatible with evidence selection/context relevance contributing to citation failures. Since this is a cross-phase comparison, it does not establish that context noise rather than generator behavior alone causes the differences.

Even with gold passages, D2 token F1 remains approximately 0.46 (Phase 3 Pass B mean 0.4555). Evidence availability alone therefore does not eliminate the reference-overlap gap. Specialized aerospace vocabulary, answer phrasing, and lexical mismatch are plausible contributors, but aggregate scores cannot identify their causal shares. D2 Phase 3 Pass B semantic similarity was 0.6290, so token F1 alone is not a complete semantic-quality measure.

### 3. Knowledge-graph sparsity and adaptive fallback

The active Neo4j populations contain 10 D2 relationships and 11 D3 relationships despite 5,671 and 11,145 nodes. In Phase 6 answerable runs, no D2 or D3 query returned graph facts; D0 returned graph facts for two of 20 answerable questions. Mean learned alpha was 0.7894, 0.8009, and 0.8118 for D0, D2, and D3, respectively, over 25 answerable/refusal queries per dataset. These observations are consistent with vector-dominant fallback under sparse graph evidence, with retrieval metrics not falling below the Phase 4 reranker baseline. They do not prove robust fallback under other query distributions or denser graph conditions. Graph retrieval did not improve measured Hit Rate, MRR, or NDCG beyond Phase 4 here.

## Reporting limitations

- Dataset identity and corpus size are confounded; the trend is descriptive, not a causal corpus-size experiment.
- Phase 6 equals Phase 4 reranked-vector retrieval metrics; gains over Phase 2 cannot be credited as an incremental hybrid contribution.
- Phase 3 gold-context values are a cross-phase descriptive comparison, not a single-run randomized context ablation.
- Refusal values are phrase-detector triggers; corpus-wide answer absence was not independently established.
- D0 is validation-derived. Consult the Phase 6 report and manifest for model-adapter provenance uncertainty and the previously recorded Chroma SQLite metadata integrity caveat.

## Figure captions for paper

{chr(10).join(captions)}

Phase 6 input status: `{manifest['status']}`; {len(p6)} rows.
'''
(OUT/'ablation_study_2_insights.md').write_text(report,encoding='utf-8')
print(json.dumps({'output_dir':str(OUT),'phase6_rows':len(p6),'figure_pngs':sorted(p.name for p in OUT.glob('fig*.png')),'figure_svgs':sorted(p.name for p in OUT.glob('fig*.svg')),'report':'ablation_study_2_insights.md'},indent=2))
