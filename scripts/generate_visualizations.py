"""Generate reporting-only plots from the completed RAG evaluation artifacts.

The script never runs or changes retrieval, generation, training, or evaluation.
It only reads the existing JSON files in ``outputs/``.

Usage: python scripts/generate_visualizations.py
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

# Keep Matplotlib's cache within the project, avoiding user-profile permissions
# in restricted or CI environments.
_MPL_CACHE = Path("outputs") / "visualizations" / ".matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE.resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = ["faithfulness", "answer_relevance", "citation_precision", "citation_compliance", "token_f1", "exact_match", "semantic_similarity"]
PALETTE = ["#2563EB", "#F97316", "#16A34A", "#9333EA", "#DC2626", "#0891B2"]


def configure() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 220, "savefig.bbox": "tight", "font.size": 10, "axes.titlesize": 13, "axes.titleweight": "bold"})


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [item for item in payload.get("per_query_results", []) if isinstance(item, dict)]
    return []


def save(fig: plt.Figure, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.svg")
    plt.close(fig)


def bar_labels(ax: plt.Axes, bars: Iterable[Any], spec: str = "{:.3f}") -> None:
    for bar in bars:
        y = bar.get_height()
        ax.annotate(spec.format(y), (bar.get_x() + bar.get_width() / 2, y), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)


def production(summary: dict[str, Any], out: Path) -> None:
    ingest, negative = summary.get("ingestion", {}), summary.get("negative_evaluation", {})
    labels = ["Documents", "Chunks", "Entities", "Relations", "Vector entries"]
    values = [ingest.get(key, 0) for key in ["documents_processed", "chunks_created", "entities_extracted", "relations_extracted", "vector_store_count"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bars = axes[0].bar(labels, values, color=PALETTE[:5]); bar_labels(axes[0], bars, "{:.0f}")
    axes[0].set_title("Production corpus and knowledge assets"); axes[0].set_ylabel("Count"); axes[0].tick_params(axis="x", rotation=25)
    safety = [100 * float(negative.get("refusal_rate", 0)), 100 if negative.get("unsupported_citations", 0) == 0 else 0]
    axes[1].bar(["Refusal rate", "No unsupported citations"], safety, color=[PALETTE[2], PALETTE[0]])
    axes[1].set_ylim(0, 110); axes[1].set_title("Safety evaluation"); axes[1].set_ylabel("Percent")
    for i, value in enumerate(safety): axes[1].text(i, value + 3, f"{value:.0f}%", ha="center")
    fig.suptitle("Enterprise RAG production evaluation overview", fontsize=15, fontweight="bold"); fig.tight_layout(); save(fig, out, "01_production_overview")


def retrieval_ablation(summary: dict[str, Any], out: Path) -> None:
    data = pd.DataFrame(summary.get("ablation_retrieval", {})).T
    metrics = [m for m in ["hit_rate", "mrr", "ndcg"] if m in data]
    if data.empty or not metrics: return
    x, width = np.arange(len(data)), 0.24
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        bars = ax.bar(x + (i - 1) * width, data[metric], width, label=metric.upper(), color=PALETTE[i]); bar_labels(ax, bars)
    ax.set_title("Retrieval ablation: architecture comparison"); ax.set_ylabel("Score"); ax.set_ylim(0, 1.12)
    ax.set_xticks(x, [name.replace("_", " ").title() for name in data.index], rotation=18, ha="right"); ax.legend(ncols=3); fig.tight_layout(); save(fig, out, "02_retrieval_ablation")


def generation_frame(input_dir: Path) -> pd.DataFrame:
    result: list[dict[str, Any]] = []
    paths = sorted(input_dir.glob("*generation*.json")) + [input_dir / "llm_reference_answer_evaluation.json"]
    for path in paths:
        if not path.exists(): continue
        for row in records(load(path)):
            item = dict(row); item["evaluation_file"] = path.stem; result.append(item)
    return pd.DataFrame(result)


def model_name(frame: pd.DataFrame) -> str | None:
    return "model_label" if "model_label" in frame else ("model" if "model" in frame else None)


def clean_label(value: Any) -> str:
    """Repair common UTF-8-as-Windows-1252 display artifacts in old outputs."""
    text = str(value)
    if "â" in text:
        try:
            return text.encode("cp1252").decode("utf-8")
        except UnicodeError:
            pass
    return text


def generation_plots(frame: pd.DataFrame, out: Path) -> None:
    model = model_name(frame)
    metrics = [m for m in METRICS if m in frame and frame[m].notna().any()]
    if not model or not metrics: return
    grouped = frame.groupby(model)[metrics].mean().sort_index()
    fig, ax = plt.subplots(figsize=(max(9, len(metrics) * 1.3), max(4, len(grouped) * .8 + 2)))
    values = np.ma.masked_invalid(grouped.values)
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad(color="#F3F4F6")
    im = ax.imshow(values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(metrics)), [m.replace("_", " ").title() for m in metrics], rotation=25, ha="right"); ax.set_yticks(range(len(grouped)), [clean_label(name) for name in grouped.index])
    for r in range(len(grouped)):
        for c in range(len(metrics)):
            value = grouped.iloc[r, c]
            if pd.notna(value): ax.text(c, r, f"{value:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Mean score"); ax.set_title("Generation quality metrics by model"); fig.tight_layout(); save(fig, out, "03_generation_metric_heatmap")

    core = [m for m in ["faithfulness", "answer_relevance", "citation_precision", "citation_compliance"] if m in frame and frame[m].notna().any()]
    if core:
        runs, x, width = frame.groupby("evaluation_file")[core].mean(), None, None
        x, width = np.arange(len(runs)), .8 / len(core); fig, ax = plt.subplots(figsize=(13, 6))
        for i, metric in enumerate(core):
            bars = ax.bar(x - .4 + width / 2 + i * width, runs[metric], width, label=metric.replace("_", " ").title(), color=PALETTE[i]); bar_labels(ax, bars)
        ax.set_ylim(0, 1.12); ax.set_ylabel("Mean score"); ax.set_title("Generation evaluation runs: quality and citation metrics")
        ax.set_xticks(x, [name.replace("llm_", "").replace("_", " ") for name in runs.index], rotation=24, ha="right"); ax.legend(ncols=2); fig.tight_layout(); save(fig, out, "04_generation_metrics_by_run")

    if "latency_ms" in frame:
        usable, names = frame.dropna(subset=["latency_ms", model]), list(frame[model].dropna().unique())
        alpha = next((c for c in ["alpha", "alpha_v1"] if c in usable), None)
        fig, axes = plt.subplots(1, 2 if alpha else 1, figsize=(14 if alpha else 8, 5)); axes = np.atleast_1d(axes)
        axes[0].boxplot([usable.loc[usable[model] == name, "latency_ms"] / 1000 for name in names], tick_labels=names, showmeans=True)
        axes[0].set_title("Generation latency distribution"); axes[0].set_ylabel("Latency (seconds)"); axes[0].tick_params(axis="x", rotation=20)
        if alpha:
            for i, name in enumerate(names):
                part = usable.loc[usable[model] == name]; axes[1].scatter(part[alpha], part["latency_ms"] / 1000, label=name, color=PALETTE[i % len(PALETTE)], alpha=.7)
            axes[1].set_title("Adaptive retrieval weight vs. latency"); axes[1].set_xlabel("Alpha"); axes[1].set_ylabel("Latency (seconds)"); axes[1].legend()
        fig.tight_layout(); save(fig, out, "05_latency_and_alpha")

    if "citation_compliance" in frame:
        rates = frame.groupby(model)["citation_compliance"].mean().sort_values(ascending=False).mul(100)
        fig, ax = plt.subplots(figsize=(10, 5)); bars = ax.bar(rates.index, rates, color=PALETTE[:len(rates)]); bar_labels(ax, bars, "{:.1f}%")
        ax.set_ylim(0, 110); ax.set_title("Citation-compliant answers by model"); ax.set_ylabel("Citation compliance (%)"); ax.tick_params(axis="x", rotation=18); fig.tight_layout(); save(fig, out, "06_citation_compliance")
    frame.groupby(model)[metrics + [c for c in ["latency_ms", "retrieved_evidence_count"] if c in frame]].mean().round(4).to_csv(out / "visualization_metric_summary.csv")


def evidence_plots(payload: dict[str, Any], out: Path) -> None:
    frame = pd.DataFrame(payload.get("per_query_results", []))
    if frame.empty: return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    overlap = frame["top5_overlap_ratio"]
    axes[0].hist(overlap, bins=np.linspace(0, 1, 11), color=PALETTE[0], edgecolor="white"); axes[0].axvline(overlap.mean(), color=PALETTE[4], ls="--", label=f"Mean {overlap.mean():.3f}")
    axes[0].set_title("V1/V2 Top-5 evidence-set overlap"); axes[0].set_xlabel("Overlap ratio"); axes[0].set_ylabel("Queries"); axes[0].legend()
    freq = frame["shared_top5_count"].value_counts().sort_index(); bars = axes[1].bar(freq.index.astype(str), freq, color=PALETTE[1]); bar_labels(axes[1], bars, "{:.0f}")
    axes[1].set_title("Shared evidence chunks per query"); axes[1].set_xlabel("Shared chunks in Top-5"); axes[1].set_ylabel("Queries"); fig.tight_layout(); save(fig, out, "07_vector_only_evidence_overlap")
    ranks = [{"rank": item["rank"], "score": item["rerank_score"]} for row in payload.get("per_query_results", []) for item in row.get("v2_evidence", []) if "rank" in item and "rerank_score" in item]
    rank_frame = pd.DataFrame(ranks)
    if not rank_frame.empty:
        stats = rank_frame.groupby("rank")["score"].agg(["mean", "std"]); fig, ax = plt.subplots(figsize=(10, 5)); ax.errorbar(stats.index, stats["mean"], yerr=stats["std"].fillna(0), fmt="o-", color=PALETTE[3], capsize=4)
        ax.set_title("Cross-encoder reranker score by evidence rank"); ax.set_xlabel("Final evidence rank"); ax.set_ylabel("Reranker score (mean ± standard deviation)"); ax.set_xticks(stats.index); fig.tight_layout(); save(fig, out, "08_reranker_score_profile")


def guardrail_plot(payload: dict[str, Any], out: Path) -> None:
    frame = pd.DataFrame(payload.get("per_query_results", []))
    if frame.empty or not {"alpha_v1", "alpha_v2_guarded"}.issubset(frame): return
    frame = frame.sort_values("query_index"); x = np.arange(len(frame)); fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(x, frame["alpha_v1"], "o-", label="V1 alpha", color=PALETTE[0]); ax.plot(x, frame["alpha_v2_guarded"], "s-", label="Guarded V2 alpha", color=PALETTE[1])
    if "alpha_floor" in frame and frame["alpha_floor"].notna().any(): ax.axhline(frame["alpha_floor"].dropna().iloc[0], color=PALETTE[4], ls="--", label="Alpha floor")
    ax.set_title("Adaptive-retrieval alpha before and after guardrail"); ax.set_xlabel("Query index"); ax.set_ylabel("Alpha"); ax.set_ylim(0, 1.05); ax.legend(ncols=3); fig.tight_layout(); save(fig, out, "09_alpha_guardrail_comparison")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visualizations from completed evaluation artifacts.")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs")); parser.add_argument("--output-dir", type=Path, default=Path("outputs/visualizations"))
    args = parser.parse_args(); configure(); args.output_dir.mkdir(parents=True, exist_ok=True)
    production_file = args.input_dir / "production_evaluation_summary.json"
    if production_file.exists():
        summary = load(production_file); production(summary, args.output_dir); retrieval_ablation(summary, args.output_dir)
    frame = generation_frame(args.input_dir)
    if not frame.empty: generation_plots(frame, args.output_dir)
    evidence_file = args.input_dir / "vector_only_v2_evidence.json"
    if evidence_file.exists(): evidence_plots(load(evidence_file), args.output_dir)
    guardrail_file = args.input_dir / "retrieval_v1_vs_guarded_v2_generation_queries.json"
    if guardrail_file.exists(): guardrail_plot(load(guardrail_file), args.output_dir)
    print(f"Visualization report written to: {args.output_dir}")


if __name__ == "__main__": main()
