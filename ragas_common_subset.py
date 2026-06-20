#!/usr/bin/env python3
"""
Common-subset RAGAS analysis.

The main benchmark excludes refusals from RAGAS, so each pipeline is scored on a
different (self-selected) set of questions — Naive abstains more, so it is judged
on an easier subset, which inflates its faithfulness relative to Advanced. This
script recomputes faithfulness AND answer_relevancy on the *common subset*: the
questions where BOTH pipelines produced a substantive answer. That holds the
question set identical for both, giving an apples-to-apples comparison.

It does NOT regenerate answers — it reuses the ones saved in the results JSON.
By default it re-retrieves each pipeline's context (same methodology as the
original RAGAS figure, so only the subset differs); `--context gold` instead
scores against the gold document (no Ollama / retrieval needed, deterministic).

Usage (on the VM, with Ollama up):
  RAG_DEVICE=cpu python ragas_common_subset.py --input results_docred.json --plot
  python ragas_common_subset.py --input results_docred.json --context gold --plot
"""

from __future__ import annotations

import argparse
import json
import os

import datasets_registry
from benchmark import REFUSAL_MARKER, _avg, _run_ragas, _sample_std

NAIVE_COLOR = "#4C72B0"
ADV_COLOR = "#DD8452"


def _scorable(answer: str) -> bool:
    return bool(answer) and REFUSAL_MARKER not in answer.lower() and not answer.startswith("[ERROR:")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="results_docred.json",
                   help="Benchmark results JSON to read answers from")
    p.add_argument("--context", choices=["retrieved", "gold"], default="retrieved",
                   help="RAGAS context source: re-retrieve each pipeline's chunks "
                        "(matches the original figure's methodology) or use the gold "
                        "document (no Ollama/retrieval needed). Default: retrieved")
    p.add_argument("--ragas-workers", type=int, default=4)
    p.add_argument("--plot", action="store_true", help="Save a comparison figure")
    p.add_argument("--figures-dir", default="figures")
    p.add_argument("--output", help="Write the common-subset summary to this JSON")
    args = p.parse_args()

    with open(args.input) as fh:
        data = json.load(fh)

    dataset = data.get("dataset", "docred")
    datasets_registry.set_active(dataset)
    seeds = data["seeds"]

    records_by_seed: dict[int, list[dict]] = {}
    for r in data["records"]:
        if r.get("answerable", True):
            records_by_seed.setdefault(r["seed"], []).append(r)

    retrievers = None
    if args.context == "retrieved":
        from advanced import pipeline as adv
        from naive import pipeline as naive
        print("Warming up pipelines for re-retrieval...")
        naive.retrieve("warm-up query")
        adv.retrieve("warm-up query")
        retrievers = {"naive": naive, "advanced": adv}
        print("Ready.\n")

    # Per-seed RAGAS on the common subset, then mean ± std across seeds.
    per_seed: dict[str, dict[str, list[float]]] = {
        "naive": {"faithfulness": [], "answer_relevancy": []},
        "advanced": {"faithfulness": [], "answer_relevancy": []},
    }
    total_common = 0

    for seed in seeds:
        records = records_by_seed.get(seed, [])
        common = [r for r in records
                  if _scorable(r["naive"].get("answer", ""))
                  and _scorable(r["advanced"].get("answer", ""))]
        total_common += len(common)
        print(f"--- seed {seed}: {len(common)} common-subset questions ---")
        if not common:
            continue

        ragas_rows: dict[str, list[dict]] = {"naive": [], "advanced": []}
        for r in common:
            for name in ("naive", "advanced"):
                if args.context == "gold":
                    contexts = [r["gold_context"]]
                else:
                    chunks = retrievers[name].retrieve(r["question"])
                    contexts = [c["text"] for c in chunks]
                ragas_rows[name].append({
                    "question": r["question"],
                    "answer": r[name]["answer"],
                    "contexts": contexts,
                })

        scores = _run_ragas(ragas_rows, ragas_workers=args.ragas_workers)
        for name in ("naive", "advanced"):
            if name in scores:
                per_seed[name]["faithfulness"].append(scores[name]["faithfulness"])
                per_seed[name]["answer_relevancy"].append(scores[name]["answer_relevancy"])

    summary = {
        "dataset": dataset,
        "context_source": args.context,
        "n_common_total": total_common,
        "seeds": seeds,
    }
    for name in ("naive", "advanced"):
        summary[name] = {
            "faithfulness": round(_avg(per_seed[name]["faithfulness"]), 4),
            "faithfulness_std": round(_sample_std(per_seed[name]["faithfulness"]), 4),
            "answer_relevancy": round(_avg(per_seed[name]["answer_relevancy"]), 4),
            "answer_relevancy_std": round(_sample_std(per_seed[name]["answer_relevancy"]), 4),
        }

    _print_summary(summary)

    if args.output:
        with open(args.output, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nSaved {args.output}")

    if args.plot:
        _plot(summary, args.figures_dir)


def _print_summary(s: dict) -> None:
    n, a = s["naive"], s["advanced"]
    w = 58
    print("\n" + "=" * w)
    print(f"  Common-subset RAGAS — {s['dataset']}  "
          f"(N={s['n_common_total']}, context={s['context_source']})")
    print("=" * w)
    print(f"{'Metric':<26}{'Naive RAG':>15}{'Advanced RAG':>15}")
    print("-" * w)
    for label, key in (("Faithfulness", "faithfulness"),
                       ("Answer Relevancy", "answer_relevancy")):
        nv = f"{n[key]:.3f} ± {n[key + '_std']:.3f}"
        av = f"{a[key]:.3f} ± {a[key + '_std']:.3f}"
        print(f"{label:<26}{nv:>15}{av:>15}")
    print("=" * w)


def _plot(s: dict, out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import numpy as np
    except ImportError:
        print("\nmatplotlib/numpy not installed — skipping figure.")
        return

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")

    n, a = s["naive"], s["advanced"]
    labels = ["Faithfulness", "Answer Relevancy"]
    nvals = [n["faithfulness"], n["answer_relevancy"]]
    avals = [a["faithfulness"], a["answer_relevancy"]]
    nerr = [n["faithfulness_std"], n["answer_relevancy_std"]]
    aerr = [a["faithfulness_std"], a["answer_relevancy_std"]]
    multi = len(s["seeds"]) > 1

    x = np.arange(len(labels))
    bw = 0.35
    fig, ax = plt.subplots(figsize=(6, 4.5))
    b1 = ax.bar(x - bw / 2, nvals, bw, label="Naive RAG", color=NAIVE_COLOR, zorder=3,
                yerr=nerr if multi else None, capsize=5 if multi else 0)
    b2 = ax.bar(x + bw / 2, avals, bw, label="Advanced RAG", color=ADV_COLOR, zorder=3,
                yerr=aerr if multi else None, capsize=5 if multi else 0)
    for bars, vals, errs in ((b1, nvals, nerr), (b2, avals, aerr)):
        for bar, v, e in zip(bars, vals, errs):
            ax.text(bar.get_x() + bar.get_width() / 2, v + (e if multi else 0) + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_ylabel("Scor RAGAS")
    ax.set_title(f"Metrici RAGAS — subset comun (N={s['n_common_total']}) — "
                 f"{s['dataset'].upper()}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    name = f"{s['dataset']}_ragas_common"
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {name}.pdf / {name}.png in {out_dir}/")


if __name__ == "__main__":
    main()
