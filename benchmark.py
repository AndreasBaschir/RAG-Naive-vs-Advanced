#!/usr/bin/env python3
"""
Benchmark: Naive RAG vs Advanced RAG on a sample from the SQuAD training set.

Retrieval metrics (always computed):
  context_recall  — fraction of questions where a gold answer span appears in
                    a top-K chunk from the correct article
  mrr             — Mean Reciprocal Rank of the first such chunk
  avg_retrieval_ms — mean query-processing latency in milliseconds. NOTE: for the
                    Advanced pipeline this includes the multi-query expansion LLM
                    call, not just vector/BM25 retrieval.

Answer quality metrics (opt-in, requires LLM calls):
  answer_f1       — SQuAD token-F1 against reference answer spans
  exact_match     — SQuAD exact match against reference answer spans

RAGAS metrics (opt-in, implies --with-generation, very slow):
  faithfulness      — how well the answer is grounded in the retrieved context
  answer_relevancy  — how relevant the answer is to the question

Results are reported as mean ± std across all seeds.

Usage:
  python benchmark.py
  python benchmark.py --samples 100 --num-seeds 4
  python benchmark.py --with-generation --samples 50
  python benchmark.py --with-ragas --samples 50 --output results.json
  python benchmark.py --with-ragas --samples 50 --plot --figures-dir figures/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import string
import sys
import time
from collections import Counter

import datasets_registry
from advanced import pipeline as adv
from naive import pipeline as naive

DEFAULT_SAMPLES = 200
DEFAULT_SEED = 42
DEFAULT_NUM_SEEDS = 4

# Substring that identifies a grounded "I can't answer from context" refusal,
# emitted by both pipelines' system prompts. Such answers must be excluded from
# RAGAS answer_relevancy, which would otherwise penalise a correct refusal.
REFUSAL_MARKER = "does not contain enough information"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help=f"Questions per seed (default: {DEFAULT_SAMPLES})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Base random seed (default: {DEFAULT_SEED})")
    parser.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS,
                        help=f"Number of seeds to run (default: {DEFAULT_NUM_SEEDS})")
    parser.add_argument("--with-generation", action="store_true",
                        help="Also evaluate answer quality (slow — one LLM call per question per pipeline)")
    parser.add_argument("--with-ragas", action="store_true",
                        help="Run RAGAS faithfulness and answer_relevancy (implies --with-generation, very slow)")
    parser.add_argument("--ragas-from", metavar="FILE",
                        help="Skip retrieval/generation — load existing results.json and run only RAGAS on stored answers")
    parser.add_argument("--plot", action="store_true",
                        help="Generate thesis-ready figures (PDF + PNG) after evaluation")
    parser.add_argument("--figures-dir", default="figures",
                        help="Directory to save figures in (default: figures/)")
    parser.add_argument("--ragas-workers", type=int, default=4,
                        help="Concurrent RAGAS judge calls; set equal to Ollama's "
                             "OLLAMA_NUM_PARALLEL (default: 4)")
    parser.add_argument("--dataset", choices=datasets_registry.available(),
                        default="squad",
                        help="Which dataset to benchmark (default: squad)")
    parser.add_argument("--neg-frac", type=float, default=0.0,
                        help="Fraction of questions made UNANSWERABLE (a real entity "
                             "paired with a relation it lacks) to measure hallucination "
                             "resistance. Needs --with-generation and a dataset that "
                             "supports negatives (docred). Default: 0.0")
    parser.add_argument("--output", metavar="FILE",
                        help="Save full results to a JSON file")
    args = parser.parse_args()

    if args.with_ragas:
        args.with_generation = True

    # Select the active dataset BEFORE any pipeline call so the lazy singletons
    # resolve the right ChromaDB collection / BM25 cache.
    datasets_registry.set_active(args.dataset)
    spec = datasets_registry.active()

    if args.ragas_from:
        _ragas_from_file(args)
        return

    # Derive reproducible seeds from the base seed
    seeds = [args.seed + i * 1000 for i in range(args.num_seeds)]

    print(f"Loading '{args.dataset}' evaluation rows...")
    rows = spec.load_eval_rows()

    print("Warming up pipelines (loads models + BM25 index)...")
    naive.retrieve("warm-up query")
    adv.retrieve("warm-up query")

    if naive._get_collection().count() == 0:
        print(f"\nCollection '{spec.collection_name}' is empty. "
              f"Run: python ingest.py --dataset {args.dataset}")
        sys.exit(1)
    print("Ready.\n")

    pipelines = [("naive", naive), ("advanced", adv)]
    per_seed_summaries: list[dict] = []
    all_records: list[dict] = []

    for seed_idx, seed in enumerate(seeds):
        print(f"--- Seed {seed_idx + 1}/{len(seeds)}  (seed={seed}, {args.samples} questions) ---")

        rng = random.Random(seed)
        # Oversample the candidate pool, then keep the first N rows that yield a
        # usable record (make_record may return None, e.g. a DocRED doc with no
        # usable triple), so each seed still produces args.samples questions.
        # A neg_frac fraction is turned into UNANSWERABLE questions to measure
        # hallucination resistance (only when generation runs and the dataset
        # provides a negative maker).
        make_neg = spec.make_negative_record if args.with_generation else None
        pool = rng.sample(rows, min(args.samples * 2, len(rows)))
        sample: list[dict] = []
        for row in pool:
            use_neg = make_neg is not None and args.neg_frac > 0 and rng.random() < args.neg_frac
            record = make_neg(row) if use_neg else spec.make_record(row)
            if record is not None:
                record["seed"] = seed
                sample.append(record)
            if len(sample) >= args.samples:
                break

        records: list[dict] = []
        ragas_rows: dict[str, list[dict]] = {"naive": [], "advanced": []}

        for i, record in enumerate(sample):
            question: str = record["question"]
            gold_answers: list[str] = record["gold_answers"]
            gold_title: str = record["gold_title"]

            for name, pipeline in pipelines:
                t0 = time.perf_counter()
                chunks = pipeline.retrieve(question)
                retrieval_ms = (time.perf_counter() - t0) * 1000

                rank = _context_rank(chunks, gold_answers, gold_title)

                entry: dict = {
                    "retrieval_ms": round(retrieval_ms, 2),
                    "recall": rank is not None,
                    "mrr": round(1.0 / rank, 4) if rank is not None else 0.0,
                    "rank": rank,
                }

                if args.with_generation:
                    try:
                        answer = "".join(pipeline.stream(question, chunks))
                    except Exception as exc:
                        answer = f"[ERROR: {exc}]"
                    entry["answer"] = answer
                    entry["abstained"] = REFUSAL_MARKER in answer.lower()

                    if record.get("answerable", True):
                        entry["f1"] = round(
                            max((_token_f1(answer, ref) for ref in gold_answers), default=0.0), 4
                        )
                        entry["em"] = any(_exact_match(answer, ref) for ref in gold_answers)

                        # Only answerable questions are scored by RAGAS.
                        if args.with_ragas:
                            ragas_rows[name].append({
                                "question": question,
                                "answer": answer,
                                "contexts": [c["text"] for c in chunks],
                            })

                record[name] = entry

            records.append(record)
            _print_progress(i + 1, len(sample))

        print()
        all_records.extend(records)

        ragas_scores: dict = {}
        if args.with_ragas:
            print(f"  Running RAGAS for seed {seed_idx + 1} (this may take several minutes)...")
            ragas_scores = _run_ragas(ragas_rows, ragas_workers=args.ragas_workers)

        per_seed_summaries.append(_build_summary(records, args.with_generation, ragas_scores))

        # Write a checkpoint after every seed so a crash doesn't lose all work.
        if args.output:
            _write_checkpoint(args.output, per_seed_summaries, all_records, seeds)

    # Aggregate across seeds. Display flags are derived from the aggregated
    # summary so a metric only shows up when it survived for every seed.
    final_summary = _aggregate_summaries(per_seed_summaries)
    multi_seed = len(seeds) > 1
    has_generation = "answer_f1" in final_summary["naive"]
    has_ragas = "faithfulness" in final_summary["naive"]

    print()
    _print_results(final_summary, has_generation, has_ragas, multi_seed, len(seeds))

    sig_tests = _significance_tests(all_records, has_generation)
    _print_significance(sig_tests)

    if args.output:
        with open(args.output, "w") as fh:
            json.dump({
                "dataset": args.dataset,
                "summary": final_summary,
                "per_seed_summaries": per_seed_summaries,
                "significance": sig_tests,
                "seeds": seeds,
                "records": all_records,
            }, fh, indent=2)
        print(f"\nDetailed results saved to {args.output}")

    if args.plot:
        _plot_results(final_summary, has_generation, has_ragas,
                      args.figures_dir, multi_seed, len(seeds), args.dataset)


# --------------------------------------------------------------------------- #
# Aggregation                                                                  #
# --------------------------------------------------------------------------- #

def _ragas_from_file(args) -> None:
    """Load a completed results.json and run RAGAS on the stored answers."""
    with open(args.ragas_from) as fh:
        data = json.load(fh)

    all_records: list[dict] = data["records"]
    per_seed_summaries: list[dict] = data["per_seed_summaries"]
    seeds: list[int] = data["seeds"]

    # Group records back by seed, build ragas_rows per seed
    records_by_seed: dict[int, list[dict]] = {}
    for r in all_records:
        records_by_seed.setdefault(r["seed"], []).append(r)

    updated_summaries = []
    for seed_idx, (seed, summary) in enumerate(zip(seeds, per_seed_summaries)):
        records = records_by_seed.get(seed, [])
        # Retrieved chunk texts aren't stored in the JSON, so RAGAS faithfulness
        # is judged against the gold context. Unanswerable (negative) records are
        # excluded — they have no gold answer to be faithful to.
        ragas_rows: dict[str, list[dict]] = {"naive": [], "advanced": []}
        for r in records:
            if not r.get("answerable", True):
                continue
            for name in ("naive", "advanced"):
                entry = r.get(name, {})
                answer = entry.get("answer", "")
                if answer:
                    ragas_rows[name].append({
                        "question": r["question"],
                        "answer": answer,
                        "contexts": [r["gold_context"]],
                    })

        print(f"Running RAGAS for seed {seed_idx + 1}/{len(seeds)} (seed={seed})...")
        ragas_scores = _run_ragas(ragas_rows, ragas_workers=args.ragas_workers)

        updated = dict(summary)
        for name in ("naive", "advanced"):
            if name in ragas_scores:
                updated[name] = dict(summary[name])
                updated[name]["faithfulness"] = ragas_scores[name]["faithfulness"]
                updated[name]["answer_relevancy"] = ragas_scores[name]["answer_relevancy"]
        updated_summaries.append(updated)

        # Checkpoint after each seed's RAGAS so a crash mid-run isn't fatal.
        if args.output:
            data["per_seed_summaries"] = updated_summaries + per_seed_summaries[len(updated_summaries):]
            tmp = args.output + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(data, fh, indent=2)
            os.replace(tmp, args.output)
            print(f"  Checkpoint saved ({len(updated_summaries)}/{len(seeds)} seeds).")

    final_summary = _aggregate_summaries(updated_summaries)
    has_generation = "answer_f1" in final_summary["naive"]
    has_ragas = "faithfulness" in final_summary["naive"]
    multi_seed = len(seeds) > 1

    print()
    _print_results(final_summary, has_generation, has_ragas, multi_seed, len(seeds))

    sig_tests = _significance_tests(all_records, has_generation)
    _print_significance(sig_tests)

    if args.output:
        data["summary"] = final_summary
        data["per_seed_summaries"] = updated_summaries
        data["significance"] = sig_tests
        with open(args.output, "w") as fh:
            json.dump(data, fh, indent=2)
        print(f"\nUpdated results saved to {args.output}")

    if args.plot:
        _plot_results(final_summary, has_generation, has_ragas,
                      args.figures_dir, multi_seed, len(seeds),
                      data.get("dataset", args.dataset))


def _write_checkpoint(path: str, per_seed_summaries: list[dict],
                      all_records: list[dict], seeds: list[int]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({
            "seeds_completed": len(per_seed_summaries),
            "seeds": seeds,
            "per_seed_summaries": per_seed_summaries,
            "records": all_records,
        }, fh, indent=2)
    os.replace(tmp, path)  # atomic replace — no partial-write corruption


def _aggregate_summaries(summaries: list[dict]) -> dict:
    """Mean ± sample-std across per-seed summaries. Single seed → std omitted.

    Only metrics present in *every* seed are aggregated, so a RAGAS failure on
    one seed can neither crash the run nor silently corrupt the output table.
    """
    if len(summaries) == 1:
        return summaries[0]

    result: dict = {}
    for name in ("naive", "advanced"):
        pipeline_runs = [s[name] for s in summaries]

        # Keep only keys present in all seeds and numeric in all seeds.
        common_keys = set(pipeline_runs[0])
        for r in pipeline_runs[1:]:
            common_keys &= set(r)
        numeric_keys = [
            k for k in common_keys
            if k != "n" and all(isinstance(r[k], (int, float)) for r in pipeline_runs)
        ]

        agg: dict = {"n": pipeline_runs[0]["n"]}
        for key in numeric_keys:
            vals = [r[key] for r in pipeline_runs]
            agg[key] = round(_avg(vals), 4)
            agg[f"{key}_std"] = round(_sample_std(vals), 4)
        result[name] = agg
    return result


# --------------------------------------------------------------------------- #
# RAGAS evaluation                                                             #
# --------------------------------------------------------------------------- #

def _run_ragas(rows_by_pipeline: dict, llm_model: str = "qwen3:8b",
               ragas_workers: int = 4) -> dict:
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.metrics import Faithfulness, ResponseRelevancy
        from langchain_ollama import ChatOllama
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError as exc:
        print(f"\nRAGAS dependencies missing: {exc}")
        print("Install with: pip install ragas langchain-ollama langchain-community")
        return {}

    # num_ctx is kept small on purpose: RAGAS prompts (a question + a few SQuAD
    # passages) are well under 4K tokens, so qwen3's 40K default just wastes
    # ~15GB of KV cache and blocks parallel slots. A small context lets several
    # requests run concurrently within the A5000's 24GB.
    llm = LangchainLLMWrapper(ChatOllama(model=llm_model, num_ctx=4096))
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    results: dict = {}
    for name, rows in rows_by_pipeline.items():
        if not rows:
            continue

        # Drop grounded refusals: scoring them for answer_relevancy unfairly
        # penalises a pipeline for correctly declining to answer.
        usable = [r for r in rows
                  if REFUSAL_MARKER not in r["answer"].lower()
                  and not r["answer"].startswith("[ERROR:")]
        excluded = len(rows) - len(usable)
        if excluded:
            print(f"    ({name}: {excluded}/{len(rows)} refusal answers excluded from RAGAS)")
        if not usable:
            print(f"    ({name}: no scorable answers — skipping RAGAS)")
            continue

        print(f"    Evaluating {name} RAG ({len(usable)} samples)...")
        samples = [
            SingleTurnSample(
                user_input=r["question"],
                response=r["answer"],
                retrieved_contexts=r["contexts"],
            )
            for r in usable
        ]
        dataset = EvaluationDataset(samples=samples)
        from ragas import RunConfig
        # max_workers must match Ollama's OLLAMA_NUM_PARALLEL: more workers than
        # Ollama can serve concurrently just makes the surplus queue past the
        # per-job timeout. A generous timeout covers a full slow generation.
        result = evaluate(
            dataset=dataset,
            metrics=[Faithfulness(), ResponseRelevancy()],
            llm=llm,
            embeddings=embeddings,
            run_config=RunConfig(max_workers=ragas_workers, timeout=600),
        )
        df = result.to_pandas()
        faith_vals = [v for v in df["faithfulness"] if not math.isnan(v)]
        relev_vals = [v for v in df["answer_relevancy"] if not math.isnan(v)]
        results[name] = {
            "faithfulness": round(_avg(faith_vals), 4),
            "answer_relevancy": round(_avg(relev_vals), 4),
        }

    return results


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #

def _plot_results(summary: dict, with_generation: bool, with_ragas: bool,
                  out_dir: str, multi_seed: bool, num_seeds: int = DEFAULT_NUM_SEEDS,
                  dataset: str = "squad") -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import numpy as np
    except ImportError:
        print("\nmatplotlib/numpy not installed — skipping figures.")
        print("Install with: pip install matplotlib numpy")
        return

    os.makedirs(out_dir, exist_ok=True)

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")

    NAIVE_COLOR = "#4C72B0"
    ADV_COLOR   = "#DD8452"
    FONT_SIZE   = 11
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 1,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 1,
    })

    n = summary["naive"]
    a = summary["advanced"]
    ds_label = dataset.upper()  # e.g. "SQUAD", "DOCRED" — appended to titles

    def _get(d: dict, key: str) -> tuple[float, float]:
        """Return (mean, std) for a metric. std=0 when no multi-seed."""
        return d[key], d.get(f"{key}_std", 0.0)

    def _save(fig: "plt.Figure", name: str) -> None:
        # Prefix filenames with the dataset so SQuAD/DocRED runs into the same
        # figures dir don't overwrite each other.
        fname = f"{dataset}_{name}"
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out_dir, f"{fname}.{ext}"), dpi=300, bbox_inches="tight")
        print(f"  Saved {fname}.pdf / {fname}.png")
        plt.close(fig)

    def _grouped_bars(ax, labels, naive_vals, adv_vals,
                      naive_errs=None, adv_errs=None) -> None:
        x   = np.arange(len(labels))
        w   = 0.35
        kw  = {"capsize": 5, "error_kw": {"elinewidth": 1.2}} if multi_seed else {}
        b1 = ax.bar(x - w / 2, naive_vals, w, label="Naive RAG",    color=NAIVE_COLOR, zorder=3,
                    yerr=naive_errs if multi_seed else None, **kw)
        b2 = ax.bar(x + w / 2, adv_vals,   w, label="Advanced RAG", color=ADV_COLOR,   zorder=3,
                    yerr=adv_errs if multi_seed else None, **kw)
        for bar, val, err in zip(list(b1) + list(b2),
                                  naive_vals + adv_vals,
                                  (naive_errs or [0]*len(naive_vals)) + (adv_errs or [0]*len(adv_vals))):
            top = val + (err if multi_seed else 0) + 0.01
            ax.text(bar.get_x() + bar.get_width() / 2, top,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=FONT_SIZE - 2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

    print(f"\nGenerating figures in '{out_dir}/'...")

    # ------------------------------------------------------------------ #
    # Figure 1 — Retrieval metrics                                        #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(6, 4.5))
    n_rc, n_rc_std = _get(n, "context_recall")
    a_rc, a_rc_std = _get(a, "context_recall")
    n_mrr, n_mrr_std = _get(n, "mrr")
    a_mrr, a_mrr_std = _get(a, "mrr")
    _grouped_bars(ax,
                  ["Context Recall@K", "MRR"],
                  [n_rc, n_mrr], [a_rc, a_mrr],
                  [n_rc_std, n_mrr_std], [a_rc_std, a_mrr_std])
    ax.set_ylim(0, 1.2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_ylabel("Scor")
    ax.set_title(f"Metrici de Regăsire — {ds_label}")
    if multi_seed:
        ax.set_xlabel(f"medie ± std  ({num_seeds} seed-uri)")
    fig.tight_layout()
    _save(fig, "retrieval_metrics")

    # ------------------------------------------------------------------ #
    # Figure 2 — Latency                                                  #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(5, 4))
    n_lat, n_lat_std = _get(n, "avg_retrieval_ms")
    a_lat, a_lat_std = _get(a, "avg_retrieval_ms")
    pipeline_labels  = ["Naive RAG", "Advanced RAG"]
    latencies        = [n_lat, a_lat]
    errs             = [n_lat_std, a_lat_std] if multi_seed else None
    bars = ax.bar(pipeline_labels, latencies,
                  color=[NAIVE_COLOR, ADV_COLOR], width=0.4, zorder=3,
                  yerr=errs, capsize=5 if multi_seed else 0,
                  error_kw={"elinewidth": 1.2})
    max_top = max(latencies[i] + (errs[i] if errs else 0) for i in range(2))
    for bar, val in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + max_top * 0.02,
                f"{val:.1f} ms", ha="center", va="bottom", fontsize=FONT_SIZE - 1)
    ax.set_ylabel("Latență medie (ms)")
    ax.set_title(f"Latența Medie de Procesare a Interogării — {ds_label}")
    ax.set_ylim(0, max_top * 1.25)
    fig.tight_layout()
    _save(fig, "latency")

    # ------------------------------------------------------------------ #
    # Figure 3 — Answer quality                                           #
    # ------------------------------------------------------------------ #
    if with_generation:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        n_f1, n_f1_std = _get(n, "answer_f1")
        a_f1, a_f1_std = _get(a, "answer_f1")
        n_em, n_em_std = _get(n, "exact_match")
        a_em, a_em_std = _get(a, "exact_match")
        _grouped_bars(ax,
                      ["Answer F1", "Exact Match"],
                      [n_f1, n_em], [a_f1, a_em],
                      [n_f1_std, n_em_std], [a_f1_std, a_em_std])
        ax.set_ylim(0, 1.2)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.set_ylabel("Scor")
        ax.set_title(f"Calitatea Răspunsurilor — {ds_label}")
        if multi_seed:
            ax.set_xlabel(f"medie ± std  ({num_seeds} seed-uri)")
        fig.tight_layout()
        _save(fig, "answer_quality")

    # ------------------------------------------------------------------ #
    # Figure 4 — RAGAS metrics                                            #
    # ------------------------------------------------------------------ #
    if with_ragas:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        n_fa, n_fa_std = _get(n, "faithfulness")
        a_fa, a_fa_std = _get(a, "faithfulness")
        n_ar, n_ar_std = _get(n, "answer_relevancy")
        a_ar, a_ar_std = _get(a, "answer_relevancy")
        _grouped_bars(ax,
                      ["Faithfulness", "Answer Relevancy"],
                      [n_fa, n_ar], [a_fa, a_ar],
                      [n_fa_std, n_ar_std], [a_fa_std, a_ar_std])
        ax.set_ylim(0, 1.2)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.set_ylabel("Scor RAGAS")
        ax.set_title(f"Metrici RAGAS — {ds_label}")
        if multi_seed:
            ax.set_xlabel(f"medie ± std  ({num_seeds} seed-uri)")
        fig.tight_layout()
        _save(fig, "ragas_metrics")

    # ------------------------------------------------------------------ #
    # Figure 4b — Hallucination rate on unanswerable questions            #
    # ------------------------------------------------------------------ #
    if "hallucination_rate" in n:
        fig, ax = plt.subplots(figsize=(5, 4))
        n_h, n_h_std = _get(n, "hallucination_rate")
        a_h, a_h_std = _get(a, "hallucination_rate")
        errs = [n_h_std, a_h_std] if multi_seed else None
        bars = ax.bar(["Naive RAG", "Advanced RAG"], [n_h, a_h],
                      color=[NAIVE_COLOR, ADV_COLOR], width=0.4, zorder=3,
                      yerr=errs, capsize=5 if multi_seed else 0,
                      error_kw={"elinewidth": 1.2})
        for bar, val in zip(bars, [n_h, a_h]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=FONT_SIZE - 1)
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.set_ylabel("Rată de halucinație (mai mic = mai bine)")
        ax.set_title(f"Halucinație la Întrebări fără Răspuns — {ds_label}")
        fig.tight_layout()
        _save(fig, "hallucination")

    # ------------------------------------------------------------------ #
    # Figure 5 — Radar overview                                           #
    # ------------------------------------------------------------------ #
    metric_names: list[str] = ["Context\nRecall@K", "MRR"]
    naive_radar:  list[float] = [n["context_recall"], n["mrr"]]
    adv_radar:    list[float] = [a["context_recall"], a["mrr"]]

    if with_generation:
        metric_names += ["Answer F1", "Exact\nMatch"]
        naive_radar  += [n["answer_f1"], n["exact_match"]]
        adv_radar    += [a["answer_f1"], a["exact_match"]]
    if with_ragas:
        metric_names += ["Faithfulness", "Answer\nRelevancy"]
        naive_radar  += [n["faithfulness"], n["answer_relevancy"]]
        adv_radar    += [a["faithfulness"], a["answer_relevancy"]]

    # NOTE: latency is intentionally omitted from the radar. It lives on a
    # different (millisecond) scale and would only be representable as a
    # relative ratio, which mixes absolute quality scores with a relative one
    # on the same 0-1 axis and misleads the reader. See the latency figure.

    num_vars = len(metric_names)
    angles = [i * 2 * math.pi / num_vars for i in range(num_vars)]
    angles += angles[:1]
    naive_radar += naive_radar[:1]
    adv_radar   += adv_radar[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    ax.plot(angles, naive_radar, color=NAIVE_COLOR, linewidth=2, label="Naive RAG")
    ax.fill(angles, naive_radar, color=NAIVE_COLOR, alpha=0.15)
    ax.plot(angles, adv_radar,   color=ADV_COLOR,   linewidth=2, label="Advanced RAG")
    ax.fill(angles, adv_radar,   color=ADV_COLOR,   alpha=0.15)
    ax.set_thetagrids([a * 180 / math.pi for a in angles[:-1]], metric_names)
    ax.set_ylim(0, 1)
    ax.set_title(f"Prezentare Generală a Metricilor — {ds_label}", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    _save(fig, "overview_radar")

    print(f"\nAll figures saved to '{out_dir}/'.")
    print("Include in LaTeX with:  \\includegraphics[width=\\linewidth]{figures/<name>.pdf}")


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #

def _context_rank(chunks: list[dict], gold_answers: list[str],
                  gold_title: str | None = None) -> int | None:
    """
    1-based rank of the first chunk that contains a gold answer span, or None.

    When gold_title is given, a chunk only counts if it also comes from the
    correct SQuAD article. This prevents short answer spans (e.g. "May", "US")
    from spuriously matching unrelated passages and inflating recall.
    """
    golds = [a.lower() for a in gold_answers]
    title = gold_title.strip().lower() if gold_title else None
    for i, chunk in enumerate(chunks):
        if title is not None and chunk.get("source", "").strip().lower() != title:
            continue
        text = chunk["text"].lower()
        if any(ans in text for ans in golds):
            return i + 1
    return None


def _normalize(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _normalize(prediction).split()
    ref_tokens  = _normalize(reference).split()
    common      = Counter(pred_tokens) & Counter(ref_tokens)
    num_common  = sum(common.values())
    if num_common == 0 or not pred_tokens or not ref_tokens:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall    = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction: str, reference: str) -> bool:
    return _normalize(reference) in _normalize(prediction)


# --------------------------------------------------------------------------- #
# Output                                                                       #
# --------------------------------------------------------------------------- #

def _avg(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


def _sample_std(values: list) -> float:
    """Sample standard deviation (ddof=1). Returns 0.0 for <2 values."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _build_summary(records: list[dict], with_generation: bool, ragas_scores: dict) -> dict:
    # Retrieval/answer-quality metrics are computed over answerable questions
    # only; the hallucination metric is computed over the unanswerable ones.
    answerable = [r for r in records if r.get("answerable", True)]
    negatives = [r for r in records if not r.get("answerable", True)]

    summary = {}
    for name in ("naive", "advanced"):
        entries = [r[name] for r in records]
        ans = [r[name] for r in answerable]
        s: dict = {
            "n": len(entries),
            "context_recall": round(_avg([e["recall"] for e in ans]), 4),
            "mrr":            round(_avg([e["mrr"]    for e in ans]), 4),
            "avg_retrieval_ms": round(_avg([e["retrieval_ms"] for e in entries]), 2),
        }
        if with_generation:
            s["answer_f1"]   = round(_avg([e.get("f1",  0.0)          for e in ans]), 4)
            s["exact_match"] = round(_avg([float(e.get("em", False))   for e in ans]), 4)
        if negatives:
            neg = [r[name] for r in negatives]
            # hallucination = produced a substantive answer instead of abstaining
            s["hallucination_rate"] = round(
                _avg([0.0 if e.get("abstained") else 1.0 for e in neg]), 4
            )
        if name in ragas_scores:
            s["faithfulness"]     = ragas_scores[name]["faithfulness"]
            s["answer_relevancy"] = ragas_scores[name]["answer_relevancy"]
        summary[name] = s
    return summary


def _fmt(summary: dict, key: str, multi_seed: bool) -> str:
    mean = summary[key]
    std  = summary.get(f"{key}_std")
    if multi_seed and std is not None:
        return f"{mean:.3f} ± {std:.3f}"
    return f"{mean:.3f}"


def _print_results(summary: dict, with_generation: bool, with_ragas: bool,
                   multi_seed: bool, num_seeds: int = DEFAULT_NUM_SEEDS) -> None:
    n = summary["naive"]
    a = summary["advanced"]
    col_w = 18 if multi_seed else 14
    w = 30 + col_w * 2 + 2

    header = f"{'Naive RAG':>{col_w}} {'Advanced RAG':>{col_w}}"
    seeds_note = f"  ({num_seeds} seeds × {n['n']} questions each)" if multi_seed else f"  ({n['n']} questions)"

    print("=" * w)
    print(f"  Benchmark results{seeds_note}")
    print("=" * w)
    print(f"{'Metric':<30} {header}")
    print("-" * w)

    def row(label: str, key: str) -> None:
        nv = _fmt(n, key, multi_seed)
        av = _fmt(a, key, multi_seed)
        print(f"{label:<30} {nv:>{col_w}} {av:>{col_w}}")

    row("Context Recall@K",   "context_recall")
    row("MRR",                "mrr")

    lat_n = f"{n['avg_retrieval_ms']:.1f}" + (f" ± {n['avg_retrieval_ms_std']:.1f}" if multi_seed and 'avg_retrieval_ms_std' in n else "") + " ms"
    lat_a = f"{a['avg_retrieval_ms']:.1f}" + (f" ± {a['avg_retrieval_ms_std']:.1f}" if multi_seed and 'avg_retrieval_ms_std' in a else "") + " ms"
    print(f"{'Avg Query Processing (ms)':<30} {lat_n:>{col_w}} {lat_a:>{col_w}}")

    if with_generation:
        print("-" * w)
        row("Answer F1",   "answer_f1")
        row("Exact Match", "exact_match")
    if with_ragas:
        print("-" * w)
        row("RAGAS Faithfulness",     "faithfulness")
        row("RAGAS Answer Relevancy", "answer_relevancy")
    if "hallucination_rate" in n:
        print("-" * w)
        row("Hallucination Rate (neg)", "hallucination_rate")
    print("=" * w)


def _significance_tests(records: list[dict], with_generation: bool) -> dict:
    """
    Paired Advanced-vs-Naive significance tests over the pooled per-question
    results (all seeds combined):

      continuous metrics (mrr, answer_f1) — Wilcoxon signed-rank test
      binary metrics (recall, exact_match) — McNemar test (exact binomial)

    Returns {} if scipy is unavailable.
    """
    try:
        from scipy.stats import binomtest, wilcoxon
    except ImportError:
        print("\nscipy not installed — skipping significance tests.")
        print("Install with: pip install scipy")
        return {}

    # Retrieval/answer tests run over answerable questions only; the
    # hallucination test runs over the unanswerable ones.
    answerable = [r for r in records if r.get("answerable", True)]
    negatives = [r for r in records if not r.get("answerable", True)]

    tests: dict = {}

    continuous = [("MRR", "mrr")]
    if with_generation:
        continuous.append(("Answer F1", "f1"))
    for label, key in continuous:
        naive_vals = [r["naive"][key] for r in answerable]
        adv_vals = [r["advanced"][key] for r in answerable]
        if not answerable or all(av == nv for av, nv in zip(adv_vals, naive_vals)):
            p = float("nan")  # wilcoxon errors on all-zero / empty differences
        else:
            try:
                _, p = wilcoxon(adv_vals, naive_vals)
            except ValueError:
                p = float("nan")
        tests[label] = {"test": "Wilcoxon", "p": p, "n": len(answerable)}

    binary = [("Context Recall@K", "recall")]
    if with_generation:
        binary.append(("Exact Match", "em"))
    for label, key in binary:
        # McNemar on discordant pairs: how often each pipeline alone succeeds.
        adv_only = sum(1 for r in answerable if r["advanced"][key] and not r["naive"][key])
        naive_only = sum(1 for r in answerable if r["naive"][key] and not r["advanced"][key])
        discordant = adv_only + naive_only
        p = 1.0 if discordant == 0 else binomtest(min(adv_only, naive_only), discordant, 0.5).pvalue
        tests[label] = {
            "test": "McNemar", "p": p, "n": len(answerable),
            "adv_better": adv_only, "naive_better": naive_only,
        }

    # Hallucination resistance on unanswerable questions: "better" = abstained.
    if negatives:
        adv_only = sum(1 for r in negatives if r["advanced"].get("abstained") and not r["naive"].get("abstained"))
        naive_only = sum(1 for r in negatives if r["naive"].get("abstained") and not r["advanced"].get("abstained"))
        discordant = adv_only + naive_only
        p = 1.0 if discordant == 0 else binomtest(min(adv_only, naive_only), discordant, 0.5).pvalue
        tests["Abstention (neg)"] = {
            "test": "McNemar", "p": p, "n": len(negatives),
            "adv_better": adv_only, "naive_better": naive_only,
        }

    return tests


def _print_significance(tests: dict) -> None:
    if not tests:
        return
    print("\nPaired significance tests (Advanced vs Naive, pooled questions):")
    for metric, t in tests.items():
        p = t["p"]
        pstr = "n/a" if p != p else f"{p:.4g}"
        verdict = "n.s." if (p != p or p >= 0.05) else "significant (p<0.05)"
        extra = ""
        if t["test"] == "McNemar":
            extra = f"   [adv-only: {t['adv_better']}, naive-only: {t['naive_better']}]"
        print(f"  {metric:<22} {t['test']:<9} p={pstr:<9} {verdict}{extra}")


def _print_progress(current: int, total: int) -> None:
    pct    = current / total
    bar_len = 38
    filled  = int(bar_len * pct)
    bar     = "█" * filled + "░" * (bar_len - filled)
    import sys
    if sys.stdout.isatty():
        print(f"\r  [{bar}] {current}/{total}", end="", flush=True)
    elif current == total:
        print(f"  [{bar}] {current}/{total}", flush=True)


if __name__ == "__main__":
    main()
