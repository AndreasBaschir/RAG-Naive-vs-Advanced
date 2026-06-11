#!/usr/bin/env python3
"""
Benchmark: Naive RAG vs Advanced RAG on a sample from the SQuAD training set.

Retrieval metrics (always computed):
  context_recall  — fraction of questions where the gold passage appears in top-K
  mrr             — Mean Reciprocal Rank of the gold passage
  avg_retrieval_ms — mean retrieval latency in milliseconds

Answer quality metrics (opt-in, requires LLM calls):
  answer_f1       — SQuAD token-F1 against reference answer spans
  exact_match     — SQuAD exact match against reference answer spans

Usage:
  python benchmark.py
  python benchmark.py --samples 100
  python benchmark.py --with-generation --samples 50 --output results.json
"""

from __future__ import annotations

import argparse
import json
import random
import string
import time
from collections import Counter

from datasets import load_dataset

from advanced import pipeline as adv
from naive import pipeline as naive

DEFAULT_SAMPLES = 200
DEFAULT_SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help=f"Questions to evaluate (default: {DEFAULT_SAMPLES})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--with-generation", action="store_true",
                        help="Also evaluate answer quality (slow — one LLM call per question per pipeline)")
    parser.add_argument("--output", metavar="FILE",
                        help="Save full results to a JSON file")
    args = parser.parse_args()

    print("Loading SQuAD training split...")
    dataset = load_dataset("rajpurkar/squad", split="train")
    rows = list(dataset)

    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.samples, len(rows)))
    print(f"Sampled {len(sample)} questions (seed={args.seed}).")

    # Warm up both pipelines so model loading / BM25 index build
    # doesn't skew the first few timing measurements.
    print("Warming up pipelines (loads models + BM25 index)...")
    naive.retrieve("warm-up query")
    adv.retrieve("warm-up query")
    print("Ready.\n")

    pipelines = [("naive", naive), ("advanced", adv)]
    records: list[dict] = []

    for i, row in enumerate(sample):
        question: str = row["question"]
        gold_context: str = row["context"]
        gold_answers: list[str] = row["answers"]["text"]

        record: dict = {
            "question": question,
            "gold_context": gold_context,
            "gold_answers": gold_answers,
        }

        for name, pipeline in pipelines:
            t0 = time.perf_counter()
            chunks = pipeline.retrieve(question)
            retrieval_ms = (time.perf_counter() - t0) * 1000

            rank = _context_rank(chunks, gold_context)

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
                entry["f1"] = round(
                    max((_token_f1(answer, ref) for ref in gold_answers), default=0.0), 4
                )
                entry["em"] = any(_exact_match(answer, ref) for ref in gold_answers)

            record[name] = entry

        records.append(record)
        _print_progress(i + 1, len(sample))

    print()
    summary = _build_summary(records, args.with_generation)
    _print_results(summary, args.with_generation)

    if args.output:
        with open(args.output, "w") as fh:
            json.dump({"summary": summary, "records": records}, fh, indent=2)
        print(f"\nDetailed results saved to {args.output}")


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #

def _context_rank(chunks: list[dict], gold_context: str) -> int | None:
    """1-based rank of the gold passage in the retrieved list, or None."""
    gold = gold_context.strip()
    for i, chunk in enumerate(chunks):
        if chunk["text"].strip() == gold:
            return i + 1
    return None


def _normalize(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _normalize(prediction).split()
    ref_tokens = _normalize(reference).split()
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0 or not pred_tokens or not ref_tokens:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction: str, reference: str) -> bool:
    return _normalize(reference) in _normalize(prediction)


# --------------------------------------------------------------------------- #
# Output                                                                       #
# --------------------------------------------------------------------------- #

def _avg(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


def _build_summary(records: list[dict], with_generation: bool) -> dict:
    summary = {}
    for name in ("naive", "advanced"):
        entries = [r[name] for r in records]
        s: dict = {
            "n": len(entries),
            "context_recall": round(_avg([e["recall"] for e in entries]), 4),
            "mrr": round(_avg([e["mrr"] for e in entries]), 4),
            "avg_retrieval_ms": round(_avg([e["retrieval_ms"] for e in entries]), 2),
        }
        if with_generation:
            s["answer_f1"] = round(_avg([e.get("f1", 0.0) for e in entries]), 4)
            s["exact_match"] = round(_avg([float(e.get("em", False)) for e in entries]), 4)
        summary[name] = s
    return summary


def _print_results(summary: dict, with_generation: bool) -> None:
    n = summary["naive"]
    a = summary["advanced"]
    w = 62

    print("=" * w)
    print(f"  Benchmark results  —  {n['n']} questions")
    print("=" * w)
    print(f"{'Metric':<30} {'Naive RAG':>14} {'Advanced RAG':>14}")
    print("-" * w)
    print(f"{'Context Recall@K':<30} {n['context_recall']:>14.3f} {a['context_recall']:>14.3f}")
    print(f"{'MRR':<30} {n['mrr']:>14.3f} {a['mrr']:>14.3f}")
    print(f"{'Avg Retrieval (ms)':<30} {n['avg_retrieval_ms']:>13.1f}ms {a['avg_retrieval_ms']:>13.1f}ms")
    if with_generation:
        print("-" * w)
        print(f"{'Answer F1':<30} {n['answer_f1']:>14.3f} {a['answer_f1']:>14.3f}")
        print(f"{'Exact Match':<30} {n['exact_match']:>14.3f} {a['exact_match']:>14.3f}")
    print("=" * w)


def _print_progress(current: int, total: int) -> None:
    pct = current / total
    bar_len = 38
    filled = int(bar_len * pct)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  [{bar}] {current}/{total}", end="", flush=True)


if __name__ == "__main__":
    main()
