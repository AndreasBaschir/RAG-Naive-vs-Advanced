"""
Advanced RAG pipeline.

Improvements over the naive baseline:
  1. Multi-query expansion  — LLM generates alternative phrasings to improve recall
  2. Hybrid retrieval       — dense (bi-encoder) + sparse (BM25) per query
  3. Reciprocal Rank Fusion — merges all result lists into a single ranking
  4. Cross-encoder reranking — re-scores top candidates for precision
  5. Grounded generation    — system prompt instructs the LLM to stay in-context
"""

from __future__ import annotations

import re
from typing import Generator

import ollama

from advanced.retrieval import bm25_retrieve, dense_retrieve, rerank, rrf_merge

LLM_MODEL = "qwen3:8b"

N_DENSE = 20     # dense candidates per query variant
N_BM25 = 20     # BM25 candidates per query variant
N_VARIANTS = 2  # extra query reformulations to generate
RERANK_POOL = 40  # max candidates fed into the cross-encoder
FINAL_K = 5     # chunks passed to the LLM

_EXPAND_PROMPT = """\
Generate {n} alternative phrasings for the following search query. \
Output only the queries, one per line, no numbering or explanation. /no_think

Original query: {query}

Alternative queries:"""

_SYSTEM_PROMPT = (
    "You are a precise, factual assistant. "
    "Answer the user's question using ONLY the information in the provided context. "
    "If the context does not contain enough information to answer, reply with exactly: "
    "\"The provided context does not contain enough information to answer this question.\""
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def retrieve(query: str) -> list[dict]:
    """
    Return FINAL_K reranked chunks for the query.

    Steps:
      1. Generate N_VARIANTS alternative query phrasings via the LLM.
      2. Run dense + BM25 retrieval for every query variant.
      3. Merge all ranked lists with Reciprocal Rank Fusion.
      4. Rerank the top-RERANK_POOL candidates with a cross-encoder.
    """
    queries = _expand_queries(query)

    ranked_lists: list[list[dict]] = []
    for q in queries:
        ranked_lists.append(dense_retrieve(q, n=N_DENSE))
        ranked_lists.append(bm25_retrieve(q, n=N_BM25))

    merged = rrf_merge(ranked_lists)
    pool = merged[:RERANK_POOL]
    reranked = rerank(query, pool, top_k=FINAL_K)

    for chunk in reranked:
        chunk.setdefault("score", chunk.get("rerank_score", 0.0))

    return reranked


def stream(query: str, chunks: list[dict]) -> Generator[str, None, None]:
    context = "\n\n".join(c["text"] for c in chunks)

    for part in ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"},
        ],
        stream=True,
    ):
        yield part["message"]["content"]


def _expand_queries(query: str) -> list[str]:
    """Ask the LLM for N_VARIANTS alternative phrasings; fall back gracefully."""
    try:
        resp = ollama.chat(
            model=LLM_MODEL,
            messages=[{
                "role": "user",
                "content": _EXPAND_PROMPT.format(n=N_VARIANTS, query=query),
            }],
            options={"num_predict": 150},
        )
        raw = _THINK_RE.sub("", resp["message"]["content"]).strip()
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        alternatives = lines[:N_VARIANTS]
    except Exception:
        alternatives = []

    return [query] + alternatives
