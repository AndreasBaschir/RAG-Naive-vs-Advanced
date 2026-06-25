"""
Hybrid retrieval components for the Advanced RAG pipeline.

Dense (ChromaDB cosine similarity) + Sparse (BM25) + RRF + Cross-encoder reranking.
"""

from __future__ import annotations

import os
import threading

import bm25s
import chromadb
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from datasets_registry import CHROMA_PATH, active, active_key

# Serialises singleton first-init: pipeline.py fans out threads per query, so concurrent cache-misses must be blocked.
_init_lock = threading.RLock()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEVICE = os.environ.get("RAG_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

_embedder: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None
_collection_key: str | None = None
_reranker: CrossEncoder | None = None
_bm25_index: bm25s.BM25 | None = None
_bm25_corpus: list[dict] | None = None
_bm25_key: str | None = None


def encode(texts: list[str]) -> list[list[float]]:
    """Batch-encode a list of texts into embedding vectors.

    :param texts: strings to embed
    :returns: list of float vectors, one per input text
    """
    return _get_embedder().encode(texts, batch_size=len(texts)).tolist()


def dense_retrieve(query: str, n: int = 20) -> list[dict]:
    """Return the top-*n* chunks via dense cosine similarity in ChromaDB.

    :param query: search query (encoded internally)
    :param n: maximum number of results
    :returns: ranked chunk dicts with ``text``, ``source``, ``dense_score``
    """
    return dense_retrieve_vec(_get_embedder().encode(query).tolist(), n)


def dense_retrieve_vec(vec: list[float], n: int = 20) -> list[dict]:
    """Return the top-*n* chunks using a pre-computed embedding vector.

    :param vec: pre-computed query embedding (skips the encoding step)
    :param n: maximum number of results
    :returns: ranked chunk dicts with ``text``, ``source``, ``dense_score``
    """
    col = _get_collection()
    if col.count() == 0:
        return []

    results = col.query(
        query_embeddings=[vec],
        n_results=min(n, col.count()),
        include=["documents", "metadatas", "distances"],
    )

    docs, metas, dists = results["documents"], results["metadatas"], results["distances"]
    if not docs or not metas or not dists:
        return []

    return [
        {
            "text": text,
            "source": meta.get("title", ""),
            "dense_score": round(1.0 - dist, 4),
        }
        for text, meta, dist in zip(docs[0], metas[0], dists[0])
    ]


def bm25_retrieve(query: str, n: int = 20) -> list[dict]:
    """Return the top-*n* chunks via BM25 keyword scoring over the full corpus.

    :param query: search query (tokenised internally)
    :param n: maximum number of results
    :returns: ranked chunk dicts with ``text``, ``source``, ``bm25_score``
    """
    index, corpus = _get_bm25()
    query_tokens = bm25s.tokenize([query])
    results, scores = index.retrieve(query_tokens, k=min(n, len(corpus)))

    return [
        {
            "text": corpus[idx]["text"],
            "source": corpus[idx]["title"],
            "bm25_score": round(float(score), 4),
        }
        for idx, score in zip(results[0], scores[0])
        if score > 0
    ]


def rrf_merge(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Merge multiple ranked result lists with Reciprocal Rank Fusion.

    Score formula: ``sum(1 / (k + rank))`` across all lists a document appears in.
    Deduplication is by text content.

    :param ranked_lists: list of ranked chunk lists (dense or BM25 results)
    :param k: RRF smoothing constant
    :returns: merged and re-sorted list with ``rrf_score`` added to each chunk
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = doc["text"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in doc_map:
                doc_map[key] = doc.copy()

    merged = sorted(doc_map.values(), key=lambda d: scores[d["text"]], reverse=True)
    for doc in merged:
        doc["rrf_score"] = round(scores[doc["text"]], 6)
    return merged


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Re-score *candidates* with a cross-encoder and return the top-*top_k*.

    Cross-encoder reranking scores each (query, passage) pair independently,
    providing more accurate relevance than bi-encoder similarity alone.

    :param query: original user query
    :param candidates: pool of chunks to rerank (each must have a ``text`` key)
    :param top_k: number of top-scoring chunks to return
    :returns: top-*top_k* chunks sorted by ``rerank_score`` descending
    """
    if not candidates:
        return []

    cross_enc = _get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = cross_enc.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = round(float(s), 4)
        c["score"] = c["rerank_score"]

    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


def _get_embedder() -> SentenceTransformer:
    """Return the bi-encoder singleton, initialising on first call (thread-safe)."""
    global _embedder
    if _embedder is not None:
        return _embedder
    with _init_lock:
        if _embedder is None:
            _embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
    return _embedder


def _get_collection() -> chromadb.Collection:
    """Return the ChromaDB collection for the active dataset, re-opening on switch.

    Also invalidates the BM25 cache when the active dataset changes so the
    two caches stay in sync.
    """
    global _collection, _collection_key, _bm25_index, _bm25_corpus
    if _collection is not None and _collection_key == active_key():
        return _collection
    with _init_lock:
        if _collection is not None and _collection_key == active_key():
            return _collection
        _bm25_index = None
        _bm25_corpus = None
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = client.get_or_create_collection(
            active().collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        _collection_key = active_key()
    return _collection


def _get_reranker() -> CrossEncoder:
    """Return the cross-encoder singleton, initialising on first call (thread-safe)."""
    global _reranker
    if _reranker is not None:
        return _reranker
    with _init_lock:
        if _reranker is None:
            _reranker = CrossEncoder(RERANKER_MODEL, device=DEVICE)
    return _reranker


def _get_bm25() -> tuple[bm25s.BM25, list[dict]]:
    """Return the BM25 index and corpus for the active dataset (thread-safe).

    Loads from disk cache if available; otherwise builds from the ChromaDB
    collection and persists the result. Uses bm25s native save/load rather
    than pickle to avoid cross-version score-index corruption.

    :returns: ``(BM25 index, corpus)`` where corpus is a list of
              ``{"text", "title"}`` dicts
    """
    global _bm25_index, _bm25_corpus, _bm25_key
    if _bm25_index is not None and _bm25_corpus is not None and _bm25_key == active_key():
        return _bm25_index, _bm25_corpus

    import json

    with _init_lock:
        if _bm25_index is not None and _bm25_corpus is not None and _bm25_key == active_key():
            return _bm25_index, _bm25_corpus

        cache_dir = active().bm25_cache_dir
        corpus_file = cache_dir / "corpus.json"

        if cache_dir.exists() and corpus_file.exists():
            _bm25_index = bm25s.BM25.load(str(cache_dir), mmap=False)
            with open(corpus_file) as f:
                _bm25_corpus = json.load(f)
            _bm25_key = active_key()
            return _bm25_index, _bm25_corpus

        col = _get_collection()
        all_docs = col.get(include=["documents", "metadatas"])

        corpus = [
            {"text": text, "title": meta.get("title", "")}
            for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
        ]

        corpus_tokens = bm25s.tokenize([doc["text"] for doc in corpus])
        index = bm25s.BM25()
        index.index(corpus_tokens)

        cache_dir.mkdir(parents=True, exist_ok=True)
        index.save(str(cache_dir))
        with open(corpus_file, "w") as f:
            json.dump(corpus, f)

        # Publish to globals only once fully built to prevent half-initialised reads.
        _bm25_corpus = corpus
        _bm25_index = index
        _bm25_key = active_key()

    _bm25_key = active_key()
    return _bm25_index, _bm25_corpus
