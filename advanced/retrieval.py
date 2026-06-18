"""
Hybrid retrieval components for the Advanced RAG pipeline.

Dense (ChromaDB cosine similarity) + Sparse (BM25) + RRF + Cross-encoder reranking.
"""

from __future__ import annotations

import os

import bm25s
import chromadb
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from datasets_registry import CHROMA_PATH, active, active_key

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Force with RAG_DEVICE=cpu (e.g. to leave the GPU entirely to Ollama);
# otherwise auto-detect CUDA.
DEVICE = os.environ.get("RAG_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

_embedder: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None
_collection_key: str | None = None
_reranker: CrossEncoder | None = None
_bm25_index: bm25s.BM25 | None = None
_bm25_corpus: list[dict] | None = None
_bm25_key: str | None = None


def encode(texts: list[str]) -> list[list[float]]:
    """Batch-encode a list of texts into embedding vectors."""
    return _get_embedder().encode(texts, batch_size=len(texts)).tolist()


def dense_retrieve(query: str, n: int = 20) -> list[dict]:
    """Top-n chunks via dense cosine similarity in ChromaDB."""
    return dense_retrieve_vec(_get_embedder().encode(query).tolist(), n)


def dense_retrieve_vec(vec: list[float], n: int = 20) -> list[dict]:
    """Top-n chunks using a pre-computed embedding vector (skips encoding)."""
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
    """Top-n chunks via BM25 keyword scoring over the full corpus."""
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
    """
    Reciprocal Rank Fusion across multiple ranked result lists.

    Score formula: sum(1 / (k + rank)) across all lists a document appears in.
    Deduplication is by text content.
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
    """
    Cross-encoder reranking: scores each (query, passage) pair independently,
    providing more accurate relevance than bi-encoder similarity alone.
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


# --- Lazy singletons ---

def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
    return _embedder


def _get_collection() -> chromadb.Collection:
    global _collection, _collection_key, _bm25_index, _bm25_corpus
    if _collection is not None and _collection_key == active_key():
        return _collection
    # Dataset switched (or first call): drop any stale BM25 state too.
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
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL, device=DEVICE)
    return _reranker


def _get_bm25() -> tuple[bm25s.BM25, list[dict]]:
    global _bm25_index, _bm25_corpus, _bm25_key
    if _bm25_index is not None and _bm25_corpus is not None and _bm25_key == active_key():
        return _bm25_index, _bm25_corpus

    import json

    # Use bm25s' native save/load (NOT pickle) — pickled BM25 objects lose the
    # `scores` index across library versions/machines. The doc corpus (text +
    # title) is stored alongside as JSON.
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

    _bm25_corpus = [
        {"text": text, "title": meta.get("title", "")}
        for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
    ]

    corpus_tokens = bm25s.tokenize([doc["text"] for doc in _bm25_corpus])
    _bm25_index = bm25s.BM25()
    _bm25_index.index(corpus_tokens)

    cache_dir.mkdir(parents=True, exist_ok=True)
    _bm25_index.save(str(cache_dir))
    with open(corpus_file, "w") as f:
        json.dump(_bm25_corpus, f)

    _bm25_key = active_key()
    return _bm25_index, _bm25_corpus
