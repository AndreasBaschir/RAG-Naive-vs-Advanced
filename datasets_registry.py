"""
Dataset registry for the RAG benchmark.

Owns:
  - the process-global "active dataset" selection (squad | docred),
  - per-dataset ChromaDB collection name and BM25 cache path,
  - the loaders (corpus for ingestion, eval rows for benchmarking),
  - the row -> benchmark-record extraction that maps each dataset into the
    common (question, gold_answers, gold_context, gold_title) schema.

The active dataset is resolved LAZILY by the pipelines (inside their singleton
getters), so selecting a dataset at runtime via set_active() works regardless
of import order. set_active() must be called before the first retrieve() call.
"""

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Callable, cast

ROOT = pathlib.Path(__file__).parent
DATA_DIR = ROOT / "data"
CHROMA_PATH = DATA_DIR / "chroma"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    collection_name: str
    bm25_cache_path: pathlib.Path
    load_corpus: Callable[[], list[dict]]       # -> [{"text", "title"}] for ingestion
    load_eval_rows: Callable[[], list[dict]]    # raw rows for benchmark sampling
    make_record: Callable[[dict], dict | None]  # row -> benchmark record (None to skip)


_REGISTRY: dict[str, DatasetSpec] = {}
_active: str = os.environ.get("RAG_DATASET", "squad")


def register(spec: DatasetSpec) -> None:
    _REGISTRY[spec.key] = spec


def available() -> list[str]:
    return sorted(_REGISTRY)


def set_active(key: str) -> None:
    global _active
    if key not in _REGISTRY:
        raise ValueError(f"unknown dataset {key!r}; have {available()}")
    _active = key
    os.environ["RAG_DATASET"] = key  # propagate to any re-import / child process


def active() -> DatasetSpec:
    return _REGISTRY[_active]


def active_key() -> str:
    return _active


# --------------------------------------------------------------------------- #
# SQuAD                                                                        #
# --------------------------------------------------------------------------- #

def _squad_corpus() -> list[dict]:
    """Unique SQuAD contexts as [{"text", "title"}] (one paragraph per entry)."""
    from datasets import load_dataset

    dataset = load_dataset("rajpurkar/squad", split="train")
    seen: set[str] = set()
    contexts: list[dict] = []
    for _row in dataset:
        row = cast("dict[str, Any]", _row)
        ctx = row["context"]
        if ctx not in seen:
            seen.add(ctx)
            contexts.append({"text": ctx, "title": row["title"]})
    return contexts


def _squad_eval_rows() -> list[dict]:
    from datasets import load_dataset

    return list(load_dataset("rajpurkar/squad", split="train"))


def _squad_record(row: dict) -> dict:
    return {
        "question": row["question"],
        "gold_context": row["context"],
        "gold_answers": row["answers"]["text"],
        "gold_title": row["title"],
    }


# --------------------------------------------------------------------------- #
# DocRED (relation extraction)                                                 #
# --------------------------------------------------------------------------- #

_DOCRED_REPO = "thunlp/docred"
_DOCRED_SPLIT = "train_annotated"
_docred_rows_cache: list[dict] | None = None


def _docred_doc_text(row: dict) -> str:
    return " ".join(" ".join(sent) for sent in row["sents"])


def _normalize_labels(labels: Any, rel_info: dict | None = None) -> list[dict]:
    """Normalize DocRED labels (either dict-of-lists or list-of-dicts) to a
    canonical list of {head, tail, relation_id, relation_text, evidence}."""
    out: list[dict] = []

    # Parquet / load_dataset form: dict of parallel lists (numpy arrays).
    if isinstance(labels, dict):
        n = len(labels.get("head", []))
        ev = labels.get("evidence", None)
        for i in range(n):
            ev_i = ev[i] if ev is not None else []
            out.append({
                "head": int(labels["head"][i]),
                "tail": int(labels["tail"][i]),
                "relation_id": str(labels["relation_id"][i]),
                "relation_text": str(labels["relation_text"][i]),
                "evidence": [int(e) for e in ev_i],
            })
        return out

    # Raw JSON form: list of {h, t, r, evidence} with no relation_text.
    for lab in labels:
        rid = lab.get("relation_id", lab.get("r", ""))
        rtext = lab.get("relation_text") or (rel_info or {}).get(rid, rid)
        out.append({
            "head": int(lab.get("head", lab.get("h"))),
            "tail": int(lab.get("tail", lab.get("t"))),
            "relation_id": rid,
            "relation_text": rtext,
            "evidence": list(lab.get("evidence", []) or []),
        })
    return out


def _normalize_row(row: dict, rel_info: dict | None = None) -> dict:
    return {
        "title": row["title"],
        "sents": row["sents"],
        "vertexSet": row["vertexSet"],
        "labels": _normalize_labels(row["labels"], rel_info),
    }


def _load_docred() -> list[dict]:
    """Load DocRED train_annotated as normalized rows, trying several
    strategies because datasets>=4 dropped script-based loading."""
    global _docred_rows_cache
    if _docred_rows_cache is not None:
        return _docred_rows_cache

    rows: list[dict] | None = None

    # 1) Auto-parquet export via load_dataset.
    try:
        from datasets import load_dataset

        ds = load_dataset(_DOCRED_REPO, "default", split=_DOCRED_SPLIT)
        rows = [_normalize_row(cast("dict[str, Any]", r)) for r in ds]
        print(f"[docred] loaded via load_dataset auto-parquet ({len(rows)} docs)")
    except Exception as exc:  # noqa: BLE001
        print(f"[docred] load_dataset failed ({exc}); trying direct parquet...")

    # 2) Direct parquet on the refs/convert/parquet branch.
    if rows is None:
        try:
            import pandas as pd
            from huggingface_hub import HfApi

            files = HfApi().list_repo_files(
                repo_id=_DOCRED_REPO, repo_type="dataset",
                revision="refs/convert/parquet",
            )
            targets = [f for f in files if _DOCRED_SPLIT in f and f.endswith(".parquet")]
            if not targets:
                raise RuntimeError("no parquet files for split on convert branch")
            frames = [
                pd.read_parquet(f"hf://datasets/{_DOCRED_REPO}@refs/convert/parquet/{f}")
                for f in targets
            ]
            df = pd.concat(frames, ignore_index=True)
            rows = [_normalize_row(cast("dict[str, Any]", r)) for r in df.to_dict("records")]
            print(f"[docred] loaded via direct parquet ({len(rows)} docs)")
        except Exception as exc:  # noqa: BLE001
            print(f"[docred] direct parquet failed ({exc}); trying raw JSON...")

    # 3) Raw JSON + rel_info.json from the repo.
    if rows is None:
        try:
            from huggingface_hub import hf_hub_download

            data_path = hf_hub_download(
                repo_id=_DOCRED_REPO, repo_type="dataset",
                filename=f"{_DOCRED_SPLIT}.json",
            )
            rel_path = hf_hub_download(
                repo_id=_DOCRED_REPO, repo_type="dataset", filename="rel_info.json",
            )
            with open(rel_path) as f:
                rel_info = json.load(f)
            with open(data_path) as f:
                raw = json.load(f)
            rows = [_normalize_row(r, rel_info) for r in raw]
            print(f"[docred] loaded via raw JSON ({len(rows)} docs)")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Could not load DocRED by any method. Manually download "
                f"{_DOCRED_SPLIT}.json and rel_info.json from "
                f"https://huggingface.co/datasets/{_DOCRED_REPO} and place them in "
                f"{DATA_DIR}/docred/, then adapt _load_docred()."
            ) from exc

    _docred_rows_cache = rows
    return rows


def _docred_corpus() -> list[dict]:
    """One entry per DocRED document, de-duped by title."""
    seen: set[str] = set()
    corpus: list[dict] = []
    for row in _load_docred():
        title = row["title"]
        if title in seen:
            continue
        seen.add(title)
        corpus.append({"text": _docred_doc_text(row), "title": title})
    return corpus


def _docred_eval_rows() -> list[dict]:
    return _load_docred()


def _entity_names(entity: list[dict]) -> list[str]:
    """De-duplicated (case-insensitive) surface forms of an entity's mentions."""
    names: list[str] = []
    seen: set[str] = set()
    for mention in entity:
        name = (mention.get("name") or "").strip()
        if name and name.lower() not in seen:
            seen.add(name.lower())
            names.append(name)
    return names


def _docred_record(row: dict) -> dict | None:
    """Pick one relation triple deterministically and build a QA-style record."""
    labels = row["labels"]
    vertex = row["vertexSet"]
    if not labels or len(vertex) == 0:
        return None

    # Prefer triples with evidence; tie-break stable on (relation_id, head, tail);
    # skip self-loops.
    candidates = [
        lab for lab in labels
        if lab["head"] != lab["tail"]
        and 0 <= lab["head"] < len(vertex)
        and 0 <= lab["tail"] < len(vertex)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda l: (not l["evidence"], l["relation_id"], l["head"], l["tail"]))
    lab = candidates[0]

    head_names = _entity_names(vertex[lab["head"]])
    tail_names = _entity_names(vertex[lab["tail"]])
    if not head_names or not tail_names:
        return None

    head_name = head_names[0]
    relation = lab["relation_text"]
    return {
        "question": f'What is the "{relation}" of {head_name}?',
        "gold_context": _docred_doc_text(row),
        "gold_answers": tail_names,
        "gold_title": row["title"],
    }


# --------------------------------------------------------------------------- #
# Registration                                                                 #
# --------------------------------------------------------------------------- #

register(DatasetSpec(
    key="squad",
    collection_name="squad",
    bm25_cache_path=DATA_DIR / "bm25_index_squad.pkl",
    load_corpus=_squad_corpus,
    load_eval_rows=_squad_eval_rows,
    make_record=_squad_record,
))

register(DatasetSpec(
    key="docred",
    collection_name="docred",
    bm25_cache_path=DATA_DIR / "bm25_index_docred.pkl",
    load_corpus=_docred_corpus,
    load_eval_rows=_docred_eval_rows,
    make_record=_docred_record,
))
