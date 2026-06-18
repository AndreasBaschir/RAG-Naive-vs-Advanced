# RAG Naive vs Advanced

A comparison between a simple RAG (Retrieval-Augmented Generation) system and an advanced one, using the **Qwen3-8B** LLM.

## Objective

Demonstrating the improvements brought by an advanced RAG architecture through concrete metrics (RAGAS faithfulness and answer_relevancy) and real-world use cases.

## Architectures

### Naive RAG

- **Chunking**: Documents split into fixed-size chunks
- **Indexing**: Dense embedding model, stored in ChromaDB
- **Retrieval**: Cosine Similarity, top K documents
- **Generation**: Concatenate into Qwen3-8B prompt
- **Purpose**: Simple baseline to highlight Advanced RAG benefits

### Advanced RAG

1. **Preprocessing**
   - Semantic chunking
   - Contextual retrieval: each chunk prefixed with LLM-generated context

2. **Indexing**
   - Dense indexing in ChromaDB
   - Sparse indexing with BM25 (keyword-based)

3. **Query Processing**
   - Multi-Query: 3 query reformulations
   - Hybrid retrieval (dense + BM25)
   - Large top-K (e.g., 50 documents)
   - Merge using Reciprocal Rank Fusion (RRF)

4. **Reranking**
   - Cross-Encoder for relevance scoring
   - Keeps top 5 chunks with highest relevance

5. **Generation**
   - Strict grounding
   - Answers only based on available documents

## Metrics

- **RAGAS Faithfulness**: How faithful the answer is to the provided context
- **RAGAS Answer Relevancy**: How relevant the answer is to the question

## Demonstration Cases

### 1. Exact/Rare Terms (numbers, codes)
```
Naive:     Hallucinates - cosine similarity misses exact terms
Advanced:  Finds answer via dense + BM25 combination
```

### 2. Question with no answer in corpus
```
Naive:     Hallucinates - generates answer anyway
Advanced:  Strict grounding - "Not found in documents"
```

### 3. Detail from long chunk
```
Naive:     Truncates chunk, loses context, partial/wrong answer
Advanced:  Semantic chunking + reranking = complete, correct answer
```

## Setup

### Prerequisites

- **Ollama** with Qwen3-8B model
```bash
ollama pull qwen3:8b
ollama serve
```

### Installation

Run the automated setup script:
```bash
bash setup.sh
```

Or manually:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python ingest.py
```

### Datasets

Two datasets are supported, each stored in its own ChromaDB collection:

- **SQuAD** (`squad`) — extractive question answering.
- **DocRED** (`docred`) — document-level **relation extraction**. Each relation
  triple `(head, relation, tail)` is turned into a question
  (`What is the "{relation}" of {head}?`) whose gold answers are the tail
  entity's surface forms. One triple per document is evaluated.

Ingest a dataset (run once per dataset):
```bash
python ingest.py                  # squad (default)
python ingest.py --dataset docred
```

Notes:
- The first advanced run per dataset builds a BM25 index cached at
  `data/bm25_index_<dataset>.pkl`. An older `data/bm25_index.pkl` from a previous
  version is now orphaned and can be deleted (it regenerates).
- Documents are embedded whole; passages longer than the embedder's ~256-token
  window are truncated at embed time. This affects both pipelines equally, so the
  Naive-vs-Advanced comparison stays fair, but interpret absolute recall with it
  in mind.

## Benchmark

```bash
python benchmark.py --dataset squad  --samples 150 --num-seeds 4 --with-generation --plot --output results_squad.json
python benchmark.py --dataset docred --samples 150 --num-seeds 4 --with-generation --plot --output results_docred.json
```

Add `--with-ragas` for RAGAS faithfulness / answer-relevancy (slow). Results are
reported as mean ± std across seeds with paired significance tests.

**Hallucination resistance (DocRED).** `--neg-frac F` turns a fraction `F` of
questions into *unanswerable* ones — a real entity paired with a relation it does
not have in the document — and reports each pipeline's **hallucination rate**
(fraction where it fabricated an answer instead of abstaining). Requires
`--with-generation`. Each DocRED record also stores its `relation`,
`relation_id`, `evidence_sents` and `multi_hop` flag, so per-relation-type and
multi-hop breakdowns can be computed post-hoc from the saved `records`.

```bash
python benchmark.py --dataset docred --samples 100 --num-seeds 3 \
    --with-ragas --neg-frac 0.3 --plot --output results_docred.json
```

## Interface

```bash
streamlit run main.py
```

Compares responses from both systems side-by-side. Use the sidebar to switch
between the `squad` and `docred` corpora.

## Project Structure

```
RAG-Naive-vs-Advanced/
├── .git/                  # Git configuration
├── .gitignore             # Git ignore rules
├── .streamlit/            # Streamlit configuration
├── naive/                 # Naive RAG implementation
│   ├── pipeline.py        # Naive pipeline logic
│   └── __init__.py
├── advanced/              # Advanced RAG implementation
│   ├── pipeline.py        # Advanced pipeline logic
│   ├── chunking.py        # Semantic chunking
│   ├── retrieval.py       # Hybrid retrieval
│   ├── reranking.py       # Cross-encoder reranking
│   └── __init__.py
├── data/                  # Dataset and ChromaDB
│   ├── raw/               # Raw dataset files
│   └── chroma_db/         # ChromaDB storage
├── main.py               # Streamlit interface
├── ingest.py             # Data ingestion script
├── setup.sh              # Automated setup script
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Main Dependencies

- `streamlit` - Web interface
- `chromadb` - Vector database
- `ollama` - LLM client
- `ragas` - Evaluation metrics
- `bm25s` - BM25 indexing

---

**Author**: Andreas
