#!/usr/bin/env bash
#
# Overnight benchmark: Naive vs Advanced RAG on SQuAD + DocRED, with RAGAS,
# hallucination test (DocRED), figures and significance tests.
#
# Usage on the VM:
#   chmod +x run_overnight.sh
#   nohup ./run_overnight.sh > overnight.log 2>&1 &
#   tail -f overnight.log
#
# Tunables (override on the command line, e.g. SAMPLES=150 ./run_overnight.sh):
SAMPLES="${SAMPLES:-100}"          # questions per seed
SEEDS="${SEEDS:-3}"                # number of seeds
NEG_FRAC="${NEG_FRAC:-0.3}"        # fraction of DocRED questions made unanswerable
PARALLEL="${PARALLEL:-4}"          # Ollama parallel slots == RAGAS workers
RESTART_OLLAMA="${RESTART_OLLAMA:-0}"   # set to 1 to restart ollama with PARALLEL slots

set -euo pipefail
cd "$(dirname "$0")"

# Keep the GPU entirely for Ollama; embedder + reranker run on CPU.
export RAG_DEVICE="${RAG_DEVICE:-cpu}"
export OLLAMA_NUM_PARALLEL="$PARALLEL"

echo "=== RAG overnight benchmark — $(date) ==="
echo "samples=$SAMPLES seeds=$SEEDS neg_frac=$NEG_FRAC parallel=$PARALLEL device=$RAG_DEVICE"

# --- venv ---------------------------------------------------------------- #
if [ -d venv ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

# --- optional: restart Ollama with parallel slots ----------------------- #
if [ "$RESTART_OLLAMA" = "1" ]; then
  echo "Restarting ollama with OLLAMA_NUM_PARALLEL=$PARALLEL ..."
  pkill -f "ollama serve" 2>/dev/null || true
  sleep 2
  OLLAMA_NUM_PARALLEL="$PARALLEL" nohup ollama serve > ollama.log 2>&1 &
  sleep 5
fi

# --- pre-flight: Ollama reachable --------------------------------------- #
if ! curl -sf http://localhost:11434/api/tags >/dev/null; then
  echo "ERROR: Ollama is not reachable on :11434. Start it (optionally with"
  echo "OLLAMA_NUM_PARALLEL=$PARALLEL ollama serve) or rerun with RESTART_OLLAMA=1."
  exit 1
fi

# --- pre-flight: ensure both corpora are ingested ----------------------- #
for ds in squad docred; do
  count=$(python - "$ds" <<'PY'
import sys, datasets_registry as dr, chromadb
dr.set_active(sys.argv[1])
c = chromadb.PersistentClient(path=str(dr.CHROMA_PATH))
col = c.get_or_create_collection(dr.active().collection_name)
print(col.count())
PY
)
  echo "collection '$ds' has $count documents"
  if [ "$count" -eq 0 ]; then
    echo "Ingesting '$ds' ..."
    python ingest.py --dataset "$ds"
  fi
done

# --- SQuAD --------------------------------------------------------------- #
echo ""
echo "=== [1/2] SQuAD — $(date) ==="
python benchmark.py --dataset squad \
    --samples "$SAMPLES" --num-seeds "$SEEDS" \
    --with-ragas --ragas-workers "$PARALLEL" \
    --plot --figures-dir figures/ \
    --output results_squad.json

# --- DocRED (relation extraction + hallucination test) ------------------- #
echo ""
echo "=== [2/2] DocRED — $(date) ==="
python benchmark.py --dataset docred \
    --samples "$SAMPLES" --num-seeds "$SEEDS" \
    --with-ragas --ragas-workers "$PARALLEL" \
    --neg-frac "$NEG_FRAC" \
    --plot --figures-dir figures/ \
    --output results_docred.json

echo ""
echo "=== DONE — $(date) ==="
echo "Outputs: results_squad.json, results_docred.json"
echo "Figures: figures/squad_*.{pdf,png}, figures/docred_*.{pdf,png}"
