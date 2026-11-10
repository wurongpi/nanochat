#!/bin/bash
# Full re-evaluation suite for the published checkpoints (SFT + RL), with fixed scoring.
#   0. harness self-check (canonical solutions must pass ~100%)
#   1. greedy pass@1                (HumanEval, MBPP; HumanEval+ from the same completions)
#   2. sampled pass@1/5/10          (T=$TEMP, n=$N; HumanEval+ from the same completions)
#   3. [MBPP_PLUS=1] MBPP+ generations for both decodings
#   4. [SWEEP=1]     temperature sweep on $SWEEP_PROBLEMS problems per task, run before step 2
#   5. EvalPlus scoring of every exported sample file
# ~1.5-2.5h on one A800 for the core (0,1,2,5); MBPP_PLUS adds ~1-1.5h, SWEEP ~15-20min.
#
# Usage (from the repo root):
#   NANOCHAT_BASE_DIR=/root/autodl-tmp/nanochat bash runs/code_reeval.sh
#   MBPP_PLUS=1 SWEEP=1 bash runs/code_reeval.sh
#   TORCH_EXTRA=cpu bash runs/code_reeval.sh      # Mac (MPS) / CPU
# Knobs: SOURCES (sft,rl) TEMP (0.6) N (10) MAX_TOK (256) MAX_PROBLEMS (all) HF_ENDPOINT (hub mirror)
#        TORCH_EXTRA (gpu|cpu, default gpu) SKIP_SYNC=1 (reuse the current .venv as is)
# Dependencies come from pyproject.toml: the `eval` extra adds evalplus. --inexact keeps the packages of other
# extras (e.g. `hub` from runs/code_rlvr.sh) instead of uninstalling them.
set -euo pipefail
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
SOURCES="${SOURCES:-sft,rl}"
TEMP="${TEMP:-0.6}"
N="${N:-10}"
MAX_TOK="${MAX_TOK:-256}"
SWEEP_PROBLEMS="${SWEEP_PROBLEMS:-50}"
TASKS="HumanEval,MBPP"
[ "${MBPP_PLUS:-0}" = "1" ] && TASKS="$TASKS,MBPPPlus"
LIMIT=()
[ -n "${MAX_PROBLEMS:-}" ] && LIMIT=(--max-problems="$MAX_PROBLEMS")
OUT="$NANOCHAT_BASE_DIR/eval_fixed"

TORCH_EXTRA="${TORCH_EXTRA:-gpu}"

# Python venv (same convention as runs/code_rlvr.sh)
if [ "${SKIP_SYNC:-0}" != "1" ]; then
    command -v uv &> /dev/null || pip install uv
    [ -d ".venv" ] || uv venv
    uv sync --inexact --extra "$TORCH_EXTRA" --extra eval
fi
source .venv/bin/activate

echo "=== 0. selfcheck ==="
python -m scripts.code_reeval --selfcheck

echo "=== 1. greedy ==="
python -m scripts.code_reeval --sources="$SOURCES" --tasks="$TASKS" --max-new-tokens="$MAX_TOK" \
    --num-samples=1 --temperature=0 ${LIMIT[@]+"${LIMIT[@]}"}

if [ "${SWEEP:-0}" = "1" ]; then
    echo "=== 4. temperature sweep ($SWEEP_PROBLEMS problems/task) ==="
    for T in 0.2 0.4 0.6 0.8; do
        python -m scripts.code_reeval --sources="$SOURCES" --tasks=HumanEval,MBPP --max-new-tokens="$MAX_TOK" \
            --num-samples="$N" --temperature="$T" --max-problems="$SWEEP_PROBLEMS" --no-evalplus-export \
            --out-name="sweep_T${T}_n${N}" | tail -n 12
    done
    echo ">>> sweep done: pick the T with the best pass@$N, rerun with TEMP=<that T> if it differs from $TEMP"
fi

echo "=== 2. sampling T=$TEMP n=$N ==="
python -m scripts.code_reeval --sources="$SOURCES" --tasks="$TASKS" --max-new-tokens="$MAX_TOK" \
    --num-samples="$N" --temperature="$TEMP" ${LIMIT[@]+"${LIMIT[@]}"}

echo "=== 5. EvalPlus scoring ==="
for d in "$OUT/T0_n1_tok$MAX_TOK" "$OUT/T${TEMP}_n${N}_tok$MAX_TOK"; do
    for f in "$d"/evalplus_*_humaneval.jsonl "$d"/evalplus_*_mbpp.jsonl; do
        [ -f "$f" ] || continue
        ds=humaneval; [[ "$f" == *_mbpp.jsonl ]] && ds=mbpp
        rm -f "${f%.jsonl}_eval_results.json"
        echo "--- $ds : $f"
        evalplus.evaluate --dataset "$ds" --samples "$f" 2>&1 | grep -E "pass@|Base|Plus|base|plus" || true
    done
done
echo "Done. Summaries: $OUT/*/summary.json ; EvalPlus results: $OUT/*/evalplus_*_eval_results.json"
