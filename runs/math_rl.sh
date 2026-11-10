#!/bin/bash
# Reward-design experiment on GSM8K, single GPU, starting from the pretrained d12 base checkpoint.
#   sft    : default chat SFT (SmolTalk + MMLU + GSM8K) -> a model that can do GSM8K-style answers + calculator
#   eval0  : held-out GSM8K accuracy of the SFT model (greedy, strict)
#   rl     : one GRPO run per reward in $REWARDS, all from the same SFT model
#              exact    = the final "#### answer" must match              (sound reward)
#              contains = the reference number appears anywhere            (naive reward, can be gamed)
#            every run aborts at step 0 if pass@k < $MIN_SIGNAL (nothing to reinforce)
#   eval   : held-out GSM8K accuracy of every RL model (greedy, strict), then a comparison table
# The comparison of interest: optimized reward vs true (strict) reward vs held-out accuracy vs sequence
# length, per reward. All of it is in the logs: $NANOCHAT_BASE_DIR/math_rl/*.log
#
# Usage (from the repo root):
#   NANOCHAT_BASE_DIR=/root/autodl-tmp/nanochat HF_ENDPOINT=https://hf-mirror.com \
#     screen -L -Logfile runs/math_rl.log -S mathrl bash runs/math_rl.sh
# Knobs: START_STAGE (sft|eval0|rl|eval) REWARDS ("exact contains") NUM_STEPS (300) MIN_SIGNAL (0.02)
#        BASE_TAG (d12) TAG (d12-math) SFT_DBS (32) TORCH_EXTRA (gpu) SKIP_SYNC=1 EVAL_PROBLEMS (all)
set -euo pipefail
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
START_STAGE="${START_STAGE:-sft}"
REWARDS="${REWARDS:-exact contains}"
NUM_STEPS="${NUM_STEPS:-300}"
MIN_SIGNAL="${MIN_SIGNAL:-0.02}"
BASE_TAG="${BASE_TAG:-d12}"
TAG="${TAG:-d12-math}"
SFT_DBS="${SFT_DBS:-32}"
TORCH_EXTRA="${TORCH_EXTRA:-gpu}"
LOGS="$NANOCHAT_BASE_DIR/math_rl"
mkdir -p "$LOGS"
EVAL_LIMIT=()
[ -n "${EVAL_PROBLEMS:-}" ] && EVAL_LIMIT=(-x "$EVAL_PROBLEMS")

STAGES=(sft eval0 rl eval)
[[ " ${STAGES[*]} " == *" $START_STAGE "* ]] || { echo "unknown START_STAGE=$START_STAGE"; exit 1; }
run_stage() { # true if $1 is at or after START_STAGE
    local s
    for s in "${STAGES[@]}"; do
        [ "$s" = "$START_STAGE" ] && return 0
        [ "$s" = "$1" ] && return 1
    done
}
gsm8k_eval() { # gsm8k_eval <source> <tag> <logname>: greedy, strict, held-out test split
    python -m scripts.chat_eval -i "$1" -g "$2" -a GSM8K -t 0 -m 256 ${EVAL_LIMIT[@]+"${EVAL_LIMIT[@]}"} 2>&1 | tee "$LOGS/$3.log"
}

if [ "${SKIP_SYNC:-0}" != "1" ]; then
    command -v uv &> /dev/null || pip install uv
    [ -d ".venv" ] || uv venv
    uv sync --inexact --extra "$TORCH_EXTRA"
fi
source .venv/bin/activate

# A separate tag keeps this experiment apart from the code checkpoints (chatsft/chatrl d12)
[ -d "$NANOCHAT_BASE_DIR/base_checkpoints/$BASE_TAG" ] || { echo "missing base_checkpoints/$BASE_TAG"; exit 1; }
ln -sfn "$NANOCHAT_BASE_DIR/base_checkpoints/$BASE_TAG" "$NANOCHAT_BASE_DIR/base_checkpoints/$TAG"

if run_stage sft; then
    echo "=== sft ==="
    python -m scripts.chat_sft --mix=default --model-tag="$TAG" --device-batch-size="$SFT_DBS" \
        --eval-every=200 --chatcore-every=-1 2>&1 | tee "$LOGS/sft.log"
fi

if run_stage eval0; then
    echo "=== eval0: SFT model on held-out GSM8K ==="
    gsm8k_eval sft "$TAG" eval_sft
fi

if run_stage rl; then
    for R in $REWARDS; do
        echo "=== rl: reward=$R ==="
        python -m scripts.chat_rl --task=gsm8k --reward="$R" --model-tag="$TAG" --output-tag="$TAG-$R" \
            --num-samples=16 --device-batch-size=16 --examples-per-step=8 --num-steps="$NUM_STEPS" \
            --eval-every=100 --eval-examples=200 --save-every=100 --min-signal="$MIN_SIGNAL" 2>&1 | tee "$LOGS/rl_$R.log"
    done
fi

if run_stage eval; then
    for R in $REWARDS; do
        echo "=== eval: RL model (reward=$R) on held-out GSM8K ==="
        gsm8k_eval rl "$TAG-$R" "eval_rl_$R"
    done
fi

echo
echo "==================== SUMMARY ===================="
echo "held-out GSM8K (greedy, strict):"
for f in eval_sft $(for R in $REWARDS; do echo "eval_rl_$R"; done); do
    [ -f "$LOGS/$f.log" ] && echo "  $f: $(grep -E 'GSM8K accuracy' "$LOGS/$f.log" | tail -1)"
done
for R in $REWARDS; do
    [ -f "$LOGS/rl_$R.log" ] || continue
    echo "rl reward=$R (first / last 20 steps: optimized reward | true reward | sequence length):"
    python - "$LOGS/rl_$R.log" <<'PY'
import re, sys
rows = [tuple(map(float, m.groups())) for m in re.finditer(
    r"Average reward: ([\d.eE+-]+) \| True \(strict\) reward: ([\d.]+) \| Average sequence length: ([\d.]+)", open(sys.argv[1]).read())]
avg = lambda xs, i: sum(x[i] for x in xs) / max(1, len(xs))
for name, part in (("first", rows[:20]), ("last ", rows[-20:])):
    print(f"  {name}: {avg(part,0):.4f} | {avg(part,1):.4f} | {avg(part,2):.1f}   ({len(part)} steps)")
PY
    grep -E "^Step [0-9]+ \| Pass@1:" "$LOGS/rl_$R.log" | sed -E 's/^(Step [0-9]+) \| (Pass@1: [0-9.]+).*(Pass@16: [0-9.]+).*/  \1 | \2 | \3/' || true
done
echo "Logs: $LOGS"
