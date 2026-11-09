#!/bin/bash

# Code-RLVR on a single GPU: pretrain from scratch -> Code-CoT SFT -> GRPO on MBPP unit tests,
# then benchmark every stage (HumanEval + MBPP) and publish to HuggingFace / ModelScope.
# Takes ~6-8 hours for d12 on one A800.

# Example launch:
# screen -L -Logfile runs/code_rlvr.log -S code_rlvr bash runs/code_rlvr.sh
#
# Environment knobs:
# NANOCHAT_BASE_DIR=...  data, checkpoints and evals (~35GB), e.g. /root/autodl-tmp/nanochat on AutoDL
# UV_CACHE_DIR=...       uv's download cache (~7GB), keep it off a small system disk too
# GPU=a800|4090          sets the per-device batch sizes (default a800)
# START_STAGE=<stage>    resume from setup|tok|data|base|sft|rl|eval|upload
# HF_ENDPOINT=...        hub mirror for downloads, e.g. https://hf-mirror.com
# HF_REPO_ID, MS_REPO_ID publish to HuggingFace ($HF_TOKEN) and/or ModelScope ($MODELSCOPE_API_TOKEN)
# HUB_PRIVATE=1          create private repos
# CODE_URL=...           code repo linked from the model card
# WANDB_RUN=...          enable wandb logging

set -euo pipefail
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

GPU="${GPU:-a800}"
START_STAGE="${START_STAGE:-setup}"
WANDB_RUN="${WANDB_RUN:-dummy}"
CODE_URL="${CODE_URL:-https://github.com/karpathy/nanochat}"
DEPTH=12
TAG="d$DEPTH"
EVAL_DIR="$NANOCHAT_BASE_DIR/eval"

case "$GPU" in
    a800) BASE_DBS=16; SFT_DBS=16; RL_DBS=8 ;;
    4090) BASE_DBS=4;  SFT_DBS=4;  RL_DBS=4 ;; # RL: each group of 8 rollouts runs in 2 passes
    *) echo "unknown GPU=$GPU (a800 or 4090)"; exit 1 ;;
esac

STAGES=(setup tok data base sft rl eval upload)
[[ " ${STAGES[*]} " == *" $START_STAGE "* ]] || { echo "unknown START_STAGE=$START_STAGE"; exit 1; }
run_stage() { # true if $1 is at or after START_STAGE
    local s
    for s in "${STAGES[@]}"; do
        [ "$s" = "$START_STAGE" ] && return 0
        [ "$s" = "$1" ] && return 1
    done
}

publish() { # publish <sources> <message>; a failed upload never stops the run
    local hubs=() private=()
    [ "${HUB_PRIVATE:-0}" = "1" ] && private=(--private)
    [ -n "${MS_REPO_ID:-}" ] && hubs+=(modelscope)
    [ -n "${HF_REPO_ID:-}" ] && hubs+=(hf)
    [ ${#hubs[@]} -eq 0 ] && return 0
    python -m scripts.hub_upload --hub="$(IFS=,; echo "${hubs[*]}")" --sources="$1" --model-tag=$TAG \
        --eval-dir="$EVAL_DIR" --code-url="$CODE_URL" --commit-message="$2" \
        --export-on-failure="$NANOCHAT_BASE_DIR/hub_export" "${private[@]}" || true
}

# -----------------------------------------------------------------------------
# Python venv (uv from PyPI: its standalone installer and managed Pythons come from GitHub)

if run_stage setup; then
    command -v uv &> /dev/null || pip install uv
    [ -d ".venv" ] || uv venv
    uv sync --extra gpu --extra hub
fi
source .venv/bin/activate

# Fetch the FA3 kernel once (hub mirrors rate-limit), then let every process load it from the cache
python -m nanochat.flash_attention
export HF_HUB_OFFLINE=1

# -----------------------------------------------------------------------------
# Tokenizer and data

if run_stage tok; then
    python -m nanochat.dataset -n 8
    python -m nanochat.dataset -n 64 & # ~2.5B tokens of pretraining data plus padding
    DATASET_DOWNLOAD_PID=$!
    python -m scripts.tok_train --vocab-size=65536
    python -m scripts.tok_eval
    wait $DATASET_DOWNLOAD_PID
fi

if run_stage data; then
    python -m scripts.prepare_code_data --sources=magicoder,stratos --num-samples=20000
fi

# -----------------------------------------------------------------------------
# Pretraining: 4800 x 524288 tokens ~= 2.5B tokens

if run_stage base; then
    python -m scripts.base_train \
        --depth=$DEPTH \
        --window-pattern=auto \
        --device-batch-size=$BASE_DBS \
        --total-batch-size=524288 \
        --num-iterations=4800 \
        --core-metric-every=-1 \
        --save-every=1000 \
        --model-tag=$TAG \
        --run=$WANDB_RUN
fi

# -----------------------------------------------------------------------------
# Code-CoT SFT, then compare against the base model and publish the first results

if run_stage sft; then
    python -m scripts.chat_sft \
        --mix=code \
        --code-epochs=3 \
        --model-tag=$TAG \
        --max-seq-len=1024 \
        --device-batch-size=$SFT_DBS \
        --total-batch-size=65536 \
        --eval-every=50 \
        --eval-tokens=1048576 \
        --chatcore-every=200 \
        --chatcore-max-sample=64 \
        --run=$WANDB_RUN
    python -m scripts.code_eval --sources=base,sft --model-tag=$TAG --output-json="$EVAL_DIR/benchmark.json"
    python -m scripts.plot_benchmarks --input="$EVAL_DIR/benchmark.json"
    publish base,sft "base + SFT checkpoints and benchmarks"
fi

# -----------------------------------------------------------------------------
# Code-RLVR: GRPO with binary unit-test rewards on MBPP (group size 8)

if run_stage rl; then
    python -m scripts.chat_rl \
        --task=mbpp \
        --model-tag=$TAG \
        --num-samples=8 \
        --device-batch-size=$RL_DBS \
        --examples-per-step=4 \
        --num-steps=1000 \
        --adv-norm=zscore \
        --max-new-tokens=256 \
        --temperature=0.8 \
        --eval-every=100 \
        --eval-examples=200 \
        --save-every=200 \
        --run=$WANDB_RUN
fi

# -----------------------------------------------------------------------------
# Benchmark all stages (base/sft results are reused) and publish

if run_stage eval; then
    python -m scripts.code_eval --sources=base,sft,rl --model-tag=$TAG --output-json="$EVAL_DIR/benchmark.json"
    python -m scripts.plot_benchmarks --input="$EVAL_DIR/benchmark.json"
fi

if run_stage upload; then
    publish base,sft,rl "base + SFT + RL checkpoints and benchmarks"
fi
