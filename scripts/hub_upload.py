"""
Publish trained checkpoints, the tokenizer and the benchmark comparison to the
HuggingFace Hub and/or ModelScope.

The repo mirrors the $NANOCHAT_BASE_DIR layout, so a download into ~/.cache/nanochat
is directly usable by chat_cli / chat_eval / code_eval.

Credentials: $HF_TOKEN (or `huggingface-cli login`) for HuggingFace,
$MODELSCOPE_API_TOKEN for ModelScope.

Examples:
python -m scripts.hub_upload --hub=modelscope --ms-repo-id=me/nanochat-d12-code --sources=base,sft,rl
python -m scripts.hub_upload --hub=hf,modelscope --hf-repo-id=me/nanochat-d12-code --ms-repo-id=me/nanochat-d12-code
python -m scripts.hub_upload --hub=hf --hf-repo-id=me/nanochat-d12-code --export-dir=./hub_export  # upload later, elsewhere
"""

import os
import json
import shutil
import argparse
import tempfile

# uploads always need the network, even when training ran with a cached, offline hub
os.environ.pop("HF_HUB_OFFLINE", None)

from nanochat.common import get_base_dir
from nanochat.checkpoint_manager import find_largest_model, find_last_step

SOURCE_DIRS = {"base": "base_checkpoints", "sft": "chatsft_checkpoints", "rl": "chatrl_checkpoints"}
EVAL_FILES = ("benchmark.json", "comparison.png", "comparison.svg", "comparison.md")

parser = argparse.ArgumentParser(description="Publish nanochat checkpoints to HuggingFace and/or ModelScope")
parser.add_argument("--hub", type=str, default="hf", help="comma separated: hf, modelscope")
parser.add_argument("--hf-repo-id", type=str, default=os.environ.get("HF_REPO_ID"), help="e.g. me/nanochat-d12-code")
parser.add_argument("--ms-repo-id", type=str, default=os.environ.get("MS_REPO_ID"), help="e.g. me/nanochat-d12-code")
parser.add_argument("--hf-endpoint", type=str, default="https://huggingface.co", help="uploads must go to the real hub, not a read-only mirror")
parser.add_argument("--sources", type=str, default="rl", help="comma separated subset of base,sft,rl")
parser.add_argument("--model-tag", type=str, default=None, help="model tag (default: largest model)")
parser.add_argument("--step", type=int, default=None, help="checkpoint step (default: last)")
parser.add_argument("--safetensors", type=int, default=1, help="also export model_<step>.safetensors")
parser.add_argument("--eval-dir", type=str, default=None, help="output of code_eval + plot_benchmarks (default: $NANOCHAT_BASE_DIR/eval)")
parser.add_argument("--code-url", type=str, default="https://github.com/karpathy/nanochat", help="code repo shown in the model card")
parser.add_argument("--private", action="store_true")
parser.add_argument("--commit-message", type=str, default="Upload nanochat checkpoints")
parser.add_argument("--export-dir", type=str, default=None, help="only write the repo contents here, don't upload")
parser.add_argument("--export-on-failure", type=str, default=None, help="if any upload fails, write the repo contents here")
args = parser.parse_args()

hubs = [h.strip() for h in args.hub.split(",") if h.strip()]
assert hubs and set(hubs) <= {"hf", "modelscope"}, f"--hub must be hf and/or modelscope, got {args.hub}"
assert "hf" not in hubs or args.hf_repo_id, "--hub=hf needs --hf-repo-id (or $HF_REPO_ID)"
assert "modelscope" not in hubs or args.ms_repo_id, "--hub=modelscope needs --ms-repo-id (or $MS_REPO_ID)"
repo_ids = {"hf": args.hf_repo_id, "modelscope": args.ms_repo_id}
sources = [s.strip() for s in args.sources.split(",") if s.strip()]
assert set(sources) <= set(SOURCE_DIRS), f"--sources must be a subset of {list(SOURCE_DIRS)}"
base_dir = get_base_dir()

# -----------------------------------------------------------------------------
# Stage the repo contents in a temp dir (hardlinks where possible, it's fast and free)

staging = tempfile.mkdtemp(prefix="nanochat_hub_")

def stage(src, rel):
    dst = os.path.join(staging, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)

for name in ("tokenizer.pkl", "token_bytes.pt"):
    stage(os.path.join(base_dir, "tokenizer", name), os.path.join("tokenizer", name))

checkpoints = []
for source in sources:
    checkpoints_dir = os.path.join(base_dir, SOURCE_DIRS[source])
    tag = args.model_tag or find_largest_model(checkpoints_dir)
    ckpt_dir = os.path.join(checkpoints_dir, tag)
    step = args.step if args.step is not None else find_last_step(ckpt_dir)
    rel_dir = os.path.join(SOURCE_DIRS[source], tag)
    for name in (f"model_{step:06d}.pt", f"meta_{step:06d}.json"):
        stage(os.path.join(ckpt_dir, name), os.path.join(rel_dir, name))
    if args.safetensors:
        import torch
        from safetensors.torch import save_file
        state = torch.load(os.path.join(ckpt_dir, f"model_{step:06d}.pt"), map_location="cpu")
        state = {k.removeprefix("_orig_mod."): v.contiguous() for k, v in state.items()}
        save_file(state, os.path.join(staging, rel_dir, f"model_{step:06d}.safetensors"))
    with open(os.path.join(ckpt_dir, f"meta_{step:06d}.json")) as f:
        model_config = json.load(f).get("model_config", {})
    checkpoints.append({"source": source, "tag": tag, "step": step, "dir": rel_dir, "model_config": model_config})
    print(f"staged {source}: {rel_dir} (step {step})")

eval_dir = args.eval_dir or os.path.join(base_dir, "eval")
eval_files = []
if os.path.isdir(eval_dir):
    for name in sorted(os.listdir(eval_dir)):
        if name in EVAL_FILES or (name.startswith("samples_") and name.endswith(".jsonl")):
            stage(os.path.join(eval_dir, name), os.path.join("eval", name))
            eval_files.append(name)

with open(os.path.join(staging, "nanochat_manifest.json"), "w") as f:
    json.dump({"checkpoints": checkpoints}, f, indent=2)
with open(os.path.join(staging, "configuration.json"), "w") as f: # ModelScope convention
    json.dump({"framework": "pytorch", "task": "text-generation"}, f, indent=2)

# -----------------------------------------------------------------------------
# Model card (front matter and download command differ per hub)

final = checkpoints[-1]
cfg = final["model_config"]
chat_source = final["source"] if final["source"] != "base" else "sft"

if "comparison.md" in eval_files:
    with open(os.path.join(eval_dir, "comparison.md"), encoding="utf-8") as f:
        evaluation = f.read().strip()
    chart = next((f"eval/{c}" for c in ("comparison.png", "comparison.svg") if c in eval_files), None)
    if chart:
        evaluation = f"![Benchmarks by training stage]({chart})\n\n{evaluation}"
    evaluation += "\n\nRaw numbers: `eval/benchmark.json`, per-problem completions: `eval/samples_<stage>_<task>.jsonl`."
else:
    evaluation = "_No evaluation results attached._"

FRONT_MATTER = {
    "hf": "---\nlicense: mit\nlibrary_name: nanochat\npipeline_tag: text-generation\ntags:\n- nanochat\n- code\n- grpo\n---",
    "modelscope": "---\nlicense: MIT\nfrontmatter_version: 1\nframeworks:\n- PyTorch\ntasks:\n- text-generation\ntags:\n- nanochat\n- code\n- grpo\n---",
}

def model_card(hub):
    repo_id = repo_ids[hub]
    if hub == "hf":
        download = f"huggingface-cli download {repo_id} --local-dir ~/.cache/nanochat"
    else:
        download = f"modelscope download --model {repo_id} --local_dir ~/.cache/nanochat"
    rows = "\n".join(f"| {c['source']} | `{c['dir']}` | {c['step']} |" for c in checkpoints)
    return f"""{FRONT_MATTER[hub]}

# {repo_id.split('/')[-1]}

A small GPT trained from scratch with [nanochat]({args.code_url}):
pretraining -> Code-CoT SFT -> Code-RLVR (GRPO with unit-test rewards on MBPP).

| depth | model dim | heads | context | vocab |
|---|---|---|---|---|
| {cfg.get('n_layer')} | {cfg.get('n_embd')} | {cfg.get('n_head')} | {cfg.get('sequence_len')} | {cfg.get('vocab_size')} |

## Evaluation by training stage

{evaluation}

## Checkpoints

| stage | path | step |
|---|---|---|
{rows}

Each checkpoint has `model_<step>.pt` (nanochat state dict), `meta_<step>.json` (config and training args)
and, if exported, `model_<step>.safetensors`. The tokenizer is in `tokenizer/`.

## Usage

These are nanochat-native checkpoints (not a `transformers` model):

```bash
git clone {args.code_url} nanochat && cd nanochat
uv sync --extra gpu && source .venv/bin/activate
{download}
python -m scripts.chat_cli -i {chat_source} -g {final['tag']} -p "Write a python function that reverses a string."
python -m scripts.code_eval --sources={','.join(c['source'] for c in checkpoints)} --model-tag={final['tag']}
```
"""

def write_card(hub):
    with open(os.path.join(staging, "README.md"), "w", encoding="utf-8") as f:
        f.write(model_card(hub))

def export(dest, hub):
    write_card(hub)
    shutil.copytree(staging, dest, dirs_exist_ok=True)
    print(f"Exported the {hub} repo contents to {dest}, upload it from a machine with access:")
    if hub == "hf":
        print(f"  huggingface-cli upload {repo_ids[hub]} {dest} . --repo-type=model")
    else:
        print(f"  modelscope upload {repo_ids[hub]} {dest}")

# -----------------------------------------------------------------------------
# Upload

def upload_hf():
    from huggingface_hub import HfApi
    api = HfApi(endpoint=args.hf_endpoint, token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id=args.hf_repo_id, private=args.private, exist_ok=True)
    api.upload_folder(repo_id=args.hf_repo_id, folder_path=staging, commit_message=args.commit_message)
    print(f"Uploaded to {args.hf_endpoint}/{args.hf_repo_id}")

def upload_modelscope():
    from modelscope.hub.api import HubApi
    from modelscope.hub.constants import ModelVisibility
    token = os.environ.get("MODELSCOPE_API_TOKEN")
    assert token, "set $MODELSCOPE_API_TOKEN (modelscope.cn -> access tokens)"
    api = HubApi()
    api.login(token)
    try:
        api.get_model(args.ms_repo_id)
    except Exception:
        visibility = ModelVisibility.PRIVATE if args.private else ModelVisibility.PUBLIC
        api.create_model(model_id=args.ms_repo_id, visibility=visibility, license="MIT")
    api.upload_folder(repo_id=args.ms_repo_id, folder_path=staging, commit_message=args.commit_message)
    print(f"Uploaded to https://modelscope.cn/models/{args.ms_repo_id}")

if args.export_dir:
    export(args.export_dir, hubs[0])
    shutil.rmtree(staging, ignore_errors=True)
    raise SystemExit(0)

failed = []
for hub in hubs:
    write_card(hub)
    try:
        upload_hf() if hub == "hf" else upload_modelscope()
    except Exception as e:
        print(f"Upload to {hub} failed: {type(e).__name__}: {e}")
        failed.append(hub)

if failed and args.export_on_failure:
    export(args.export_on_failure, failed[0])
shutil.rmtree(staging, ignore_errors=True)
if failed:
    raise SystemExit(f"upload failed for: {', '.join(failed)}")
