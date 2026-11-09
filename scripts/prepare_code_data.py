"""
Prepare the Code-CoT SFT dataset (Stage 0 of the Code-RLVR pipeline).

Sources (pick with --sources, comma separated):
  magicoder : ise-uiuc/Magicoder-OSS-Instruct-75K   (python rows only)
  stratos   : bespokelabs/Bespoke-Stratos-17k       (coding rows only; the long <|begin_of_thought|> part is dropped)

Cleaning / filtering rules:
  1. The response must contain a ```python code block with a single-function style solution (no classes by default).
  2. The free-text reasoning is cut down to <= 3 sentences and <= --max-reasoning-chars characters
     (the long-winded thinking is discarded). Output = short reasoning + ```python``` block.
  3. Rows whose rendered prompt + response exceeds --max-tokens tokens are dropped.
  4. Exact-duplicate prompts are dropped, then rows are shuffled (seed 42) and the first --num-samples are kept.

The tokenizer must already be trained (scripts.tok_train) because rule 3 counts real tokens.

Example:
python -m scripts.prepare_code_data --sources=magicoder,stratos --num-samples=20000
"""

import os
import re
import json
import random
import argparse

from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer
from tasks.common import load_hub_dataset
from tasks.code_sft import first_python_block, trim_reasoning, is_single_function_solution, format_response

parser = argparse.ArgumentParser(description="Prepare the Code-CoT SFT jsonl")
parser.add_argument("--sources", type=str, default="magicoder", help="comma separated: magicoder,stratos")
parser.add_argument("--num-samples", type=int, default=20000, help="number of rows to keep after filtering")
parser.add_argument("--max-tokens", type=int, default=1024, help="drop rows whose prompt+response exceeds this many tokens")
parser.add_argument("--max-reasoning-chars", type=int, default=200, help="max characters of free-text reasoning kept")
parser.add_argument("--max-reasoning-sentences", type=int, default=3, help="max sentences of free-text reasoning kept")
parser.add_argument("--allow-classes", type=int, default=0, help="keep solutions that define classes (0=no, 1=yes)")
parser.add_argument("--output", type=str, default=None, help="output jsonl (default: $NANOCHAT_BASE_DIR/code_sft/code_cot_sft.jsonl)")
args = parser.parse_args()

output_path = args.output or os.path.join(get_base_dir(), "code_sft", "code_cot_sft.jsonl")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
tokenizer = get_tokenizer()
stats = {}

def bump(key):
    stats[key] = stats.get(key, 0) + 1

def clean_pair(prompt, solution_text, source):
    """Apply rules 1-3. Returns a {"prompt","response"} dict or None."""
    code, before, after = first_python_block(solution_text)
    if code is None:
        bump(f"{source}:no_code_block"); return None
    if not is_single_function_solution(code, allow_classes=bool(args.allow_classes)):
        bump(f"{source}:not_single_function"); return None
    # prefer the analysis written before the code, fall back to the explanation after it
    reasoning = trim_reasoning(before, args.max_reasoning_chars, args.max_reasoning_sentences)
    if not reasoning:
        reasoning = trim_reasoning(after, args.max_reasoning_chars, args.max_reasoning_sentences)
    response = format_response(reasoning, code)
    prompt = prompt.strip()
    conversation = {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]}
    ids, _ = tokenizer.render_conversation(conversation, max_tokens=10**9)
    if len(ids) > args.max_tokens:
        bump(f"{source}:too_long"); return None
    bump(f"{source}:kept")
    return {"prompt": prompt, "response": response}

def iter_magicoder():
    ds = load_hub_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", subset="default", split="train")
    for i in range(len(ds)):
        row = ds[i]
        if (row.get("lang") or "").lower() != "python":
            bump("magicoder:not_python"); continue
        yield clean_pair(row["problem"], row["solution"], "magicoder")

STRATOS_THOUGHT_RE = re.compile(r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>", re.DOTALL)
def iter_stratos():
    ds = load_hub_dataset("bespokelabs/Bespoke-Stratos-17k", subset="default", split="train")
    for i in range(len(ds)):
        row = ds[i]
        turns = row["conversations"]
        user = next((t["value"] for t in turns if t["from"] == "user"), None)
        assistant = next((t["value"] for t in turns if t["from"] == "assistant"), None)
        if not user or not assistant:
            continue
        solution = STRATOS_THOUGHT_RE.sub("", assistant)
        solution = solution.replace("<|begin_of_solution|>", "").replace("<|end_of_solution|>", "").strip()
        yield clean_pair(user, solution, "stratos")

SOURCES = {"magicoder": iter_magicoder, "stratos": iter_stratos}

rows, seen = [], set()
for name in [s.strip() for s in args.sources.split(",") if s.strip()]:
    assert name in SOURCES, f"Unknown source {name}, choose from {list(SOURCES)}"
    print(f"Processing source: {name}")
    for item in SOURCES[name]():
        if item is None:
            continue
        key = item["prompt"]
        if key in seen:
            bump(f"{name}:duplicate"); continue
        seen.add(key)
        rows.append(item)

random.Random(42).shuffle(rows)
rows = rows[:args.num_samples]
with open(output_path, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("Filter statistics:")
for k in sorted(stats):
    print(f"  {k}: {stats[k]:,}")
print(f"Wrote {len(rows):,} rows to {output_path}")
if len(rows) < args.num_samples:
    print(f"NOTE: only {len(rows):,} rows survived the filters (< --num-samples={args.num_samples}). "
          f"Consider adding --sources=magicoder,stratos or relaxing --allow-classes / --max-reasoning-chars.")
