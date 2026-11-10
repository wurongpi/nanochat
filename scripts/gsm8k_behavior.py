"""
Behavior statistics of the GSM8K models from the reward-design experiment (runs/math_rl.sh).

Greedy generation on the held-out GSM8K test split, then per model:
  strict      : the final "#### answer" matches (= chat_eval's GSM8K accuracy, a sanity check)
  contains    : the naive reward fires (the reference number appears anywhere)
  hacked      : contains but not strict (the naive reward paid for a wrong answer)
  has_final   : the response writes a "#### answer" at all
  truncated   : the response hit the token budget without <|assistant_end|>
  tokens, numbers, distinct_numbers, calc_calls : averages per response

python -m scripts.gsm8k_behavior --models=sft:d12-math,rl:d12-math-exact,rl:d12-math-contains
Outputs: $NANOCHAT_BASE_DIR/math_rl/behavior/{summary.json,samples_<source>_<tag>.jsonl}
"""

import os
import json
import time
import argparse

from nanochat.common import compute_init, compute_cleanup, autodetect_device_type, get_base_dir, print0
from tasks.gsm8k import GSM8K, NUM_RE

parser = argparse.ArgumentParser()
parser.add_argument("--models", type=str, default="sft:d12-math,rl:d12-math-exact,rl:d12-math-contains",
                    help="comma separated source:model_tag pairs")
parser.add_argument("--max-problems", type=int, default=None)
parser.add_argument("--max-new-tokens", type=int, default=256, help="same budget as chat_eval in runs/math_rl.sh")
parser.add_argument("--device-type", type=str, default="")
args = parser.parse_args()

KEYS = ["strict", "contains", "hacked", "has_final", "truncated", "tokens", "numbers", "distinct_numbers", "calc_calls"]


def analyze(task, conversation, text, n_tokens, ended):
    numbers = NUM_RE.findall(text)
    strict = int(task.evaluate(conversation, text))
    contains = int(task.reward_contains(conversation, text))
    return dict(
        strict=strict,
        contains=contains,
        hacked=int(contains and not strict),
        has_final=int("####" in text),
        truncated=int(not ended),
        tokens=n_tokens,
        numbers=len(numbers),
        distinct_numbers=len({n.replace(",", "").rstrip(".") for n in numbers}),
        calc_calls=text.count("<|python_start|>"),
    )


def generate(engine, tokenizer, prompt_tokens, seed):
    """Greedy, one row. Returns (text, n_tokens, ended): ended=False means the budget ran out."""
    stop_ids = {tokenizer.encode_special("<|assistant_end|>"), tokenizer.get_bos_token_id()}
    out, ended = [], False
    for col, _ in engine.generate(prompt_tokens, num_samples=1, max_tokens=args.max_new_tokens,
                                  temperature=0.0, top_k=50, seed=seed):
        if col[0] in stop_ids:
            ended = True
            break
        out.append(col[0])
    return tokenizer.decode(out), len(out), ended


if __name__ == "__main__":
    from nanochat.checkpoint_manager import load_model
    from nanochat.engine import Engine

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _, _, _, device = compute_init(device_type)
    out_dir = os.path.join(get_base_dir(), "math_rl", "behavior")
    os.makedirs(out_dir, exist_ok=True)

    task = GSM8K(subset="main", split="test")
    n = len(task) if args.max_problems is None else min(len(task), args.max_problems)
    summary = {"settings": vars(args), "problems": n, "results": {}}
    for spec in [s.strip() for s in args.models.split(",") if s.strip()]:
        source, tag = spec.split(":")
        print0(f"=== {source}:{tag} ===")
        model, tokenizer, _ = load_model(source, device, phase="eval", model_tag=tag)
        engine = Engine(model, tokenizer)
        recs, t0 = [], time.time()
        for i in range(n):
            conv = task[i]
            text, ntok, ended = generate(engine, tokenizer, tokenizer.render_for_completion(conv), seed=1234 + i)
            r = analyze(task, conv, text, ntok, ended)
            r.update(idx=i, completion=text[:2000])
            recs.append(r)
            if (i + 1) % 100 == 0 or i + 1 == n:
                print0(f"  {i+1}/{n} | strict {100*sum(x['strict'] for x in recs)/len(recs):.2f}% | "
                       f"hacked {100*sum(x['hacked'] for x in recs)/len(recs):.2f}% | {time.time()-t0:.0f}s")
        summary["results"][f"{source}:{tag}"] = {k: sum(r[k] for r in recs) / n for k in KEYS}
        with open(os.path.join(out_dir, f"samples_{source}_{tag}.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        del model, engine

    pct = lambda x: f"{100*x:6.2f}%"
    print0(f"\n==================== BEHAVIOR (GSM8K test, greedy, n={n}) ====================")
    print0(f"{'model':28s} {'strict':>8s} {'contains':>9s} {'hacked':>8s} {'has####':>8s} {'trunc':>8s} "
           f"{'tokens':>7s} {'numbers':>8s} {'distinct':>9s} {'calc':>6s}")
    for name, a in summary["results"].items():
        print0(f"{name:28s} {pct(a['strict'])} {pct(a['contains']):>9s} {pct(a['hacked'])} {pct(a['has_final'])} "
               f"{pct(a['truncated'])} {a['tokens']:7.1f} {a['numbers']:8.1f} {a['distinct_numbers']:9.1f} {a['calc_calls']:6.1f}")
    print0(f"\nWrote {out_dir}/summary.json")
    compute_cleanup()
