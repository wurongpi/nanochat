"""
Re-evaluate the published checkpoints with FIXED, more objective scoring, side by side with the
original scoring, with greedy or sampled decoding (unbiased pass@k), plus an EvalPlus export.

Scores computed on the SAME completions:
  orig    : the original pipeline (task.evaluate, pass = returncode==0) -> reproduces eval/benchmark.json
  fixed   : robust extraction (closed / unclosed-because-truncated / no block) + sanitize (keep imports, defs,
            classes, top-level assignments; drop prints, asserts, demo calls, `if __name__`, top-level exits;
            trim a half-written trailing line) + a pass sentinel printed AFTER the tests -> headline number
  hidden  : MBPP only, `fixed` scoring on challenge_test_list (tests absent from the prompt)
  EvalPlus: HumanEval+ reuses the HumanEval completions; MBPP+ (--tasks ...,MBPPPlus) gets its own
            generations. Both are exported as EvalPlus samples, then scored by `evalplus.evaluate`.
Diagnostics: truncated, unclosed/no block, salvaged-by-trimming, exit-style hacks, hard-coding suspects.

Decoding: --num-samples 1 --temperature 0 = greedy pass@1. --num-samples n --temperature T>0 = n samples per
problem decoded as one batch (shared prefill), unbiased pass@k (Chen et al. 2021) for k in --ks (k <= n).
Each row stops at <|assistant_end|> or as soon as its first ```python block is closed (scores unaffected).

Examples:
  python -m scripts.code_reeval --selfcheck                                   # harness check, no model
  python -m scripts.code_reeval --sources=sft,rl                              # greedy pass@1
  python -m scripts.code_reeval --sources=sft,rl --num-samples=10 --temperature=0.6   # pass@1, pass@5, pass@10
  python -m scripts.code_reeval --tasks=HumanEval,MBPP,MBPPPlus ...           # also generate for MBPP+
  # RL signal check: pass@8 of the SFT model on the RL training problems, with the RL rollout settings
  python -m scripts.code_reeval --sources=sft --tasks=MBPP --mbpp-split=rl_train --num-samples=8 --temperature=1.0 --ks=1,8
Outputs go to $NANOCHAT_BASE_DIR/eval_fixed/<T..._n..._tok...>/ ; see runs/code_reeval.sh for the full suite.
Scoring logic: tasks/code_scoring.py (tests: tests/test_code_scoring.py). EvalPlus: `uv sync --extra eval`.
"""

import os
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor

from nanochat.common import compute_init, compute_cleanup, autodetect_device_type, get_base_dir, print0
from tasks.humaneval import HumanEval
from tasks.mbpp import MBPP
from tasks.code_scoring import score_completion, pass_at_k, CLOSED_BLOCK_RE

parser = argparse.ArgumentParser()
parser.add_argument("--sources", type=str, default="sft,rl")
parser.add_argument("--tasks", type=str, default="HumanEval,MBPP", help="subset of HumanEval,MBPP,MBPPPlus")
parser.add_argument("--model-tag", type=str, default="d12")
parser.add_argument("--max-new-tokens", type=int, default=256)
parser.add_argument("--num-samples", type=int, default=1, help="samples per problem (1 = greedy)")
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--top-k", type=int, default=50)
parser.add_argument("--ks", type=str, default="1,5,10", help="pass@k values to report (only k <= num-samples)")
parser.add_argument("--max-problems", type=int, default=None)
parser.add_argument("--mbpp-split", type=str, default="test",
                    help="MBPP split: test (benchmark) | rl_train (the 474 RL training problems: is there any RL signal?)")
parser.add_argument("--workers", type=int, default=8, help="parallel code executions")
parser.add_argument("--no-early-stop", action="store_true", help="do not stop a row when its first code block closes")
parser.add_argument("--no-evalplus-export", action="store_true")
parser.add_argument("--out-name", type=str, default=None, help="output sub-directory (default: derived from the settings)")
parser.add_argument("--selfcheck", action="store_true", help="score the canonical solutions only (no model)")
parser.add_argument("--device-type", type=str, default="")
args = parser.parse_args()
if args.temperature == 0.0 and args.num_samples > 1:
    print0("temperature=0 is deterministic: forcing --num-samples=1")
    args.num_samples = 1
KS = [k for k in (int(x) for x in args.ks.split(",")) if k <= args.num_samples] or [1]

# -----------------------------------------------------------------------------
# scoring (all the logic lives in tasks/code_scoring.py)

def score(task_name, task, i, conv, completion):
    challenge = (task.rows[i].get("challenge_test_list") or None) if task_name == "MBPP" else None
    return score_completion(task_name, conv, completion, challenge_tests=challenge, timeout=getattr(task, "timeout", 5.0))

# -----------------------------------------------------------------------------
# tasks

class MBPPPlus:
    """MBPP+ prompts from EvalPlus, rendered in the same chat format as tasks/mbpp.py. Scored by EvalPlus."""
    def __init__(self):
        from evalplus.data import get_mbpp_plus
        self.items = sorted(get_mbpp_plus().items(), key=lambda kv: int(kv[0].split("/")[-1]))
    def __len__(self):
        return len(self.items)
    def task_id(self, i):
        return self.items[i][0]
    def __getitem__(self, i):
        prompt = self.items[i][1]["prompt"].strip().strip('"').strip()
        tests = [l.strip() for l in prompt.splitlines() if l.strip().startswith("assert")]
        text = "\n".join(l for l in prompt.splitlines() if not l.strip().startswith("assert")).strip()
        user = (f"{text}\nYour code should pass these tests:\n\n```python\n" + "\n".join(tests) +
                "\n```\n\nWrite the Python function inside a ```python code block.")
        return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": ""}]}

def build_task(name):
    if name == "HumanEval":
        return HumanEval()
    if name == "MBPP":
        return MBPP(split=args.mbpp_split)
    if name == "MBPPPlus":
        return MBPPPlus()
    raise ValueError(name)

def evalplus_task_id(task_name, task, i):
    if task_name == "HumanEval":
        return task.ds[i]["task_id"]          # e.g. HumanEval/17 (the dataset is shuffled, keep the real id)
    if task_name == "MBPPPlus":
        return task.task_id(i)                # e.g. Mbpp/2
    return None                               # original MBPP test split: not exported (use MBPPPlus)

tasks = {t.strip(): build_task(t.strip()) for t in args.tasks.split(",") if t.strip()}

if args.selfcheck:
    for tname, task in tasks.items():
        if tname == "MBPPPlus":
            print0("[selfcheck] MBPPPlus: scored by EvalPlus, skipped")
            continue
        n = len(task) if args.max_problems is None else min(len(task), args.max_problems)
        fails = []
        for i in range(n):
            conv = task[i]
            ref = conv["messages"][-1]["content"]
            if tname == "HumanEval":  # the reference is prompt + body; wrap it like a chat answer
                ref = f"```python\n{ref}\n```"
            s = score(tname, task, i, conv, ref)
            if not s["fixed"]:
                fails.append((i, s["error"]))
        print0(f"[selfcheck] {tname}: canonical solutions pass {n-len(fails)}/{n}")
        for i, e in fails[:10]:
            print0(f"    idx {i}: {e}")
    raise SystemExit(0)

# -----------------------------------------------------------------------------
# generation

import torch  # noqa: E402
from nanochat.checkpoint_manager import load_model  # noqa: E402
from nanochat.engine import Engine  # noqa: E402

def generate(engine, tokenizer, prompt_tokens, seed):
    """n rows decoded as one batch. A row ends on <|assistant_end|>/<|bos|> or when its first code block closes."""
    n = args.num_samples
    stop_ids = {tokenizer.encode_special("<|assistant_end|>"), tokenizer.get_bos_token_id()}
    outs, done = [[] for _ in range(n)], [False] * n
    for step, (col, _) in enumerate(engine.generate(prompt_tokens, num_samples=n, max_tokens=args.max_new_tokens,
                                                    temperature=args.temperature, top_k=args.top_k, seed=seed)):
        for r in range(n):
            if done[r]:
                continue
            if col[r] in stop_ids:
                done[r] = True
                continue
            outs[r].append(col[r])
            if not args.no_early_stop and step % 4 == 3 and CLOSED_BLOCK_RE.search(tokenizer.decode(outs[r])):
                done[r] = True
        if all(done):
            break
    results = []
    for r in range(n):
        text = tokenizer.decode(outs[r])
        if not done[r] and not args.no_early_stop and CLOSED_BLOCK_RE.search(text):
            done[r] = True
        results.append((text, len(outs[r]), (not done[r]) and len(outs[r]) >= args.max_new_tokens))
    return results

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
_, _, _, _, device = compute_init(device_type)
out_name = args.out_name or (f"T{args.temperature:g}_n{args.num_samples}_tok{args.max_new_tokens}"
                             + ("" if args.mbpp_split == "test" else f"_mbpp-{args.mbpp_split}"))
out_dir = os.path.join(get_base_dir(), "eval_fixed", out_name)
os.makedirs(out_dir, exist_ok=True)
pool = ThreadPoolExecutor(max_workers=args.workers)  # each execution is its own subprocess

RATE_KEYS = ["truncated", "closed_block", "unclosed_block", "no_block", "salvaged_by_trim", "dropped_toplevel",
             "exit_hack", "exit_hack_passed_orig", "hardcode", "hardcode_passed_fixed"]
summary = {"settings": vars(args), "ks": KS, "results": {}}
for source in [s.strip() for s in args.sources.split(",") if s.strip()]:
    print0(f"=== {source} ({out_name}) ===")
    model, tokenizer, _ = load_model(source, device, phase="eval", model_tag=args.model_tag)
    engine = Engine(model, tokenizer)
    summary["results"][source] = {}
    for tname, task in tasks.items():
        n_prob = len(task) if args.max_problems is None else min(len(task), args.max_problems)
        scored = tname != "MBPPPlus"
        agg = {k: 0 for k in RATE_KEYS}
        per_problem, errors, recs, evalplus_rows = [], {}, [], []
        gen_tokens, n_samples_total, t0 = 0, 0, time.time()
        for i in range(n_prob):
            conv = task[i]
            ptoks = tokenizer.render_for_completion(conv)
            gens = generate(engine, tokenizer, ptoks, seed=1234 + i)
            scores = list(pool.map(lambda g: score(tname, task, i, conv, g[0]), gens))
            origs = list(pool.map(lambda g: bool(task.evaluate(conv, g[0])), gens)) if scored else [None] * len(gens)
            c_orig = sum(bool(o) for o in origs)
            c_fixed = sum(bool(s["fixed"]) for s in scores)
            hid = [s["hidden"] for s in scores if s["hidden"] is not None]
            per_problem.append(dict(n=len(gens), c_orig=c_orig, c_fixed=c_fixed,
                                    c_hidden=(sum(hid) if hid else None)))
            tid = evalplus_task_id(tname, task, i)
            samples = []
            for (comp, ntok, trunc), s, o in zip(gens, scores, origs):
                gen_tokens += ntok; n_samples_total += 1
                agg["truncated"] += trunc; agg[s["extract"]] += 1
                agg["salvaged_by_trim"] += int(s["trimmed"] > 0 and bool(s["fixed"]))
                agg["dropped_toplevel"] += int(s["dropped"] > 0)
                agg["exit_hack"] += s["exit_hack"]; agg["exit_hack_passed_orig"] += int(s["exit_hack"] and bool(o))
                agg["hardcode"] += s["hardcode"]; agg["hardcode_passed_fixed"] += int(s["hardcode"] and bool(s["fixed"]))
                if scored and not s["fixed"]:
                    key = (s["error"] or "")[:60]
                    errors[key] = errors.get(key, 0) + 1
                if tid is not None:
                    evalplus_rows.append({"task_id": tid, "solution": s["solution"]})
                samples.append(dict(orig=o, fixed=s["fixed"], hidden=s["hidden"], truncated=trunc, gen_tokens=ntok,
                                    extract=s["extract"], exit_hack=s["exit_hack"], hardcode=s["hardcode"],
                                    error=s["error"], completion=comp[:1500]))
            recs.append(dict(idx=i, task_id=tid, c_orig=c_orig, c_fixed=c_fixed, samples=samples))
            if (i + 1) % 25 == 0 or i + 1 == n_prob:
                msg = f"  [{source}] {tname} {i+1}/{n_prob} | {time.time()-t0:.0f}s"
                if scored:
                    msg += (f" | pass@1 orig {100*sum(p['c_orig']/p['n'] for p in per_problem)/len(per_problem):.1f}%"
                            f" fixed {100*sum(p['c_fixed']/p['n'] for p in per_problem)/len(per_problem):.1f}%")
                print0(msg)
        res = {"problems": n_prob, "samples": n_samples_total, "avg_gen_tokens": round(gen_tokens / max(1, n_samples_total), 1),
               "elapsed_s": round(time.time() - t0)}
        if scored:
            for k in KS:
                res[f"orig_pass@{k}"] = sum(pass_at_k(p["n"], p["c_orig"], k) for p in per_problem) / n_prob
                res[f"fixed_pass@{k}"] = sum(pass_at_k(p["n"], p["c_fixed"], k) for p in per_problem) / n_prob
            hp = [p for p in per_problem if p["c_hidden"] is not None]
            if hp:
                res["hidden_problems"] = len(hp)
                for k in KS:
                    res[f"hidden_pass@{k}"] = sum(pass_at_k(p["n"], p["c_hidden"], k) for p in hp) / len(hp)
            res["top_errors"] = dict(sorted(errors.items(), key=lambda kv: -kv[1])[:8])
        res["per_sample_rates"] = {k: round(v / max(1, n_samples_total), 4) for k, v in agg.items()}
        summary["results"][source][tname] = res
        with open(os.path.join(out_dir, f"samples_{source}_{tname}.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if evalplus_rows and not args.no_evalplus_export:
            ep_name = "humaneval" if tname == "HumanEval" else "mbpp"
            ep_path = os.path.join(out_dir, f"evalplus_{source}_{ep_name}.jsonl")
            with open(ep_path, "w") as f:
                for r in evalplus_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            res["evalplus_samples"] = ep_path
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    del model, engine
    if device_type == "cuda":
        torch.cuda.empty_cache()
pool.shutdown()

pct = lambda x: f"{100*x:5.1f}%"
print0(f"\n==================== SUMMARY ({out_name}) ====================")
for s, tr in summary["results"].items():
    for t, r in tr.items():
        if f"fixed_pass@{KS[0]}" in r:
            line = " | ".join(f"pass@{k} orig {pct(r[f'orig_pass@{k}'])} FIXED {pct(r[f'fixed_pass@{k}'])}" for k in KS)
            if "hidden_problems" in r:
                line += " | hidden " + " ".join(f"@{k} {pct(r[f'hidden_pass@{k}'])}" for k in KS) + f" (n={r['hidden_problems']})"
            print0(f"{s:4s} {t:9s} {line}")
            rt = r["per_sample_rates"]
            print0(f"     per-sample: truncated {pct(rt['truncated'])} | unclosed {pct(rt['unclosed_block'])} | "
                   f"no-block {pct(rt['no_block'])} | exit-hack {pct(rt['exit_hack'])} | hardcode {pct(rt['hardcode'])} | "
                   f"avg tokens {r['avg_gen_tokens']} | {r['elapsed_s']}s")
            print0(f"     top errors: {r['top_errors']}")
        else:
            print0(f"{s:4s} {t:9s} generated {r['samples']} samples -> score with evalplus ({r['elapsed_s']}s)")
        if "evalplus_samples" in r:
            print0(f"     evalplus samples: {r['evalplus_samples']}")
print0(f"\nWrote {out_dir}/summary.json")
compute_cleanup()
