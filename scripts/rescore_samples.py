"""
Re-score the generations that scripts/code_eval.py saved (eval/samples_<stage>_<task>.jsonl, the ones behind the
published benchmark.json) with the fixed grader in tasks/code_scoring.py. CPU only, no model needed.

The original numbers are recomputed from the saved `passed` flags, so the table shows both graders on exactly the
same completions. Only the chat stages (sft, rl) are re-scored: base used completion-style prompting.

python -m scripts.rescore_samples --eval-dir=$NANOCHAT_BASE_DIR/eval
"""

import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor

from tasks.humaneval import HumanEval
from tasks.mbpp import MBPP
from tasks.code_scoring import score_completion

parser = argparse.ArgumentParser()
parser.add_argument("--eval-dir", type=str, required=True, help="directory with samples_<stage>_<task>.jsonl")
parser.add_argument("--stages", type=str, default="sft,rl")
parser.add_argument("--workers", type=int, default=8)
args = parser.parse_args()

tasks = {"HumanEval": HumanEval(), "MBPP": MBPP(split="test")}
pool = ThreadPoolExecutor(max_workers=args.workers)
summary = {}
for stage in args.stages.split(","):
    for tname, task in tasks.items():
        path = os.path.join(args.eval_dir, f"samples_{stage}_{tname}.jsonl")
        if not os.path.exists(path):
            print(f"missing {path}, skipped")
            continue
        recs = [json.loads(l) for l in open(path)]

        def rescore(r):
            conv = task[r["idx"]] # same task object and order as code_eval (HumanEval is shuffled with seed 42)
            challenge = (task.rows[r["idx"]].get("challenge_test_list") or None) if tname == "MBPP" else None
            return score_completion(tname, conv, r["completion"], challenge_tests=challenge,
                                    timeout=getattr(task, "timeout", 5.0))

        scores = list(pool.map(rescore, recs))
        n = len(recs)
        hidden = [s["hidden"] for s in scores if s["hidden"] is not None]
        row = {
            "n": n,
            "orig": sum(bool(r["passed"]) for r in recs) / n,
            "fixed": sum(bool(s["fixed"]) for s in scores) / n,
            "exit_hack": sum(s["exit_hack"] for s in scores) / n,
            "exit_hack_and_orig_pass": sum(s["exit_hack"] and bool(r["passed"]) for s, r in zip(scores, recs)) / n,
            "hardcode": sum(s["hardcode"] for s in scores) / n,
            "unclosed_block": sum(s["extract"] == "unclosed_block" for s in scores) / n,
            "hidden": (sum(hidden) / len(hidden)) if hidden else None,
            "hidden_n": len(hidden),
        }
        summary[f"{stage}/{tname}"] = row
        with open(os.path.join(args.eval_dir, f"rescored_{stage}_{tname}.jsonl"), "w") as f:
            for r, s in zip(recs, scores):
                f.write(json.dumps({"idx": r["idx"], "orig": bool(r["passed"]), **{k: s[k] for k in
                        ("fixed", "hidden", "exit_hack", "hardcode", "extract", "error")}}) + "\n")
pool.shutdown()

with open(os.path.join(args.eval_dir, "rescored_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
pct = lambda x: "n/a" if x is None else f"{100*x:.1f}%"
print("| stage | benchmark | n | original grader | fixed grader | exit call | exit call and passed original | "
      "hard-coding suspect | unclosed code block | MBPP hidden tests |")
print("|---|---|---|---|---|---|---|---|---|---|")
for key, r in summary.items():
    stage, tname = key.split("/")
    hid = f"{pct(r['hidden'])} (n={r['hidden_n']})" if r["hidden_n"] else "n/a"
    print(f"| {stage} | {tname} | {r['n']} | {pct(r['orig'])} | {pct(r['fixed'])} | {pct(r['exit_hack'])} | "
          f"{pct(r['exit_hack_and_orig_pass'])} | {pct(r['hardcode'])} | {pct(r['unclosed_block'])} | {hid} |")
