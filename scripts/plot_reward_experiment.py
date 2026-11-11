"""
Figure + tables for the GSM8K reward-design experiment (runs/math_rl.sh, scripts/gsm8k_behavior.py).

Reads the RL logs (rl_<reward>[-s<seed>].log) and the behavior samples (behavior/samples_*.jsonl) from a
math_rl directory, writes a PNG figure and prints markdown tables.

python -m scripts.plot_reward_experiment --logs=runs/math_rl --out=docs/figures/reward_experiment.png
"""

import os
import re
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--logs", type=str, default=os.path.expanduser("~/.cache/nanochat/math_rl"))
parser.add_argument("--out", type=str, default="docs/figures/reward_experiment.png")
parser.add_argument("--smooth", type=int, default=15, help="moving-average window over RL steps")
args = parser.parse_args()

STEP_RE = re.compile(r"Step (\d+)/\d+ \| Average reward: ([\d.eE+-]+) \| True \(strict\) reward: ([\d.]+) \| "
                     r"Average sequence length: ([\d.]+)")
RUNS = [("exact", "", "exact, seed 0"), ("exact", "-s1", "exact, seed 1"),
        ("contains", "", "contains, seed 0"), ("contains", "-s1", "contains, seed 1")]
COLOR = {"exact": "#2a78d6", "contains": "#eb6834"}   # categorical slots 1 and 2 of the reference palette
STYLE = {"": "-", "-s1": "--"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def read_steps(path):
    rows = [m.groups() for m in STEP_RE.finditer(open(path).read())]
    return [int(r[0]) for r in rows], [[float(r[i]) for r in rows] for i in (1, 2, 3)]


def smooth(xs, w):
    out, acc = [], 0.0
    for i, x in enumerate(xs):
        acc += x
        if i >= w:
            acc -= xs[i - w]
        out.append(acc / min(i + 1, w))
    return out


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    panels = [(0, "Optimized reward (what RL maximizes)", None),
              (1, "True reward (strict '#### answer' match)", None),
              (2, "Response length (tokens incl. prompt)", None)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=150)
    for ax, (k, title, _) in zip(axes, panels):
        for reward, sfx, label in RUNS:
            path = os.path.join(args.logs, f"rl_{reward}{sfx}.log")
            if not os.path.exists(path):
                continue
            steps, series = read_steps(path)
            ax.plot(steps, smooth(series[k], args.smooth), STYLE[sfx], color=COLOR[reward], lw=2,
                    label=label, solid_capstyle="round")
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("RL step")
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if k < 2:
            ax.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=4) # one row under the panels, clear of the data
    fig.suptitle(f"GSM8K GRPO from the same SFT model: exact vs naive 'contains' reward "
                 f"({args.smooth}-step moving average)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, facecolor="#fcfcfb")
    print(f"Wrote {args.out}")


def tables():
    beh = os.path.join(args.logs, "behavior")
    models = [("sft_d12-math", "SFT (start)"), ("rl_d12-math-exact", "RL exact, seed 0"),
              ("rl_d12-math-exact-s1", "RL exact, seed 1"), ("rl_d12-math-contains", "RL contains, seed 0"),
              ("rl_d12-math-contains-s1", "RL contains, seed 1")]
    cols = [("strict", "accuracy (strict)", True), ("contains", "naive reward fires", True),
            ("hacked", "naive reward on a wrong answer", True), ("has_final", "writes '####'", True),
            ("truncated", "hits token limit", True), ("tokens", "tokens", False),
            ("numbers", "numbers / answer", False), ("calc_calls", "calculator calls", False)]
    print("| model | " + " | ".join(c[1] for c in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for f, name in models:
        p = os.path.join(beh, f"samples_{f}.jsonl")
        if not os.path.exists(p):
            continue
        rs = [json.loads(l) for l in open(p)]
        cells = []
        for key, _, is_pct in cols:
            v = sum(r[key] for r in rs) / len(rs)
            cells.append(f"{100*v:.2f}%" if is_pct else f"{v:.1f}")
        print(f"| {name} | " + " | ".join(cells) + " |")
    print(f"\n(n = {len(rs)} GSM8K test problems, greedy, 256 new tokens)\n")
    print("| run | optimized reward, first → last 20 steps | true reward, first → last 20 steps |")
    print("|---|---|---|")
    for reward, sfx, label in RUNS:
        path = os.path.join(args.logs, f"rl_{reward}{sfx}.log")
        if not os.path.exists(path):
            continue
        _, (opt, true, _) = read_steps(path)
        avg = lambda xs: sum(xs) / len(xs)
        print(f"| {label} | {avg(opt[:20]):.3f} → {avg(opt[-20:]):.3f} | {avg(true[:20]):.3f} → {avg(true[-20:]):.3f} |")


if __name__ == "__main__":
    plot()
    tables()
