"""
Turn eval/benchmark.json (written by scripts.code_eval) into an at-a-glance comparison:
  eval/comparison.svg  grouped bar chart: one group per benchmark, one bar per stage (base / sft / rl),
                       each bar labeled with its pass@1 and the change vs. the previous stage
  eval/comparison.png  the same chart rendered with matplotlib (only if matplotlib is installed)
  eval/comparison.md   markdown table with absolute scores and deltas (embedded in the HF model card)

python -m scripts.plot_benchmarks [--input path/to/benchmark.json]
"""

import os
import json
import argparse
from nanochat.common import get_base_dir

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default=None, help="default: $NANOCHAT_BASE_DIR/eval/benchmark.json")
parser.add_argument("--outdir", type=str, default=None, help="default: same folder as --input")
args = parser.parse_args()

in_path = args.input or os.path.join(get_base_dir(), "eval", "benchmark.json")
outdir = args.outdir or os.path.dirname(os.path.abspath(in_path))
with open(in_path) as f:
    data = json.load(f)
results = data["results"]

STAGE_ORDER = ["base", "sft", "rl"]
STAGE_LABEL = {"base": "Base (before SFT)", "sft": "After Code-CoT SFT", "rl": "After Code-RLVR (GRPO)"}
# categorical slots 1-3 of the validated reference palette (fixed order, follows the stage, never the rank)
STAGE_COLOR = {"base": "#2a78d6", "sft": "#eb6834", "rl": "#1baf7a"}
TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#7a7974", "#e6e5e1", "#fcfcfb"

stages = [s for s in STAGE_ORDER if s in results]
tasks = []
for s in stages:
    for t in results[s]:
        if t not in tasks:
            tasks.append(t)
assert stages and tasks, f"no results in {in_path}"

def score(stage, task):
    r = results.get(stage, {}).get(task)
    return None if r is None else 100.0 * r["pass@1"]

def delta_vs_prev(stage, task):
    i = stages.index(stage)
    if i == 0:
        return None
    cur, prev = score(stage, task), score(stages[i - 1], task)
    return None if cur is None or prev is None else cur - prev

# --------------------------------------------------------------------------- markdown table
lines = ["| Benchmark | " + " | ".join(STAGE_LABEL[s] for s in stages) + (" | Total gain |" if len(stages) > 1 else " |"),
         "|---|" + "---|" * len(stages) + ("---|" if len(stages) > 1 else "")]
for t in tasks:
    cells = []
    for s in stages:
        v, d = score(s, t), delta_vs_prev(s, t)
        if v is None:
            cells.append("–"); continue
        r = results[s][t]
        cell = f"**{v:.1f}%** ({r['passed']}/{r['total']})"
        if d is not None:
            cell += f" {'🔺' if d > 0 else ('🔻' if d < 0 else '▪️')} {d:+.1f}pp"
        cells.append(cell)
    row = f"| {t} | " + " | ".join(cells) + " |"
    if len(stages) > 1:
        first, last = score(stages[0], t), score(stages[-1], t)
        row += f" **{last - first:+.1f}pp** |" if first is not None and last is not None else " – |"
    lines.append(row)
settings = data.get("settings", {})
lines.append("")
lines.append(f"_pass@1, temperature={settings.get('temperature', 0.0)}, max_new_tokens={settings.get('max_new_tokens')}. "
             f"Base model is prompted as plain code completion, chat models with the chat template. "
             f"pp = percentage points vs. the previous stage._")
table_md = "\n".join(lines) + "\n"
with open(os.path.join(outdir, "comparison.md"), "w", encoding="utf-8") as f:
    f.write(table_md)

# --------------------------------------------------------------------------- SVG (dependency free)
max_v = max([score(s, t) or 0 for s in stages for t in tasks] + [1.0])
y_max = min(100.0, max(10.0, (int(max_v * 1.25) // 5 + 1) * 5))   # headroom for labels, round to 5
W, H = 760, 440
left, right, top, bottom = 64, 24, 92, 56
pw, ph = W - left - right, H - top - bottom
group_w = pw / len(tasks)
bar_w = min(64, group_w * 0.7 / len(stages))
gap = 2  # surface gap between adjacent bars
y = lambda v: top + ph * (1 - v / y_max)

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif">',
       f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
       f'<text x="{left}" y="30" font-size="18" font-weight="600" fill="{TEXT_PRIMARY}">Coding benchmarks by training stage (pass@1)</text>',
       f'<text x="{left}" y="52" font-size="13" fill="{TEXT_SECONDARY}">nanochat {data.get("model_tag") or ""} · greedy decoding · labels show score and change vs. previous stage</text>']
# legend (always present for >= 2 series)
lx = left
for s in stages:
    svg.append(f'<rect x="{lx}" y="66" width="12" height="12" rx="3" fill="{STAGE_COLOR[s]}"/>')
    svg.append(f'<text x="{lx + 18}" y="77" font-size="12" fill="{TEXT_SECONDARY}">{STAGE_LABEL[s]}</text>')
    lx += 18 + 7.2 * len(STAGE_LABEL[s]) + 22
# recessive grid + y axis labels
step = 5 if y_max <= 30 else (10 if y_max <= 60 else 20)
v = 0
while v <= y_max + 1e-9:
    svg.append(f'<line x1="{left}" x2="{W - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{GRID}" stroke-width="1"/>')
    svg.append(f'<text x="{left - 8}" y="{y(v) + 4:.1f}" font-size="11" text-anchor="end" fill="{TEXT_MUTED}">{v:g}%</text>')
    v += step
# bars
for ti, t in enumerate(tasks):
    gx = left + group_w * ti + (group_w - len(stages) * bar_w) / 2
    for si, s in enumerate(stages):
        val = score(s, t)
        if val is None:
            continue
        x0 = gx + si * bar_w + gap / 2
        bw = bar_w - gap
        y0, h = y(val), max(0.0, y(0) - y(val))
        r = min(4, h / 2, bw / 2)
        # 4px rounded data-end, square at the baseline
        path = (f"M{x0:.1f},{y(0):.1f} L{x0:.1f},{y0 + r:.1f} Q{x0:.1f},{y0:.1f} {x0 + r:.1f},{y0:.1f} "
                f"L{x0 + bw - r:.1f},{y0:.1f} Q{x0 + bw:.1f},{y0:.1f} {x0 + bw:.1f},{y0 + r:.1f} L{x0 + bw:.1f},{y(0):.1f} Z")
        res = results[s][t]
        svg.append(f'<path d="{path}" fill="{STAGE_COLOR[s]}"><title>{STAGE_LABEL[s]} · {t}: {val:.1f}% ({res["passed"]}/{res["total"]})</title></path>')
        cx = x0 + bw / 2
        svg.append(f'<text x="{cx:.1f}" y="{y0 - 8:.1f}" font-size="13" font-weight="600" text-anchor="middle" fill="{TEXT_PRIMARY}">{val:.1f}%</text>')
        d = delta_vs_prev(s, t)
        if d is not None:
            svg.append(f'<text x="{cx:.1f}" y="{y0 - 24:.1f}" font-size="11" text-anchor="middle" fill="{TEXT_SECONDARY}">{"▲" if d > 0 else ("▼" if d < 0 else "■")} {d:+.1f}pp</text>')
    svg.append(f'<text x="{left + group_w * ti + group_w / 2:.1f}" y="{y(0) + 24:.1f}" font-size="13" text-anchor="middle" fill="{TEXT_PRIMARY}">{t}</text>')
svg.append(f'<line x1="{left}" x2="{W - right}" y1="{y(0):.1f}" y2="{y(0):.1f}" stroke="{TEXT_MUTED}" stroke-width="1"/>')
svg.append("</svg>")
with open(os.path.join(outdir, "comparison.svg"), "w", encoding="utf-8") as f:
    f.write("\n".join(svg))

# --------------------------------------------------------------------------- PNG (optional)
png_written = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    n = len(stages)
    width = 0.7 / n
    for si, s in enumerate(stages):
        xs, vals = [], []
        for ti, t in enumerate(tasks):
            val = score(s, t)
            if val is None:
                continue
            xs.append(ti - 0.35 + width * (si + 0.5)); vals.append(val)
        bars = ax.bar(xs, vals, width=width * 0.94, color=STAGE_COLOR[s], label=STAGE_LABEL[s], zorder=3)
        for b, t_idx in zip(bars, [ti for ti, t in enumerate(tasks) if score(s, t) is not None]):
            t = tasks[t_idx]
            val = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, val + y_max * 0.015, f"{val:.1f}%", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=TEXT_PRIMARY)
            d = delta_vs_prev(s, t)
            if d is not None:
                ax.text(b.get_x() + b.get_width() / 2, val + y_max * 0.065, f"{'▲' if d > 0 else ('▼' if d < 0 else '■')} {d:+.1f}pp",
                        ha="center", va="bottom", fontsize=8, color=TEXT_SECONDARY)
    ax.set_xticks(range(len(tasks))); ax.set_xticklabels(tasks, fontsize=10, color=TEXT_PRIMARY)
    ax.set_ylim(0, y_max); ax.set_ylabel("pass@1 (%)", color=TEXT_SECONDARY, fontsize=9)
    ax.yaxis.grid(True, color=GRID, zorder=0); ax.tick_params(axis="y", colors=TEXT_MUTED, labelsize=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(TEXT_MUTED)
    ax.set_title("Coding benchmarks by training stage (pass@1, greedy)", loc="left", fontsize=12, color=TEXT_PRIMARY, pad=28)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.1), ncol=n, frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "comparison.png"), facecolor=SURFACE)
    png_written = True
except ImportError:
    pass

print(table_md)
print(f"Wrote {outdir}/comparison.svg" + (", comparison.png" if png_written else " (no matplotlib: PNG skipped)") + ", comparison.md")
