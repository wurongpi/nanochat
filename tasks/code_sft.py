"""
Code-CoT SFT task: reads a local JSONL file produced by scripts/prepare_code_data.py.

Each line is either
  {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
or
  {"prompt": ..., "response": ...}

The assistant responses follow the format we want the small model to learn:
  2-3 short sentences of edge-case analysis (<= ~200 chars)
  + a single standard Python function inside a ```python markdown block.

Loss masking: tokenizer.render_conversation() marks only assistant tokens with mask=1,
so prompt tokens never contribute to the gradient (Assistant Response-Only Loss).
"""

import re
import json
from tasks.common import Task

CODE_BLOCK_RE = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)\n```", re.DOTALL)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")


def first_python_block(text):
    """Return (code, text_before, text_after) for the first fenced code block, or (None, text, '')."""
    m = CODE_BLOCK_RE.search(text)
    if m is None:
        return None, text, ""
    return m.group(1).strip("\n"), text[:m.start()], text[m.end():]


def trim_reasoning(text, max_chars=200, max_sentences=3):
    """Keep at most `max_sentences` leading sentences that fit within `max_chars` characters."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return ""
    kept = []
    for sentence in SENTENCE_SPLIT_RE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = " ".join(kept + [sentence])
        if len(candidate) > max_chars or len(kept) >= max_sentences:
            break
        kept.append(sentence)
    return " ".join(kept)


def is_single_function_solution(code, allow_classes=False):
    """Heuristic for 'single-function solution logic': has a top-level def, no classes, not too many defs."""
    if not re.search(r"^def \w+\(", code, re.MULTILINE):
        return False
    if not allow_classes and re.search(r"^class \w+", code, re.MULTILINE):
        return False
    num_defs = len(re.findall(r"^def \w+\(", code, re.MULTILINE))
    return num_defs <= 3


def format_response(reasoning, code):
    reasoning = reasoning.strip()
    block = f"```python\n{code.strip()}\n```"
    return f"{reasoning}\n\n{block}" if reasoning else block


class CodeSFT(Task):
    """
    split="train" keeps all rows except every `val_every`-th row, split="val" keeps only those.
    """

    def __init__(self, path, split="train", val_every=50, **kwargs):
        super().__init__(**kwargs)
        assert split in {"train", "val", "all"}
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if split == "train":
            rows = [r for i, r in enumerate(rows) if i % val_every != 0]
        elif split == "val":
            rows = [r for i, r in enumerate(rows) if i % val_every == 0]
        assert len(rows) > 0, f"No rows for split={split} in {path}"
        self.rows = rows

    def num_examples(self):
        return len(self.rows)

    def get_example(self, index):
        row = self.rows[index]
        if "messages" in row:
            return {"messages": row["messages"]}
        return {"messages": [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ]}
