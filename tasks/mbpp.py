"""
MBPP (Mostly Basic Python Problems), used both as an RL environment with a
verifiable reward (unit tests) and as an evaluation.
https://huggingface.co/datasets/google-research-datasets/mbpp

Splits of the "full" config: train (374), validation (90), prompt (10), test (500).
We additionally expose split="rl_train" = train + validation + prompt (474 problems),
which is what we use for Code-RLVR so that the 500 test problems stay held out.

Reward (deterministic, no reward model):
  all asserts pass                                   -> 1.0
  timeout / syntax error / exception / assert fails  -> 0.0
"""

from nanochat.execution import execute_code
from tasks.common import Task, load_hub_dataset
from tasks.humaneval import extract_program

MBPP_SPLITS = ("train", "validation", "prompt", "test")


def _normalize(code):
    return code.replace("\r\n", "\n").replace("\r", "\n").strip()


def render_mbpp_prompt(text, test_list):
    """The prompt the model sees. Showing the tests pins down the function name and signature."""
    tests = "\n".join(test_list)
    return (
        f"{text.strip()}\n"
        f"Your code should pass these tests:\n\n"
        f"```python\n{tests}\n```\n\n"
        f"Write the Python function inside a ```python code block."
    )


def build_test_program(code, test_setup_code, test_list):
    """Concatenate the candidate solution with the setup code and the assert statements."""
    parts = []
    if test_setup_code and test_setup_code.strip():
        parts.append(_normalize(test_setup_code))
    parts.append(code)
    parts.extend(_normalize(t) for t in test_list)
    return "\n\n".join(parts) + "\n"


class MBPP(Task):

    def __init__(self, split="test", subset="full", timeout=5.0, include_challenge_tests=False, **kwargs):
        super().__init__(**kwargs)
        splits = ("train", "validation", "prompt") if split == "rl_train" else (split,)
        assert all(s in MBPP_SPLITS for s in splits), f"MBPP split must be one of {MBPP_SPLITS} or rl_train, got {split}"
        self.rows = []
        for s in splits:
            ds = load_hub_dataset("google-research-datasets/mbpp", subset=subset, split=s)
            self.rows.extend(ds[i] for i in range(len(ds)))
        self.timeout = timeout
        self.include_challenge_tests = include_challenge_tests

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.rows)

    def get_example(self, index):
        row = self.rows[index]
        test_list = [_normalize(t) for t in row["test_list"]]
        reference = _normalize(row["code"])
        messages = [
            {"role": "user", "content": render_mbpp_prompt(row["text"], test_list)},
            {"role": "assistant", "content": f"```python\n{reference}\n```"},
        ]
        tests = list(test_list)
        if self.include_challenge_tests:
            tests += [_normalize(t) for t in (row.get("challenge_test_list") or [])]
        return {
            "messages": messages,
            "task_id": row["task_id"],
            "text": row["text"].strip(),
            "test_list": tests,
            "test_setup_code": row.get("test_setup_code") or "",
        }

    def evaluate(self, conversation, completion):
        """Return 1 if the extracted program passes all unit tests, else 0."""
        code = extract_program(completion)
        if not code:
            return 0
        program = build_test_program(code, conversation["test_setup_code"], conversation["test_list"])
        result = execute_code(program, timeout=self.timeout)
        return int(result.success)

    def reward(self, conversation, completion):
        """Binary verifiable reward used by Code-RLVR."""
        return float(self.evaluate(conversation, completion))
