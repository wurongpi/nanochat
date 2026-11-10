"""
Offline tests for the fixed code scoring used by scripts/code_reeval.py (no network, no GPU).

python -m pytest tests/test_code_scoring.py -v
"""

import pytest

from tasks.code_scoring import (extract_code, sanitize, run_with_sentinel, humaneval_solution,
                                hardcode_suspect, score_completion, pass_at_k, mbpp_program)
from tasks.mbpp import build_test_program
from nanochat.execution import execute_code

MBPP_CONV = {"test_list": ["assert add(1, 2) == 3", "assert add(2, 5) == 7"], "test_setup_code": ""}
HIDDEN = ["assert add(10, 20) == 30"]
HE_PROMPT = 'from typing import List\n\n\ndef add(a: int, b: int) -> int:\n    """Add two numbers."""\n'
HE_CONV = {"messages": [{"role": "user", "content": HE_PROMPT}], "entry_point": "add",
           "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(5, 5) == 10\n"}


def block(code):
    return f"Short analysis.\n\n```python\n{code}\n```\nTrailing explanation."


def test_extract_code():
    assert extract_code(block("x = 1")) == ("x = 1", "closed_block")
    assert extract_code("Analysis.\n```python\ndef f():\n    return 1") == ("def f():\n    return 1", "unclosed_block")
    assert extract_code("def f():\n    return 1")[1] == "no_block"


def test_sanitize_drops_demo_code_and_trims_truncation():
    code, trimmed, dropped = sanitize("import os\ndef f():\n    return 1\nprint(f())\nf(")
    assert trimmed == 1 and dropped == 1
    assert "print" not in code and "def f" in code and "import os" in code
    code, _, dropped = sanitize("def f():\n    return 1\nif __name__ == '__main__':\n    assert f() == 2")
    assert dropped == 1 and "__main__" not in code


def test_sentinel_blocks_early_exit():
    tests = MBPP_CONV["test_list"]
    wrong = "def add(a, b):\n    return 0"
    for hack in ["import sys\n" + wrong + "\nsys.exit(0)", wrong + "\nraise SystemExit",
                 "import os\n" + wrong + "\nos._exit(0)", "import sys\ndef add(a, b):\n    sys.exit(0)"]:
        program = build_test_program(hack, "", tests)
        assert execute_code(program).success, "the original returncode check is fooled"
        assert not run_with_sentinel(program)[0], "the sentinel check is not"
    assert run_with_sentinel(build_test_program("def add(a, b):\n    return a + b", "", tests))[0]


def test_humaneval_solution_body_only_and_full_function():
    body = humaneval_solution(HE_PROMPT, "add", "    return a + b")
    full = humaneval_solution(HE_PROMPT, "add", "def add(a, b):\n    return a + b")
    assert body.startswith(HE_PROMPT)
    assert full.startswith("from typing import List")
    for sol in (body, full):
        assert run_with_sentinel(f"{sol}\n\n{HE_CONV['test']}\ncheck(add)")[0]


@pytest.mark.parametrize("code,expected", [
    ("def add(a, b):\n    return {(1, 2): 3, (2, 5): 7}[(a, b)]", True),
    ("def add(a, b):\n    return a + b", False),
])
def test_hardcode_suspect_mbpp(code, expected):
    assert hardcode_suspect(code, MBPP_CONV["test_list"]) is expected


def test_hardcode_suspect_strings():
    tests = ["assert rev('hello') == 'olleh'", "assert rev('abc') == 'cba'"]
    assert hardcode_suspect("def rev(s):\n    if s == 'hello':\n        return 'olleh'\n    return 'cba'", tests)
    assert not hardcode_suspect("def rev(s):\n    return s[::-1]", tests)


def test_score_completion_mbpp():
    ok = score_completion("MBPP", MBPP_CONV, block("def add(a, b):\n    return a + b"), challenge_tests=HIDDEN)
    assert ok["fixed"] and ok["hidden"] and not ok["exit_hack"] and not ok["hardcode"]
    hack = score_completion("MBPP", MBPP_CONV, block("import sys\ndef add(a, b):\n    sys.exit(0)"), challenge_tests=HIDDEN)
    assert not hack["fixed"] and hack["exit_hack"]
    memo = score_completion("MBPP", MBPP_CONV, block("def add(a, b):\n    return {(1, 2): 3, (2, 5): 7}[(a, b)]"),
                            challenge_tests=HIDDEN)
    assert memo["fixed"] and memo["hardcode"] and memo["hidden"] is False
    cut = score_completion("MBPP", MBPP_CONV, "```python\ndef add(a, b):\n    return a + b\n\nprint(add(1,")
    assert cut["fixed"] and cut["extract"] == "unclosed_block" and cut["trimmed"] > 0


def test_score_completion_humaneval():
    assert score_completion("HumanEval", HE_CONV, block("def add(a, b):\n    return a + b"))["fixed"]
    assert not score_completion("HumanEval", HE_CONV, block("def add(a, b):\n    return 3"))["fixed"]


def test_pass_at_k():
    assert pass_at_k(10, 0, 1) == 0.0
    assert pass_at_k(10, 10, 5) == 1.0
    assert pass_at_k(10, 3, 1) == pytest.approx(0.3)
    assert pass_at_k(4, 2, 2) == pytest.approx(1 - 1 / 6)
    assert pass_at_k(10, 1, 10) == 1.0


def test_mbpp_setup_uses_class_defined_by_the_solution():
    # like MBPP's tree problems: the setup builds a tree from the `Node` class that the solution defines
    conv = {"test_list": ["assert depth(root) == 2"],
            "test_setup_code": "root = Node(1)\nroot.left = Node(2)"}
    code = ("class Node:\n    def __init__(self, v):\n        self.v, self.left, self.right = v, None, None\n"
            "def depth(n):\n    return 0 if n is None else 1 + max(depth(n.left), depth(n.right))")
    assert not execute_code(build_test_program(code, conv["test_setup_code"], conv["test_list"])).success
    assert run_with_sentinel(mbpp_program(code, conv["test_setup_code"], conv["test_list"]))[0]
    assert score_completion("MBPP", conv, block(code))["fixed"]
