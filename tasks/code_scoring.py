"""
Robust scoring of generated code for HumanEval / MBPP (used by scripts/code_reeval.py).

Fixes over the original pipeline (tasks.humaneval.extract_program + "returncode == 0"):
  - extraction also handles a ```python block left unclosed because the token budget ran out
  - sanitize: keep only imports, defs, classes and top-level assignments, so model-written prints, asserts,
    demo calls, `if __name__` blocks and top-level exits cannot break or fake the run; a half-written
    trailing line is trimmed until the code parses
  - a sentinel is printed AFTER the tests: exiting early (sys.exit, raise SystemExit, os._exit) is a fail
  - MBPP programs are assembled solution -> setup -> tests (the original puts the setup first)
Also: a hard-coding heuristic for the visible MBPP tests and the unbiased pass@k estimator.
"""

import re
import ast
import math

from nanochat.execution import execute_code
from tasks.humaneval import extract_imports
from tasks.mbpp import _normalize

SENTINEL = "__ALL_TESTS_PASSED_9c2e41__"
EXIT_HACK_RE = re.compile(r"sys\.exit|SystemExit|os\._exit|\bquit\(|(?<![\w.])exit\(")
CLOSED_BLOCK_RE = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)\n?```", re.DOTALL)
OPEN_BLOCK_RE = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*)", re.DOTALL)
KEEP_NODES = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
              ast.Assign, ast.AnnAssign)
TRIVIAL_LITERALS = {"True", "False", "None", "0", "1", "-1", "''", '""', "[]", "()", "{}", "0.0"}


def extract_code(completion):
    """Returns (code, how) with how in {closed_block, unclosed_block, no_block}."""
    m = CLOSED_BLOCK_RE.search(completion)
    if m:
        return m.group(1).strip("\n"), "closed_block"
    m = OPEN_BLOCK_RE.search(completion)
    if m:  # block opened but never closed (cut by the token budget)
        return m.group(1).strip("\n"), "unclosed_block"
    return completion.strip("\n"), "no_block"


def sanitize(code):
    """Returns (clean_code, trimmed_lines, dropped_statements). Falls back to the raw code if nothing parses."""
    lines = code.split("\n")
    tree, trimmed = None, 0
    while lines:
        try:
            tree = ast.parse("\n".join(lines))
            break
        except SyntaxError:
            lines.pop()
            trimmed += 1
    if tree is None:
        return code, 0, 0
    kept = [n for n in tree.body if isinstance(n, KEEP_NODES)]
    dropped = len(tree.body) - len(kept)
    if not kept:
        return "\n".join(lines), trimmed, 0
    return "\n\n".join(ast.unparse(n) for n in kept), trimmed, dropped


def run_with_sentinel(program, timeout=5.0):
    """Returns (passed, error). Passed only if the program ran to the end, i.e. past every test."""
    r = execute_code(program + f"\nprint({SENTINEL!r})\n", timeout=timeout)
    ok = r.success and SENTINEL in r.stdout
    return ok, (None if ok else (r.error or ("exited before tests finished" if r.success else "failed")))


def mbpp_program(code, test_setup_code, test_list):
    """
    Solution FIRST, then the setup code, then the asserts. tasks.mbpp.build_test_program puts the setup first,
    which breaks problems whose setup builds objects from a class the solution defines (e.g. a tree `Node`).
    Setup definitions are only used when the asserts run, so this order is safe for every problem.
    """
    parts = [code]
    if test_setup_code and test_setup_code.strip():
        parts.append(_normalize(test_setup_code))
    parts.extend(_normalize(t) for t in test_list)
    return "\n\n".join(parts) + "\n"


def humaneval_solution(prompt, entry_point, code):
    """A full, self-contained solution (also what EvalPlus expects in its `solution` field)."""
    if re.search(rf"^\s*def {re.escape(entry_point)}\s*\(", code, re.MULTILINE):
        imports = extract_imports(prompt)
        return (imports + "\n\n" + code) if imports else code
    return prompt + code  # the model only wrote the body: re-attach signature + docstring


def _assert_compares(test_list):
    out = []
    for t in test_list:
        try:
            node = ast.parse(t).body[0]
        except SyntaxError:
            continue
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare) and node.test.comparators:
            out.append(node.test)
    return out


def hardcode_suspect(code, test_list):
    """Heuristic: the solution spells out >=2 visible test inputs or >=2 non-trivial expected outputs."""
    flat = code.replace(" ", "")
    parts = _assert_compares(test_list)
    lits = [ast.unparse(p.comparators[-1]).replace(" ", "") for p in parts]
    lits = [l for l in lits if l not in TRIVIAL_LITERALS and len(l) >= 2]
    calls = [",".join(ast.unparse(a) for a in p.left.args).replace(" ", "") for p in parts if isinstance(p.left, ast.Call)]
    calls = [a for a in calls if len(a) >= 3]
    return (len(lits) >= 2 and sum(l in flat for l in lits) >= 2) or \
           (len(calls) >= 2 and sum(a in flat for a in calls) >= 2)


def score_completion(task_name, conversation, completion, challenge_tests=None, timeout=5.0):
    """
    All scores for one chat completion. task_name in {HumanEval, MBPP, MBPPPlus}; MBPPPlus is scored by EvalPlus,
    so only the extracted solution is returned for it. challenge_tests: MBPP tests that were not in the prompt.
    """
    code_raw, how = extract_code(completion)
    code, trimmed, dropped = sanitize(code_raw)
    s = dict(extract=how, trimmed=trimmed, dropped=dropped, exit_hack=bool(EXIT_HACK_RE.search(code_raw)),
             hardcode=False, hidden=None, fixed=None, error=None, solution=code)
    if task_name == "HumanEval":
        s["solution"] = humaneval_solution(conversation["messages"][0]["content"], conversation["entry_point"], code)
        program = f"{s['solution']}\n\n{conversation['test']}\ncheck({conversation['entry_point']})"
        s["fixed"], s["error"] = run_with_sentinel(program, timeout=timeout)
    elif task_name == "MBPP":
        setup = conversation["test_setup_code"]
        s["fixed"], s["error"] = run_with_sentinel(mbpp_program(code, setup, conversation["test_list"]), timeout=timeout)
        if challenge_tests:
            tests = [_normalize(t) for t in challenge_tests]
            s["hidden"], _ = run_with_sentinel(mbpp_program(code, setup, tests), timeout=timeout)
        s["hardcode"] = hardcode_suspect(code, conversation["test_list"])
    return s


def pass_at_k(n, c, k):
    """Unbiased estimator of pass@k from n samples with c correct (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - j) / (n - j) for j in range(k))
