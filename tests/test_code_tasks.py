"""
Offline tests for the Code-RLVR pieces (no network, no GPU).

python -m pytest tests/test_code_tasks.py -v
"""

from tasks.code_sft import first_python_block, trim_reasoning, is_single_function_solution, format_response
from tasks.mbpp import MBPP, build_test_program, render_mbpp_prompt
from nanochat.execution import execute_code


def test_first_python_block():
    text = "Handle the empty list first.\n\n```python\ndef f(x):\n    return x\n```\nMore text."
    code, before, after = first_python_block(text)
    assert code == "def f(x):\n    return x"
    assert before.strip() == "Handle the empty list first."
    assert after.strip() == "More text."
    assert first_python_block("no code here")[0] is None


def test_trim_reasoning_limits():
    long = "First sentence here. Second one. Third one! Fourth should be dropped."
    assert trim_reasoning(long, max_chars=200, max_sentences=3) == "First sentence here. Second one. Third one!"
    assert trim_reasoning(long, max_chars=25) == "First sentence here."
    assert trim_reasoning("x" * 300, max_chars=200) == ""   # a single over-long sentence is dropped entirely


def test_single_function_filter():
    assert is_single_function_solution("def f():\n    pass")
    assert not is_single_function_solution("class A:\n    pass\ndef f():\n    pass")
    assert not is_single_function_solution("x = 1")


def test_format_response():
    assert format_response("", "def f(): pass") == "```python\ndef f(): pass\n```"
    assert format_response("Edge case: n=0.", "def f(): pass").startswith("Edge case: n=0.\n\n```python")


def _fake_mbpp_conversation():
    tests = ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"]
    return {
        "messages": [{"role": "user", "content": render_mbpp_prompt("Add two numbers.", tests)},
                     {"role": "assistant", "content": "```python\ndef add(a, b):\n    return a + b\n```"}],
        "test_list": tests, "test_setup_code": "", "text": "Add two numbers.", "task_id": 0,
    }


def test_mbpp_reward_pass_fail_timeout_syntax():
    task = MBPP.__new__(MBPP)   # skip the dataset download
    task.timeout = 2.0
    conv = _fake_mbpp_conversation()
    assert task.reward(conv, "Simple.\n\n```python\ndef add(a, b):\n    return a + b\n```") == 1.0
    assert task.reward(conv, "```python\ndef add(a, b):\n    return a - b\n```") == 0.0          # assert fails
    assert task.reward(conv, "```python\ndef add(a, b)\n    return a + b\n```") == 0.0           # syntax error
    assert task.reward(conv, "```python\ndef add(a, b):\n    while True: pass\n```") == 0.0     # timeout
    assert task.reward(conv, "") == 0.0


def test_build_test_program_runs():
    program = build_test_program("def sq(x):\n    return x * x", "import math", ["assert sq(3) == 9"])
    assert execute_code(program).success
