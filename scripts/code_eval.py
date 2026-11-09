"""
Coding benchmark comparison across training stages: base (before SFT) vs sft vs rl.

The same benchmarks (HumanEval 164, MBPP test 500), the same decoding (greedy by default), but each stage is
prompted the way it was trained:
  - base: plain code completion (the base model has never seen chat special tokens)
      HumanEval: the function signature + docstring, the model continues the body (standard base-model HumanEval)
      MBPP     : a docstring with the task + tests, the model writes the function
      generations are cut at the usual stop sequences (\\nclass, \\nif __name__, \\nprint(, \\nassert, ...)
  - sft / rl: chat format (<|user_start|> ... <|assistant_start|>), code extracted from the ```python block

Results are merged into --output-json (existing stages are kept unless --overwrite), so you can run
`--sources=base,sft` right after SFT and `--sources=rl` later. Then `python -m scripts.plot_benchmarks`
turns the json into the comparison chart + table that `scripts.hub_upload` puts on the model card.

python -m scripts.code_eval --sources=base,sft,rl --model-tag=d12
"""

import os
import gc
import json
import time
import argparse

import torch
from nanochat.common import compute_init, compute_cleanup, autodetect_device_type, get_base_dir, print0
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from nanochat.execution import execute_code
from tasks.humaneval import HumanEval, extract_imports
from tasks.mbpp import MBPP, build_test_program

parser = argparse.ArgumentParser(description="Compare coding benchmarks across base/sft/rl")
parser.add_argument("--sources", type=str, default="base,sft,rl", help="comma separated subset of base,sft,rl")
parser.add_argument("--tasks", type=str, default="HumanEval,MBPP", help="comma separated subset of HumanEval,MBPP")
parser.add_argument("--model-tag", type=str, default=None)
parser.add_argument("--max-new-tokens", type=int, default=256)
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--top-k", type=int, default=50)
parser.add_argument("--max-problems", type=int, default=None, help="limit problems per task (debugging)")
parser.add_argument("--output-json", type=str, default=None, help="default: $NANOCHAT_BASE_DIR/eval/benchmark.json")
parser.add_argument("--overwrite", action="store_true", help="re-run stages already present in the json")
parser.add_argument("--save-samples", type=int, default=1, help="also store per-problem pass/fail + completion (1=yes)")
parser.add_argument("--device-type", type=str, default="")
args = parser.parse_args()

output_json = args.output_json or os.path.join(get_base_dir(), "eval", "benchmark.json")
os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
assert not ddp, "code_eval runs on a single GPU, use `python -m scripts.code_eval`"

BASE_STOPS = {
    "HumanEval": ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\nassert", "\n```", '\n"""'],
    "MBPP": ["\nclass ", "\nif __name__", "\nprint(", "\nassert", "\n```", '\n"""', "\n\n\n\n"],
}

def truncate_at_stops(text, stops):
    cut = len(text)
    for s in stops:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]

def base_prompt_and_program(task_name, conversation, completion=None):
    """Returns the completion-style prompt, or (if completion is given) the full program to execute."""
    if task_name == "HumanEval":
        prompt = conversation["messages"][0]["content"]
        if completion is None:
            return prompt
        body = truncate_at_stops(completion, BASE_STOPS["HumanEval"])
        return f"{prompt}{body}\n\n{conversation['test']}\ncheck({conversation['entry_point']})\n"
    else:
        tests = "\n".join(conversation["test_list"])
        prompt = f'"""\n{conversation["text"]}\nYour code should pass these tests:\n{tests}\n"""\n'
        if completion is None:
            return prompt
        code = truncate_at_stops(completion, BASE_STOPS["MBPP"])
        return build_test_program(code, conversation["test_setup_code"], conversation["test_list"])

def evaluate_stage(source, task_name, task, model, tokenizer, engine):
    n = len(task) if args.max_problems is None else min(len(task), args.max_problems)
    passed, records = 0, []
    t0 = time.time()
    for i in range(n):
        conversation = task[i]
        if source == "base":
            prompt_tokens = tokenizer.encode(base_prompt_and_program(task_name, conversation), prepend=tokenizer.get_bos_token_id())
        else:
            prompt_tokens = tokenizer.render_for_completion(conversation)
        results, _ = engine.generate_batch(prompt_tokens, num_samples=1, max_tokens=args.max_new_tokens,
                                           temperature=args.temperature, top_k=args.top_k, seed=1234 + i)
        completion = tokenizer.decode(results[0][len(prompt_tokens):])
        if source == "base":
            ok = execute_code(base_prompt_and_program(task_name, conversation, completion)).success
        else:
            ok = bool(task.evaluate(conversation, completion))
        passed += int(ok)
        if args.save_samples:
            records.append({"idx": i, "passed": ok, "completion": completion[:2000]})
        if (i + 1) % 20 == 0 or i + 1 == n:
            print0(f"  [{source}] {task_name}: {i+1}/{n} | pass@1 so far {100*passed/(i+1):.2f}% | {time.time()-t0:.0f}s")
    return {"pass@1": passed / n, "passed": passed, "total": n}, records

# load existing results
data = {"model_tag": args.model_tag, "settings": {}, "results": {}}
if os.path.exists(output_json):
    with open(output_json) as f:
        data = json.load(f)
data["settings"] = {"temperature": args.temperature, "max_new_tokens": args.max_new_tokens, "top_k": args.top_k,
                    "metric": "pass@1", "base_prompting": "code completion", "chat_prompting": "chat template"}
task_objects = {}
for name in [t.strip() for t in args.tasks.split(",") if t.strip()]:
    task_objects[name] = HumanEval() if name == "HumanEval" else MBPP(split="test")

for source in [s.strip() for s in args.sources.split(",") if s.strip()]:
    if source in data["results"] and not args.overwrite and all(t in data["results"][source] for t in task_objects):
        print0(f"Skipping {source}: already in {output_json} (use --overwrite to re-run)")
        continue
    print0(f"=== Evaluating stage: {source} ===")
    model, tokenizer, meta = load_model(source, device, phase="eval", model_tag=args.model_tag)
    engine = Engine(model, tokenizer)
    stage_results = data["results"].get(source, {})
    for task_name, task in task_objects.items():
        res, records = evaluate_stage(source, task_name, task, model, tokenizer, engine)
        stage_results[task_name] = res
        print0(f"[{source}] {task_name} pass@1 = {100*res['pass@1']:.2f}% ({res['passed']}/{res['total']})")
        if args.save_samples:
            samples_path = os.path.join(os.path.dirname(os.path.abspath(output_json)), f"samples_{source}_{task_name}.jsonl")
            with open(samples_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    data["results"][source] = stage_results
    with open(output_json, "w") as f:
        json.dump(data, f, indent=2)
    del model, engine
    gc.collect()
    if device_type == "cuda":
        torch.cuda.empty_cache()

print0(json.dumps(data["results"], indent=2))
print0(f"Wrote {output_json}")
compute_cleanup()
