#!/usr/bin/env python3
"""
Evaluation script for math reasoning benchmarks.

Grader: verify.py — cascade of MathD -> SymPy -> Math-Verify.
Generation defaults (aligned with training rollout config):
  temperature=0.7, top_p=0.95, n_samples=8, max_new_tokens=8192

Benchmarks: math500, aime25, amc23, minervamath

Output: Each model's results are saved in output_dir/<model_name>/ where model_name
  is the last path component (e.g. Qwen3-8B or qwen3-4b_aime_grpo_structure_only_step45).

Usage:
    # Single model, all benchmarks
    python eval_all.py --model /path/to/model --use-vllm

    # Multiple models listed in a file (one path per line)
    python eval_all.py --model-list models.txt --use-vllm

    # Specific benchmarks
    python eval_all.py --model /path/to/model --datasets aime25 math500 --use-vllm

    # Quick sanity check (5 problems per benchmark)
    python eval_all.py --model /path/to/model --limit 5 --use-vllm
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

# Add this directory to path so verify.py is importable
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from verify import grade, get_answer_str  # noqa: E402

# ---------------------------------------------------------------------------
# Benchmark configs
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).parent.parent
DATA_ROOT = Path(os.environ.get("SARL_DATA_ROOT", str(_repo_root / "data")))

DATASET_CONFIGS = {
    "math500":       str(DATA_ROOT / "math500/test.parquet"),
    "aime25":      str(DATA_ROOT / "aime25/test.parquet"),
    "amc23":       str(DATA_ROOT / "amc23/test.parquet"),
    "minervamath": str(DATA_ROOT / "minervamath/test.parquet"),
}

ALL_DATASETS = list(DATASET_CONFIGS.keys())

# Evaluation defaults — aligned with training rollout config
# (max_response_length=8192, temperature=1.0/top_p=0.7 in ppo_trainer default;
#  temperature=0.7 is used here as a reasonable eval value matching training)
DEFAULT_N_SAMPLES    = 8
DEFAULT_TEMPERATURE  = 0.7
DEFAULT_TOP_P        = 0.95
DEFAULT_MAX_TOKENS   = 8192


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def _extract_answer_lenient(response: str) -> str:
    """
    Format-agnostic extraction: try \\boxed{} first, then fallback patterns.
    Used for fair comparison when models use different output formats.
    """
    extracted_list = get_answer_str(response, num_answers=1)
    extracted = extracted_list[0] if extracted_list else ""
    if extracted:
        return extracted.strip()

    # Fallback patterns for common answer formats (order matters)
    fallbacks = [
        r"(?:The|Therefore|Thus)[,\s]+(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\s]*\s*([^\n]+)",
        r"(?:Final|The)\s+answer\s*[:\s]+\s*([^\n]+)",
        r"[Aa]nswer\s*[:\s]+\s*([^\n]+)",
        r"<answer>([^<]+)</answer>",
        r"\\boxed\s*\{\s*([^}]+)\s*\}",  # unescaped in some outputs
        r"\[([^\]]+)\]\s*$",  # last line in brackets
    ]
    for pat in fallbacks:
        m = re.search(pat, response, re.IGNORECASE | re.DOTALL)
        if m:
            ans = m.group(1).strip()
            if ans and len(ans) < 200:  # sanity: avoid capturing full solution
                return ans

    # Last non-empty line (often contains the answer)
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    if lines:
        last = lines[-1]
        if last and not last.startswith("<think>") and len(last) < 200:
            return last
    return ""


def check_answer(response: str, ground_truth: str, format_agnostic: bool = False) -> tuple:
    """
    Grade a model response against the ground truth.

    Extracts answer from response (\\boxed{} first, or fallback patterns if format_agnostic),
    then applies the cascade: MathD -> SymPy -> Math-Verify.

    Returns:
        correct (bool): whether the answer is correct
        pred    (str | None): the extracted prediction string
    """
    if format_agnostic:
        extracted = _extract_answer_lenient(response)
    else:
        extracted_list = get_answer_str(response, num_answers=1)
        extracted = extracted_list[0] if extracted_list else ""

    if not extracted:
        return False, None
    try:
        correct, _method, pred, _gt = grade(extracted, ground_truth, is_matched=True)
        return bool(correct), str(pred) if pred is not None else None
    except Exception:
        return False, None


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompt(tokenizer, messages: List[Dict], enable_thinking: bool = False) -> str:
    """Apply the model's chat template to a list of role/content messages."""
    try:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if not enable_thinking:
            kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            content = messages[0]["content"] if messages else ""
            return f"{content}\n\nAnswer:"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_benchmark(name: str, limit: Optional[int] = None) -> List[Dict]:
    """Load a benchmark from its local parquet file (veRL format)."""
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown benchmark '{name}'. Available: {ALL_DATASETS}")

    path = DATASET_CONFIGS[name]
    print(f"  Loading {name}: {path}")
    ds = load_dataset("parquet", data_files=path, split="train")

    items = []
    for idx, item in enumerate(ds):
        rm = item["reward_model"]
        ground_truth = rm["ground_truth"] if isinstance(rm, dict) else str(rm)
        items.append({
            "prompt_messages": item["prompt"],
            "answer": str(ground_truth),
            "idx": idx,
            "data_source": item.get("data_source", name),
        })

    if limit:
        items = items[:limit]
    return items


def _batched(iterable, n):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    tokenizer,
    benchmark_name: str,
    data: List[Dict],
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
    model_device,
    global_batch_size: int,
    use_vllm: bool = False,
    SamplingParams=None,
    format_agnostic: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate *model* on all problems in *data*.

    Two metrics are computed:
      avg@n  — mean over problems of (correct_count / n)
      pass@n — fraction of problems where at least one of n samples is correct
    """
    problem_accs = []
    problems_passed = 0
    total_correct = 0
    total_samples = 0
    extraction_fails = 0
    results = []

    # Align batch size to a multiple of num_samples
    if global_batch_size % num_samples != 0:
        global_batch_size = max(num_samples, (global_batch_size // num_samples) * num_samples)
    problems_per_batch = global_batch_size // num_samples

    num_batches = (len(data) + problems_per_batch - 1) // problems_per_batch
    pbar = tqdm(
        _batched(data, problems_per_batch),
        total=num_batches,
        desc=f"{benchmark_name}",
        leave=False,
    )

    for batch in pbar:
        if use_vllm:
            # vLLM: expand prompts so each problem is repeated num_samples times
            expanded_prompts = []
            for ex in batch:
                prompt = build_prompt(tokenizer, ex["prompt_messages"], enable_thinking)
                expanded_prompts.extend([prompt] * num_samples)

            sp = SamplingParams(
                temperature=0.0 if num_samples == 1 else temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
            )
            outputs = model.generate(expanded_prompts, sp)

            for bi, ex in enumerate(batch):
                prompt_text = build_prompt(tokenizer, ex["prompt_messages"], enable_thinking)
                correct_count = 0
                preds = []
                for si in range(num_samples):
                    out = outputs[bi * num_samples + si]
                    text = (
                        out.outputs[0].text
                        if hasattr(out, "outputs")
                        else getattr(out, "text", str(out))
                    )
                    ok, pred = check_answer(text, ex["answer"], format_agnostic=format_agnostic)
                    if ok:
                        correct_count += 1
                    if pred is None:
                        extraction_fails += 1
                    preds.append({"response": text, "pred": pred, "correct": ok})

                acc = correct_count / num_samples if num_samples > 1 else float(correct_count > 0)
                problem_accs.append(acc)
                passed = int(correct_count > 0)
                problems_passed += passed
                total_correct += correct_count
                total_samples += num_samples
                results.append({
                    "idx": ex["idx"], "prompt": prompt_text, "gold": ex["answer"],
                    "predictions": preds, "correct_count": correct_count,
                    "num_samples": num_samples, "problem_acc": acc, "passed": passed,
                })

        else:
            # Transformers: batched generation with num_return_sequences
            prompts = [
                build_prompt(tokenizer, ex["prompt_messages"], enable_thinking)
                for ex in batch
            ]
            inputs = tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=4096,
            )
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            with torch.no_grad():
                if num_samples == 1:
                    outs = model.generate(
                        **inputs, max_new_tokens=max_new_tokens,
                        do_sample=False, pad_token_id=tokenizer.pad_token_id,
                    )
                else:
                    outs = model.generate(
                        **inputs, max_new_tokens=max_new_tokens,
                        do_sample=True, temperature=temperature, top_p=top_p,
                        pad_token_id=tokenizer.pad_token_id,
                        num_return_sequences=num_samples,
                    )

            for bi, ex in enumerate(batch):
                in_len = inputs["input_ids"][bi].shape[0]
                prompt_text = build_prompt(tokenizer, ex["prompt_messages"], enable_thinking)
                correct_count = 0
                preds = []
                for si in range(num_samples):
                    text = tokenizer.decode(
                        outs[bi * num_samples + si][in_len:], skip_special_tokens=True
                    )
                    ok, pred = check_answer(text, ex["answer"], format_agnostic=format_agnostic)
                    if ok:
                        correct_count += 1
                    if pred is None:
                        extraction_fails += 1
                    preds.append({"response": text, "pred": pred, "correct": ok})

                acc = correct_count / num_samples if num_samples > 1 else float(correct_count > 0)
                problem_accs.append(acc)
                passed = int(correct_count > 0)
                problems_passed += passed
                total_correct += correct_count
                total_samples += num_samples
                results.append({
                    "idx": ex["idx"], "prompt": prompt_text, "gold": ex["answer"],
                    "predictions": preds, "correct_count": correct_count,
                    "num_samples": num_samples, "problem_acc": acc, "passed": passed,
                })

        if problem_accs:
            pbar.set_postfix({
                "avg": f"{sum(problem_accs)/len(problem_accs):.3f}",
                "pass": f"{problems_passed/len(problem_accs):.3f}",
            })

    pbar.close()
    n = len(data)
    extractable = total_samples - extraction_fails
    extractable_accuracy = (
        total_correct / extractable if extractable > 0 else None
    )
    return {
        "dataset": benchmark_name,
        "num_problems": n,
        "num_samples": num_samples,
        "avg_at_n":  sum(problem_accs) / n if n else 0.0,
        "pass_at_n": problems_passed / n if n else 0.0,
        "total_correct": total_correct,
        "total_samples": total_samples,
        "problems_passed": problems_passed,
        "extraction_failures": extraction_fails,
        "extractable_accuracy": extractable_accuracy,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Model loading / unloading
# ---------------------------------------------------------------------------

def load_model(model_path: str, use_vllm: bool, vllm_mem: float):
    """Load a model and tokenizer. Returns (model, tokenizer, SamplingParams|None)."""
    if use_vllm:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        print(f"Loading with vLLM: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        llm = LLM(
            model=model_path,
            dtype="bfloat16",
            gpu_memory_utilization=vllm_mem,
            trust_remote_code=True,
            tensor_parallel_size=1,
        )
        return llm, tokenizer, SamplingParams

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading with transformers: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer, None


def unload_model(model, tokenizer, use_vllm: bool):
    """Release GPU memory after a model is done."""
    import gc, time

    if use_vllm:
        try:
            if hasattr(model, "llm_engine"):
                engine = model.llm_engine
                if hasattr(engine, "shutdown"):
                    engine.shutdown()
        except Exception:
            pass
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(5)
    else:
        del model, tokenizer
        torch.cuda.empty_cache()


def model_tag(path: str) -> str:
    """
    Extract a short folder name from a model path for display and file naming.
    For long paths like /scratch/.../qwen3-4b_aime_grpo_structure_only_step45,
    returns only the last component: qwen3-4b_aime_grpo_structure_only_step45.
    For HuggingFace IDs like Qwen/Qwen3-8B, returns Qwen3-8B.
    """
    return path.rstrip("/").split("/")[-1]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate math reasoning models on standard benchmarks."
    )

    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--model",      help="Single model path or HuggingFace ID")
    grp.add_argument("--model-list", help="Path to file with one model per line")

    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS,
        help="Benchmarks to evaluate (default: all)",
    )
    parser.add_argument("--num-samples",       type=int,   default=DEFAULT_N_SAMPLES,
                        help=f"Samples per problem (default: {DEFAULT_N_SAMPLES})")
    parser.add_argument("--global-batch-size", type=int,   default=32,
                        help="Total generations per batch = problems × num-samples")
    parser.add_argument("--temperature",       type=float, default=DEFAULT_TEMPERATURE,
                        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--top-p",             type=float, default=DEFAULT_TOP_P,
                        help=f"Nucleus sampling top-p (default: {DEFAULT_TOP_P})")
    parser.add_argument("--max-new-tokens",    type=int,   default=DEFAULT_MAX_TOKENS,
                        help=f"Max generated tokens (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--output-dir",        default="./results")
    parser.add_argument("--limit",             type=int,   default=None,
                        help="Limit problems per benchmark (for quick testing)")
    parser.add_argument("--enable-thinking",   action="store_true",
                        help="Enable thinking mode (e.g. Qwen3 instruct)")
    parser.add_argument("--use-vllm",          action="store_true",
                        help="Use vLLM for generation (recommended)")
    parser.add_argument("--vllm-mem",          type=float, default=0.85,
                        help="vLLM gpu_memory_utilization (default: 0.85)")
    parser.add_argument("--format-agnostic",   action="store_true",
                        help="Use format-agnostic extraction (fallback patterns when \\boxed{} empty) for fairer comparison")

    args = parser.parse_args()

    model_paths = (
        [args.model]
        if args.model
        else [l.strip() for l in open(args.model_list) if l.strip() and not l.startswith("#")]
    )

    print(f"\n{'='*70}")
    print(f"Math Reasoning Evaluation")
    print(f"{'='*70}")
    print(f"Models     : {len(model_paths)}")
    print(f"Benchmarks : {args.datasets}")
    print(f"n_samples  : {args.num_samples}   temperature: {args.temperature}"
          f"   top_p: {args.top_p}   max_tokens: {args.max_new_tokens}")
    print(f"vLLM       : {args.use_vllm}   mem: {args.vllm_mem}")
    print(f"Format-agnostic extraction: {args.format_agnostic}")
    print(f"{'='*70}\n")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all benchmarks once (before iterating models)
    benchmarks = {}
    for ds in args.datasets:
        benchmarks[ds] = load_benchmark(ds, args.limit)
        print(f"  {ds}: {len(benchmarks[ds])} problems")

    all_rows = []
    n = args.num_samples

    for mi, mp in enumerate(model_paths):
        print(f"\n{'='*70}")
        print(f"[{mi+1}/{len(model_paths)}] {mp}")
        print(f"{'='*70}")

        model, tokenizer, SP = load_model(mp, args.use_vllm, args.vllm_mem)
        device = None if args.use_vllm else next(model.parameters()).device
        tag = model_tag(mp)
        row = {"model": mp, "model_tag": tag}

        model_out_dir = out_dir / tag
        model_out_dir.mkdir(parents=True, exist_ok=True)

        for ds in args.datasets:
            print(f"\n  [{ds}] {len(benchmarks[ds])} problems ...")
            r = evaluate_model(
                model, tokenizer, ds, benchmarks[ds],
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                enable_thinking=args.enable_thinking,
                model_device=device,
                global_batch_size=args.global_batch_size,
                use_vllm=args.use_vllm,
                SamplingParams=SP,
                format_agnostic=args.format_agnostic,
            )
            row[f"{ds}_avg@{n}"]  = round(r["avg_at_n"],  4)
            row[f"{ds}_pass@{n}"] = round(r["pass_at_n"], 4)
            ext_acc = r.get("extractable_accuracy")
            ext_str = f"   extractable_acc: {ext_acc:.4f}" if ext_acc is not None else ""
            print(f"  avg@{n}: {r['avg_at_n']:.4f}   pass@{n}: {r['pass_at_n']:.4f}{ext_str}")

            with open(model_out_dir / f"summary_{ds}.json", "w") as f:
                json.dump({k: v for k, v in r.items() if k != "results"} | {"model": mp},
                          f, indent=2, default=str)
            with open(model_out_dir / f"details_{ds}.json", "w") as f:
                json.dump(r, f, indent=2, default=str)

        all_rows.append(row)
        unload_model(model, tokenizer, args.use_vllm)

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    header = f"{'Model':<45}" + "".join(f"  {ds[:12]:<12}" for ds in args.datasets)
    print(header)
    print("-" * max(len(header), 70))
    for row in all_rows:
        line = f"{row['model_tag']:<45}"
        for ds in args.datasets:
            val = row.get(f"{ds}_avg@{n}", "N/A")
            line += f"  {str(val):<12}"
        print(line)

    with open(out_dir / "all_results.json", "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    print(f"\nResults saved to: {out_dir}/ (each model in subfolder named after model)")


if __name__ == "__main__":
    main()
