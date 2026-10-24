"""
evaluate_v2.py — Full evaluation of the retrained model.

Loads ./surgical_repair_v2, generates repairs for a held-out test set,
validates through the two-stage pipeline (AST + sandbox), and reports
Pass@k with per-category breakdowns.

Run:
    python evaluate_v2.py
"""

import os
import sys
import ast
import json
import time
import random
import subprocess
import textwrap
from pathlib import Path
from typing import List, Dict, Tuple
from math import comb

# ---- Local cache (before HF imports) ----
os.environ.setdefault("HF_HOME", str(Path("./hf_cache").resolve()))
os.environ.setdefault("TRANSFORMERS_CACHE", str(Path("./hf_cache/models").resolve()))

_hf_token = os.environ.get("HF_TOKEN")
try:
    from huggingface_hub import login as _hf_login

    if _hf_token:
        _hf_login(token=_hf_token, add_to_git_credential=False)
except Exception:
    pass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

MODEL_DIR = "./surgical_repair_v2"
CACHE_DIR = str(Path("./hf_cache/models").resolve())
SANDBOX_DIR = Path("./sandbox_eval").resolve()
SANDBOX_DIR.mkdir(exist_ok=True, parents=True)

FIM_PREFIX = "<fim_prefix>"
FIM_SUFFIX = "<fim_suffix>"
FIM_MIDDLE = "<fim_middle>"

NUM_CANDIDATES = 5  # Pass@1, Pass@5
TEMPERATURE = 0.4


# ===========================================================================
# TEST BATTERY — held-out test cases
# ===========================================================================
# Each entry: (name, category, buggy_code, expected_fixed_code, test_code)
# The test_code contains assertions that both the buggy and fixed code run
# against. Repairs count as correct iff assertions pass.

TEST_CASES = [
    # ---------------- In-distribution (patterns similar to training) -------
    (
        "add_fix",
        "arithmetic",
        "def add_test(a, b):\n    result = a - b\n    return result",
        "def add_test(a, b):\n    result = a + b\n    return result",
        "assert add_test(2, 3) == 5\nassert add_test(-1, 1) == 0\nprint('OK')",
    ),
    (
        "sub_fix",
        "arithmetic",
        "def sub_test(a, b):\n    result = a + b\n    return result",
        "def sub_test(a, b):\n    result = a - b\n    return result",
        "assert sub_test(5, 3) == 2\nassert sub_test(0, 4) == -4\nprint('OK')",
    ),
    (
        "mul_fix",
        "arithmetic",
        "def mul_test(a, b):\n    result = a + b\n    return result",
        "def mul_test(a, b):\n    result = a * b\n    return result",
        "assert mul_test(3, 4) == 12\nassert mul_test(0, 5) == 0\nprint('OK')",
    ),
    (
        "div_fix",
        "arithmetic",
        "def div_test(a, b):\n    result = a * b\n    return result",
        "def div_test(a, b):\n    result = a / b\n    return result",
        "assert div_test(10, 2) == 5\nassert div_test(9, 3) == 3\nprint('OK')",
    ),
    (
        "mod_fix",
        "arithmetic",
        "def mod_test(a, b):\n    result = a // b\n    return result",
        "def mod_test(a, b):\n    result = a % b\n    return result",
        "assert mod_test(10, 3) == 1\nassert mod_test(20, 7) == 6\nprint('OK')",
    ),
    (
        "pow_fix",
        "arithmetic",
        "def pow_test(a, b):\n    result = a * b\n    return result",
        "def pow_test(a, b):\n    result = a ** b\n    return result",
        "assert pow_test(2, 3) == 8\nassert pow_test(5, 2) == 25\nprint('OK')",
    ),
    (
        "factorial_base",
        "base_case",
        "def factorial_test(n):\n    if n == 0:\n        return 0\n    return n * factorial_test(n - 1)",
        "def factorial_test(n):\n    if n == 0:\n        return 1\n    return n * factorial_test(n - 1)",
        "assert factorial_test(0) == 1\nassert factorial_test(5) == 120\nprint('OK')",
    ),
    (
        "max_of_two",
        "comparison",
        "def max2_test(x, y):\n    return min(x, y)",
        "def max2_test(x, y):\n    return max(x, y)",
        "assert max2_test(3, 5) == 5\nassert max2_test(10, 2) == 10\nprint('OK')",
    ),
    (
        "min_of_two",
        "comparison",
        "def min2_test(x, y):\n    return max(x, y)",
        "def min2_test(x, y):\n    return min(x, y)",
        "assert min2_test(3, 5) == 3\nassert min2_test(10, 2) == 2\nprint('OK')",
    ),
    (
        "is_even",
        "parity",
        "def is_even_test(n):\n    return n % 2 == 1",
        "def is_even_test(n):\n    return n % 2 == 0",
        "assert is_even_test(2) == True\nassert is_even_test(3) == False\nprint('OK')",
    ),
    (
        "is_odd",
        "parity",
        "def is_odd_test(n):\n    return n % 2 == 0",
        "def is_odd_test(n):\n    return n % 2 == 1",
        "assert is_odd_test(3) == True\nassert is_odd_test(4) == False\nprint('OK')",
    ),
    (
        "abs_val",
        "arithmetic",
        "def absval_test(x):\n    return -x",
        "def absval_test(x):\n    return abs(x)",
        "assert absval_test(-5) == 5\nassert absval_test(3) == 3\nprint('OK')",
    ),
    (
        "negate",
        "arithmetic",
        "def neg_test(x):\n    return x",
        "def neg_test(x):\n    return -x",
        "assert neg_test(5) == -5\nassert neg_test(-3) == 3\nprint('OK')",
    ),
    (
        "inc",
        "off_by_one",
        "def inc_test(n):\n    return n - 1",
        "def inc_test(n):\n    return n + 1",
        "assert inc_test(5) == 6\nassert inc_test(0) == 1\nprint('OK')",
    ),
    (
        "dec",
        "off_by_one",
        "def dec_test(n):\n    return n + 1",
        "def dec_test(n):\n    return n - 1",
        "assert dec_test(5) == 4\nassert dec_test(1) == 0\nprint('OK')",
    ),
    (
        "first_elem",
        "list",
        "def first_test(lst):\n    return lst[-1]",
        "def first_test(lst):\n    return lst[0]",
        "assert first_test([1,2,3]) == 1\nassert first_test(['a','b']) == 'a'\nprint('OK')",
    ),
    (
        "last_elem",
        "list",
        "def last_test(lst):\n    return lst[0]",
        "def last_test(lst):\n    return lst[-1]",
        "assert last_test([1,2,3]) == 3\nassert last_test(['a','b']) == 'b'\nprint('OK')",
    ),
    (
        "list_size",
        "list",
        "def size_test(lst):\n    return len(lst) - 1",
        "def size_test(lst):\n    return len(lst)",
        "assert size_test([1,2,3]) == 3\nassert size_test([]) == 0\nprint('OK')",
    ),
    (
        "upper_fix",
        "string",
        "def shout_test(text):\n    return text.lower()",
        "def shout_test(text):\n    return text.upper()",
        "assert shout_test('hi') == 'HI'\nassert shout_test('AbC') == 'ABC'\nprint('OK')",
    ),
    (
        "lower_fix",
        "string",
        "def quiet_test(text):\n    return text.upper()",
        "def quiet_test(text):\n    return text.lower()",
        "assert quiet_test('HI') == 'hi'\nassert quiet_test('AbC') == 'abc'\nprint('OK')",
    ),
    (
        "always_true",
        "boolean",
        "def yes_test(x):\n    return False",
        "def yes_test(x):\n    return True",
        "assert yes_test(0) == True\nassert yes_test('anything') == True\nprint('OK')",
    ),
    (
        "always_false",
        "boolean",
        "def no_test(x):\n    return True",
        "def no_test(x):\n    return False",
        "assert no_test(0) == False\nassert no_test('x') == False\nprint('OK')",
    ),
    # ---------------- Out-of-distribution (unseen variants) ----------------
    (
        "swap_gt",
        "ood_comparison",
        "def gt_test(x, y):\n    if x < y:\n        return True\n    return False",
        "def gt_test(x, y):\n    if x > y:\n        return True\n    return False",
        "assert gt_test(5, 3) == True\nassert gt_test(2, 7) == False\nprint('OK')",
    ),
    (
        "fib_base",
        "ood_base_case",
        "def fib_test(n):\n    if n <= 1:\n        return 0\n    return fib_test(n-1) + fib_test(n-2)",
        "def fib_test(n):\n    if n <= 1:\n        return 1\n    return fib_test(n-1) + fib_test(n-2)",
        "assert fib_test(0) == 1\nassert fib_test(5) == 8\nprint('OK')",
    ),
    (
        "drop_first",
        "ood_list",
        "def df_test(lst):\n    return lst[:-1]",
        "def df_test(lst):\n    return lst[1:]",
        "assert df_test([1,2,3]) == [2,3]\nassert df_test([9,8,7]) == [8,7]\nprint('OK')",
    ),
    (
        "trim_left",
        "ood_string",
        "def tl_test(text):\n    return text.rstrip()",
        "def tl_test(text):\n    return text.lstrip()",
        "assert tl_test('  hi  ') == 'hi  '\nprint('OK')",
    ),
]


# ===========================================================================
# MODEL LOADING
# ===========================================================================


def load_model():
    print("[eval] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_cfg = PeftConfig.from_pretrained(MODEL_DIR)
    base_name = peft_cfg.base_model_name_or_path

    print(f"[eval] Loading base model: {base_name}")
    base = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    )

    target_vocab = len(tokenizer)
    current_vocab = base.get_input_embeddings().weight.shape[0]
    if current_vocab != target_vocab:
        print(f"[eval] Resizing embeddings: {current_vocab} -> {target_vocab}")
        base.resize_token_embeddings(target_vocab)

    print("[eval] Attaching LoRA adapter...")
    model = PeftModel.from_pretrained(base, MODEL_DIR)
    model.eval()
    return model, tokenizer


# ===========================================================================
# FIM MASKING + CANDIDATE GENERATION
# ===========================================================================


def _find_mask_line(lines: List[str]) -> int:
    """Find the line most likely containing the bug."""
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s or s.startswith(
            (
                "#",
                "def ",
                "class ",
                "import ",
                "from ",
                "return ",
                "if ",
                "for ",
                "while ",
                "with ",
                "else",
                "elif",
            )
        ):
            continue
        if "=" in ln:
            return i
    # Fallback: find any non-def, non-blank line
    for i, ln in enumerate(lines):
        if ln.strip() and not ln.strip().startswith(("def ", "class ")):
            return i
    return 0


def _clean(text: str) -> str:
    """Strip leaked tokens, take first line only."""
    first = text.split("\n")[0]
    for tok in ["User:", "Assistant:", FIM_MIDDLE, FIM_SUFFIX, FIM_PREFIX]:
        first = first.replace(tok, "")
    return first.rstrip()


def generate_candidates(
    model, tokenizer, buggy_code: str, n: int = 5, temperature: float = 0.4
) -> List[str]:
    """Generate n candidate repairs via FIM masking."""
    lines = buggy_code.splitlines()
    mask_idx = _find_mask_line(lines)
    prefix = "\n".join(lines[:mask_idx])
    suffix = "\n".join(lines[mask_idx + 1 :])

    prompt = f"{FIM_PREFIX}{prefix}\n{FIM_SUFFIX}{suffix}\n{FIM_MIDDLE}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384)

    candidates = []
    for i in range(n):
        temp = temperature + 0.15 * i
        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=48,
                    do_sample=True,
                    temperature=min(temp, 1.0),
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        except Exception as e:
            candidates.append(buggy_code)
            continue

        raw = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        frag = _clean(raw)

        # Preserve original indentation if the fragment lost it
        original_indent = len(lines[mask_idx]) - len(lines[mask_idx].lstrip())
        if frag and not frag.startswith(" "):
            frag = " " * original_indent + frag

        new_lines = lines[:mask_idx] + [frag] + lines[mask_idx + 1 :]
        candidates.append("\n".join(new_lines))

    return candidates


# ===========================================================================
# STAGE 1 — AST validation
# ===========================================================================


def stage1_ast_ok(code: str) -> Tuple[bool, str]:
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"


# ===========================================================================
# STAGE 2 — sandboxed execution
# ===========================================================================


def stage2_exec_ok(code: str, tests: str, timeout: int = 10) -> Tuple[bool, str]:
    f = SANDBOX_DIR / f"t_{random.randint(0,9999999)}.py"
    try:
        with open(f, "w", encoding="utf-8") as fh:
            fh.write(code + "\n\n" + tests)
        r = subprocess.run(
            [sys.executable, str(f)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(SANDBOX_DIR),
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        if r.returncode == 0:
            return True, "PASS"
        err = (r.stderr.strip() or r.stdout.strip())[:200]
        return False, err
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, f"EXC: {e}"
    finally:
        try:
            f.unlink(missing_ok=True)
        except Exception:
            pass


# ===========================================================================
# PASS@K
# ===========================================================================


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased Pass@k estimator: 1 - C(n-c, k) / C(n, k)."""
    if n < k or c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


# ===========================================================================
# EVALUATION LOOP
# ===========================================================================


def exact_match(candidate: str, expected: str) -> bool:
    """Normalize whitespace and compare."""

    def norm(s):
        return "\n".join(ln.rstrip() for ln in s.splitlines() if ln.strip())

    return norm(candidate) == norm(expected)


def main():
    model, tokenizer = load_model()

    print()
    print("=" * 78)
    print(
        f"EVALUATION — {len(TEST_CASES)} test cases, "
        f"{NUM_CANDIDATES} candidates each"
    )
    print("=" * 78)

    results = []
    t_start = time.time()

    for idx, (name, category, buggy, expected, tests) in enumerate(TEST_CASES, 1):
        t0 = time.time()
        candidates = generate_candidates(
            model,
            tokenizer,
            buggy,
            n=NUM_CANDIDATES,
            temperature=TEMPERATURE,
        )

        # Two-stage validation per candidate
        s1_pass = 0
        s2_pass = 0
        exact = 0
        first_ok = None

        for i, c in enumerate(candidates):
            ok1, _ = stage1_ast_ok(c)
            if ok1:
                s1_pass += 1
                ok2, _ = stage2_exec_ok(c, tests)
                if ok2:
                    s2_pass += 1
                    if first_ok is None:
                        first_ok = i
            if exact_match(c, expected):
                exact += 1

        pk1 = pass_at_k(NUM_CANDIDATES, s2_pass, 1)
        pk5 = pass_at_k(NUM_CANDIDATES, s2_pass, 5)

        results.append(
            {
                "name": name,
                "category": category,
                "s1_pass": s1_pass,
                "s2_pass": s2_pass,
                "exact_match": exact,
                "pass@1": pk1,
                "pass@5": pk5,
                "seconds": time.time() - t0,
            }
        )

        elapsed = time.time() - t_start
        print(
            f"[{idx:2d}/{len(TEST_CASES)}] {name:<16} "
            f"S1={s1_pass}/{NUM_CANDIDATES}  S2={s2_pass}/{NUM_CANDIDATES}  "
            f"exact={exact}/{NUM_CANDIDATES}  "
            f"Pass@1={pk1:.2f}  Pass@5={pk5:.2f}  "
            f"({elapsed:.0f}s)"
        )

    # ---------------- Summary ----------------
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)

    by_cat: Dict[str, Dict[str, int]] = {}
    for r in results:
        c = r["category"]
        d = by_cat.setdefault(c, {"n": 0, "s1": 0, "s2": 0, "exact": 0})
        d["n"] += 1
        d["s1"] += 1 if r["s1_pass"] > 0 else 0
        d["s2"] += 1 if r["s2_pass"] > 0 else 0
        d["exact"] += 1 if r["exact_match"] > 0 else 0

    print(
        f"{'category':<20} {'N':>3} {'S1 ok':>7} {'S2 ok':>7} "
        f"{'exact':>7} {'Pass@1':>8} {'Pass@5':>8}"
    )
    print("-" * 78)
    for cat, d in sorted(by_cat.items()):
        n = d["n"]
        # Average Pass@k over cases in this category
        p1 = sum(r["pass@1"] for r in results if r["category"] == cat) / n
        p5 = sum(r["pass@5"] for r in results if r["category"] == cat) / n
        print(
            f"{cat:<20} {n:>3} {d['s1']:>7} {d['s2']:>7} "
            f"{d['exact']:>7} {p1:>8.2f} {p5:>8.2f}"
        )

    total_n = len(results)
    total_s1 = sum(1 for r in results if r["s1_pass"] > 0)
    total_s2 = sum(1 for r in results if r["s2_pass"] > 0)
    total_exact = sum(1 for r in results if r["exact_match"] > 0)
    avg_p1 = sum(r["pass@1"] for r in results) / total_n
    avg_p5 = sum(r["pass@5"] for r in results) / total_n

    print("-" * 78)
    print(
        f"{'OVERALL':<20} {total_n:>3} {total_s1:>7} {total_s2:>7} "
        f"{total_exact:>7} {avg_p1:>8.2f} {avg_p5:>8.2f}"
    )

    print()
    print(f"Total wall-clock time: {time.time() - t_start:.0f}s")
    print()

    # Save full results
    out = Path("./eval_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "model": MODEL_DIR,
                    "num_candidates": NUM_CANDIDATES,
                    "temperature": TEMPERATURE,
                },
                "overall": {
                    "cases_with_s1_pass": total_s1,
                    "cases_with_s2_pass": total_s2,
                    "cases_with_exact_match": total_exact,
                    "avg_pass_at_1": avg_p1,
                    "avg_pass_at_5": avg_p5,
                    "total_cases": total_n,
                },
                "per_case": results,
            },
            f,
            indent=2,
        )
    print(f"Detailed results saved to {out}")


if __name__ == "__main__":
    main()
