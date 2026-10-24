"""
smoke_test.py — Verify the trained Surgical Code Repair model actually works.

Run:
    python smoke_test.py
"""

import os
import sys
import time
from pathlib import Path

# ---- Local HF cache + auth (must come before transformers/peft import) ----
os.environ.setdefault("HF_HOME", str(Path("./hf_cache").resolve()))
os.environ.setdefault("TRANSFORMERS_CACHE", str(Path("./hf_cache/models").resolve()))

_hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
try:
    from huggingface_hub import login as _hf_login

    if _hf_token:
        _hf_login(token=_hf_token, add_to_git_credential=False)
        print("[auth] HuggingFace token accepted")
    else:
        print("[auth] No HF_TOKEN set — using anonymous requests")
except Exception as _e:
    print(f"[auth] HF login skipped: {_e}")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

MODEL_DIR = "./surgical_repair_output"
BASE = "deepseek-ai/deepseek-coder-1.3b-base"
CACHE_DIR = str(Path("./hf_cache/models").resolve())


def sep(title=""):
    print("\n" + "=" * 70)
    if title:
        print(title)
        print("=" * 70)


def main():
    # ------------------------------------------------------------------ #
    # STEP 1 — Tokenizer
    # ------------------------------------------------------------------ #
    sep("STEP 1 — Load tokenizer")
    t = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    print(f"  Loaded in {time.time()-t:.1f}s")
    print(f"  Vocab size: {tok.vocab_size}")
    print(f"  Pad token:  {tok.pad_token!r}")

    print("\n  FIM token check:")
    for name in ["<fim_prefix>", "<fim_suffix>", "<fim_middle>", "<extra_id_0>"]:
        ids = tok.encode(name, add_special_tokens=False)
        status = "OK (single token)" if len(ids) == 1 else f"BAD ({len(ids)} tokens)"
        print(f"    {name:<15} -> ids={ids}  [{status}]")

    # ------------------------------------------------------------------ #
    # STEP 2 — Base model + adapter
    # ------------------------------------------------------------------ #
    sep("STEP 2 — Load base model + adapter")
    t = time.time()
    print("  Loading base model (from local cache)...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    )
    print(f"  Base loaded in {time.time()-t:.1f}s")

    # Critical: resize base embeddings to match the trained tokenizer
    # BEFORE attaching the adapter. Otherwise vocab sizes mismatch.
    target_vocab = len(tok)
    current_vocab = base.get_input_embeddings().weight.shape[0]
    if current_vocab != target_vocab:
        print(f"  Resizing base embeddings: {current_vocab} -> {target_vocab}")
        base.resize_token_embeddings(target_vocab)

    t = time.time()
    print("  Attaching LoRA adapter...")
    model = PeftModel.from_pretrained(base, MODEL_DIR)
    model.eval()
    print(f"  Adapter attached in {time.time()-t:.1f}s")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} ({100*trainable/total:.4f}% of total)")
    print(f"  Total params    : {total:,}")

    # ------------------------------------------------------------------ #
    # STEP 3 — Generation tests
    # ------------------------------------------------------------------ #
    sep("STEP 3 — FIM generation tests (raw model output)")
    print("  Reading these outputs tells you whether the model learned anything.")

    tests = [
        {
            "name": "Simple addition (buggy: a - b, expect: a + b)",
            "prompt": (
                "<fim_prefix>def add_numbers(a, b):\n    result = "
                "<fim_suffix>\n    return result\n<fim_middle>"
            ),
            "expect_contains": "a + b",
        },
        {
            "name": "Factorial base case (buggy: return 0, expect: return 1)",
            "prompt": (
                "<fim_prefix>def factorial(n):\n    if n == 0:\n        return "
                "<fim_suffix>\n    return n * factorial(n - 1)\n<fim_middle>"
            ),
            "expect_contains": "1",
        },
        {
            "name": "Multiplication (buggy: a + b, expect: a * b)",
            "prompt": (
                "<fim_prefix>def multiply(a, b):\n    return "
                "<fim_suffix>\n\n<fim_middle>"
            ),
            "expect_contains": "a * b",
        },
        {
            "name": "Find max init (buggy: max_val = 0, expect: max_val = lst[0])",
            "prompt": (
                "<fim_prefix>def find_max(lst):\n    max_val = "
                "<fim_suffix>\n    for x in lst:\n        if x > max_val:\n"
                "            max_val = x\n    return max_val\n<fim_middle>"
            ),
            "expect_contains": "lst[0]",
        },
        {
            "name": "Loop range (buggy: range(len), expect: range(1, len)",
            "prompt": (
                "<fim_prefix>def sum_rest(lst):\n    total = 0\n    for i in "
                "<fim_suffix>:\n        total += lst[i]\n    return total\n<fim_middle>"
            ),
            "expect_contains": "range(1",
        },
    ]

    verdicts = []

    for i, tc in enumerate(tests, 1):
        print(f"\n--- Test {i}/{len(tests)}: {tc['name']} ---")
        print(f"  Prompt: {tc['prompt']!r}")

        inputs = tok(tc["prompt"], return_tensors="pt")

        # --- Greedy (deterministic) ---
        t = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=24,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        greedy = tok.decode(
            out[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        print(f"  Greedy ({time.time()-t:.1f}s): {greedy!r}")

        # --- Sampled (3 attempts) ---
        samples = []
        for j in range(3):
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=24,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.95,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )
            s = tok.decode(
                out[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            samples.append(s)
            print(f"  Sample {j+1} : {s!r}")

        # --- Verdict ---
        expected = tc["expect_contains"]
        greedy_ok = expected in greedy
        any_sample_ok = any(expected in s for s in samples)

        if greedy_ok:
            verdict = "PASS (greedy)"
        elif any_sample_ok:
            verdict = "PARTIAL (sample only)"
        else:
            verdict = "FAIL"

        verdicts.append(verdict)
        print(f"  VERDICT: {verdict}  — expected to contain {expected!r}")

    # ------------------------------------------------------------------ #
    # SUMMARY
    # ------------------------------------------------------------------ #
    sep("SUMMARY")
    for i, (tc, v) in enumerate(zip(tests, verdicts), 1):
        print(f"  Test {i}: {v:<22}  {tc['name'][:50]}")

    n_pass = sum(1 for v in verdicts if v.startswith("PASS"))
    n_partial = sum(1 for v in verdicts if v.startswith("PARTIAL"))
    n_fail = sum(1 for v in verdicts if v.startswith("FAIL"))
    print(f"\n  Totals: PASS={n_pass}, PARTIAL={n_partial}, FAIL={n_fail}")

    sep("HOW TO READ THIS")
    print("""
  If outputs look like code fragments (e.g. 'a + b', 'return 1', 'lst[0]')
  → model learned the FIM repair task. Infrastructure is validated.

  If outputs are empty '', newlines, or gibberish
  → model did not generalize; training corpus was too narrow.
     This is a DATA problem, not a code problem — the pipeline works.

  If 0/5 pass but samples occasionally produce code fragments
  → model is under-confident. More training data or more steps needed.
""")


if __name__ == "__main__":
    main()
