"""
check_model.py — Verification and inspection script for the trained
Surgical Code Repair model.

Run:
    python check_model.py
    python check_model.py --model-path ./surgical_repair_output

What it does:
    1. Verifies all expected files exist and have sane sizes
    2. Loads the tokenizer (checks FIM special tokens are present)
    3. Loads the base model + LoRA adapter
    4. Runs sample generations to inspect what the model actually produces
    5. Runs the two-stage validation on the generated candidates
    6. Prints a diagnostic summary
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict

import torch

# ---- Local cache (must be set before transformers/datasets import) ----
os.environ.setdefault("HF_HOME", str(Path("./hf_cache").resolve()))
os.environ.setdefault("TRANSFORMERS_CACHE", str(Path("./hf_cache/models").resolve()))
os.environ.setdefault("HF_DATASETS_CACHE", str(Path("./hf_cache/datasets").resolve()))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [CHECK] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# TEST CASES (same 3 as training-time validation)
# =============================================================================

TEST_CASES = [
    {
        "name": "add_numbers (should be a + b, currently a - b)",
        "buggy": "def add_numbers(a, b):\n    result = a - b\n    return result",
        "tests": (
            "assert add_numbers(2, 3) == 5\n"
            "assert add_numbers(-1, 1) == 0\n"
            "print('PASS_ADD')"
        ),
    },
    {
        "name": "factorial (base case should return 1, currently 0)",
        "buggy": (
            "def factorial(n):\n"
            "    if n == 0:\n"
            "        return 0\n"
            "    return n * factorial(n - 1)"
        ),
        "tests": (
            "assert factorial(0) == 1\n"
            "assert factorial(5) == 120\n"
            "print('PASS_FACT')"
        ),
    },
    {
        "name": "find_max (max_val init should be lst[0], currently works but check)",
        "buggy": (
            "def find_max(lst):\n"
            "    max_val = 0\n"
            "    for x in lst:\n"
            "        if x > max_val:\n"
            "            max_val = x\n"
            "    return max_val"
        ),
        "tests": (
            "assert find_max([1, 2, 3]) == 3\n"
            "assert find_max([-1, -5, -3]) == -1\n"
            "print('PASS_MAX')"
        ),
    },
]


# =============================================================================
# STEP 1 — File existence check
# =============================================================================


def check_files(model_dir: Path) -> bool:
    logger.info("=" * 70)
    logger.info("STEP 1: Checking saved files")
    logger.info("=" * 70)

    required = {
        "adapter_model.safetensors": (1_000_000, 200_000_000),  # 1MB – 200MB
        "adapter_model.bin": (1_000_000, 200_000_000),
        "adapter_config.json": (100, 100_000),
        "tokenizer_config.json": (100, 100_000),
        "tokenizer.json": (100_000, 20_000_000),
        "special_tokens_map.json": (50, 10_000),
    }
    optional = [
        "train_metrics.json",
        "pass_at_k_results.json",
        "README.md",
    ]

    if not model_dir.exists():
        logger.error(f"Model directory does not exist: {model_dir}")
        return False

    found_adapter = False
    for fname, (min_sz, max_sz) in required.items():
        f = model_dir / fname
        if f.exists():
            size = f.stat().st_size
            status = "OK" if min_sz <= size <= max_sz else "SUSPICIOUS SIZE"
            logger.info(f"  [FOUND] {fname:<35} {size:>12,} bytes   {status}")
            if "adapter_model" in fname:
                found_adapter = True
        else:
            if "adapter_model.bin" in fname:
                pass  # .bin may not exist if safetensors is used
            else:
                logger.warning(f"  [MISSING] {fname}")

    for fname in optional:
        f = model_dir / fname
        if f.exists():
            logger.info(f"  [found optional] {fname}")

    if not found_adapter:
        logger.error("No adapter_model.safetensors or adapter_model.bin found.")
        logger.error("Training likely did not save. Cannot proceed.")
        return False

    # Print any checkpoint folders
    ckpts = sorted(
        [
            d.name
            for d in model_dir.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ]
    )
    if ckpts:
        logger.info(f"  Intermediate checkpoints present: {ckpts}")

    logger.info("STEP 1: PASSED")
    return True


# =============================================================================
# STEP 2 — Config inspection
# =============================================================================


def check_config(model_dir: Path) -> bool:
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 2: Inspecting adapter_config.json and training metrics")
    logger.info("=" * 70)

    cfg_path = model_dir / "adapter_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        logger.info(f"  base_model_name_or_path : {cfg.get('base_model_name_or_path')}")
        logger.info(f"  peft_type               : {cfg.get('peft_type')}")
        logger.info(f"  task_type               : {cfg.get('task_type')}")
        logger.info(f"  r (rank)                : {cfg.get('r')}")
        logger.info(f"  lora_alpha              : {cfg.get('lora_alpha')}")
        logger.info(f"  lora_dropout            : {cfg.get('lora_dropout')}")
        logger.info(f"  target_modules          : {cfg.get('target_modules')}")

    metrics_path = model_dir / "train_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        logger.info("")
        logger.info("  --- Training metrics ---")
        for k, v in m.items():
            logger.info(f"  {k:<30}: {v}")

    passk_path = model_dir / "pass_at_k_results.json"
    if passk_path.exists():
        with open(passk_path) as f:
            p = json.load(f)
        logger.info("")
        logger.info("  --- Pass@k from training-time validation ---")
        for k, v in p.items():
            logger.info(f"  Pass@{k:<3} : {v}")

    logger.info("STEP 2: PASSED")
    return True


# =============================================================================
# STEP 3 — Load model and check tokenizer
# =============================================================================


def load_model(model_dir: Path):
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 3: Loading base model + LoRA adapter")
    logger.info("=" * 70)

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    logger.info(f"  Tokenizer loaded in {time.time()-t0:.1f}s")
    logger.info(f"  Vocab size            : {tokenizer.vocab_size}")
    logger.info(f"  Pad token             : {tokenizer.pad_token!r}")
    logger.info(f"  EOS token             : {tokenizer.eos_token!r}")

    # Check FIM special tokens
    fim_tokens = ["<fim_prefix>", "<fim_suffix>", "<fim_middle>", "<extra_id_0>"]
    logger.info("  FIM special tokens:")
    all_present = True
    for tok in fim_tokens:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        present = len(ids) == 1  # must be a single token
        mark = "OK" if present else "MISSING or multi-token"
        logger.info(f"    {tok:<15} -> ids={ids}  [{mark}]")
        if not present:
            all_present = False

    if not all_present:
        logger.error("  FIM tokens are not correctly registered. Model will not work.")

    # Load base model
    peft_cfg = PeftConfig.from_pretrained(str(model_dir))
    base_name = peft_cfg.base_model_name_or_path
    logger.info(f"  Loading base model: {base_name}")

    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        cache_dir=str(Path("./hf_cache/models").resolve()),
    )
    logger.info(f"  Base model loaded in {time.time()-t0:.1f}s")

    # Attach adapter
    t0 = time.time()
    model = PeftModel.from_pretrained(base, str(model_dir))
    model.eval()
    logger.info(f"  Adapter attached in {time.time()-t0:.1f}s")

    # Count trainable params (should be ~3M for rank-8 LoRA)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"  Trainable params: {trainable:,} ({100*trainable/total:.4f}% of total)"
    )
    logger.info(f"  Total params    : {total:,}")

    logger.info("STEP 3: PASSED")
    return model, tokenizer


# =============================================================================
# STEP 4 — Generation probe
# =============================================================================


def generate_for_case(
    model, tokenizer, buggy_code: str, num_candidates: int = 3
) -> List[str]:
    """
    Mask each line in turn, ask the model to fill, reconstruct the full code.
    Returns a list of full-code candidates.
    """
    lines = buggy_code.splitlines()
    candidates = []

    for mask_idx in range(len(lines)):
        prefix = "\n".join(lines[:mask_idx])
        suffix = "\n".join(lines[mask_idx + 1 :])

        prompt = f"<fim_prefix>{prefix}\n" f"<fim_suffix>{suffix}\n" f"<fim_middle>"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.4,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()

        # Reconstruct: original lines, but line mask_idx replaced with generated
        new_lines = lines[:mask_idx] + [generated] + lines[mask_idx + 1 :]
        candidates.append("\n".join(new_lines))

        if len(candidates) >= num_candidates:
            break

    return candidates


def probe_generation(model, tokenizer):
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 4: Generation probe — what does the model actually output?")
    logger.info("=" * 70)

    # Simple FIM test: ask it to complete something obvious
    simple_prompt = (
        "<fim_prefix>def add(a, b):\n    <fim_suffix>\n    return result\n<fim_middle>"
    )
    inputs = tokenizer(simple_prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    logger.info(f"  Greedy sample (T=0):")
    logger.info(f"    Prompt: def add(a, b):\\n    <MASK>\\n    return result")
    logger.info(f"    Output: {generated!r}")

    # Sample generation
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    logger.info(f"  Sampled (T=0.7):")
    logger.info(f"    Output: {generated!r}")

    logger.info("STEP 4: COMPLETE (this is informational, not pass/fail)")


# =============================================================================
# STEP 5 — Full two-stage validation
# =============================================================================


def run_sandbox(code: str, tests: str, timeout: int = 10) -> bool:
    """Inline sandbox — same behavior as the training pipeline."""
    import subprocess
    import random

    temp_dir = Path("./sandbox_tmp").resolve()
    temp_dir.mkdir(exist_ok=True, parents=True)
    temp_file = (temp_dir / f"chk_{random.randint(0,999999)}.py").resolve()
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(code + "\n\n" + tests)
        result = subprocess.run(
            [sys.executable, str(temp_file)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(temp_dir),
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        try:
            temp_file.unlink(missing_ok=True)
        except Exception:
            pass


def validate_all(model, tokenizer):
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 5: Two-stage validation on test cases")
    logger.info("=" * 70)

    try:
        import ast
    except ImportError:
        ast = None

    summary = []

    for tc in TEST_CASES:
        logger.info("")
        logger.info(f"  --- {tc['name']} ---")
        logger.info(f"  Buggy code:")
        for line in tc["buggy"].splitlines():
            logger.info(f"      {line}")

        candidates = generate_for_case(model, tokenizer, tc["buggy"], num_candidates=5)

        stage1_pass = 0
        stage2_pass = 0
        for i, cand in enumerate(candidates):
            # Stage 1: AST
            try:
                ast.parse(cand)
                s1 = True
            except SyntaxError:
                s1 = False

            if s1:
                stage1_pass += 1
                # Stage 2: sandbox
                s2 = run_sandbox(cand, tc["tests"])
                if s2:
                    stage2_pass += 1

            logger.info(
                f"    Candidate {i+1}: stage1={'OK' if s1 else 'FAIL'}  "
                f"stage2={'OK' if s1 and s2 else 'FAIL'}"
            )

        logger.info(f"  Stage 1 passed: {stage1_pass}/{len(candidates)}")
        logger.info(f"  Stage 2 passed: {stage2_pass}/{len(candidates)}")
        summary.append(
            {
                "case": tc["name"],
                "stage1": f"{stage1_pass}/{len(candidates)}",
                "stage2": f"{stage2_pass}/{len(candidates)}",
            }
        )

    logger.info("")
    logger.info("  === Summary ===")
    for row in summary:
        logger.info(
            f"    {row['case'][:50]:<50}  " f"S1={row['stage1']:<6}  S2={row['stage2']}"
        )
    return summary


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="./surgical_repair_output")
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Only check files + config, do not load model",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_path).resolve()
    logger.info(f"Model directory: {model_dir}")

    # 1) Files
    if not check_files(model_dir):
        sys.exit(1)

    # 2) Config
    check_config(model_dir)

    if args.skip_validation:
        logger.info("")
        logger.info("Skipping model load (--skip-validation). Done.")
        return

    # 3) Load
    model, tokenizer = load_model(model_dir)

    # 4) Probe
    probe_generation(model, tokenizer)

    # 5) Validate
    validate_all(model, tokenizer)

    logger.info("")
    logger.info("=" * 70)
    logger.info("VERIFICATION COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
