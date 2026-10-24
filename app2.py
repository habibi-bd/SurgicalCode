"""
app.py — Gradio ChatInterface for the Surgical Code Repair model (v2).

Uses the retrained adapter in ./surgical_repair_v2.

Run:
    python app.py

Opens at http://127.0.0.1:7860
"""

import os
import inspect
from pathlib import Path
from typing import List

# ---- Local HF cache + auth (before transformers/peft imports) ----
os.environ.setdefault("HF_HOME", str(Path("./hf_cache").resolve()))
os.environ.setdefault("TRANSFORMERS_CACHE", str(Path("./hf_cache/models").resolve()))

_hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
try:
    from huggingface_hub import login as _hf_login

    if _hf_token:
        _hf_login(token=_hf_token, add_to_git_credential=False)
except Exception:
    pass

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# ===========================================================================
# CONFIG — now points at the retrained v2 model
# ===========================================================================

MODEL_DIR = "habibCSEBRUR/surgical-code-repair-v2"


CACHE_DIR = str(Path("./hf_cache/models").resolve())

FIM_PREFIX = "<fim_prefix>"
FIM_SUFFIX = "<fim_suffix>"
FIM_MIDDLE = "<fim_middle>"

# Each row = [message, temperature, max_new_tokens]
EXAMPLES = [
    ["def add(a, b):\n    result = a - b\n    return result", 0.3, 48],
    ["def mul(a, b):\n    result = a + b\n    return result", 0.3, 48],
    ["def max2(x, y):\n    return min(x, y)", 0.3, 48],
    ["def first(lst):\n    return lst[-1]", 0.3, 48],
    ["def is_even(n):\n    return n % 2 == 1", 0.3, 48],
    ["def shout(text):\n    return text.lower()", 0.3, 48],
    [
        "def factorial(n):\n    if n == 0:\n        return 0\n    return n * factorial(n - 1)",
        0.3,
        48,
    ],
]


# ===========================================================================
# MODEL LOADING
# ===========================================================================


def load_model():
    print("[app] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_cfg = PeftConfig.from_pretrained(MODEL_DIR)
    base_name = peft_cfg.base_model_name_or_path

    print(f"[app] Loading base model: {base_name}")
    base = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    )

    # Resize embeddings BEFORE attaching adapter so vocab sizes match.
    target_vocab = len(tokenizer)
    current_vocab = base.get_input_embeddings().weight.shape[0]
    if current_vocab != target_vocab:
        print(f"[app] Resizing base embeddings: {current_vocab} -> {target_vocab}")
        base.resize_token_embeddings(target_vocab)

    print("[app] Attaching LoRA adapter...")
    model = PeftModel.from_pretrained(base, MODEL_DIR)
    model.eval()

    n_lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"[app] Model ready. LoRA params={n_lora:,} "
        f"({100 * n_lora / n_total:.3f}% of {n_total:,})"
    )
    return model, tokenizer


print("=" * 70)
print("SURGICAL CODE REPAIR — Gradio ChatInterface (v2)")
print("=" * 70)
model, tokenizer = load_model()
print("=" * 70)


# ===========================================================================
# FIM MASKING — fixed to handle return-only functions
# ===========================================================================


def _find_mask_line(lines: List[str]) -> int:
    """
    Find the line most likely containing the bug.

    Priority order:
      1. Assignment line (contains '=' but not a control-flow keyword)
      2. Return statement
      3. Any non-def, non-blank line
    """
    # Pass 1: assignment lines
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s or s.startswith(("#", "def ", "class ", "import ", "from ")):
            continue
        if s.startswith(("if ", "for ", "while ", "with ", "elif ", "else")):
            continue
        if "=" in ln and "==" not in ln.split("=")[0]:  # avoid comparisons
            return i

    # Pass 2: return statements (fixes the eval-v2 failure mode)
    for i, ln in enumerate(lines):
        if ln.strip().startswith("return "):
            return i

    # Pass 3: fall back to first non-def line
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s and not s.startswith(("def ", "class ")):
            return i

    return 0


def _clean_fragment(text: str) -> str:
    """
    Strip leaked markers and truncate to the first logical statement.

    The model sometimes continues generating past a single statement,
    producing output like:  "    return max(x, y)    return max(x, y)    if x > y:"
    on one physical line. We cut at the second occurrence of a
    statement-starting keyword (return/if/for/while/def) to keep
    only the first repair.
    """
    import re

    # 1) Cut at first newline (if any)
    first = text.split("\n")[0]

    # 2) Remove leaked FIM / chat tokens
    for tok in ["User:", "Assistant:", FIM_MIDDLE, FIM_SUFFIX, FIM_PREFIX]:
        first = first.replace(tok, "")

    # 3) Truncate at the SECOND occurrence of a statement keyword.
    #    The first occurrence is expected; the second signals the model
    #    kept generating past the repair.
    for kw in [r"\breturn\b", r"\bif\b", r"\bfor\b", r"\bwhile\b", r"\bdef\b"]:
        matches = [m.start() for m in re.finditer(kw, first)]
        if len(matches) > 1:
            first = first[: matches[1]].rstrip()

    return first


def repair_code(code: str, temperature: float, max_new_tokens: int) -> str:
    """Repair the input code using FIM masking."""
    if not code or not code.strip():
        return "⚠️ Please paste some Python code."

    lines = code.splitlines()
    if len(lines) < 2:
        return "⚠️ Paste a full function (at least 2 lines)."

    mask_idx = _find_mask_line(lines)
    prefix = "\n".join(lines[:mask_idx])
    suffix = "\n".join(lines[mask_idx + 1 :])

    # Match training-time FIM format exactly: only add '\n' before
    # <fim_middle> when the suffix is non-empty.
    suffix_part = f"{suffix}\n" if suffix else ""
    prompt = f"{FIM_PREFIX}{prefix}\n{FIM_SUFFIX}{suffix_part}{FIM_MIDDLE}"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384)

    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=temperature > 0.01,
                temperature=max(temperature, 0.01),
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    except Exception as e:
        return f"⚠️ Generation failed: {e}"

    raw = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )
    fragment = _clean_fragment(raw)

    # Preserve the original indentation if the fragment lost it
    original_indent = len(lines[mask_idx]) - len(lines[mask_idx].lstrip())
    if fragment and not fragment.startswith(" "):
        fragment = " " * original_indent + fragment

    # Reconstruct: put the fragment back in place of the masked line
    new_lines = lines[:mask_idx] + [fragment] + lines[mask_idx + 1 :]
    repaired = "\n".join(new_lines)

    return (
        f"**Masked line:** `{lines[mask_idx]}`\n\n"
        f"**Model output fragment:** `{fragment}`\n\n"
        f"**Repaired function:**\n```python\n{repaired}\n```"
    )


# ===========================================================================
# CHAT FUNCTION
# ===========================================================================


def chat_fn(message, history, temperature, max_new_tokens):
    msg = message.strip()
    if not msg:
        return "⚠️ Please paste some Python code."

    if not (
        msg.startswith(("def ", "class ", "import ", "from "))
        or "=" in msg
        or "return " in msg
    ):
        return (
            "⚠️ This model only repairs Python functions. "
            "Paste a function definition, indented with 4 spaces."
        )

    return repair_code(msg, temperature, max_new_tokens)


# ===========================================================================
# UI
# ===========================================================================


def build_demo():
    sig = inspect.signature(gr.ChatInterface.__init__)
    supports_additional = "additional_inputs" in sig.parameters

    kwargs = dict(
        fn=chat_fn,
        title="🩺 Surgical Code Repair (v2)",
        description=(
            "Fine-tuned `deepseek-coder-1.3b-base` (QLoRA) for localized "
            "bug repair using Fill-in-the-Middle masking.\n\n"
            "**How to use:** Paste a Python function with 4-space indentation. "
            "The model masks the buggy line, generates a replacement, and "
            "reconstructs the function.\n\n"
            "Trained on 40 diverse bug templates — supports arithmetic, "
            "comparison, base-case, list, string, boolean, and off-by-one "
            "repairs."
        ),
        examples=EXAMPLES,
        cache_examples=False,
    )

    if supports_additional:
        kwargs["additional_inputs"] = [
            gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Temperature"),
            gr.Slider(8, 128, value=48, step=8, label="Max new tokens"),
        ]
        if "additional_inputs_accordion" in sig.parameters:
            kwargs["additional_inputs_accordion"] = gr.Accordion(
                "Generation settings", open=False
            )

    return gr.ChatInterface(**kwargs)


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    print(f"[ui] Gradio {gr.__version__}")
    demo = build_demo()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
