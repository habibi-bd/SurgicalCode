# """
# app.py — Gradio ChatInterface for the Surgical Code Repair model.

# Run:
#     python app.py

# Opens at http://127.0.0.1:7860

# Compatible with Gradio 6.x (and 4.x/3.x with adaptive fallbacks).
# """

# import os
# import inspect
# from pathlib import Path

# # ---------------------------------------------------------------------------
# # Local HF cache + auth (must run before transformers/peft imports)
# # ---------------------------------------------------------------------------
# os.environ.setdefault("HF_HOME", str(Path("./hf_cache").resolve()))
# os.environ.setdefault("TRANSFORMERS_CACHE", str(Path("./hf_cache/models").resolve()))

# _hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
# try:
#     from huggingface_hub import login as _hf_login

#     if _hf_token:
#         _hf_login(token=_hf_token, add_to_git_credential=False)
# except Exception:
#     pass

# import torch
# import gradio as gr
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel, PeftConfig

# # ===========================================================================
# # CONFIG
# # ===========================================================================

# MODEL_DIR = "./surgical_repair_output"
# CACHE_DIR = str(Path("./hf_cache/models").resolve())

# SYSTEM_PREAMBLE = (
#     "You are a code repair assistant. "
#     "When given buggy Python code, output the corrected version.\n\n"
# )

# # Each row = [message, temperature, max_new_tokens]
# # Gradio requires this shape when additional_inputs are present.
# EXAMPLES = [
#     ["def compute(a, b):\n    var = a + b\n    return var", 0.3, 96],
#     ["def compute_1(a, b):\n    var_1 = a + b\n    return var_1", 0.3, 96],
#     ["def compute_42(a, b):\n    var_42 = a + b\n    return var_42", 0.3, 96],
#     ["def add_numbers(a, b):\n    result = a + b\n    return result", 0.3, 96],
#     ["def multiply(a, b):\n    product = a + b\n    return product", 0.3, 96],
# ]


# # ===========================================================================
# # MODEL LOADING
# # ===========================================================================


# def load_model():
#     print("[app] Loading tokenizer...")
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     peft_cfg = PeftConfig.from_pretrained(MODEL_DIR)
#     base_name = peft_cfg.base_model_name_or_path

#     print(f"[app] Loading base model: {base_name}")
#     base = AutoModelForCausalLM.from_pretrained(
#         base_name,
#         torch_dtype=torch.float32,
#         device_map="cpu",
#         trust_remote_code=True,
#         cache_dir=CACHE_DIR,
#     )

#     # Resize embeddings BEFORE attaching adapter so vocab sizes match.
#     target_vocab = len(tokenizer)
#     current_vocab = base.get_input_embeddings().weight.shape[0]
#     if current_vocab != target_vocab:
#         print(f"[app] Resizing base embeddings: {current_vocab} -> {target_vocab}")
#         base.resize_token_embeddings(target_vocab)

#     print("[app] Attaching LoRA adapter...")
#     model = PeftModel.from_pretrained(base, MODEL_DIR)
#     model.eval()

#     # Count LoRA params regardless of requires_grad (they're frozen at inference)
#     n_lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
#     n_total = sum(p.numel() for p in model.parameters())
#     print(
#         f"[app] Model ready. LoRA params={n_lora:,} "
#         f"({100 * n_lora / n_total:.3f}% of {n_total:,})"
#     )

#     return model, tokenizer


# print("=" * 70)
# print("SURGICAL CODE REPAIR — Gradio ChatInterface")
# print("=" * 70)
# model, tokenizer = load_model()
# print("=" * 70)


# # ===========================================================================
# # INFERENCE
# # ===========================================================================


# def generate_reply(message: str, temperature: float, max_new_tokens: int) -> str:
#     if not message or not message.strip():
#         return "Please type something."

#     prompt = SYSTEM_PREAMBLE + "User: " + message.strip() + "\nAssistant:"

#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384)

#     try:
#         with torch.no_grad():
#             out = model.generate(
#                 **inputs,
#                 max_new_tokens=int(max_new_tokens),
#                 do_sample=temperature > 0.01,
#                 temperature=max(temperature, 0.01),
#                 top_p=0.95,
#                 pad_token_id=tokenizer.pad_token_id,
#                 eos_token_id=tokenizer.eos_token_id,
#             )
#     except Exception as e:
#         return f"⚠️ Generation failed: {e}"

#     text = tokenizer.decode(
#         out[0][inputs["input_ids"].shape[1] :],
#         skip_special_tokens=True,
#     ).strip()

#     return text or "(empty response)"


# # ===========================================================================
# # CHAT FUNCTION
# # ===========================================================================


# def chat_fn(message, history, temperature, max_new_tokens):
#     if not message.strip().startswith(("def ", "class ", "import ")):
#         return "⚠️ This model only repairs code. Paste a Python function."
#     return generate_reply(message, temperature, max_new_tokens)


# # ===========================================================================
# # UI
# # ===========================================================================


# def build_demo():
#     sig = inspect.signature(gr.ChatInterface.__init__)
#     supports_additional = "additional_inputs" in sig.parameters

#     kwargs = dict(
#         fn=chat_fn,
#         title="🩺 Surgical Code Repair",
#         description=(
#             "Fine-tuned `deepseek-coder-1.3b-base` (QLoRA) for localized "
#             "bug repair. Paste buggy Python code and the model will attempt "
#             "a fix.\n\n"
#             "⚠️ **Note:** Trained on a narrow synthetic corpus "
#             "(`a + b` → `a * b`). Expect correct behavior only on inputs "
#             "similar to training data."
#         ),
#         examples=EXAMPLES,
#         cache_examples=False,
#     )

#     if supports_additional:
#         kwargs["additional_inputs"] = [
#             gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Temperature"),
#             gr.Slider(16, 256, value=96, step=8, label="Max new tokens"),
#         ]
#         if "additional_inputs_accordion" in sig.parameters:
#             kwargs["additional_inputs_accordion"] = gr.Accordion(
#                 "Generation settings", open=False
#             )

#     return gr.ChatInterface(**kwargs)


# # ===========================================================================
# # ENTRY POINT
# # ===========================================================================

# if __name__ == "__main__":
#     print(f"[ui] Gradio {gr.__version__}")
#     demo = build_demo()
#     demo.launch(
#         server_name="127.0.0.1",
#         server_port=7860,
#         share=False,
#         show_error=True,
#     )


"""
app.py — Gradio ChatInterface for the Surgical Code Repair model.

Run:
    python app.py

Uses FIM (Fill-in-the-Middle) prompts matching the training format.
No chat-style prompting — the model wasn't trained for it.
"""

import os
import inspect
from pathlib import Path

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
# CONFIG
# ===========================================================================

MODEL_DIR = "./surgical_repair_output"
CACHE_DIR = str(Path("./hf_cache/models").resolve())

FIM_PREFIX = "<fim_prefix>"
FIM_SUFFIX = "<fim_suffix>"
FIM_MIDDLE = "<fim_middle>"

EXAMPLES = [
    ["def compute(a, b):\n    var = a + b\n    return var", 0.3, 48],
    ["def compute_N(a, b):\n    var_N = a + b\n    return var_N", 0.3, 48],
    ["def compute_42(a, b):\n    var_42 = a + b\n    return var_42", 0.3, 48],
    ["def add_numbers(a, b):\n    result = a + b\n    return result", 0.3, 48],
    ["def multiply(a, b):\n    product = a + b\n    return product", 0.3, 48],
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
print("SURGICAL CODE REPAIR — Gradio ChatInterface")
print("=" * 70)
model, tokenizer = load_model()
print("=" * 70)


# ===========================================================================
# FIM MASKING + REPAIR
# ===========================================================================


def _find_mask_line(lines: list) -> int:
    """
    Find the line most likely to contain the bug.
    Prefer the first line containing '=' that isn't a def/import/comment.
    """
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(
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
            )
        ):
            continue
        if "=" in line:
            return i
    # Fallback: the middle line
    return len(lines) // 2 if lines else 0


def _clean_fragment(text: str) -> str:
    """Strip leaked chat tokens, FIM tokens, and take only the first line."""
    # Cut at first newline — we only need the replacement line
    first_line = text.split("\n")[0]
    # Remove any leaked markers
    for tok in ["User:", "Assistant:", FIM_MIDDLE, FIM_SUFFIX, FIM_PREFIX]:
        first_line = first_line.replace(tok, "")
    return first_line.rstrip()


def repair_code(code: str, temperature: float, max_new_tokens: int) -> str:
    """Repair the input code using FIM masking. Returns the repaired code."""
    if not code or not code.strip():
        return "⚠️ Please paste some Python code."

    lines = code.splitlines()
    if len(lines) < 2:
        return "⚠️ Paste a full function (at least 2 lines)."

    mask_idx = _find_mask_line(lines)
    prefix = "\n".join(lines[:mask_idx])
    suffix = "\n".join(lines[mask_idx + 1 :])

    # FIM prompt exactly matching training format
    prompt = f"{FIM_PREFIX}{prefix}\n{FIM_SUFFIX}{suffix}\n{FIM_MIDDLE}"

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
    # Only handle code-looking input
    if not (msg.startswith(("def ", "class ", "import ", "from ")) or "=" in msg):
        return (
            "⚠️ This model only repairs Python code. "
            "Paste a function definition (indented with 4 spaces)."
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
        title="🩺 Surgical Code Repair",
        description=(
            "Fine-tuned `deepseek-coder-1.3b-base` (QLoRA) for localized "
            "bug repair using Fill-in-the-Middle masking.\n\n"
            "**Paste a Python function with proper 4-space indentation.** "
            "The model masks the assignment line, generates a repair, and "
            "reconstructs the function.\n\n"
            "⚠️ Trained on a single bug pattern (`a + b` → `a * b`). "
            "Expect correct repairs only on inputs matching the training "
            "distribution."
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
