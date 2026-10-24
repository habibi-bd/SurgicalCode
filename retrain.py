"""
retrain.py — Retrain the Surgical Code Repair model on a broader corpus.

This script:
  - Generates ~3000 bug/fix pairs from ~40 diverse templates
  - Fine-tunes deepseek-coder-1.3b-base with QLoRA (rank 16)
  - Saves to ./surgical_repair_v2/ (does NOT touch the old adapter)

Run:
    python retrain.py

Duration: ~2.5-3 hours on CPU.
"""

import os
import sys
import json
import time
import random
import logging
import inspect
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

# ---- Local HF cache ----
os.environ.setdefault("HF_HOME", str(Path("./hf_cache").resolve()))
os.environ.setdefault("TRANSFORMERS_CACHE", str(Path("./hf_cache/models").resolve()))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_hf_token = os.environ.get("HF_TOKEN")
try:
    from huggingface_hub import login as _hf_login

    if _hf_token:
        _hf_login(token=_hf_token, add_to_git_credential=False)
except Exception:
    pass

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
    TrainerCallback,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [RETRAIN] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================
# CONFIG
# =============================================================


@dataclass
class RetrainConfig:
    base_model: str = "deepseek-ai/deepseek-coder-1.3b-base"
    output_dir: str = "./surgical_repair_v2"
    cache_dir: str = ""

    num_examples: int = 3000
    train_ratio: float = 0.9
    max_seq_length: int = 512

    fim_prefix: str = "<fim_prefix>"
    fim_suffix: str = "<fim_suffix>"
    fim_middle: str = "<fim_middle>"
    mask_token: str = "<extra_id_0>"

    # LoRA — bumped to r=16 for more capacity
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    batch_size: int = 1
    grad_accum: int = 8
    max_steps: int = 400
    learning_rate: float = 2e-4
    warmup_steps: int = 30
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_hours: float = 3.0
    seed: int = 42

    def __post_init__(self):
        self.cache_dir = str(Path("./hf_cache/models").resolve())


# =============================================================
# DATASET — ~40 bug templates
# =============================================================

BUG_TEMPLATES: List[Tuple[str, str]] = [
    # --- Arithmetic ---
    (
        "def add_{N}(a, b):\n    result = a - b\n    return result",
        "def add_{N}(a, b):\n    result = a + b\n    return result",
    ),
    (
        "def sub_{N}(a, b):\n    result = a + b\n    return result",
        "def sub_{N}(a, b):\n    result = a - b\n    return result",
    ),
    (
        "def mul_{N}(a, b):\n    result = a + b\n    return result",
        "def mul_{N}(a, b):\n    result = a * b\n    return result",
    ),
    (
        "def mul_{N}(a, b):\n    result = a - b\n    return result",
        "def mul_{N}(a, b):\n    result = a * b\n    return result",
    ),
    (
        "def div_{N}(a, b):\n    result = a * b\n    return result",
        "def div_{N}(a, b):\n    result = a / b\n    return result",
    ),
    (
        "def idiv_{N}(a, b):\n    result = a / b\n    return result",
        "def idiv_{N}(a, b):\n    result = a // b\n    return result",
    ),
    (
        "def mod_{N}(a, b):\n    result = a // b\n    return result",
        "def mod_{N}(a, b):\n    result = a % b\n    return result",
    ),
    (
        "def pow_{N}(a, b):\n    result = a * b\n    return result",
        "def pow_{N}(a, b):\n    result = a ** b\n    return result",
    ),
    ("def neg_{N}(x):\n    return x", "def neg_{N}(x):\n    return -x"),
    ("def absval_{N}(x):\n    return -x", "def absval_{N}(x):\n    return abs(x)"),
    # --- Comparison ---
    (
        "def gt_{N}(x, y):\n    if x < y:\n        return True\n    return False",
        "def gt_{N}(x, y):\n    if x > y:\n        return True\n    return False",
    ),
    (
        "def lt_{N}(x, y):\n    if x > y:\n        return True\n    return False",
        "def lt_{N}(x, y):\n    if x < y:\n        return True\n    return False",
    ),
    (
        "def eq_{N}(x, y):\n    if x != y:\n        return True\n    return False",
        "def eq_{N}(x, y):\n    if x == y:\n        return True\n    return False",
    ),
    (
        "def ne_{N}(x, y):\n    if x == y:\n        return True\n    return False",
        "def ne_{N}(x, y):\n    if x != y:\n        return True\n    return False",
    ),
    (
        "def max2_{N}(x, y):\n    return min(x, y)",
        "def max2_{N}(x, y):\n    return max(x, y)",
    ),
    (
        "def min2_{N}(x, y):\n    return max(x, y)",
        "def min2_{N}(x, y):\n    return min(x, y)",
    ),
    # --- Base cases ---
    (
        "def fact_{N}(n):\n    if n == 0:\n        return 0\n    return n * fact_{N}(n - 1)",
        "def fact_{N}(n):\n    if n == 0:\n        return 1\n    return n * fact_{N}(n - 1)",
    ),
    (
        "def fib_{N}(n):\n    if n <= 1:\n        return 0\n    return fib_{N}(n - 1) + fib_{N}(n - 2)",
        "def fib_{N}(n):\n    if n <= 1:\n        return 1\n    return fib_{N}(n - 1) + fib_{N}(n - 2)",
    ),
    (
        "def sum_list_{N}(lst):\n    if not lst:\n        return 1\n    return lst[0] + sum_list_{N}(lst[1:])",
        "def sum_list_{N}(lst):\n    if not lst:\n        return 0\n    return lst[0] + sum_list_{N}(lst[1:])",
    ),
    # --- List ops ---
    (
        "def find_max_{N}(lst):\n    max_val = 0\n    for v in lst:\n        if v > max_val:\n            max_val = v\n    return max_val",
        "def find_max_{N}(lst):\n    max_val = lst[0]\n    for v in lst:\n        if v > max_val:\n            max_val = v\n    return max_val",
    ),
    (
        "def first_{N}(lst):\n    return lst[-1]",
        "def first_{N}(lst):\n    return lst[0]",
    ),
    ("def last_{N}(lst):\n    return lst[0]", "def last_{N}(lst):\n    return lst[-1]"),
    (
        "def drop_first_{N}(lst):\n    return lst[:-1]",
        "def drop_first_{N}(lst):\n    return lst[1:]",
    ),
    (
        "def drop_last_{N}(lst):\n    return lst[1:]",
        "def drop_last_{N}(lst):\n    return lst[:-1]",
    ),
    (
        "def size_{N}(lst):\n    return len(lst) - 1",
        "def size_{N}(lst):\n    return len(lst)",
    ),
    # --- String ops ---
    (
        "def shout_{N}(text):\n    return text.lower()",
        "def shout_{N}(text):\n    return text.upper()",
    ),
    (
        "def quiet_{N}(text):\n    return text.upper()",
        "def quiet_{N}(text):\n    return text.lower()",
    ),
    (
        "def trim_left_{N}(text):\n    return text.rstrip()",
        "def trim_left_{N}(text):\n    return text.lstrip()",
    ),
    (
        "def trim_right_{N}(text):\n    return text.lstrip()",
        "def trim_right_{N}(text):\n    return text.rstrip()",
    ),
    (
        "def is_even_{N}(n):\n    return n % 2 == 1",
        "def is_even_{N}(n):\n    return n % 2 == 0",
    ),
    (
        "def is_odd_{N}(n):\n    return n % 2 == 0",
        "def is_odd_{N}(n):\n    return n % 2 == 1",
    ),
    # --- Off-by-one ---
    ("def next_{N}(n):\n    return n", "def next_{N}(n):\n    return n + 1"),
    ("def prev_{N}(n):\n    return n", "def prev_{N}(n):\n    return n - 1"),
    ("def inc_{N}(n):\n    return n - 1", "def inc_{N}(n):\n    return n + 1"),
    ("def dec_{N}(n):\n    return n + 1", "def dec_{N}(n):\n    return n - 1"),
    # --- Boolean / None ---
    (
        "def always_true_{N}(x):\n    return False",
        "def always_true_{N}(x):\n    return True",
    ),
    (
        "def always_false_{N}(x):\n    return True",
        "def always_false_{N}(x):\n    return False",
    ),
    ("def nothing_{N}(x):\n    return True", "def nothing_{N}(x):\n    return None"),
]


def generate_dataset(cfg: RetrainConfig) -> Tuple[Dataset, Dataset]:
    random.seed(cfg.seed)
    examples = []
    per_tmpl = max(1, cfg.num_examples // len(BUG_TEMPLATES))

    for tmpl_idx, (buggy_t, fixed_t) in enumerate(BUG_TEMPLATES):
        for k in range(per_tmpl):
            n = f"{tmpl_idx}_{k}"
            examples.append(
                {
                    "buggy": buggy_t.format(N=n),
                    "fixed": fixed_t.format(N=n),
                }
            )

    random.shuffle(examples)
    logger.info(
        f"Generated {len(examples)} examples from " f"{len(BUG_TEMPLATES)} templates"
    )

    split = int(len(examples) * cfg.train_ratio)
    return (
        Dataset.from_list(examples[:split]),
        Dataset.from_list(examples[split:]),
    )


# =============================================================
# FIM CONVERSION
# =============================================================


def convert_to_fim(example: Dict, cfg: RetrainConfig) -> Dict:
    import difflib

    bl = example["buggy"].splitlines(keepends=True)
    fl = example["fixed"].splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(None, bl, fl)

    prefix_lines, suffix_lines, fix_lines = [], [], []
    seen_change = False

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            if not seen_change:
                prefix_lines.extend(bl[i1:i2])
            else:
                suffix_lines.extend(bl[i1:i2])
        else:
            seen_change = True
            fix_lines.extend(fl[j1:j2])

    prefix = "".join(prefix_lines)
    suffix = "".join(suffix_lines)
    fix = "".join(fix_lines)

    fim_input = f"{cfg.fim_prefix}{prefix}{cfg.fim_suffix}{suffix}{cfg.fim_middle}"
    return {"fim_text": fim_input, "target_text": fix}


def tokenize_fim(example: Dict, cfg: RetrainConfig, tok) -> Dict:
    fim = example["fim_text"]
    tgt = example["target_text"]
    if not fim or not tgt:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    inp_ids = tok(
        fim, truncation=True, max_length=cfg.max_seq_length, add_special_tokens=False
    )["input_ids"]
    tgt_ids = tok(
        tgt, truncation=True, max_length=cfg.max_seq_length, add_special_tokens=False
    )["input_ids"]

    combined = (inp_ids + tgt_ids)[: cfg.max_seq_length]
    labels = ([-100] * len(inp_ids) + tgt_ids)[: cfg.max_seq_length]
    attn = [1] * len(combined)
    return {"input_ids": combined, "attention_mask": attn, "labels": labels}


# =============================================================
# CALLBACKS
# =============================================================


class TimeBudget(TrainerCallback):
    def __init__(self, max_hours):
        self.max_sec = max_hours * 3600
        self.t0 = None

    def on_train_begin(self, args, state, control, **kw):
        self.t0 = time.time()
        logger.info(f"Time budget: {self.max_sec/3600:.1f}h")

    def on_step_end(self, args, state, control, **kw):
        if self.t0 and (time.time() - self.t0) > self.max_sec:
            logger.warning("Time budget exceeded. Stopping.")
            control.should_training_stop = True

    def on_log(self, args, state, control, logs=None, **kw):
        if logs and self.t0:
            h = (time.time() - self.t0) / 3600
            logger.info(f"[{h:.2f}h] {logs}")


# =============================================================
# MODEL LOADING
# =============================================================


def load_base_model(cfg: RetrainConfig):
    logger.info("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.add_special_tokens(
        {
            "additional_special_tokens": [
                cfg.fim_prefix,
                cfg.fim_suffix,
                cfg.fim_middle,
                cfg.mask_token,
            ]
        }
    )

    logger.info(f"Loading base model: {cfg.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        cache_dir=cfg.cache_dir,
    )
    model.resize_token_embeddings(len(tok))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_targets,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {n_train:,} ({100*n_train/n_total:.3f}%)")
    return model, tok


# =============================================================
# TRAINER
# =============================================================


def build_trainer(cfg, model, tok, train_ds, eval_ds):
    collator = DataCollatorForLanguageModeling(
        tokenizer=tok, mlm=False, pad_to_multiple_of=8
    )

    candidate_kwargs = {
        "output_dir": cfg.output_dir,
        "do_train": True,
        "do_eval": True,
        "num_train_epochs": 5,
        "max_steps": cfg.max_steps,
        "per_device_train_batch_size": cfg.batch_size,
        "per_device_eval_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.grad_accum,
        "learning_rate": cfg.learning_rate,
        "warmup_steps": cfg.warmup_steps,
        "weight_decay": cfg.weight_decay,
        "max_grad_norm": cfg.max_grad_norm,
        "eval_strategy": "steps",
        "evaluation_strategy": "steps",
        "eval_steps": 200,
        "logging_steps": 5,
        "logging_dir": os.path.join(cfg.output_dir, "logs"),
        "report_to": ["none"],
        "save_steps": 200,
        "save_strategy": "steps",
        "save_total_limit": 2,
        "use_cpu": True,
        "no_cuda": True,
        "fp16": False,
        "bf16": False,
        "optim": "adamw_torch",
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
        "gradient_checkpointing": True,
        "seed": cfg.seed,
        "remove_unused_columns": False,
        "label_names": ["labels"],
    }
    sig = inspect.signature(TrainingArguments.__init__)
    valid = set(sig.parameters.keys())
    kwargs = {k: v for k, v in candidate_kwargs.items() if k in valid}
    dropped = set(candidate_kwargs) - valid
    if dropped:
        logger.warning(f"Dropped unsupported kwargs: {sorted(dropped)}")

    args = TrainingArguments(**kwargs)
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=[TimeBudget(cfg.max_hours)],
    )


# =============================================================
# MAIN
# =============================================================


def main():
    cfg = RetrainConfig()
    set_seed(cfg.seed)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("RETRAIN — Broader Surgical Code Repair Corpus")
    logger.info("=" * 70)

    model, tok = load_base_model(cfg)

    logger.info("Generating dataset...")
    raw_train, raw_eval = generate_dataset(cfg)

    logger.info("Converting to FIM format...")
    train_fim = raw_train.map(
        lambda ex: convert_to_fim(ex, cfg),
        remove_columns=raw_train.column_names,
    )
    eval_fim = raw_eval.map(
        lambda ex: convert_to_fim(ex, cfg),
        remove_columns=raw_eval.column_names,
    )

    logger.info("Tokenizing...")
    train_ds = train_fim.map(
        lambda ex: tokenize_fim(ex, cfg, tok),
        remove_columns=train_fim.column_names,
    )
    eval_ds = eval_fim.map(
        lambda ex: tokenize_fim(ex, cfg, tok),
        remove_columns=eval_fim.column_names,
    )

    trainer = build_trainer(cfg, model, tok, train_ds, eval_ds)
    logger.info("Training...")
    trainer.train()

    logger.info("Saving adapter...")
    trainer.save_model(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)
    logger.info(f"Saved to {cfg.output_dir}")
    logger.info("DONE.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
