"""
merge_model.py — Merge the LoRA adapter into the base model.

Produces a single self-contained model folder that can be loaded with
plain transformers (no PEFT required).

Run:
    python merge_model.py

Output:
    ./surgical_repair_merged/   (~5.4 GB)
"""

import os
import shutil
from pathlib import Path

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

ADAPTER_DIR = "./surgical_repair_output"
OUTPUT_DIR = "./surgical_repair_merged"
CACHE_DIR = str(Path("./hf_cache/models").resolve())


def main():
    adapter_dir = Path(ADAPTER_DIR).resolve()
    output_dir = Path(OUTPUT_DIR).resolve()

    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_dir}")

    if output_dir.exists():
        print(f"[merge] Removing existing {output_dir}...")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    print("[merge] Loading tokenizer from adapter...")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[merge] Reading adapter config...")
    peft_cfg = PeftConfig.from_pretrained(str(adapter_dir))
    base_name = peft_cfg.base_model_name_or_path

    print(f"[merge] Loading base model: {base_name}")
    base = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    )

    # Resize to match trained tokenizer BEFORE attaching adapter
    target_vocab = len(tokenizer)
    current_vocab = base.get_input_embeddings().weight.shape[0]
    if current_vocab != target_vocab:
        print(f"[merge] Resizing embeddings: {current_vocab} -> {target_vocab}")
        base.resize_token_embeddings(target_vocab)

    print("[merge] Attaching adapter...")
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))

    print("[merge] Merging LoRA into base weights...")
    merged = peft_model.merge_and_unload()

    print(f"[merge] Saving merged model to {output_dir}...")
    merged.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    # Copy adapter config for reference (optional)
    cfg_src = adapter_dir / "adapter_config.json"
    if cfg_src.exists():
        shutil.copy(cfg_src, output_dir / "adapter_config_source.json")

    # Report
    size_mb = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file()) / (
        1024 * 1024
    )
    print(f"[merge] Done. Total size: {size_mb:,.1f} MB")
    print(f"[merge] Output: {output_dir}")


if __name__ == "__main__":
    main()
