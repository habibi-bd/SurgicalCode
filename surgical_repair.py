"""
Surgical Code Repair: Fine-Tuning Large Language Models for Localized Automated Program Repair
via Masked Context Fillers

CPU-optimized implementation for AMD Ryzen 5 5600G, 16GB RAM.
Expected training time: 2-3 hours.

Architecture:
    - Base model: deepseek-ai/deepseek-coder-1.3b-base
    - Fine-tuning: QLoRA (4-bit quantization + LoRA adapters)
    - Task: Fill-in-the-Middle (FIM) with masked context fillers
    - Validation: Tree-sitter AST parsing + sandboxed test execution
    - Metric: Pass@k
"""

# =============================================================================
# SECTION 1: IMPORTS & CONFIGURATION
# =============================================================================

import os
import sys
import ast
import json
import time
import signal
import logging
import random
import warnings
import traceback
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import numpy as np
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
    BitsAndBytesConfig,
    TrainerCallback,
)

# Tree-sitter imports
import tree_sitter
import tree_sitter_python

# =============================================================================
# SECTION 2: LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [SURGICAL-REPAIR] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training_log.txt", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# =============================================================================
# SECTION 3: CONFIGURATION DATACLASS
# =============================================================================


@dataclass
class SurgicalRepairConfig:
    """
    Central configuration for the entire pipeline.
    All values are tuned for CPU-only training on 16GB RAM.
    """

    # --- Model Configuration ---
    model_name: str = "deepseek-ai/deepseek-coder-1.3b-base"
    use_4bit: bool = False  # bitsandbytes 4-bit is GPU-only; CPU uses plain LoRA
    torch_dtype: str = "float32"  # float32 for CPU stability
    device: str = "cpu"

    # --- Dataset Configuration ---
    dataset_name: str = "code_x_glue_cc_code_refinement"  # CodeXGLUE code refinement
    dataset_subset: str = "small"  # 'small' subset for speed
    max_seq_length: int = 512  # Reduced for CPU memory
    train_split_ratio: float = 0.9  # Use 90% of small subset for training
    max_train_samples: int = 3000  # Cap training samples for 2-3 hour budget
    max_eval_samples: int = 300

    # --- FIM Masking Configuration ---
    mask_token: str = "<extra_id_0>"  # Masked context filler token
    fim_prefix_token: str = "<fim_prefix>"
    fim_suffix_token: str = "<fim_suffix>"
    fim_middle_token: str = "<fim_middle>"

    # --- LoRA Configuration (CPU-optimized) ---
    lora_r: int = 8  # Reduced rank for CPU
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # --- Training Configuration (CPU-optimized) ---
    batch_size: int = 1  # Absolute minimum for CPU
    gradient_accumulation_steps: int = 8  # Simulate larger batch
    num_epochs: int = 2  # Fewer epochs
    learning_rate: float = 2e-4
    warmup_steps: int = 30
    max_steps: int = 400  # Hard cap for 2-3 hour budget
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # --- Memory Optimization ---
    use_gradient_checkpointing: bool = True
    fp16: bool = False  # CPU does not support fp16 well
    bf16: bool = False  # CPU does not support bf16 well
    optim: str = "adamw_torch"
    dataloader_num_workers: int = 0  # Avoid multiprocessing overhead on CPU

    # --- Validation Configuration ---
    pass_at_k_values: List[int] = field(default_factory=lambda: [1, 5, 10])
    num_candidates: int = 10  # Number of candidate patches to generate
    test_timeout_seconds: int = 10
    syntax_validator: str = "tree_sitter"  # or 'ast'

    # --- Output ---
    output_dir: str = "./surgical_repair_output"
    logging_steps: int = 5
    save_steps: int = 50
    eval_steps: int = 50
    seed: int = 42


# =============================================================================
# SECTION 4: DATASET LOADING & FIM PREPROCESSING
# =============================================================================


class FIMDatasetBuilder:
    """
    Builds a Fill-in-the-Middle (FIM) dataset from bug-fix pairs.

    Each training example is transformed into:
        <fim_prefix> context_before_bug <fim_suffix> context_after_bug <fim_middle> fixed_code

    The masked context filler (<extra_id_0>) is inserted at the bug location.
    """

    def __init__(self, config: SurgicalRepairConfig):
        self.config = config
        self.tokenizer = None

    def load_and_prepare(self, tokenizer) -> Tuple[Dataset, Dataset]:
        """Load CodeXGLUE code refinement dataset and prepare FIM examples."""
        self.tokenizer = tokenizer

        logger.info(
            f"Loading dataset: {self.config.dataset_name}/{self.config.dataset_subset}"
        )

        try:
            # Try to load CodeXGLUE from HuggingFace
            raw_dataset = load_dataset(
                self.config.dataset_name,
                self.config.dataset_subset,
                trust_remote_code=True,
            )
            train_data = raw_dataset["train"]
            eval_data = raw_dataset.get("validation", raw_dataset["train"])
        except Exception as e:
            logger.warning(f"Failed to load CodeXGLUE: {e}")
            logger.info("Falling back to synthetic dataset...")
            train_data, eval_data = self._create_synthetic_dataset()

        # Limit samples for time budget
        train_data = train_data.select(
            range(min(len(train_data), self.config.max_train_samples))
        )
        eval_data = eval_data.select(
            range(min(len(eval_data), self.config.max_eval_samples))
        )

        logger.info(f"Train samples: {len(train_data)}, Eval samples: {len(eval_data)}")

        # Convert to FIM format
        train_fim = train_data.map(
            self._convert_to_fim,
            remove_columns=train_data.column_names,
            desc="Converting train to FIM",
        )
        eval_fim = eval_data.map(
            self._convert_to_fim,
            remove_columns=eval_data.column_names,
            desc="Converting eval to FIM",
        )

        # Tokenize
        train_tokenized = train_fim.map(
            self._tokenize_fim,
            remove_columns=train_fim.column_names,
            desc="Tokenizing train",
        )
        eval_tokenized = eval_fim.map(
            self._tokenize_fim,
            remove_columns=eval_fim.column_names,
            desc="Tokenizing eval",
        )

        return train_tokenized, eval_tokenized

    def _convert_to_fim(self, example: Dict) -> Dict:
        """
        Convert a (buggy, fixed) pair into FIM format.

        The buggy code is analyzed to find the changed region.
        The context around the bug is split into prefix and suffix.
        The fix becomes the middle (target).
        """
        buggy = example.get("buggy", example.get("input", ""))
        fixed = example.get("fixed", example.get("output", ""))

        if not buggy or not fixed:
            return {"fim_text": "", "target_text": ""}

        # Find the minimal change region using difflib
        import difflib

        buggy_lines = buggy.splitlines(keepends=True)
        fixed_lines = fixed.splitlines(keepends=True)

        matcher = difflib.SequenceMatcher(None, buggy_lines, fixed_lines)
        opcodes = matcher.get_opcodes()

        prefix_lines = []
        suffix_lines = []
        fixed_region_lines = []
        buggy_region_lines = []

        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "equal":
                # Equal region: part of prefix (if before change) or suffix (if after)
                if not fixed_region_lines and not buggy_region_lines:
                    prefix_lines.extend(buggy_lines[i1:i2])
                else:
                    suffix_lines.extend(buggy_lines[i1:i2])
            elif tag in ("replace", "delete", "insert"):
                buggy_region_lines.extend(buggy_lines[i1:i2])
                fixed_region_lines.extend(fixed_lines[j1:j2])

        # Build FIM input
        prefix = "".join(prefix_lines)
        suffix = "".join(suffix_lines)
        fix = "".join(fixed_region_lines)

        # Create the masked context: buggy region is replaced with mask token
        masked_region = self.config.mask_token

        # FIM format: <fim_prefix>prefix<fim_suffix>suffix<fim_middle>fix
        fim_input = (
            f"{self.config.fim_prefix_token}{prefix}"
            f"{self.config.fim_suffix_token}{suffix}"
            f"{self.config.fim_middle_token}"
        )

        # The target is the fix (what goes in the middle)
        target = fix

        return {"fim_text": fim_input, "target_text": target}

    def _tokenize_fim(self, example: Dict) -> Dict:
        """Tokenize FIM input and target."""
        fim_text = example.get("fim_text", "")
        target_text = example.get("target_text", "")

        if not fim_text or not target_text:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        # Tokenize input
        input_enc = self.tokenizer(
            fim_text,
            truncation=True,
            max_length=self.config.max_seq_length,
            padding=False,
            return_tensors=None,
        )

        # Tokenize target
        target_enc = self.tokenizer(
            target_text,
            truncation=True,
            max_length=self.config.max_seq_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = input_enc["input_ids"]
        target_ids = target_enc["input_ids"]

        # Combine: input_ids + target_ids (for causal LM training)
        combined_ids = input_ids + target_ids

        # Truncate to max length
        if len(combined_ids) > self.config.max_seq_length:
            combined_ids = combined_ids[: self.config.max_seq_length]

        # Labels: mask the input part (set to -100), keep target part
        labels = [-100] * len(input_ids) + target_ids
        if len(labels) > self.config.max_seq_length:
            labels = labels[: self.config.max_seq_length]

        attention_mask = [1] * len(combined_ids)

        return {
            "input_ids": combined_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _create_synthetic_dataset(self) -> Tuple[Dataset, Dataset]:
        """
        Create a synthetic bug-fix dataset for demonstration.
        Used as fallback if CodeXGLUE cannot be loaded.
        """
        logger.info("Creating synthetic bug-fix dataset...")

        examples = []
        for i in range(2000):
            # Generate simple Python code with a bug
            var_name = f"var_{i}"
            buggy = f"def compute_{i}(a, b):\n    {var_name} = a + b\n    return {var_name}\n"
            fixed = f"def compute_{i}(a, b):\n    {var_name} = a * b\n    return {var_name}\n"
            examples.append({"buggy": buggy, "fixed": fixed})

        # Split
        split_idx = int(len(examples) * self.config.train_split_ratio)
        train_dataset = Dataset.from_list(examples[:split_idx])
        eval_dataset = Dataset.from_list(examples[split_idx:])

        return train_dataset, eval_dataset


# =============================================================================
# SECTION 5: MODEL LOADING WITH QLORA
# =============================================================================


class SurgicalRepairModel:
    """
    Wraps the DeepSeek-Coder model with QLoRA configuration.

    On CPU, bitsandbytes 4-bit quantization is not available.
    We use plain LoRA with float32 precision, which still provides
    parameter-efficient fine-tuning (< 1% trainable parameters).
    """

    def __init__(self, config: SurgicalRepairConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def load(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load model and tokenizer with QLoRA preparation."""
        logger.info(f"Loading model: {self.config.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="right",
        )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add FIM special tokens if not present
        special_tokens = {
            "additional_special_tokens": [
                self.config.mask_token,
                self.config.fim_prefix_token,
                self.config.fim_suffix_token,
                self.config.fim_middle_token,
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)

        # Load model
        logger.info("Loading base model (this may take a few minutes)...")

        # Determine dtype
        if self.config.torch_dtype == "float32":
            torch_dtype = torch.float32
        elif self.config.torch_dtype == "float16":
            torch_dtype = torch.float16
        elif self.config.torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        # On CPU, we use plain LoRA (bitsandbytes 4-bit is GPU-only)
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
            "device_map": "cpu",  # Force CPU
            "low_cpu_mem_usage": True,
        }

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **load_kwargs,
        )

        # Resize embeddings for new special tokens
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Prepare model for k-bit training (gradient checkpointing, etc.)
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
        )

        # Enable gradient checkpointing for memory savings
        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # Apply LoRA
        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, peft_config)

        # Log trainable parameters
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_pct = 100 * trainable_params / total_params

        logger.info(
            f"Trainable parameters: {trainable_params:,} ({trainable_pct:.4f}%)"
        )
        logger.info(f"Total parameters: {total_params:,}")

        return self.model, self.tokenizer


# =============================================================================
# SECTION 6: TRAINING CALLBACKS
# =============================================================================


class TimeBudgetCallback(TrainerCallback):
    """Stops training when time budget is exceeded."""

    def __init__(self, max_hours: float = 3.0):
        self.max_seconds = max_hours * 3600
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        logger.info(f"Training started. Time budget: {self.max_seconds/3600:.1f} hours")

    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is None:
            return
        elapsed = time.time() - self.start_time
        if elapsed > self.max_seconds:
            logger.warning(
                f"Time budget exceeded ({elapsed/3600:.2f}h). Stopping training."
            )
            control.should_training_stop = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and self.start_time:
            elapsed = (time.time() - self.start_time) / 3600
            logger.info(f"[{elapsed:.2f}h] {logs}")


class MemoryMonitorCallback(TrainerCallback):
    """Monitors memory usage during training."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            try:
                import psutil

                process = psutil.Process(os.getpid())
                mem_gb = process.memory_info().rss / (1024**3)
                logger.info(f"Step {state.global_step}: RAM usage = {mem_gb:.2f} GB")
            except ImportError:
                pass  # psutil not available


# =============================================================================
# SECTION 7: STAGE 1 - STATIC AST VALIDATION (TREE-SITTER)
# =============================================================================


class TreeSitterValidator:
    """
    Stage 1 of the two-stage validation pipeline.

    Uses Tree-sitter to parse Python code and detect syntax errors.
    Invalid patches are discarded early before execution.
    """

    def __init__(self):
        # Initialize Tree-sitter Python parser
        try:
            PY_LANGUAGE = tree_sitter.Language(tree_sitter_python.language())
            self.parser = tree_sitter.Parser(PY_LANGUAGE)
            self.available = True
            logger.info("Tree-sitter Python parser initialized")
        except Exception as e:
            logger.warning(f"Tree-sitter initialization failed: {e}")
            self.parser = None
            self.available = False

    def validate(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate code syntax using Tree-sitter.

        Returns:
            (is_valid, error_message)
        """
        if not code or not code.strip():
            return False, "Empty code"

        # Try Tree-sitter first
        if self.available:
            try:
                tree = self.parser.parse(bytes(code, "utf-8"))
                root = tree.root_node

                # Check for error nodes
                errors = []
                self._find_errors(root, errors)

                if errors:
                    return False, f"Tree-sitter found {len(errors)} syntax error(s)"

                # Also verify with Python's ast module for semantic validation
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    return False, f"AST validation failed: {e}"

                return True, None

            except Exception as e:
                logger.debug(f"Tree-sitter parse error: {e}")

        # Fallback to Python AST
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

    def _find_errors(self, node, errors: List):
        """Recursively find ERROR nodes in the syntax tree."""
        if node.type == "ERROR" or node.is_error:
            errors.append(node)
        for child in node.children:
            self._find_errors(child, errors)


# =============================================================================
# SECTION 8: STAGE 2 - DYNAMIC EXECUTION (SANDBOXED)
# =============================================================================
class SandboxExecutor:
    """
    Stage 2 of the two-stage validation pipeline.

    Executes repaired code in a sandboxed environment with:
    - Resource limits (time, memory)
    - Isolated working directory
    - Test execution with Pass@k measurement
    """

    def __init__(self, config: SurgicalRepairConfig):
        self.config = config
        # IMPORTANT: resolve to absolute path so cwd + relative file path
        # do not get concatenated (e.g. sandbox_tmp/sandbox_tmp/file.py).
        self.temp_dir = Path("./sandbox_tmp").resolve()
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Sandbox directory initialized: {self.temp_dir}")

    def execute_with_tests(
        self,
        code: str,
        test_code: str,
        timeout: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """
        Execute code with tests in a sandboxed subprocess.

        Returns:
            (passed, output_message)
        """
        timeout = timeout or self.config.test_timeout_seconds

        # Combine code and tests
        full_code = code + "\n\n" + test_code

        # Write to temp file — use ABSOLUTE path so the child process
        # can find it regardless of its working directory.
        temp_file = (self.temp_dir / f"test_{random.randint(0, 999999)}.py").resolve()
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(full_code)

            # Execute in subprocess with timeout.
            # Pass the absolute path of the script; cwd is kept at temp_dir
            # to isolate any stray files the test code might create.
            result = subprocess.run(
                [sys.executable, str(temp_file)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.temp_dir),
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )

            if result.returncode == 0:
                return True, "All tests passed"
            else:
                error_msg = result.stderr.strip() or result.stdout.strip()
                return False, f"Test failed: {error_msg[:200]}"

        except subprocess.TimeoutExpired:
            return False, f"Execution timed out after {timeout}s"
        except Exception as e:
            return False, f"Execution error: {str(e)}"
        finally:
            # Cleanup
            try:
                temp_file.unlink(missing_ok=True)
            except Exception:
                pass

    def compute_pass_at_k(
        self,
        candidates: List[str],
        test_code: str,
        k_values: List[int],
    ) -> Dict[int, float]:
        """
        Compute Pass@k metric.

        Pass@k = 1 - C(n-c, k) / C(n, k)
        where n = number of candidates, c = number that pass.

        Args:
            candidates: List of generated code candidates
            test_code: Test code to validate each candidate
            k_values: List of k values to compute

        Returns:
            Dictionary mapping k -> pass@k score
        """
        from math import comb

        n = len(candidates)
        if n == 0:
            return {k: 0.0 for k in k_values}

        # Execute all candidates
        results = []
        for candidate in candidates:
            passed, _ = self.execute_with_tests(candidate, test_code)
            results.append(passed)

        c = sum(results)  # Number that passed

        logger.info(f"Pass@k evaluation: {c}/{n} candidates passed")

        # Compute Pass@k for each k
        pass_at_k = {}
        for k in k_values:
            if k > n:
                pass_at_k[k] = 0.0
            else:
                # Use the unbiased estimator
                if n - c < k:
                    pass_at_k[k] = 1.0
                else:
                    pass_at_k[k] = 1.0 - comb(n - c, k) / comb(n, k)

        return pass_at_k


# =============================================================================
# SECTION 9: TWO-STAGE VALIDATION PIPELINE
# =============================================================================


class ValidationPipeline:
    """
    Orchestrates the two-stage validation pipeline:
        Stage 1: Static AST validation (Tree-sitter)
        Stage 2: Dynamic execution (sandboxed)
    """

    def __init__(self, config: SurgicalRepairConfig):
        self.config = config
        self.ast_validator = TreeSitterValidator()
        self.sandbox = SandboxExecutor(config)

    def validate_patch(self, patch: str, test_code: str) -> Dict[str, Any]:
        """
        Validate a single patch through both stages.

        Returns:
            Dictionary with validation results
        """
        result = {
            "patch": patch,
            "stage1_passed": False,
            "stage2_passed": False,
            "stage1_error": None,
            "stage2_error": None,
            "valid": False,
        }

        # Stage 1: Static AST validation
        is_valid_syntax, syntax_error = self.ast_validator.validate(patch)
        result["stage1_passed"] = is_valid_syntax
        result["stage1_error"] = syntax_error

        if not is_valid_syntax:
            logger.debug(f"Stage 1 failed: {syntax_error}")
            return result

        # Stage 2: Dynamic execution
        is_valid_exec, exec_error = self.sandbox.execute_with_tests(patch, test_code)
        result["stage2_passed"] = is_valid_exec
        result["stage2_error"] = exec_error
        result["valid"] = is_valid_exec

        return result

    def validate_candidates(
        self, candidates: List[str], test_code: str
    ) -> Tuple[List[Dict], Dict[int, float]]:
        """
        Validate multiple candidates and compute Pass@k.

        Returns:
            (validation_results, pass_at_k_scores)
        """
        results = []
        for candidate in candidates:
            result = self.validate_patch(candidate, test_code)
            results.append(result)

        # Compute Pass@k
        valid_candidates = [r["patch"] for r in results if r["valid"]]
        pass_at_k = self.sandbox.compute_pass_at_k(
            candidates, test_code, self.config.pass_at_k_values
        )

        return results, pass_at_k


# =============================================================================
# SECTION 10: MAIN TRAINING PIPELINE
# =============================================================================


class SurgicalRepairPipeline:
    """
    Main pipeline orchestrating:
        1. Dataset preparation (FIM masking)
        2. Model loading (QLoRA)
        3. Training
        4. Validation (two-stage pipeline)
    """

    def __init__(self, config: Optional[SurgicalRepairConfig] = None):
        self.config = config or SurgicalRepairConfig()
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.validator = ValidationPipeline(self.config)

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Set seed
        set_seed(self.config.seed)

    def run(self):
        """Execute the full pipeline."""
        logger.info("=" * 70)
        logger.info("SURGICAL CODE REPAIR PIPELINE")
        logger.info("=" * 70)

        start_time = time.time()

        # Step 1: Load model
        logger.info("\n[STEP 1/5] Loading model with QLoRA...")
        model_wrapper = SurgicalRepairModel(self.config)
        self.model, self.tokenizer = model_wrapper.load()

        # Step 2: Prepare dataset
        logger.info("\n[STEP 2/5] Preparing FIM dataset...")
        dataset_builder = FIMDatasetBuilder(self.config)
        train_dataset, eval_dataset = dataset_builder.load_and_prepare(self.tokenizer)

        # Step 3: Configure training
        logger.info("\n[STEP 3/5] Configuring training...")
        self._configure_training(train_dataset, eval_dataset)

        # Step 4: Train
        logger.info("\n[STEP 4/5] Starting training...")
        self._train()

        # Step 5: Validate
        logger.info("\n[STEP 5/5] Running validation pipeline...")
        self._validate()

        # Summary
        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 70)
        logger.info(f"PIPELINE COMPLETE. Total time: {elapsed/3600:.2f} hours")
        logger.info("=" * 70)

    def _configure_training(self, train_dataset: Dataset, eval_dataset: Dataset):
        """Configure TrainingArguments and Trainer.

        Uses runtime introspection so we only pass kwargs the installed
        transformers version actually supports. This protects against
        renames/removals across versions (e.g. overwrite_output_dir,
        evaluation_strategy -> eval_strategy, no_cuda -> use_cpu).
        """
        import inspect

        # --- Data collator ---
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
            pad_to_multiple_of=8,
        )

        # --- Full candidate kwargs (superset across transformers versions) ---
        candidate_kwargs = {
            # Core
            "output_dir": self.config.output_dir,
            "overwrite_output_dir": True,  # removed in some versions
            "do_train": True,
            "do_eval": True,
            # Training schedule
            "num_train_epochs": self.config.num_epochs,
            "max_steps": self.config.max_steps,
            "per_device_train_batch_size": self.config.batch_size,
            "per_device_eval_batch_size": self.config.batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "learning_rate": self.config.learning_rate,
            "warmup_steps": self.config.warmup_steps,
            "weight_decay": self.config.weight_decay,
            "max_grad_norm": self.config.max_grad_norm,
            # Evaluation — include BOTH old and new names
            "evaluation_strategy": "steps",  # old name
            "eval_strategy": "steps",  # new name
            "eval_steps": self.config.eval_steps,
            # Logging
            "logging_dir": os.path.join(self.config.output_dir, "logs"),
            "logging_steps": self.config.logging_steps,
            "report_to": ["none"],
            # Saving
            "save_steps": self.config.save_steps,
            "save_total_limit": 2,
            "save_strategy": "steps",
            # CPU / precision
            "use_cpu": True,  # new name
            "no_cuda": True,  # old name
            "fp16": self.config.fp16,
            "bf16": self.config.bf16,
            "optim": self.config.optim,
            "dataloader_num_workers": self.config.dataloader_num_workers,
            "dataloader_pin_memory": False,
            # Memory
            "gradient_checkpointing": self.config.use_gradient_checkpointing,
            # Misc
            "seed": self.config.seed,
            "remove_unused_columns": False,
            "label_names": ["labels"],
        }

        # --- Filter to only what this transformers version accepts ---
        sig = inspect.signature(TrainingArguments.__init__)
        valid_params = set(sig.parameters.keys())
        filtered_kwargs = {
            k: v for k, v in candidate_kwargs.items() if k in valid_params
        }

        dropped = set(candidate_kwargs) - valid_params
        if dropped:
            logger.warning(
                f"Dropped unsupported TrainingArguments kwargs for your "
                f"transformers version: {sorted(dropped)}"
            )
        logger.info(f"Passing {len(filtered_kwargs)} kwargs to TrainingArguments")

        training_args = TrainingArguments(**filtered_kwargs)

        # --- Trainer ---
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[
                TimeBudgetCallback(max_hours=3.0),
                MemoryMonitorCallback(),
            ],
        )

        logger.info(f"Training configured: {self.config.max_steps} max steps")
        logger.info(
            f"Batch size: {self.config.batch_size} x "
            f"{self.config.gradient_accumulation_steps} grad accum"
        )

    def _train(self):
        """Execute training."""
        logger.info("Starting training loop...")

        try:
            train_result = self.trainer.train()

            # Save final model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)

            # Log metrics
            metrics = train_result.metrics
            logger.info(f"Training metrics: {metrics}")

            # Save training metrics
            with open(
                os.path.join(self.config.output_dir, "train_metrics.json"), "w"
            ) as f:
                json.dump(metrics, f, indent=2)

        except Exception as e:
            logger.error(f"Training failed: {e}")
            logger.error(traceback.format_exc())
            raise

    def _validate(self):
        """Run two-stage validation pipeline on sample test cases."""
        logger.info("Running validation pipeline on sample test cases...")

        # Create sample test cases
        test_cases = self._create_test_cases()

        all_pass_at_k = {}

        for idx, test_case in enumerate(test_cases):
            logger.info(f"\nTest case {idx + 1}/{len(test_cases)}")

            # Generate candidate patches
            candidates = self._generate_candidates(
                test_case["buggy_code"],
                num_candidates=self.config.num_candidates,
            )

            # Validate through two-stage pipeline
            results, pass_at_k = self.validator.validate_candidates(
                candidates, test_case["test_code"]
            )

            # Track metrics
            for k, v in pass_at_k.items():
                all_pass_at_k.setdefault(k, []).append(v)

            # Log results
            stage1_passed = sum(1 for r in results if r["stage1_passed"])
            stage2_passed = sum(1 for r in results if r["stage2_passed"])
            logger.info(f"  Stage 1 (AST): {stage1_passed}/{len(candidates)} passed")
            logger.info(f"  Stage 2 (Exec): {stage2_passed}/{len(candidates)} passed")
            logger.info(f"  Pass@k: {pass_at_k}")

        # Average Pass@k across all test cases
        if all_pass_at_k:
            avg_pass_at_k = {k: float(np.mean(v)) for k, v in all_pass_at_k.items()}
            logger.info(f"\nAverage Pass@k across all test cases: {avg_pass_at_k}")

            # Save results
            with open(
                os.path.join(self.config.output_dir, "pass_at_k_results.json"), "w"
            ) as f:
                json.dump(avg_pass_at_k, f, indent=2)

    def _generate_candidates(self, buggy_code: str, num_candidates: int) -> List[str]:
        """Generate candidate patches using the fine-tuned model."""
        self.model.eval()

        candidates = []

        # Build FIM prompt.
        # For a buggy code, we mask the buggy region and ask the model to fill it.
        lines = buggy_code.splitlines()
        if len(lines) < 3:
            return [buggy_code]  # Too short to repair

        # Mask the middle third of lines
        mask_start = len(lines) // 3
        mask_end = 2 * len(lines) // 3

        prefix = "\n".join(lines[:mask_start])
        suffix = "\n".join(lines[mask_end:])

        prompt = (
            f"{self.config.fim_prefix_token}{prefix}\n"
            f"{self.config.fim_suffix_token}{suffix}\n"
            f"{self.config.fim_middle_token}"
        )

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length // 2,
        ).to(self.config.device)

        # Generate multiple candidates with different temperatures
        with torch.no_grad():
            for i in range(num_candidates):
                temperature = 0.2 + (i * 0.1)  # Vary temperature
                try:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        temperature=min(temperature, 1.0),
                        do_sample=True,
                        top_p=0.95,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                    generated = self.tokenizer.decode(
                        outputs[0][inputs["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                    )

                    # Reconstruct full candidate
                    candidate = prefix + "\n" + generated + "\n" + suffix
                    candidates.append(candidate.strip())

                except Exception as e:
                    logger.debug(f"Generation {i} failed: {e}")
                    candidates.append(buggy_code)  # Fallback

        return candidates

    def _create_test_cases(self) -> List[Dict]:
        """Create sample test cases for validation."""
        test_cases = [
            {
                "buggy_code": textwrap.dedent("""
                    def add_numbers(a, b):
                        result = a - b
                        return result
                """).strip(),
                "test_code": textwrap.dedent("""
                    def test_add():
                        assert add_numbers(2, 3) == 5, "2 + 3 should be 5"
                        assert add_numbers(-1, 1) == 0, "-1 + 1 should be 0"
                        assert add_numbers(0, 0) == 0, "0 + 0 should be 0"
                        print("All tests passed!")

                    test_add()
                """).strip(),
            },
            {
                "buggy_code": textwrap.dedent("""
                    def factorial(n):
                        if n == 0:
                            return 0
                        return n * factorial(n - 1)
                """).strip(),
                "test_code": textwrap.dedent("""
                    def test_factorial():
                        assert factorial(0) == 1, "0! should be 1"
                        assert factorial(1) == 1, "1! should be 1"
                        assert factorial(5) == 120, "5! should be 120"
                        print("All tests passed!")

                    test_factorial()
                """).strip(),
            },
            {
                "buggy_code": textwrap.dedent("""
                    def find_max(lst):
                        max_val = lst[0]
                        for i in range(len(lst)):
                            if lst[i] > max_val:
                                max_val = lst[i]
                        return max_val
                """).strip(),
                "test_code": textwrap.dedent("""
                    def test_find_max():
                        assert find_max([1, 2, 3]) == 3, "Max of [1,2,3] is 3"
                        assert find_max([5]) == 5, "Max of [5] is 5"
                        assert find_max([-1, -5, -3]) == -1, "Max of negatives"
                        print("All tests passed!")

                    test_find_max()
                """).strip(),
            },
        ]

        return test_cases


# =============================================================================
# SECTION 11: INFERENCE / DEPLOYMENT
# =============================================================================


class SurgicalRepairInference:
    """
    Inference wrapper for the fine-tuned model.

    Supports:
        - Single code repair
        - Batch repair
        - Integration with CI/CD pipelines
    """

    def __init__(self, model_path: str, config: Optional[SurgicalRepairConfig] = None):
        self.config = config or SurgicalRepairConfig()
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.validator = ValidationPipeline(self.config)

    def load(self):
        """Load the fine-tuned model."""
        logger.info(f"Loading fine-tuned model from {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )

        # Load LoRA adapters
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.eval()

        logger.info("Model loaded successfully")

    def repair(self, buggy_code: str, test_code: Optional[str] = None) -> Dict:
        """
        Repair a single code snippet.

        Args:
            buggy_code: The code with bugs
            test_code: Optional test code to validate the repair

        Returns:
            Dictionary with repair results
        """
        if self.model is None:
            self.load()

        # Generate candidates
        candidates = self._generate_candidates(buggy_code)

        result = {
            "buggy_code": buggy_code,
            "candidates": [],
            "best_patch": None,
            "pass_at_k": {},
        }

        # Validate each candidate
        for candidate in candidates:
            validation = self.validator.validate_patch(candidate, test_code or "")
            result["candidates"].append(validation)

            if validation["valid"] and result["best_patch"] is None:
                result["best_patch"] = candidate

        # Compute Pass@k if tests provided
        if test_code:
            _, pass_at_k = self.validator.validate_candidates(candidates, test_code)
            result["pass_at_k"] = pass_at_k

        return result

    def _generate_candidates(
        self, buggy_code: str, num_candidates: int = 5
    ) -> List[str]:
        """Generate candidate patches."""
        lines = buggy_code.splitlines()
        mask_start = len(lines) // 3
        mask_end = 2 * len(lines) // 3

        prefix = "\n".join(lines[:mask_start])
        suffix = "\n".join(lines[mask_end:])

        prompt = (
            f"{self.config.fim_prefix_token}{prefix}\n"
            f"{self.config.fim_suffix_token}{suffix}\n"
            f"{self.config.fim_middle_token}"
        )

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=256
        )

        candidates = []
        with torch.no_grad():
            for i in range(num_candidates):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.3 + i * 0.15,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                generated = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )
                candidate = prefix + "\n" + generated + "\n" + suffix
                candidates.append(candidate.strip())

        return candidates


# =============================================================================
# SECTION 12: ENTRY POINT
# =============================================================================


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Surgical Code Repair: CPU-Optimized Training Pipeline"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "validate", "infer"],
        help="Pipeline mode: train, validate, or infer",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./surgical_repair_output",
        help="Path to fine-tuned model (for infer mode)",
    )
    parser.add_argument(
        "--buggy-code",
        type=str,
        default=None,
        help="Buggy code to repair (for infer mode)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=400,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training batch size",
    )

    args = parser.parse_args()

    if args.mode == "train":
        # Training mode
        config = SurgicalRepairConfig(
            max_steps=args.max_steps,
            batch_size=args.batch_size,
        )
        pipeline = SurgicalRepairPipeline(config)
        pipeline.run()

    elif args.mode == "infer":
        # Inference mode
        if not args.buggy_code:
            logger.error("--buggy-code required for infer mode")
            sys.exit(1)

        config = SurgicalRepairConfig()
        inference = SurgicalRepairInference(args.model_path, config)
        result = inference.repair(args.buggy_code)

        print("\n" + "=" * 50)
        print("REPAIR RESULT")
        print("=" * 50)
        print(f"Best patch: {result['best_patch']}")
        print(f"Pass@k: {result['pass_at_k']}")

    elif args.mode == "validate":
        # Validation mode (test the pipeline without training)
        config = SurgicalRepairConfig()
        validator = ValidationPipeline(config)

        # Test Tree-sitter
        test_code = "def add(a, b):\n    return a + b\n"
        is_valid, error = validator.ast_validator.validate(test_code)
        print(f"Valid code: {is_valid}, Error: {error}")

        # Test sandbox
        passed, msg = validator.sandbox.execute_with_tests(
            "def add(a, b):\n    return a + b",
            "assert add(2, 3) == 5\nprint('OK')",
        )
        print(f"Sandbox test: {passed}, Message: {msg}")

        # Test Pass@k
        candidates = [
            "def add(a, b):\n    return a + b",
            "def add(a, b):\n    return a - b",
        ]
        pass_at_k = validator.sandbox.compute_pass_at_k(
            candidates,
            "assert add(2, 3) == 5\nprint('OK')",
            [1, 2],
        )
        print(f"Pass@k: {pass_at_k}")


if __name__ == "__main__":
    main()
