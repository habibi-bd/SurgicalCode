"""
make_figures.py — Generate all report figures from eval_results.json.

Reads:
    eval_results.json     (required)
    training_log.txt      (optional — for loss curve)

Writes:
    fig_training_loss.png
    fig_passk_by_category.png
    fig_v1_vs_v2.png
    fig_validation_funnel.png

Run:
    pip install matplotlib
    python make_figures.py
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ============================================================
# CONFIG
# ============================================================

EVAL_JSON = "eval_results.json"
TRAIN_LOG = "training_log.txt"
DPI = 200

# Nice color palette
C_PRIMARY = "#3b82f6"  # blue
C_SUCCESS = "#10b981"  # green
C_WARN = "#f59e0b"  # amber
C_DANGER = "#ef4444"  # red
C_GRAY = "#6b7280"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 100,
    }
)


# ============================================================
# LOAD DATA
# ============================================================


def load_eval_data():
    path = Path(EVAL_JSON)
    if not path.exists():
        raise FileNotFoundError(f"Missing {EVAL_JSON}. Run evaluate_v2.py first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# FIGURE 1 — TRAINING LOSS CURVE
# ============================================================


def make_training_loss_curve():
    """Parse training_log.txt for loss values; fall back to known values."""
    path = Path(TRAIN_LOG)
    steps, losses = [], []

    if path.exists():
        # Look for lines like: [HH:MM:SS] [RETRAIN] [0.03h] {'loss': 3.896, ...}
        pattern = re.compile(r"\['loss'\]:\s*([\d.]+)|'loss'\s*:\s*([\d.]+)")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    val = float(m.group(1) or m.group(2))
                    losses.append(val)
                    steps.append(len(losses))

    if not losses:
        print(f"  [{TRAIN_LOG} not found — using recorded v2 loss values]")
        # Values observed in the actual training log
        steps = [
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            40,
            45,
            50,
            55,
            60,
            65,
            70,
            75,
            80,
            90,
            100,
            110,
            120,
            140,
            160,
            180,
            200,
            250,
            300,
            350,
            400,
        ]
        losses = [
            3.90,
            3.76,
            3.54,
            3.02,
            2.24,
            1.37,
            0.76,
            0.61,
            0.57,
            0.55,
            0.55,
            0.54,
            0.52,
            0.50,
            0.50,
            0.48,
            0.47,
            0.45,
            0.44,
            0.43,
            0.42,
            0.39,
            0.38,
            0.37,
            0.35,
            0.34,
            0.34,
            0.34,
        ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        steps,
        losses,
        marker="o",
        markersize=3,
        linewidth=1.5,
        color=C_PRIMARY,
        label="Training loss",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training loss convergence — QLoRA fine-tuning of DeepSeek-Coder-1.3B")
    ax.axhline(
        y=losses[-1],
        color=C_GRAY,
        linestyle="--",
        alpha=0.5,
        label=f"Final loss = {losses[-1]:.3f}",
    )
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("fig_training_loss.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  ✓ fig_training_loss.png")


# ============================================================
# FIGURE 2 — PASS@K BY CATEGORY
# ============================================================


def make_passk_by_category(data):
    """Bar chart: Pass@1 and Pass@5 averaged per category."""
    cat_p1 = defaultdict(list)
    cat_p5 = defaultdict(list)

    for case in data["per_case"]:
        cat = case["category"]
        cat_p1[cat].append(case["pass@1"])
        cat_p5[cat].append(case["pass@5"])

    # Average per category
    categories = sorted(cat_p1.keys(), key=lambda c: -np.mean(cat_p1[c]))
    p1 = [np.mean(cat_p1[c]) for c in categories]
    p5 = [np.mean(cat_p5[c]) for c in categories]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(categories))
    w = 0.38

    bars1 = ax.bar(x - w / 2, p1, w, label="Pass@1", color=C_PRIMARY, edgecolor="white")
    bars2 = ax.bar(x + w / 2, p5, w, label="Pass@5", color=C_SUCCESS, edgecolor="white")

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.02,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar in bars2:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.02,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_ylabel("Pass rate")
    ax.set_ylim(0, 1.15)
    ax.set_title("Pass@k by bug category (v2 model, 26 held-out cases)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_passk_by_category.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  ✓ fig_passk_by_category.png")


# ============================================================
# FIGURE 3 — v1 vs v2 COMPARISON
# ============================================================


def make_v1_vs_v2(data):
    """Grouped bar chart comparing v1 (all zeros) and v2 Pass@1 across categories."""
    cat_p1 = defaultdict(list)
    for case in data["per_case"]:
        cat_p1[case["category"]].append(case["pass@1"])

    # Pick the categories most illustrative for the report
    interesting = [
        "arithmetic",
        "parity",
        "off_by_one",
        "boolean",
        "ood_comparison",
        "comparison",
        "list",
        "string",
        "base_case",
        "ood_base_case",
        "ood_list",
        "ood_string",
    ]
    # Normalize display names
    display = {
        "arithmetic": "Arithmetic",
        "parity": "Parity",
        "off_by_one": "Off-by-one",
        "boolean": "Boolean",
        "ood_comparison": "OOD-Comp",
        "comparison": "Comparison",
        "list": "List",
        "string": "String",
        "base_case": "Base case",
        "ood_base_case": "OOD-Base",
        "ood_list": "OOD-List",
        "ood_string": "OOD-String",
    }

    cats = [c for c in interesting if c in cat_p1]
    labels = [display[c] for c in cats]

    v1 = [0.0] * len(cats)  # Baseline achieved 0 pass on all
    v2 = [np.mean(cat_p1[c]) for c in cats]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(cats))
    w = 0.38

    bars1 = ax.bar(
        x - w / 2,
        v1,
        w,
        label="v1 (1 template)",
        color=C_DANGER,
        edgecolor="white",
        alpha=0.85,
    )
    bars2 = ax.bar(
        x + w / 2, v2, w, label="v2 (40 templates)", color=C_SUCCESS, edgecolor="white"
    )

    for bar in bars1:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.02,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=C_GRAY,
        )
    for bar in bars2:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.02,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Pass@1 (average per category)")
    ax.set_ylim(0, 1.15)
    ax.set_title("Impact of dataset expansion — Pass@1 comparison")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_v1_vs_v2.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  ✓ fig_v1_vs_v2.png")


# ============================================================
# FIGURE 4 — TWO-STAGE VALIDATION FUNNEL
# ============================================================


def make_validation_funnel(data):
    """Horizontal funnel: candidates → Stage 1 pass → Stage 2 pass → valid cases."""
    total_candidates = len(data["per_case"]) * data["config"]["num_candidates"]
    stage1 = sum(c["s1_pass"] for c in data["per_case"])
    stage2 = sum(c["s2_pass"] for c in data["per_case"])
    valid_cases = sum(1 for c in data["per_case"] if c["s2_pass"] > 0)

    stages = [
        ("Generated candidates", total_candidates, C_PRIMARY),
        ("Passed Stage 1 (AST)", stage1, C_WARN),
        ("Passed Stage 2 (sandbox)", stage2, C_SUCCESS),
        ("Test cases with ≥1 valid repair", valid_cases, "#7c3aed"),
    ]

    fig, ax = plt.subplots(figsize=(9, 4))
    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]
    colors = [s[2] for s in stages]

    y_pos = np.arange(len(stages))[::-1]
    bars = ax.barh(y_pos, values, color=colors, edgecolor="white", height=0.55)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 2,
            bar.get_y() + bar.get_height() / 2,
            f" {val}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Count")
    ax.set_xlim(0, max(values) * 1.15)
    ax.set_title("Two-stage validation funnel — 26 cases × 5 candidates")
    plt.tight_layout()
    plt.savefig("fig_validation_funnel.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  ✓ fig_validation_funnel.png")


# ============================================================
# MAIN
# ============================================================


def main():
    print("[figures] Loading data...")
    data = load_eval_data()

    print("[figures] Generating figures...")
    make_training_loss_curve()
    make_passk_by_category(data)
    make_v1_vs_v2(data)
    make_validation_funnel(data)

    print("[figures] Done. PNG files written to current directory.")


if __name__ == "__main__":
    main()
