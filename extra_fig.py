"""
make_figures_extra.py — Additional report figures from eval_results.json.

Generates:
    fig_passk_curve.png           — Pass@k curve for k = 1..5
    fig_category_heatmap.png      — heatmap of S1/S2/Exact by category
    fig_outcome_pie.png           — pie chart of repair outcomes
    fig_latency.png               — inference time per test case
    fig_lora_size.png             — LoRA vs base model parameter footprint

Run:
    python make_figures_extra.py
"""

import json
from pathlib import Path
from collections import defaultdict
from math import comb

import matplotlib.pyplot as plt
import numpy as np

EVAL_JSON = "eval_results.json"
DPI = 200

C_PRIMARY = "#3b82f6"
C_SUCCESS = "#10b981"
C_WARN = "#f59e0b"
C_DANGER = "#ef4444"
C_PURPLE = "#7c3aed"
C_GRAY = "#6b7280"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


def load_data():
    with open(EVAL_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# FIGURE 5 — PASS@K CURVE (k = 1..5)
# ============================================================


def pass_at_k(n, c, k):
    if n < k or c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def make_passk_curve(data):
    n = data["config"]["num_candidates"]
    ks = list(range(1, n + 1))

    # Aggregate across all cases
    curve_avg = []
    for k in ks:
        scores = [pass_at_k(n, c["s2_pass"], k) for c in data["per_case"]]
        curve_avg.append(np.mean(scores))

    # Also compute per-category curves (top 4 categories)
    cat_curves = defaultdict(list)
    for c in data["per_case"]:
        cat_curves[c["category"]].append(c)

    top_cats = sorted(
        cat_curves.keys(),
        key=lambda cat: -np.mean(
            [pass_at_k(n, c["s2_pass"], n) for c in cat_curves[cat]]
        ),
    )[:5]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Main average curve
    ax.plot(
        ks,
        curve_avg,
        marker="o",
        linewidth=2.5,
        markersize=8,
        color=C_PRIMARY,
        label="Overall average",
        zorder=10,
    )

    # Per-category curves for top 5
    cat_colors = [C_SUCCESS, C_WARN, C_PURPLE, C_DANGER, "#06b6d4"]
    for i, cat in enumerate(top_cats):
        scores = []
        for k in ks:
            cat_scores = [pass_at_k(n, c["s2_pass"], k) for c in cat_curves[cat]]
            scores.append(np.mean(cat_scores))
        ax.plot(
            ks,
            scores,
            marker="s",
            linewidth=1.2,
            markersize=5,
            color=cat_colors[i],
            alpha=0.7,
            label=f"{cat} (n={len(cat_curves[cat])})",
        )

    ax.set_xticks(ks)
    ax.set_xlabel("k (number of candidates)")
    ax.set_ylabel("Pass@k")
    ax.set_ylim(0, 1.05)
    ax.set_title("Pass@k curve — overall and top-5 categories")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_passk_curve.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  ✓ fig_passk_curve.png")


# ============================================================
# FIGURE 6 — CATEGORY HEATMAP
# ============================================================


def make_category_heatmap(data):
    """Heatmap: category × (S1 pass rate, S2 pass rate, exact match rate)."""
    agg = defaultdict(lambda: {"s1": 0, "s2": 0, "exact": 0, "n": 0})
    for c in data["per_case"]:
        cat = c["category"]
        agg[cat]["s1"] += c["s1_pass"]
        agg[cat]["s2"] += c["s2_pass"]
        agg[cat]["exact"] += c["exact_match"]
        agg[cat]["n"] += data["config"]["num_candidates"]

    # Sort categories by S2 pass rate descending
    cats = sorted(agg.keys(), key=lambda c: -agg[c]["s2"] / agg[c]["n"])

    matrix = []
    for cat in cats:
        d = agg[cat]
        matrix.append(
            [
                d["s1"] / d["n"],
                d["s2"] / d["n"],
                d["exact"] / d["n"],
            ]
        )
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(5.5, max(4, len(cats) * 0.4)))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Stage 1\n(AST)", "Stage 2\n(Sandbox)", "Exact\nmatch"])
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats)

    # Annotate each cell with the numeric value
    for i in range(len(cats)):
        for j in range(3):
            val = matrix[i, j]
            color = "white" if val < 0.35 or val > 0.75 else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
                fontweight="bold",
            )

    ax.set_title("Category × validation stage — pass rates")
    fig.colorbar(im, ax=ax, label="Pass rate", fraction=0.035)
    plt.tight_layout()
    plt.savefig("fig_category_heatmap.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  ✓ fig_category_heatmap.png")


# ============================================================
# FIGURE 7 — OUTCOME PIE / DONUT
# ============================================================


def make_outcome_donut(data):
    """Donut chart: breakdown of test-case outcomes."""
    n_cases = len(data["per_case"])

    perfect = sum(
        1 for c in data["per_case"] if c["s2_pass"] == data["config"]["num_candidates"]
    )
    partial = sum(
        1
        for c in data["per_case"]
        if 0 < c["s2_pass"] < data["config"]["num_candidates"]
    )
    failed = n_cases - perfect - partial

    sizes = [perfect, partial, failed]
    labels = [
        f"All candidates correct\n({perfect} cases)",
        f"Partial success\n({partial} cases)",
        f"No valid repair\n({failed} cases)",
    ]
    colors = [C_SUCCESS, C_WARN, C_DANGER]
    explode = (0.03, 0.03, 0.03)

    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        explode=explode,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
        pctdistance=0.78,
        textprops=dict(fontsize=10),
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")

    ax.text(
        0,
        0,
        f"{n_cases}\ntest cases",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )

    ax.set_title("Distribution of repair outcomes (26 held-out cases)")
    plt.tight_layout()
    plt.savefig("fig_outcome_pie.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  ✓ fig_outcome_pie.png")


# ============================================================
# FIGURE 8 — INFERENCE LATENCY
# ============================================================


def make_latency(data):
    """Horizontal bar chart: seconds per test case."""
    # Sort by time
    cases = sorted(data["per_case"], key=lambda c: c["seconds"])
    names = [c["name"] for c in cases]
    times = [c["seconds"] for c in cases]
    colors = [C_SUCCESS if c["s2_pass"] > 0 else C_DANGER for c in cases]

    fig, ax = plt.subplots(figsize=(8.5, max(5, len(cases) * 0.28)))

    y = np.arange(len(cases))
    ax.barh(y, times, color=colors, edgecolor="white", height=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Seconds (5 candidates × inference time)")
    ax.set_title("Inference latency per test case (CPU, batch size = 1)")
    ax.grid(axis="x", alpha=0.3)

    # Mean line
    mean_t = np.mean(times)
    ax.axvline(
        mean_t, color=C_GRAY, linestyle="--", alpha=0.7, label=f"Mean = {mean_t:.1f}s"
    )
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig("fig_latency.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  ✓ fig_latency.png")


# ============================================================
# FIGURE 9 — LORA VS BASE MODEL SIZE
# ============================================================


def make_lora_size():
    """Stacked bar showing trainable vs frozen parameter counts."""
    base_params = 1_342_384_128  # 1.35B - 6.29M
    lora_params = 6_291_456
    total = base_params + lora_params

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: absolute counts (log scale)
    labels = ["Base model\n(frozen)", "LoRA adapter\n(trainable)", "Total"]
    values = [base_params, lora_params, total]
    colors = [C_GRAY, C_PURPLE, C_PRIMARY]
    bars = ax1.bar(labels, values, color=colors, edgecolor="white")
    ax1.set_yscale("log")
    ax1.set_ylabel("Parameters (log scale)")
    ax1.set_title("Parameter footprint")
    for bar, val in zip(bars, values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.15,
            f"{val/1e6:,.1f}M",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax1.grid(axis="y", alpha=0.3)

    # Right: percentage pie
    sizes = [base_params, lora_params]
    labels2 = [
        f"Frozen\n({base_params/total*100:.3f}%)",
        f"Trainable\n({lora_params/total*100:.3f}%)",
    ]
    ax2.pie(
        sizes,
        labels=labels2,
        colors=[C_GRAY, C_PURPLE],
        autopct="",
        startangle=90,
        explode=(0, 0.05),
        wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops=dict(fontsize=10),
    )
    ax2.set_title("Trainable fraction")
    ax2.text(
        0,
        0,
        f"{lora_params/total*100:.2f}%",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=C_PURPLE,
    )

    plt.suptitle("QLoRA parameter efficiency — 0.47% trainable", fontsize=12)
    plt.tight_layout()
    plt.savefig("fig_lora_size.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  ✓ fig_lora_size.png")


# ============================================================
# MAIN
# ============================================================


def main():
    print("[figures-extra] Loading data...")
    data = load_data()

    print("[figures-extra] Generating extra figures...")
    make_passk_curve(data)
    make_category_heatmap(data)
    make_outcome_donut(data)
    make_latency(data)
    make_lora_size()

    print("[figures-extra] Done.")


if __name__ == "__main__":
    main()
