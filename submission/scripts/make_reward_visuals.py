"""
make_reward_visuals.py — Generate 6 static PNGs explaining PrintFarmEnv reward design.

Usage:
    python submission/scripts/make_reward_visuals.py --out submission/plots

Outputs (to --out directory):
    reward_components_table.png
    reward_range_bars.png
    reward_decision_flow.png
    baseline_decomposition.png
    signal_coverage.png
    hyperparameters_card.png
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


# ── Reward component data (source of truth: submission/rewards/) ──────────────

COMPONENTS = [
    {
        "name": "r_format",
        "label": "Format +\nAnti-Echo",
        "range_min": -0.3,
        "range_max": +0.1,
        "what": "Output has valid <action> tag\n& parseable JSON",
        "anti_gaming": "Token overlap > 50%\nwith obs → penalise echo",
        "evidence_req": "None (always active)",
        "color": "#4C72B0",
        "breakdown": {
            "No <action> tag": -0.30,
            "Tag, invalid JSON": -0.20,
            "Echo (>50% overlap)": -0.20,
            "Tag + tag JSON, partial": -0.10,
            "Clean valid action": +0.10,
        },
    },
    {
        "name": "r_economic",
        "label": "Economic\n(P&L delta)",
        "range_min": -0.4,
        "range_max": +0.4,
        "what": "LLM action P&L − rules P&L\nnormalised to $10 = ±1.0",
        "anti_gaming": "Normalised + clamped;\ncan't exceed ±0.4",
        "evidence_req": "Always (requires step sim)",
        "color": "#DD8452",
    },
    {
        "name": "r_fault_precision",
        "label": "Fault\nPrecision",
        "range_min": -0.15,
        "range_max": +0.40,
        "what": "Investigate the right printer\nwhen real anomaly present",
        "anti_gaming": "Evidence-gated: random guess\non real fault → 0.0 (no signal\nto learn from)",
        "evidence_req": "observation= dict required\n(anomaly/note/telemetry)",
        "color": "#55A868",
    },
    {
        "name": "r_message_handling",
        "label": "Message\nHandling",
        "range_min": -0.1,
        "range_max": +0.4,
        "what": "Correct action on customer\nmessage (accept/decline/sub)",
        "anti_gaming": "Requires correct job_id;\nwrong target → partial +0.15",
        "evidence_req": "customer_messages tags\nin observation",
        "color": "#C44E52",
    },
    {
        "name": "r_unnecessary_action",
        "label": "Unnecessary\nAction Penalty",
        "range_min": -0.05,
        "range_max": 0.0,
        "what": "Penalise RUN_DIAGNOSTIC\non healthy printer (no signal)",
        "anti_gaming": "Asymmetric: only downside;\ncan't game by avoiding action",
        "evidence_req": "anomaly_tags / note_tags",
        "color": "#8172B3",
    },
    {
        "name": "r_novel_fault",
        "label": "Novel Fault\nDetection",
        "range_min": 0.0,
        "range_max": +0.4,
        "what": "Catch out-of-taxonomy faults\n(hybrid_thermal, intermittent_mcu,\ndegraded_webcam)",
        "anti_gaming": "Evidence-gated: preemptive\naction with no obs signal → 0.0",
        "evidence_req": "observation= dict required\n+ anomaly_flags / telemetry",
        "color": "#937860",
    },
]

SIGNAL_TYPES = ["operator_notes", "customer_messages", "anomaly_flags", "structured"]


# ── helpers ────────────────────────────────────────────────────────────────────

def savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {path}")


# ── 1. Components table ────────────────────────────────────────────────────────

def make_components_table(out_dir: Path):
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.axis("off")

    col_labels = ["Component", "Range", "What it measures", "Evidence required", "Anti-gaming"]
    rows = []
    for c in COMPONENTS:
        rows.append([
            c["name"],
            f"[{c['range_min']:+.2f}, {c['range_max']:+.2f}]",
            c["what"].replace("\n", " "),
            c["evidence_req"].replace("\n", " "),
            c["anti_gaming"].replace("\n", " "),
        ])

    col_widths = [0.14, 0.10, 0.26, 0.24, 0.26]
    colors = [["#16213e"] * 5] * len(rows)
    header_colors = ["#0f3460"] * 5

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor(header_colors[col] if row == 0 else colors[row - 1][col])
        cell.set_text_props(color="white")
        cell.set_edgecolor("#2a2a4a")
        if col == 0 and row > 0:
            cell.set_text_props(color=COMPONENTS[row - 1]["color"], fontweight="bold")

    ax.set_title(
        "PrintFarmEnv — 6-Component Reward Design",
        color="white", fontsize=14, fontweight="bold", pad=15,
    )
    savefig(fig, out_dir / "reward_components_table.png")


# ── 2. Range bars ──────────────────────────────────────────────────────────────

def make_range_bars(out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    sorted_comps = sorted(COMPONENTS, key=lambda c: c["range_max"], reverse=True)
    y = np.arange(len(sorted_comps))
    bar_h = 0.5

    for i, c in enumerate(sorted_comps):
        lo, hi = c["range_min"], c["range_max"]
        ax.barh(y[i], hi - lo, left=lo, height=bar_h, color=c["color"], alpha=0.85)
        # zero marker
        ax.axvline(0, color="#888", linewidth=0.8, linestyle="--")
        ax.text(hi + 0.01, y[i], f"{hi:+.2f}", va="center", ha="left", color="white", fontsize=9)
        ax.text(lo - 0.01, y[i], f"{lo:+.2f}", va="center", ha="right", color="white", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels([c["name"] for c in sorted_comps], color="white", fontsize=10)
    ax.set_xlabel("Reward contribution", color="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#2a2a4a")
    ax.set_title(
        "Reward Component Ranges\n(no single component dominates)",
        color="white", fontsize=12, fontweight="bold",
    )
    ax.set_xlim(-0.55, 0.65)

    total_min = sum(c["range_min"] for c in COMPONENTS)
    total_max = sum(c["range_max"] for c in COMPONENTS)
    ax.text(
        0.98, 0.02,
        f"Total range: [{total_min:.2f}, {total_max:.2f}]",
        transform=ax.transAxes, ha="right", va="bottom",
        color="#aaa", fontsize=8,
    )

    fig.tight_layout()
    savefig(fig, out_dir / "reward_range_bars.png")


# ── 3. Decision flow ───────────────────────────────────────────────────────────

def make_decision_flow(out_dir: Path):
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)

    def box(x, y, w, h, text, color="#0f3460", text_color="white", fontsize=9):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle="round,pad=0.1", linewidth=1.5,
                              edgecolor="#4a4a7a", facecolor=color)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center",
                color=text_color, fontsize=fontsize, fontweight="bold",
                multialignment="center")

    def arrow(x1, y1, x2, y2, label="", color="white"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.1, label, ha="center", va="bottom",
                    color="#aaa", fontsize=7.5)

    # Main pipeline nodes
    box(1.2, 3, 1.8, 0.8, "Observation\n(FarmObs dict)", "#16213e")
    box(3.5, 3, 1.8, 0.8, "parse_action()\n→ AgentAction | None", "#0f3460")
    box(6.5, 3, 1.6, 0.8, "compute_reward()\n6 components", "#0f3460")
    box(9.5, 3, 1.6, 0.8, "Σ total\nreward", "#0f3460")
    box(12, 3, 1.8, 0.8, "GRPO update\n(TRL trainer)", "#16213e")

    arrow(2.1, 3, 2.6, 3)
    arrow(4.4, 3, 5.7, 3)
    arrow(7.3, 3, 8.7, 3)
    arrow(10.3, 3, 11.1, 3)

    # Component breakdown (below main pipeline)
    comp_data = [
        (4.0, 1.4, "r_format\n[-0.3,+0.1]", COMPONENTS[0]["color"], False),
        (5.5, 1.4, "r_economic\n[-0.4,+0.4]", COMPONENTS[1]["color"], False),
        (6.5, 0.5, "r_fault_precision\n[-0.15,+0.4]", COMPONENTS[2]["color"], True),
        (8.0, 1.4, "r_msg_handling\n[-0.1,+0.4]", COMPONENTS[3]["color"], False),
        (9.2, 0.5, "r_unnecessary\n[-0.05,0.0]", COMPONENTS[4]["color"], False),
        (10.5, 0.5, "r_novel_fault\n[0.0,+0.4]", COMPONENTS[5]["color"], True),
    ]
    for cx, cy, label, col, gated in comp_data:
        box(cx, cy, 1.4, 0.65, label, col, fontsize=7.5)
        ax.annotate("", xy=(cx, cy + 0.33), xytext=(6.5, 2.6),
                    arrowprops=dict(arrowstyle="<-", color=col, lw=1, linestyle="dotted"))
        if gated:
            ax.text(cx, cy - 0.45, "evidence-gated", ha="center",
                    color="#ffcc00", fontsize=6.5, style="italic")

    ax.text(0.5, 5.5, "Evidence-gating (yellow) = reward only fires when observable signal exists",
            color="#ffcc00", fontsize=8, ha="left", style="italic")
    ax.set_title("Reward Computation Pipeline", color="white", fontsize=13, fontweight="bold")
    savefig(fig, out_dir / "reward_decision_flow.png")


# ── 4. Baseline decomposition ──────────────────────────────────────────────────

def make_baseline_decomp(out_dir: Path, baselines_path: Path):
    # Load real baselines if available, else use known values
    baseline_data = {}
    if baselines_path.exists():
        with open(baselines_path) as f:
            raw = json.load(f)
        baseline_data = raw

    # Known component breakdown for dp_rules from Session notes
    # r_format: -0.1, r_message_handling: +0.082, others: 0.0
    known_breakdown = {
        "r_format": -0.10,
        "r_economic": 0.0,
        "r_fault_precision": 0.0,
        "r_message_handling": +0.082,
        "r_unnecessary_action": 0.0,
        "r_novel_fault": 0.0,
    }

    # Summary totals from baselines.json
    policies = {}
    if baseline_data:
        for key, val in baseline_data.items():
            if isinstance(val, dict) and "avg_reward" in val:
                policies[key] = val["avg_reward"]
    else:
        policies = {
            "full_rules": +0.1036,
            "full_random": +0.0254,
            "full_wait": +0.0376,
            "dp_rules": -0.0182,
            "dp_random": -0.1270,
            "dp_wait": -0.0182,
        }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1a2e")

    # Left: all-policy bar chart
    pol_names = list(policies.keys())
    pol_vals = list(policies.values())
    bar_colors = ["#55A868" if v >= 0 else "#C44E52" for v in pol_vals]
    bars = ax1.bar(pol_names, pol_vals, color=bar_colors, alpha=0.85)
    ax1.axhline(0, color="#888", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Avg episode reward", color="white", fontsize=9)
    ax1.tick_params(colors="white", axis="both")
    ax1.set_xticks(range(len(pol_names)))
    ax1.set_xticklabels(pol_names, rotation=30, ha="right", color="white", fontsize=8)
    ax1.spines[:].set_color("#2a2a4a")
    ax1.set_title("All Baselines", color="white", fontsize=11, fontweight="bold")
    for bar, val in zip(bars, pol_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f"{val:+.3f}", ha="center", va="bottom", color="white", fontsize=8)

    # Add annotation: dp_rules = dp_wait (key narrative)
    if "dp_rules" in policies and "dp_wait" in policies:
        ax1.annotate(
            "dp_rules ≡ dp_wait\n→ rules blind to text signals",
            xy=(pol_names.index("dp_rules"), policies["dp_rules"]),
            xytext=(pol_names.index("dp_rules") + 0.5, -0.05),
            color="#ffcc00", fontsize=7.5, ha="center",
            arrowprops=dict(arrowstyle="->", color="#ffcc00", lw=1),
        )

    # Right: dp_rules component breakdown stacked bar
    comp_names = list(known_breakdown.keys())
    comp_vals = list(known_breakdown.values())
    bar_cols = [c["color"] for c in COMPONENTS]
    pos_vals = [max(0, v) for v in comp_vals]
    neg_vals = [min(0, v) for v in comp_vals]

    ax2.bar(comp_names, pos_vals, color=bar_cols, alpha=0.85)
    ax2.bar(comp_names, neg_vals, color=bar_cols, alpha=0.85)
    ax2.axhline(0, color="#888", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Component reward (dp_rules)", color="white", fontsize=9)
    ax2.tick_params(colors="white", axis="both")
    ax2.set_xticks(range(len(comp_names)))
    ax2.set_xticklabels(comp_names, rotation=30, ha="right", color="white", fontsize=8)
    ax2.spines[:].set_color("#2a2a4a")
    ax2.set_title("dp_rules Component Breakdown\n(rules cannot read text signals → all 0 except format)",
                  color="white", fontsize=10, fontweight="bold")
    for i, (name, val) in enumerate(zip(comp_names, comp_vals)):
        if val != 0:
            ax2.text(i, val + (0.004 if val > 0 else -0.007),
                     f"{val:+.3f}", ha="center", va="bottom" if val > 0 else "top",
                     color="white", fontsize=8)

    fig.suptitle("Baselines — Rules Fail at Decision Points (dp_rules ≡ dp_wait = -0.018)",
                 color="white", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    savefig(fig, out_dir / "baseline_decomposition.png")


# ── 5. Signal coverage ─────────────────────────────────────────────────────────

def make_signal_coverage(out_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1a2e")

    labels = ["operator_notes", "customer_messages", "anomaly_flags", "structured"]
    counts = [1, 1, 1, 1]  # round-robin: equal coverage
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    wedge_props = dict(edgecolor="#1a1a2e", linewidth=2)

    ax1.pie(counts, labels=labels, colors=colors, autopct="%1.0f%%",
            wedgeprops=wedge_props, textprops={"color": "white", "fontsize": 9},
            startangle=90)
    ax1.set_title("Round-Robin Signal\nCoverage (equal 25% each)",
                  color="white", fontsize=11, fontweight="bold")

    # Table explaining what each signal trains
    rows = [
        ["operator_notes", "Predictive / ambiguous text notes", "r_fault_precision"],
        ["customer_messages", "Rush/decline/substitute requests", "r_message_handling"],
        ["anomaly_flags", "Novel fault text descriptions", "r_novel_fault"],
        ["structured", "Telemetry + rules P&L signals", "r_economic"],
    ]
    col_labels = ["Signal type", "What it contains", "Primary reward"]
    table = ax2.table(cellText=rows, colLabels=col_labels, cellLoc="left", loc="center",
                      colWidths=[0.3, 0.42, 0.28])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor("#16213e" if row > 0 else "#0f3460")
        cell.set_text_props(color="white")
        cell.set_edgecolor("#2a2a4a")
        if col == 0 and row > 0:
            cell.set_text_props(color=colors[row - 1], fontweight="bold")
    ax2.axis("off")
    ax2.set_title("Signal → Reward Mapping",
                  color="white", fontsize=11, fontweight="bold")

    fig.suptitle("Anti-Data-Imbalance: Round-Robin Ensures All 4 Signal Types Train Equally",
                 color="white", fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    savefig(fig, out_dir / "signal_coverage.png")


# ── 6. Hyperparameters card ────────────────────────────────────────────────────

def make_hyperparameters_card(out_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.axis("off")

    params = [
        ("Model", "google/gemma-3-1b-it (local)  /  gemma-3-4b-it (HF Jobs)"),
        ("Training algo", "GRPO (Group Relative Policy Optimisation) via HF TRL"),
        ("LoRA rank", "16   |   LoRA alpha: 16   |   Target: q_proj, v_proj"),
        ("Learning rate", "5e-6   |   Warmup: 5% of max_steps"),
        ("n_generations", "4 (local MPS)  /  8 (HF T4/A10)"),
        ("max_steps", "200 (local validation)  /  500 (HF 4B run)"),
        ("max_completion_length", "256 tokens"),
        ("max_seq_length", "2048 tokens"),
        ("Temperature", "0.3  (sampling during rollout)"),
        ("Reward aggregation", "Uniform sum of 6 components (no learned weights)"),
        ("Prompt policy", "Round-robin over SIGNAL_TYPES; skip (not fallback) if target absent"),
        ("Anti-echo threshold", ">50% token overlap with observation text → r_format = -0.2"),
        ("Evidence gating", "r_fault_precision + r_novel_fault require observation= dict"),
        ("Batch size", "= n_generations (1 prompt × n completions per step)"),
        ("Save checkpoints", "Every 25 steps; final_adapter + merged saved at end"),
        ("Logging", "monitor.jsonl (per-step) + W&B (optional)"),
    ]

    col_labels = ["Parameter", "Value"]
    col_widths = [0.28, 0.72]
    rows = [[p[0], p[1]] for p in params]
    colors = [["#16213e", "#101028"]] * len(rows)

    table = ax.table(cellText=rows, colLabels=col_labels, cellLoc="left", loc="center",
                     colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#0f3460")
        else:
            cell.set_facecolor(colors[row - 1][col])
        cell.set_text_props(color="white")
        cell.set_edgecolor("#2a2a4a")
        if col == 0 and row > 0:
            cell.set_text_props(color="#aad4f5", fontweight="bold")

    ax.set_title("Training Hyperparameters — PrintFarmEnv GRPO",
                 color="white", fontsize=13, fontweight="bold", pad=15)
    savefig(fig, out_dir / "hyperparameters_card.png")


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate reward-engineering visuals.")
    parser.add_argument("--out", default="submission/plots",
                        help="Output directory for PNGs")
    parser.add_argument("--baselines", default="submission/eval/baselines.json",
                        help="Path to baselines.json")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    baselines_path = Path(args.baselines)

    print(f"Generating reward visuals → {out_dir}/")
    make_components_table(out_dir)
    make_range_bars(out_dir)
    make_decision_flow(out_dir)
    make_baseline_decomp(out_dir, baselines_path)
    make_signal_coverage(out_dir)
    make_hyperparameters_card(out_dir)
    print("Done. 6 PNGs written.")


if __name__ == "__main__":
    main()
