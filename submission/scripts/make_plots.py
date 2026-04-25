"""
Generate the four submission plots from training artifacts.

Usage (from repo root after a real GRPO run):
    python submission/scripts/make_plots.py --run grpo_runs/overnight_v1
    python submission/scripts/make_plots.py --run grpo_runs/hf_4b_v1 --out submission/plots

Output:
    submission/plots/reward_curve.png
    submission/plots/component_decomposition.png
    submission/plots/action_distribution.png
    submission/plots/before_after_generations.png

Reads:
    <run>/checkpoint-*/trainer_state.json  — reward curve data
    <run>/monitor.jsonl or wandb export    — per-component data (optional)
    submission/eval/baselines.json         — baseline reference lines
"""

import argparse
import json
import os
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
for _candidate in [_here.parent, _here.parent.parent]:
    if (_candidate / "submission" / "__init__.py").exists():
        sys.path.insert(0, str(_candidate))
        break


def _load_trainer_log_history(run_dir: Path):
    """Extract log_history from the latest trainer_state.json in the run."""
    checkpoints = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if not checkpoints:
        # Try the run dir itself
        ts = run_dir / "trainer_state.json"
        if ts.exists():
            with open(ts) as f:
                return json.load(f).get("log_history", [])
        return []
    # Use the last checkpoint
    ts = checkpoints[-1] / "trainer_state.json"
    if ts.exists():
        with open(ts) as f:
            return json.load(f).get("log_history", [])
    return []


def _load_monitor_jsonl(run_dir: Path):
    """Load per-step monitor data from monitor.jsonl if present."""
    mf = run_dir / "monitor.jsonl"
    if not mf.exists():
        return []
    records = []
    with open(mf) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _load_baselines(submission_dir: Path):
    bf = submission_dir / "eval" / "baselines.json"
    if not bf.exists():
        return {}
    with open(bf) as f:
        return json.load(f)


def plot_reward_curve(log_history, baselines, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [e["step"] for e in log_history if "step" in e and "reward" in e]
    rewards = [e["reward"] for e in log_history if "step" in e and "reward" in e]

    if not steps:
        print("[make_plots] No reward data in trainer log — skipping reward curve")
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, rewards, "b-", linewidth=1.5, label="GRPO (trained)", alpha=0.8)

    # Smoothed trend
    if len(rewards) >= 5:
        from itertools import accumulate
        window = max(1, len(rewards) // 10)
        smoothed = []
        for i in range(len(rewards)):
            start = max(0, i - window // 2)
            end = min(len(rewards), i + window // 2 + 1)
            smoothed.append(sum(rewards[start:end]) / (end - start))
        ax.plot(steps, smoothed, "b-", linewidth=2.5, alpha=0.4)

    # Baseline horizontal lines
    colors = {"full_rules": "green", "full_random": "gray", "dp_rules": "orange"}
    labels = {"full_rules": "Rules (full ep)", "full_random": "Random", "dp_rules": "Rules (dp)"}
    for key, color in colors.items():
        val = baselines.get(key, {}).get("avg_reward")
        if val is not None:
            ax.axhline(val, color=color, linestyle="--", linewidth=1, alpha=0.7,
                       label=f"{labels[key]} = {val:.3f}")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Reward")
    ax.set_title("GRPO Training Reward Curve — PrintFarm Dispatcher")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[make_plots] Saved {out_path}")


def plot_component_decomposition(monitor_records, log_history, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Try monitor.jsonl first, fall back to log_history custom keys
    component_keys = ["r_format", "r_economic", "r_fault_precision",
                      "r_message_handling", "r_unnecessary_action", "r_novel_fault"]

    steps = []
    components_by_key = {k: [] for k in component_keys}

    for rec in monitor_records:
        s = rec.get("step")
        if s is None:
            continue
        any_found = any(k in rec for k in component_keys)
        if not any_found:
            continue
        steps.append(s)
        for k in component_keys:
            components_by_key[k].append(rec.get(k, 0.0))

    if not steps:
        print("[make_plots] No per-component data — skipping component decomposition")
        return

    colors = {
        "r_format": "steelblue",
        "r_economic": "forestgreen",
        "r_fault_precision": "darkorange",
        "r_message_handling": "orchid",
        "r_unnecessary_action": "salmon",
        "r_novel_fault": "gold",
    }
    nice_names = {
        "r_format": "Format",
        "r_economic": "Economic",
        "r_fault_precision": "Fault Precision",
        "r_message_handling": "Message Handling",
        "r_unnecessary_action": "Unnecessary Action",
        "r_novel_fault": "Novel Fault",
    }

    fig, ax = plt.subplots(figsize=(9, 4))
    for k in component_keys:
        vals = components_by_key[k]
        if any(v != 0 for v in vals):
            ax.plot(steps, vals, label=nice_names[k], color=colors[k], linewidth=1.5, alpha=0.8)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Component Reward")
    ax.set_title("Reward Component Decomposition over Training")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[make_plots] Saved {out_path}")


def plot_action_distribution(monitor_records, baselines, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import Counter

    # Get action distributions from baselines and monitor
    baseline_dist = baselines.get("full_rules", {}).get("action_distribution", {})
    random_dist = baselines.get("full_random", {}).get("action_distribution", {})

    # Get trained model distribution from monitor (end of training)
    trained_dist = Counter()
    for rec in monitor_records[-50:]:  # last 50 steps
        for k, v in rec.get("action_distribution", {}).items():
            trained_dist[k] += v

    all_actions = [
        "ASSIGN_JOB", "CANCEL_JOB", "PAUSE_JOB", "RESUME_JOB",
        "RUN_DIAGNOSTIC", "DISPATCH_TICKET", "REQUEST_SPOOL_SWAP",
        "REQUEST_MAINTENANCE", "OVERRIDE_OPERATOR", "WAIT",
    ]

    if not any(trained_dist.values()):
        print("[make_plots] No action distribution data — skipping action distribution plot")
        return

    def normalise(d):
        total = sum(d.values()) or 1
        return {k: d.get(k, 0) / total for k in all_actions}

    import numpy as np
    x = np.arange(len(all_actions))
    width = 0.25

    r_norm = normalise(random_dist)
    b_norm = normalise(baseline_dist)
    t_norm = normalise(trained_dist)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - width, [r_norm[a] for a in all_actions], width, label="Random", color="gray", alpha=0.7)
    ax.bar(x, [b_norm[a] for a in all_actions], width, label="Rules", color="forestgreen", alpha=0.7)
    ax.bar(x + width, [t_norm[a] for a in all_actions], width, label="GRPO (step 200)", color="steelblue", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("_", "\n") for a in all_actions], fontsize=7)
    ax.set_ylabel("Fraction of actions")
    ax.set_title("Action Distribution: Random vs Rules vs GRPO (end of training)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[make_plots] Saved {out_path}")


def plot_before_after_generations(monitor_records, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    early = [r.get("sample_completion", "") for r in monitor_records[:5] if r.get("sample_completion")]
    late = [r.get("sample_completion", "") for r in monitor_records[-5:] if r.get("sample_completion")]

    if not early and not late:
        print("[make_plots] No sample completions in monitor data — skipping before/after plot")
        return

    fig, (ax_early, ax_late) = plt.subplots(1, 2, figsize=(14, 5))

    def render_panel(ax, samples, title):
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")
        text = "\n\n---\n\n".join(s[:300] for s in samples[:3])
        ax.text(0.02, 0.95, text, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", fontfamily="monospace",
                wrap=True, bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))

    render_panel(ax_early, early, "Step ~1 (untrained output)")
    render_panel(ax_late, late, "Step 200 (trained output)")

    fig.suptitle("Sample Generations: Before vs After GRPO Training", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[make_plots] Saved {out_path}")


def main():
    p = argparse.ArgumentParser(description="Generate submission plots")
    p.add_argument("--run", default="grpo_runs/overnight_v1",
                   help="Path to GRPO run directory containing checkpoints")
    p.add_argument("--out", default="submission/plots",
                   help="Output directory for PNG plots")
    args = p.parse_args()

    run_dir = Path(args.run)
    out_dir = Path(args.out)
    submission_dir = Path(__file__).resolve().parent.parent

    if not run_dir.exists():
        print(f"[make_plots] Run directory not found: {run_dir}")
        print("[make_plots] Run real GRPO training first (Phase C), then re-run this script.")
        sys.exit(1)

    print(f"[make_plots] Loading training artifacts from {run_dir}")
    log_history = _load_trainer_log_history(run_dir)
    monitor_records = _load_monitor_jsonl(run_dir)
    baselines = _load_baselines(submission_dir)

    print(f"  log_history entries: {len(log_history)}")
    print(f"  monitor records:     {len(monitor_records)}")
    print(f"  baselines loaded:    {list(baselines.keys())}")

    plot_reward_curve(log_history, baselines, out_dir / "reward_curve.png")
    plot_component_decomposition(monitor_records, log_history, out_dir / "component_decomposition.png")
    plot_action_distribution(monitor_records, baselines, out_dir / "action_distribution.png")
    plot_before_after_generations(monitor_records, out_dir / "before_after_generations.png")

    print(f"\n[make_plots] All plots written to {out_dir}/")
    print("Next: embed them in submission/README.md")


if __name__ == "__main__":
    main()
