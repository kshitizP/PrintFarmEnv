"""
watch_training.py — Live red/yellow/green dashboard for GRPO training runs.

Tails monitor.jsonl and prints a colour-coded status update each time a new
row arrives. Implements the early-warning decision gates from the plan:

  Gate @ step ~30: FORMAT_FAIL → abort and fix prompt template
  Gate @ step ~75: ECHO_HACKING | SIGNAL_IMBALANCE → abort and tighten reward
  Gate @ step 200: all green → ready for HF Jobs escalation

Usage:
    python submission/scripts/watch_training.py --run grpo_runs/m4_local_v1
    python submission/scripts/watch_training.py --run grpo_runs/m4_local_v1 --interval 5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"
CYAN   = "\033[96m"
MAGENTA = "\033[95m"


def _c(text, colour):
    return f"{colour}{text}{RESET}"


# ── Decision gate thresholds (from plan) ──────────────────────────────────────

GATES = {
    30: [
        ("tag_pct",   "<", 40,  RED,    "FORMAT not learned — abort, fix prompt template"),
        ("parse_pct", "<", 40,  RED,    "JSON parse rate too low — abort, fix prompt template"),
    ],
    75: [
        ("echo_pct",  ">", 30,  RED,    "ECHO_HACKING — abort, tighten anti-echo penalty in r_format"),
        ("reward_avg","<", -0.02, YELLOW, "Reward not trending up after 75 steps — inspect generations"),
    ],
    200: [
        ("reward_avg",">", 0.05, GREEN, "Reward climbing — ready for HF Jobs 4B escalation"),
    ],
}

# Continuous health thresholds
THRESHOLDS = {
    "reward_avg":    [(0.05, GREEN, "good"), (0.0, YELLOW, "flat"), (None, RED, "low")],
    "tag_pct":       [(70, GREEN, "ok"), (40, YELLOW, "low"), (None, RED, "FAIL")],
    "parse_pct":     [(70, GREEN, "ok"), (40, YELLOW, "low"), (None, RED, "FAIL")],
    "echo_pct":      [(None, None, None), (15, GREEN, "ok"), (30, YELLOW, "elevated"), (None, RED, "HACKING")],
    "unique_actions":[(4, GREEN, "diverse"), (2, YELLOW, "limited"), (None, RED, "COLLAPSED")],
}


def _colour_val(field, val):
    if field == "echo_pct":
        if val <= 15:
            return _c(f"{val:.1f}%", GREEN)
        elif val <= 30:
            return _c(f"{val:.1f}%", YELLOW)
        else:
            return _c(f"{val:.1f}%", RED)
    elif field == "unique_actions":
        if val >= 4:
            return _c(str(val), GREEN)
        elif val >= 2:
            return _c(str(val), YELLOW)
        else:
            return _c(str(val), RED)
    elif field in ("tag_pct", "parse_pct"):
        if val >= 70:
            return _c(f"{val:.1f}%", GREEN)
        elif val >= 40:
            return _c(f"{val:.1f}%", YELLOW)
        else:
            return _c(f"{val:.1f}%", RED)
    elif field == "reward_avg":
        if val >= 0.05:
            return _c(f"{val:+.4f}", GREEN)
        elif val >= -0.01:
            return _c(f"{val:+.4f}", YELLOW)
        else:
            return _c(f"{val:+.4f}", RED)
    return str(val)


def _health_colour(health: str) -> str:
    if "WARNING" in health:
        if any(kw in health for kw in ("ECHO_HACKING", "SIGNAL_IMBALANCE", "FORMAT_FAIL",
                                        "ACTION_COLLAPSE", "TASK_NOT_LEARNING")):
            return _c(health, RED)
        return _c(health, YELLOW)
    return _c(health, GREEN)


def _render_row(row: dict, history: list):
    step = row.get("step", "?")
    reward = row.get("reward_avg", 0.0)
    tag = row.get("tag_pct", 0.0)
    parse = row.get("parse_pct", 0.0)
    echo = row.get("echo_pct", 0.0)
    unique = row.get("unique_actions", 0)
    health = row.get("health", "?")
    p50 = row.get("completion_len_p50", "—")
    p95 = row.get("completion_len_p95", "—")
    action_dist = row.get("action_dist", {})
    per_signal = row.get("per_signal_reward", {})
    components = row.get("reward_components", {})
    sample = row.get("sample", "")

    width = 72
    print("\n" + "─" * width)
    print(_c(f"  STEP {step}", BOLD) +
          f"  │  reward {_colour_val('reward_avg', reward)}"
          f"  │  tag {_colour_val('tag_pct', tag)}"
          f"  │  parse {_colour_val('parse_pct', parse)}"
          f"  │  echo {_colour_val('echo_pct', echo)}")

    # Reward trend arrow
    if len(history) >= 3:
        recent = [r.get("reward_avg", 0) for r in history[-5:]]
        trend = recent[-1] - recent[0]
        arrow = "↑" if trend > 0.005 else ("↓" if trend < -0.005 else "→")
        trend_col = GREEN if trend > 0.005 else (RED if trend < -0.005 else YELLOW)
        print(f"  trend: {_c(arrow + f' {trend:+.4f} over {len(recent)} checkpoints', trend_col)}"
              f"  │  unique_actions: {_colour_val('unique_actions', unique)}"
              f"  │  len p50={p50} p95={p95}")
    else:
        print(f"  unique_actions: {_colour_val('unique_actions', unique)}"
              f"  │  len p50={p50} p95={p95}")

    # Action distribution
    if action_dist:
        top5 = sorted(action_dist.items(), key=lambda x: x[1], reverse=True)[:5]
        dist_str = "  ".join(f"{a}={c}" for a, c in top5)
        print(f"  actions: {_c(dist_str, CYAN)}")

    # Per-signal reward breakdown
    if per_signal:
        sig_parts = []
        for sig, val in sorted(per_signal.items()):
            col = GREEN if val > 0.01 else (RED if val < -0.01 else YELLOW)
            sig_parts.append(_c(f"{sig.replace('_', '-')}={val:+.3f}", col))
        print(f"  per_signal: " + "  ".join(sig_parts))

    # Per-component reward means
    if components:
        comp_parts = []
        for k, v in components.items():
            short = k.replace("r_", "")
            col = GREEN if v > 0.01 else (RED if v < -0.01 else DIM)
            comp_parts.append(_c(f"{short}={v:+.3f}", col))
        print(f"  components: " + "  ".join(comp_parts))

    # Health
    print(f"  health: {_health_colour(health)}")

    # Decision gate advice
    _check_gates(step, row)

    # Sample completion (truncated)
    if sample:
        print(f"  {_c('sample:', DIM)} {sample[:100]}")

    print("─" * width)


def _check_gates(step: int, row: dict):
    advice = []
    for gate_step, checks in GATES.items():
        if abs(step - gate_step) <= 5:
            for field, op, threshold, col, msg in checks:
                val = row.get(field)
                if val is None:
                    continue
                triggered = (
                    (op == "<" and val < threshold) or
                    (op == ">" and val > threshold)
                )
                if triggered:
                    advice.append((col, f"[GATE @step {gate_step}] {msg}"))
                elif col == GREEN and op == ">":
                    advice.append((GREEN, f"[GATE @step {gate_step}] PASS — {field}={val:.3f} > {threshold}"))

    # Always-on signal imbalance check
    per_signal = row.get("per_signal_reward", {})
    if per_signal and len(per_signal) > 1:
        total_abs = sum(abs(v) for v in per_signal.values())
        if total_abs > 0:
            dominant = max(per_signal, key=lambda k: abs(per_signal[k]))
            frac = abs(per_signal[dominant]) / total_abs
            if frac >= 0.80:
                advice.append((RED, f"[SIGNAL_IMBALANCE] '{dominant}' carries {frac:.0%} of reward — check round-robin targeting"))

    for col, msg in advice:
        print(f"  {_c('▶ ' + msg, col)}")


def watch(run_dir: Path, interval: float = 2.0):
    monitor_file = run_dir / "monitor.jsonl"

    print(f"\n{_c('PrintFarmEnv Training Watchdog', BOLD)}")
    print(f"Monitoring: {monitor_file}")
    print(f"Refresh: {interval}s  │  Ctrl-C to exit\n")

    if not run_dir.exists():
        print(_c(f"Run directory does not exist yet: {run_dir}", YELLOW))
        print("Waiting for training to start...")

    history = []
    last_size = 0

    try:
        while True:
            if not monitor_file.exists():
                print(".", end="", flush=True)
                time.sleep(interval)
                continue

            size = monitor_file.stat().st_size
            if size == last_size:
                time.sleep(interval)
                continue

            last_size = size

            # Re-read all rows (file is append-only JSONL)
            rows = []
            with open(monitor_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

            if len(rows) > len(history):
                new_rows = rows[len(history):]
                for row in new_rows:
                    history.append(row)
                    _render_row(row, history)

                # Print summary at end
                _print_summary(history)

            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n\n{_c('Watchdog stopped.', DIM)}")
        if history:
            print(f"\nFinal summary ({len(history)} checkpoints):")
            _print_summary(history)


def _print_summary(history: list):
    if not history:
        return
    last = history[-1]
    step = last.get("step", "?")
    reward = last.get("reward_avg", 0.0)

    if len(history) >= 2:
        first = history[0]
        delta = reward - first.get("reward_avg", 0.0)
        print(f"\n  {_c(f'Summary: {len(history)} checkpoints through step {step}', BOLD)}")
        print(f"  Reward start={first.get('reward_avg',0):+.4f}  "
              f"latest={reward:+.4f}  "
              f"Δ={_c(f'{delta:+.4f}', GREEN if delta > 0 else RED)}")


def main():
    parser = argparse.ArgumentParser(description="Watch live GRPO training via monitor.jsonl")
    parser.add_argument("--run", required=True, help="Path to grpo run dir (contains monitor.jsonl)")
    parser.add_argument("--interval", type=float, default=2.0, help="Poll interval in seconds")
    args = parser.parse_args()

    watch(Path(args.run), interval=args.interval)


if __name__ == "__main__":
    main()
