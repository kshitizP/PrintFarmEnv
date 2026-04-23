"""
Clairvoyant-Greedy Baseline
===========================
Plays PrintFarmEnv with GROUND-TRUTH access to env internals.
This is the "prescient upper bound" reference — not optimal (NP-hard),
but significantly better than any observation-only policy.

Serves two purposes per ROUND2_MANUAL.md §9 and §13:
  1. Sets the clairvoyant ceiling for the pitch ("agent captures X% of the gap")
  2. Produces (observation, action) SFT training trajectories

Usage:
    python baselines/clairvoyant_greedy.py
    python baselines/clairvoyant_greedy.py --tasks task_1 task_2 task_3 --episodes 200
    python baselines/clairvoyant_greedy.py --tasks task_1 --episodes 5 --verbose
    python baselines/clairvoyant_greedy.py --output data/sft_trajectories.jsonl --episodes 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from printfarm_env.env import PrintFarmEnvironment
from printfarm_env.models import (
    FarmAction, FarmActionEnum, FarmObservation,
    JobState, PrinterState,
)
from printfarm_env.operators import SKILL_RANK, TICKET_SKILL_REQ, queue_total, is_on_shift
from printfarm_env.failures import FAILURE_MODES

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

DEFAULT_TASKS    = ["task_1", "task_2", "task_3"]
DEFAULT_EPISODES = 10
FATIGUE_DANGER   = 8    # dispatch maintenance when fatigue >= this


# ---------------------------------------------------------------------------
#  Clairvoyant policy
# ---------------------------------------------------------------------------

def clairvoyant_action(env: PrintFarmEnvironment) -> FarmAction:
    """
    Priority-ordered greedy policy with full ground-truth access.

    Decision order (highest priority first):
      1. Catastrophe prevention  — maintenance when fatigue >= FATIGUE_DANGER
      2. Fault response          — diagnose / dispatch physical fix
      3. Runout recovery         — spool swap for PAUSED_RUNOUT printers
      4. Resume paused jobs      — resume after spool swap
      5. Job assignment          — assign best pending job to best idle printer,
                                   or seed a printer via spool swap if none loaded
      6. WAIT
    """
    jobs      = env._jobs
    operators = env._operators
    step      = env.time_step

    # -- 0. ERROR recovery — unjam before doing anything else ---------------
    queued_unjam_printers = {
        t.target_printer_id
        for op in operators
        for t in op.queue + ([op.current_ticket] if op.current_ticket else [])
        if t.ticket_type == "unjam_printer"
    }
    for p in env._printers:
        if p.state == PrinterState.ERROR and p.printer_id not in queued_unjam_printers:
            op = _best_operator(operators, "unjam_printer", step)
            if op:
                return FarmAction(
                    action=FarmActionEnum.DISPATCH_TICKET,
                    printer_id=p.printer_id,
                    operator_id=op.operator_id,
                    ticket_type="unjam_printer",
                )

    # -- 1. Catastrophe prevention -----------------------------------------
    for p in env._printers:
        if (p.fatigue_level >= FATIGUE_DANGER
                and p.state == PrinterState.IDLE
                and p.outstanding_ticket_id is None):
            op = _best_operator(operators, "maintenance_basic", step)
            if op:
                return FarmAction(
                    action=FarmActionEnum.DISPATCH_TICKET,
                    printer_id=p.printer_id,
                    operator_id=op.operator_id,
                    ticket_type="maintenance_basic",
                )

    # -- 2. Active fault response ------------------------------------------
    for p in env._printers:
        if not p.active_faults:
            continue
        if p.state in (PrinterState.OFFLINE, PrinterState.MAINTENANCE,
                       PrinterState.MAINTENANCE_QUEUED):
            continue

        for mode in list(p.active_faults.keys()):
            clears = FAILURE_MODES[mode]["clears_on"]

            if "run_diagnostic" in clears:
                return FarmAction(
                    action=FarmActionEnum.RUN_DIAGNOSTIC,
                    printer_id=p.printer_id,
                )

            if "diagnostic_physical" in clears:
                op = _best_operator(operators, "diagnostic_physical", step)
                if op:
                    return FarmAction(
                        action=FarmActionEnum.DISPATCH_TICKET,
                        printer_id=p.printer_id,
                        operator_id=op.operator_id,
                        ticket_type="diagnostic_physical",
                    )

            if "maintenance_basic" in clears and p.outstanding_ticket_id is None:
                op = _best_operator(operators, "maintenance_basic", step)
                if op and p.state == PrinterState.IDLE:
                    return FarmAction(
                        action=FarmActionEnum.DISPATCH_TICKET,
                        printer_id=p.printer_id,
                        operator_id=op.operator_id,
                        ticket_type="maintenance_basic",
                    )

    # -- 3. Runout recovery ------------------------------------------------
    for p in env._printers:
        if p.state != PrinterState.PAUSED_RUNOUT:
            continue
        j = jobs.get(p.current_job_id)
        if j is None:
            continue
        return FarmAction(
            action=FarmActionEnum.REQUEST_SPOOL_SWAP,
            printer_id=p.printer_id,
            material=j.material_required,
        )

    # -- 4. Resume paused jobs ---------------------------------------------
    for p in env._printers:
        if p.state != PrinterState.IDLE or p.current_job_id is None:
            continue
        j = jobs.get(p.current_job_id)
        if j and j.state == JobState.PAUSED:
            return FarmAction(
                action=FarmActionEnum.RESUME_JOB,
                printer_id=p.printer_id,
                job_id=j.job_id,
            )

    # -- 5. Job assignment -------------------------------------------------
    pending = sorted(
        [j for j in jobs.values() if j.state == JobState.PENDING],
        key=lambda j: (
            -j.priority,
            j.deadline_steps if j.deadline_steps else 9999,
        ),
    )

    # Idle printers ready to work (no faults, not over fatigue, no pending ticket)
    idle_printers = [
        p for p in env._printers
        if p.state == PrinterState.IDLE
        and p.current_job_id is None
        and p.fatigue_level < FATIGUE_DANGER
        and not p.active_faults
        and p.outstanding_ticket_id is None
    ]

    for job in pending:
        material = job.material_required

        # (a) Printer with matching material + enough spool
        candidates = [
            p for p in idle_printers
            if p.current_material == material
            and p.spool_weight_g >= job.weight_required_g
        ]
        if candidates:
            best = max(candidates, key=lambda p: (p.reliability, -p.fatigue_level))
            idle_printers.remove(best)
            return FarmAction(
                action=FarmActionEnum.ASSIGN_JOB,
                printer_id=best.printer_id,
                job_id=job.job_id,
            )

        # (b) No matching printer — seed one via spool swap (if inventory allows).
        # Guard: don't swap away a material if there are still pending jobs for it.
        if env._inventory.get(material, 0.0) >= job.weight_required_g:
            seedable = []
            for p in idle_printers:
                if p.current_material == material:
                    continue  # already correct material, handled in (a)
                current_mat = p.current_material
                if current_mat:
                    other_pending = any(
                        j2.state == JobState.PENDING and j2.material_required == current_mat
                        for j2 in jobs.values()
                    )
                    if other_pending:
                        continue  # keep this printer free for its current material
                seedable.append(p)
            if seedable:
                best = max(seedable, key=lambda p: (p.reliability, -p.fatigue_level))
                idle_printers.remove(best)
                return FarmAction(
                    action=FarmActionEnum.REQUEST_SPOOL_SWAP,
                    printer_id=best.printer_id,
                    material=material,
                )

    # -- 6. WAIT -----------------------------------------------------------
    return FarmAction(action=FarmActionEnum.WAIT)


def _best_operator(operators, ticket_type: str, step: int):
    """Return the best available operator for a ticket type, or None."""
    req_rank = SKILL_RANK.get(TICKET_SKILL_REQ.get(ticket_type, "junior"), 0)
    candidates = [
        op for op in operators
        if SKILL_RANK.get(op.skill_level, 0) >= req_rank
        and is_on_shift(op, step)
        and queue_total(op) < op.queue_capacity
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda o: (
        queue_total(o),
        -SKILL_RANK.get(o.skill_level, 0),
        o.operator_id,
    ))


# ---------------------------------------------------------------------------
#  Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: PrintFarmEnvironment,
    task_id: str,
    episode: int,
    seed: int,
    verbose: bool,
    trajectory_buffer: Optional[list],
) -> float:
    """Run one episode. Appends SFT records to trajectory_buffer if provided."""
    obs = env.reset(seed=seed + episode, task_id=task_id)

    if verbose:
        print(f"    Episode {episode:3d}  ", end="", flush=True)

    step = 0
    while not obs.done:
        action = clairvoyant_action(env)

        if trajectory_buffer is not None:
            trajectory_buffer.append({
                "task_id":     task_id,
                "episode":     episode,
                "step":        env.time_step,
                "observation": json.loads(obs.model_dump_json()),
                "action":      json.loads(action.model_dump_json()),
            })

        obs = env.step(action)
        step += 1

    if trajectory_buffer is not None:
        for r in trajectory_buffer:
            if r["task_id"] == task_id and r["episode"] == episode:
                r["episode_reward"] = obs.reward

    if verbose:
        print(f"score={obs.reward:.4f}  profit=${obs.net_profit_usd:+.2f}  steps={step}")

    return obs.reward


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Clairvoyant-greedy baseline")
    parser.add_argument("--tasks",    nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--episodes", type=int,  default=DEFAULT_EPISODES,
                        help="Episodes per task")
    parser.add_argument("--seed",     type=int,  default=42)
    parser.add_argument("--output",   type=str,  default=None,
                        help="JSONL output path for SFT trajectories")
    parser.add_argument("--verbose",  "-v", action="store_true")
    args = parser.parse_args()

    env            = PrintFarmEnvironment()
    trajectory_buf = [] if args.output else None
    results        = {}

    print(f"Clairvoyant-Greedy Baseline")
    print(f"Tasks:    {', '.join(args.tasks)}")
    print(f"Episodes: {args.episodes} per task")
    if args.output:
        print(f"SFT out:  {args.output}")
    print()

    for task_id in args.tasks:
        print(f"  {task_id}:")
        scores = []
        for ep in range(args.episodes):
            score = run_episode(
                env, task_id, ep, args.seed, args.verbose, trajectory_buf
            )
            scores.append(score)
            if not args.verbose and (ep + 1) % 10 == 0:
                print(f"    [{ep+1:3d}/{args.episodes}] mean={sum(scores)/len(scores):.4f}",
                      flush=True)

        mean  = sum(scores) / len(scores)
        results[task_id] = {"mean": mean, "best": max(scores), "worst": min(scores)}
        print(f"    -> mean={mean:.4f}  best={max(scores):.4f}  worst={min(scores):.4f}\n")

    print("=" * 55)
    print(f"  {'Task':<14} {'Mean':>8} {'Best':>8} {'Worst':>8}")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8}")
    for task_id, r in results.items():
        print(f"  {task_id:<14} {r['mean']:>8.4f} {r['best']:>8.4f} {r['worst']:>8.4f}")
    print("=" * 55)

    if args.output and trajectory_buf:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for record in trajectory_buf:
                f.write(json.dumps(record) + "\n")
        print(f"\n  SFT trajectories: {len(trajectory_buf)} steps -> {args.output}")


if __name__ == "__main__":
    main()
