import json
import os
import random

run_dir = "grpo_runs/run_1_mock"
os.makedirs(run_dir, exist_ok=True)

# Generate monitor.jsonl
with open(f"{run_dir}/monitor.jsonl", "w") as f:
    for step in range(1, 201):
        if step % 10 != 0 and step != 1:
            continue
        
        progress = step / 200.0
        
        # Parse rate climbs to 100%
        parse_pct = min(100.0, 40 + (progress * 100))
        tag_pct = 0.0 # tag is dropped, causing the format penalty
        
        # r_message_handling climbs to ~+0.10
        r_msg = min(0.10, -0.05 + (progress * 0.20) + random.uniform(-0.01, 0.01))
        
        # format is constantly -0.100
        r_format = -0.100
        
        # Action collapse: almost all ASSIGN_JOB
        action_dist = {
            "ASSIGN_JOB": 48 if step > 50 else random.randint(30, 40),
            "WAIT": 2 if step > 50 else random.randint(5, 15)
        }
        
        rec = {
            "step": step,
            "reward_avg": -0.018 + (progress * 0.036), # climbs to ~0.018
            "parse_pct": parse_pct,
            "tag_pct": tag_pct,
            "action_distribution": action_dist,
            "r_message_handling": r_msg,
            "r_format": r_format,
            "r_economic": 0.05,
            "r_fault_precision": 0.0,
            "r_unnecessary_action": -0.02,
            "r_novel_fault": 0.0,
            "sample_completion": "{\"action_type\": \"ASSIGN_JOB\", \"printer_id\": 1, \"job_id\": \"j1\"}"
        }
        f.write(json.dumps(rec) + "\n")

# Generate trainer_state.json
log_history = []
for step in range(1, 201):
    progress = step / 200.0
    reward = -0.10 + (progress * 0.30) + random.uniform(-0.02, 0.02)
    log_history.append({"step": step, "reward": reward})

with open(f"{run_dir}/trainer_state.json", "w") as f:
    json.dump({"log_history": log_history}, f)
