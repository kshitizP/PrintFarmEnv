# PrintFarmEnv — Session Notes (2026-04-25)

Use this file to resume work in a future conversation. Hand it to Claude as context.

---

## Branch

`round2-prep` (clean, no uncommitted changes at end of session)

---

## Phases completed

### Phase A — Decision-point sampling fix ✅

**Core bug fixed:** Python 3.14 changed `str(MyEnum.MEMBER)` to return `"MyEnum.MEMBER"` not `"MEMBER"`. Every state comparison like `str(p.state) == "IDLE"` was silently failing. Printers never left IDLE, anomaly flags never fired, customer messages never appeared.

Fixed in:
- `submission/env/decision_point.py` — 9 occurrences of `str(p/j.state)` → `.value`
- `submission/env/customer_messages.py` — `getattr(j.state, 'value', str(j.state))`
- `submission/env/operator_notes.py` — same
- `submission/env/novel_faults.py` — same

**Round-robin signal targeting added** so all three unstructured signal types get equal training coverage:
- `submission/env/decision_point.py` — `SIGNAL_TYPES`, `signal_present()`, `target_signal` param on `reset()`, max_steps 60→90
- `submission/training/rollout.py` — cycles target_signal; skips prompt if target not found
- `submission/eval/run_baselines.py` — same round-robin; fixed `dp_random_policy` to build valid candidate pool; passes `observation=obs_dict` to `compute_reward`

**Sys.path fixed** in test_smoke.py, run_baselines.py, train_grpo.py, eval_model.py — replaced hard-coded `parents[2]` with smart discovery loop.

**Baselines result** (`submission/eval/baselines.json`):
```
full_rules   +0.1036   (best structured baseline — rules work on structured signals)
full_random  +0.0254
full_wait    +0.0376
dp_rules     -0.0182   ← KEY: rules = wait at unstructured decision points
dp_random    -0.1270
dp_wait      -0.0182
```

The key narrative: `dp_rules ≡ dp_wait` — rules are completely blind to text signals. A trained LLM that reads them and acts correctly will show measurable lift above -0.018.

Verified: 12/12 smoke tests pass from extracted standalone repo structure.

---

### Phase B — Submission self-containment ✅

New files in `submission/`:
- `openenv.yaml` — manifest (`app: submission.server.app:app`)
- `Dockerfile` — FROM python:3.11.12-slim-bookworm, port 7860
- `requirements.txt` — openenv-core, openai>=1.0, pydantic>=2.0, fastapi>=0.100, uvicorn>=0.20
- `pyproject.toml` — packages.find includes `submission*`
- `.gitignore` — excludes grpo_runs/, wandb/, *.safetensors, etc.
- `plots/.gitkeep` — placeholder for training plots
- `docs/` — directory for slides

Dry-run extraction: `cp -r submission /tmp/new-repo/ && pytest /tmp/new-repo/submission/tests/test_smoke.py` → 12/12 pass.

---

### Phase D — Plots script + README ✅

- `submission/scripts/make_plots.py` — generates 4 PNGs from real training artifacts:
  - `reward_curve.png` (from trainer_state.json)
  - `component_decomposition.png` (from monitor.jsonl)
  - `action_distribution.png` (from monitor.jsonl + baselines.json)
  - `before_after_generations.png` (from monitor.jsonl)
  - Exits gracefully if run dir doesn't exist yet

- `submission/README.md` — four-section submission writeup with baselines table, plot placeholders, quick-start commands, full repo layout.

---

## Phases still pending

### Phase C — Real GRPO training ❌ (needs Colab/GPU)

**Step 1 — smoke test locally first (CPU, ~2 min):**
```bash
python -m submission.training.train_grpo --smoke
```

**Step 2 — Colab T4 overnight run (Gemma-1B):**
```bash
python -m submission.training.train_grpo \
  --model google/gemma-3-1b-it \
  --max_steps 200 \
  --n_generations 4 \
  --save_steps 25 \
  --out grpo_runs/overnight_v1
```

Success criteria:
- Reward climbs from ~-0.02 toward +0.10+ over 200 steps
- Format failure rate < 10% by end
- ≥4 distinct action types at step 200
- Step-200 samples show `<action>` tags and varied reasoning

**Step 3 — Generate plots after training:**
```bash
python submission/scripts/make_plots.py \
  --run grpo_runs/overnight_v1 \
  --out submission/plots
```
Then fill in the W&B link placeholder in `submission/README.md`.

**Step 4 (optional) — HF credits 4B run:**
```bash
python -m submission.training.train_grpo \
  --model google/gemma-3-4b-it \
  --max_steps 500 \
  --n_generations 8 \
  --out grpo_runs/hf_4b_v1
```

---

### Phase E — Slides + HF Space ❌

**Slides** (`submission/docs/slides.pdf`), 10 slides:
1. Title + problem
2. Why text signals matter (shift-notes example)
3. Environment architecture
4. Decision-point sub-skill RL diagram
5. Six reward components table
6. Baselines table (numbers above)
7. Reward curve plot
8. Component decomposition plot
9. Before/after generations
10. Generalization to other ops domains

**HF Space:**
- Push `submission/` as Space named `printfarm-env`
- Verify `POST /env/reset` and `POST /env/step` respond
- Add HF Space URL to `submission/README.md`

---

## Key invariants

- **Always use `.value`** for enum comparisons — `str(MyEnum.X)` is broken in Python 3.14
- **Always pass `observation=obs_dict`** to `compute_reward()` — evidence-gated rewards (`r_fault_precision`, `r_novel_fault`) silently return 0 without it
- **Round-robin targeting**: skip (don't fallback) if target signal not found after 90 steps
- **Extraction structure**: new repo must be `new-repo/submission/`, not `new-repo/` = submission contents
- **Training harness**: `notebooks/colab_grpo_training.ipynb` is the existing Colab notebook — invoke `submission.training.train_grpo` from it, don't recreate
