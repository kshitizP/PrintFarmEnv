# PrintFarmEnv — Pre-Hackathon Implementation Plan

**Strategic goal:** Walk into the onsite fully shipped. The two hackathon days are for **interacting with judges, organizers, sponsors, and other teams** — not for building or training. A trained model, a packaged environment, and a rehearsed pitch are already in the bag before we arrive.

**Fallback posture:** If organizers mandate a specific HuggingFace model during the onsite, our pipeline is designed to retrain on a new base model in ≤ 3 hours with a single config change.

**Companion docs:**
- Architecture, reward design, task specs → [ROUND2_MANUAL.md](ROUND2_MANUAL.md)
- This doc = what, when, who, done criteria.

---

## 1. Organizer Guideline Compliance (Minimum Requirements)

| Requirement | Our deliverable | Status |
|---|---|---|
| Use OpenEnv (latest release) | Pin latest in `pyproject.toml`; verify API surface matches | [x] pinned |
| Minimal training script (Unsloth or HF TRL) in Colab | `notebooks/train_dispatcher.ipynb` — opens cleanly in free Colab | [ ] ship |
| Mini-blog on HuggingFace OR <2 min YouTube video | Both. Blog on HF as primary, video as redundancy | [ ] ship |
| 3-min pitch | Rehearsed deck + recorded trajectory demo | [ ] ship |
| 2-min Q&A | Prepared FAQ doc for team | [ ] ship |

### Judging-criteria-to-deliverable mapping

| Criterion (weight) | What specifically earns the points |
|---|---|
| Environment Innovation (40%) | Zero-Trust failure taxonomy + human-in-the-loop multi-agent + OctoPrint deployability bridge |
| Storytelling (30%) | Pitch deck + demo video with pre-baked Oversight narration overlay |
| Reward improvement (20%) | Reward curves PNG + before/after trajectory comparison + action distribution shift stats |
| Reward/Training pipeline (10%) | Clean SFT→DPO notebook + reproducible with one config swap |

---

## 2. Work Streams

Four parallel tracks. Each is owned by a team member; dependencies are called out explicitly.

### Stream A — Environment Core

Refactor Round 1's env into the Round 2 architecture. This is the foundation everything else depends on.

| Component | File(s) | Definition of done |
|---|---|---|
| Refactor env to Dispatcher/Operator/Oversight architecture | `printfarm_env/env.py` | Env exposes `FarmAction` with new action space (§3.1 of manual). Passes `pytest`. |
| Sensor failure taxonomy | `printfarm_env/failures.py` + `config/failure_modes.yaml` | All 9 failure modes inject correctly. Each has a YAML-configurable arrival rate + duration. |
| Printer profiles (4 models) | `config/profiles/*.yaml` | Bambu X1C, Prusa MK4, Creality K1, Voron 2.4. Loaded at env init. |
| Operator NPC policy | `printfarm_env/operators.py` | 3 skill levels; stochastic latency; shift windows; ticket queue with capacity. Target ≤ 150 LOC. |
| Economic layer (P&L reward) | `printfarm_env/economics.py` | Dollar-denominated reward per §9 of manual. Break-even ratio table checked in. |
| Tasks 1–5 | `printfarm_env/tasks.py` | Tasks 1, 2, 3 are blocking; 4, 5 are stretch. Each has a grader. |
| OctoPrint adapter (mocked) | `printfarm_env/adapters/octoprint.py` | Same method signatures as env; logs HTTP calls to stdout; no real network. |
| OpenEnv compliance | `server/app.py` + `openenv.yaml` | `/reset`, `/step`, `/state`, `/schema`, `/health`, `/metadata` all work with latest OpenEnv spec. |
| Dockerfile + HF Space | `Dockerfile` | `docker build && docker run -p 7860:7860` → env live. Deploys to HF Space cleanly. |

### Stream B — Training Pipeline

Everything needed to produce a trained model. Must be **model-agnostic** — see Stream D for why.

**Training path (revised 2026-04-22):** SFT (format + warm-start) → **GRPO (primary RL, uses env reward directly)** → DPO (secondary, preference-based polish). GRPO is promoted to primary because organizer Help Guide §10–11 frames RL with verifiable rewards as the intended stack, and our env is already a verifier.

| Component | File(s) | Definition of done |
|---|---|---|
| Naive-greedy baseline | `baselines/naive_greedy.py` | Plays env FIFO, ignores sensors. Logs trajectories + P&L per episode. |
| Clairvoyant-greedy baseline | `baselines/clairvoyant_greedy.py` | Plays env with ground-truth state access. Logs trajectories. Target: 30 min to write, 200 episodes in ~1 hr. |
| Teacher rollout generator | `scripts/generate_teacher_rollouts.py` | Uses Claude Opus or GPT-4o via API. Produces high-quality SFT trajectories. |
| SFT dataset builder | `scripts/build_sft_dataset.py` | Merges clairvoyant-greedy + teacher trajectories into ChatML JSONL. |
| DPO preference-pair generator | `scripts/build_dpo_pairs.py` | Heuristic labeler: detects (state, chosen_action, rejected_action) tuples from the 9 failure modes + operator trust scenarios. Target: ≥ 2,000 pairs. |
| **GRPO rollout wrapper** | `scripts/grpo_rollout.py` | Given a model + task_id, samples N trajectories through the OpenEnv client; returns `(prompts, completions, rewards)` tuples in TRL GRPOTrainer format. ≤ 200 LOC. |
| SFT training notebook | `notebooks/sft_dispatcher.ipynb` | Unsloth-based; one-cell model swap; runs to completion in ≤ 1 hr on A100 / ≤ 3 hr on T4. |
| **GRPO training notebook (PRIMARY)** | `notebooks/grpo_dispatcher.ipynb` | TRL `GRPOTrainer` + Unsloth; consumes SFT checkpoint; rollouts go through env; reward = P&L from `economics.py`. Task 1 only for first run; ≤ 50 steps to prove loop; one-cell model swap. |
| DPO training notebook (SECONDARY) | `notebooks/dpo_dispatcher.ipynb` | TRL DPOTrainer; one-cell model swap; ≤ 2 hr on A100. Runs after GRPO as polish pass. |
| Evaluation harness | `scripts/eval_dispatcher.py` | Runs trained model on Tasks 1–5, computes P&L, SLA rate, false-positive tickets, hallucination catch rate. Outputs JSON + PNG curves. |
| Reward curve plotter | `scripts/plot_rewards.py` | Before/after reward curves + action-distribution shift bar charts. GRPO run produces the primary curve. |

### Stream C — Deliverables (storytelling & submission)

These are what judges actually see. Quality here wins the 30% storytelling slice.

| Deliverable | Location | Definition of done |
|---|---|---|
| Pitch deck (3-min spec) | `presentation/pitch.key` | 8–10 slides. Rehearsed to 2:50. |
| Demo trajectory video | `presentation/demo.mp4` | 60 sec. Side-by-side baseline vs trained. Pre-baked Oversight narration overlay. |
| Reward curve PNGs | `presentation/curves/*.png` | 3 charts: reward over training steps, P&L per task, action-distribution before/after. |
| Oversight narration script | `presentation/oversight_narration.md` | Pre-computed LLM outputs for the demo trajectory. No live API in pitch. |
| HF blog post | Published on HF | < 2 min read. Links env + model + demo video. |
| YouTube video (redundancy) | Unlisted link | < 2 min. Same content as blog, recorded voiceover. |
| README + OpenEnv metadata | `README.md`, `openenv.yaml` | Reflects Round 2 architecture, not Round 1. Usage instructions verified. |
| Q&A prep doc | `presentation/qa_prep.md` | 10 anticipated questions + crisp answers for team. |

### Stream E — Reward-Hacking Audit (new)

Organizer Help Guide §8 + FAQ #13, #43, #57 treat this as a first-class evaluation dimension. Judges steeped in these docs expect to see evidence that the team tried to break their own reward. Cheap to build, high-signal.

| Component | File(s) | Definition of done |
|---|---|---|
| Adversarial policies | `tests/test_reward_hacking.py` | 5–10 policies that try to game reward (spam-OK, ignore sensors, fake completions, abuse operator trust, edit-timer analogues). Each asserts reward stays below honest-baseline floor. |
| Audit writeup | `docs/REWARD_HACKING_AUDIT.md` | One page: table of hack attempt → reward component that caught it → invariant test that enforces it. |
| Pitch-deck slide | `presentation/pitch.key` slide "How we stress-tested the reward" | 3 bullets + one chart (adversarial reward vs. honest baseline). |
| Blog paragraph | HF blog | Cite FAQ #57 framing ("don't optimize a reward you haven't tried to break yourself"). |

### Stream D — Contingency Artifacts (model-swap kit)

Organizers may mandate a specific base model onsite. Build so that's a 10-minute change, not a day of refactoring.

| Artifact | Purpose |
|---|---|
| `config/models.yaml` | Single source of truth: `base_model`, `tokenizer`, `chat_template`, `max_seq_len`, `lora_r`. All notebook cells read from this. |
| Pre-tested model profiles | Verify the pipeline runs end-to-end on at least **3 base models** before the onsite (see §4). |
| Pre-generated SFT + DPO datasets | Saved to HF datasets hub. **Model-independent.** Swap model → retrain, no data regen. |
| Offline-cached Unsloth installs | `requirements-frozen.txt` with pinned versions. If HF package hub is slow onsite, we install from wheel cache. |
| Fallback local training box | Document what runs on Mac M-series / any CUDA box you can commandeer if Colab fails onsite. |

---

## 3. Master Pre-Hackathon Checklist

Order by dependency. Check off as you go.

### Phase 1: Foundation (Days 1–2) — ✅ COMPLETE (2026-04-22)
- [x] Pin OpenEnv latest in `pyproject.toml` + verify API surface
- [x] Refactor `printfarm_env/env.py` to new action space *(9 states × 8 actions; ~1070 LOC)*
- [x] Write `failures.py` injector with all 9 modes *(deviation: modes coded inline, no YAML — ship-blocker only if organizers ask for config tunability)*
- [x] 4 printer profile loader in `env.py` *(deviation: profiles defined as Python constants, not YAML)*
- [x] Implement operator NPC policy (`operators.py`)
- [x] Implement economic layer (`economics.py`) + commit break-even-ratios table *(`docs/BREAK_EVEN_RATIOS.md`)*
- [x] Port + expand Round 1 tasks → Tasks 1, 2, 3, plus warm-up 0_1/0_2/0_3 and stretch 4, 5 in `tasks.py` *(calibrated floor/ceiling from 20-ep naive + clairvoyant runs, seed=42)*
- [x] `pytest` green across all env modules *(56/56 invariant tests, 0.12s — covers §8 physical, semantic, economic, observation, grader)*

### Phase 2: Baselines + Data (Days 2–3) — 🟡 IN PROGRESS
- [x] Write `naive_greedy.py` *(FIFO dispatcher ignoring sensors)*
- [x] Write `clairvoyant_greedy.py` *(ground-truth state access + ERROR recovery)*
- [x] Calibration run on all 8 tasks, 20 eps each *(committed in `docs/BREAK_EVEN_RATIOS.md`)*
- [ ] **Formal baseline run: Tasks 1–3, 100 eps naive + 200 eps clairvoyant, save trajectories** ← next
- [ ] Teacher rollout generation (Claude/GPT-4o, ~500 eps) ← **needs user decisions (see §9)**
- [ ] Build SFT JSONL dataset (clairvoyant + teacher merged)
- [ ] Build DPO preference-pair dataset (target ≥ 2,000 pairs)
- [ ] Upload both datasets to HF datasets hub (private or gated)

### Phase 2.5: Reward-Hacking Audit (parallelizable with Phase 2)
- [ ] Audit `printfarm_env/env.py` `step()` — confirm reward is per-step; if not, add lightweight per-step shaping (sensor-ignore penalty, correct-routing bonus)
- [ ] Write `tests/test_reward_hacking.py` with 5–10 adversarial policies
- [ ] Commit `docs/REWARD_HACKING_AUDIT.md`
- [ ] Document per-step reward as "lightweight process supervision" in README (FAQ #11 framing)

### Phase 3: Training (Day 3) — revised path: SFT → GRPO → DPO
- [ ] SFT notebook runs end-to-end on **Qwen2.5-7B** in Colab
- [ ] `scripts/grpo_rollout.py` wired to OpenEnv client, returns TRL-format tuples
- [ ] GRPO notebook runs end-to-end on SFT output — Task 1 only, 50 steps, small model (Qwen2.5-3B) to prove loop
- [ ] Scale GRPO to Tasks 1–3, Qwen2.5-7B, larger step count once loop is stable
- [ ] DPO notebook runs end-to-end on GRPO output as polish pass
- [ ] Evaluate trained model on Tasks 1–3 → commit result JSON
- [ ] Verify trained model shows **visible improvement** over naive-greedy baseline
- [ ] If reward curves are flat: pull the fallback action-distribution-shift chart

### Phase 4: Model-Swap Verification (Day 4)
- [ ] Swap `config/models.yaml` to **Llama-3.1-8B** → rerun SFT+DPO → confirm it trains
- [ ] Swap to **Mistral-7B-v0.3** → same
- [ ] Commit the 3 trained checkpoints to HF model hub
- [ ] Document which model performed best on our tasks (will be our primary submission)

### Phase 5: Deliverables (Days 4–5)
- [ ] Build OctoPrint mock adapter + record terminal-output demo
- [ ] Record 60-sec demo trajectory video with Oversight narration overlay (pre-baked)
- [ ] Generate reward curve PNGs + action distribution chart
- [ ] Write HF blog post
- [ ] Record YouTube video
- [ ] Build pitch deck (8–10 slides)
- [ ] Rehearse pitch to 2:50 — **twice minimum**
- [ ] Write Q&A prep doc
- [ ] Update `README.md` + `openenv.yaml` for Round 2

### Phase 6: Final Hardening (Day 5 / night before)
- [ ] Clone the repo fresh on a different machine → verify reproducibility
- [ ] Copy everything to the demo laptop (don't rely on cloning onsite)
- [ ] Download all HF weights locally (don't rely on venue Wi-Fi)
- [ ] Pre-bake every demo video, chart, and narration file offline
- [ ] Test pitch laptop on external projector + HDMI + sound
- [ ] Team dry run of full 5-min pitch + Q&A

---

## 4. Model-Swap Contingency Plan

### Scenario A — Organizers mandate a specific model

**Action on day of hackathon:**
1. Update `config/models.yaml` → set `base_model` to the mandated HF ID
2. Run `notebooks/sft_dispatcher.ipynb` on our pre-built SFT dataset (~1 hr)
3. Run `notebooks/dpo_dispatcher.ipynb` on our pre-built DPO dataset (~2 hr)
4. Run `scripts/eval_dispatcher.py` → generate new reward curves (~30 min)
5. Swap the curves + model link in deck
6. Total time: ~4 hours. Pitch in afternoon slot.

**Prerequisite:** We've verified the pipeline works on 3 different base models before arriving. Any mandate is likely a model our pipeline already handles.

### Scenario B — Organizers are flexible

Submit our **primary trained checkpoint** (whichever of the 3 scored highest). No onsite training. Use the time for pitch polish and sponsor conversations.

### Scenario C — Mandated model doesn't fit the pipeline (edge case)

If they mandate something weird (e.g., a non-instruct base, or a model Unsloth doesn't support):
1. Fall back to HF TRL's native SFTTrainer (slower but covers more models)
2. Cut DPO, ship SFT-only. Be upfront in the pitch: "they mandated X; we SFT'd on our clairvoyant trajectories; here's the before/after."
3. Total time: ~5 hours if Unsloth fails.

### Pre-approved model shortlist

We'll verify these work before the onsite:

| Model | HF ID | Unsloth support | Rationale |
|---|---|---|---|
| **Primary: Qwen2.5-7B-Instruct** | `Qwen/Qwen2.5-7B-Instruct` | ✅ | Strong JSON output reliability, fits free Colab |
| Secondary: Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | ✅ | Most commonly mandated in hackathons |
| Tertiary: Mistral-7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | ✅ | Broad compatibility |
| Contingency: Gemma-2-9B-IT | `google/gemma-2-9b-it` | ✅ | If Google/Kaggle sponsor mandates |
| Small fallback: SmolLM2-1.7B | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | ✅ | If compute is constrained; fits CPU-only inference |

---

## 5. Day-of-Hackathon Checklist (Minimal)

Goal: spend the day talking to people, not typing.

### Morning

- [ ] Arrive early; set up laptop at team table
- [ ] Verify demo laptop runs pitch deck + video offline
- [ ] Smoke-test OctoPrint adapter terminal demo
- [ ] Check if organizers mandated a model (slack / email / whiteboard)
  - If YES → run Scenario A (§4)
  - If NO → we're done. Proceed to afternoon.

### Afternoon

- [ ] Network. Talk to sponsors (Fleet AI, Halluminate, Scale AI, Patronus, Scaler AI, Mercor, Snorkel).
- [ ] Specifically pitch **Fleet AI** for the Scalable Oversight bonus prize
- [ ] Refine Q&A answers based on what other teams are building
- [ ] Final pitch rehearsal 1 hour before slot

### Pitch slot

- [ ] 3 min pitch + 2 min Q&A
- [ ] Demo video runs **locally from laptop** — no Wi-Fi dependency
- [ ] Have the QR code for HF repo / blog / video ready on a slide

---

## 6. Pre-Hackathon Risk Register

| Risk | Mitigation |
|---|---|
| Colab free tier rate-limits mid-training | Use Colab Pro ($10 one-time) for the training week. Plan B: local Mac/CUDA box. |
| HF gated model access delays (Llama, Gemma) | Request access 48 hr in advance. Have Qwen2.5 as unblocked primary. |
| Teacher API costs spiral | Budget $50 for Claude/GPT-4o teacher rollouts. Cap episode count. |
| Dataset generation reveals env bugs late | Run clairvoyant-greedy FIRST; it exercises the env harder than training does. |
| OpenEnv spec changes between now and onsite | Pin to a specific release SHA. Re-test 48 hr before onsite. |
| Trained model doesn't improve over baseline | Have the action-distribution-shift fallback chart ready. Also: be honest in the pitch — a model that learns correct *behavior* without beating baseline in *dollars* is still a story, if told right. |
| Pitch runs long | Rehearse with a stopwatch. Cut ruthlessly. 2:50 is the target, not 3:00. |
| Demo video file corrupts / laptop dies | Two USB sticks. Second laptop as hot spare. |
| Organizer logistics surprise (different slot, format change) | Assign one team member to be the "liaison" so the builders aren't interrupted. |

---

## 7. Ownership Template

Fill in with team members' names before starting.

| Stream | Owner | Backup |
|---|---|---|
| A. Environment Core | | |
| B. Training Pipeline | | |
| C. Deliverables (deck/video/blog) | | |
| D. Contingency Artifacts | | |
| Liaison / organizer comms (day-of) | | |

---

## 8. Definition of "Ready"

We are ready to walk into the onsite when **all of the following are true**:

1. ✅ Trained model (primary: Qwen2.5-7B) is uploaded to HF and publicly loadable.
2. ✅ Pipeline has been verified end-to-end on ≥ 2 other base models.
3. ✅ Reward improvement vs naive-greedy is **demonstrably visible** on at least one task.
4. ✅ OctoPrint mock adapter demo runs from local terminal with no internet.
5. ✅ 60-sec demo video with pre-baked Oversight narration is recorded.
6. ✅ Pitch deck rehearsed to 2:50, twice.
7. ✅ HF blog post is published (even if minor edits happen later).
8. ✅ Everything is copied to the demo laptop AND a USB stick.
9. ✅ Q&A prep doc circulated to the team.
10. ✅ Team has done a full dry run the night before.

If any of these is ❌ the night before, cut scope, don't push through. Better to walk in with a smaller but tight submission than a broken kitchen-sink one.

---

## 9. Phase 2 Data-Generation Playbook

Purpose: everything the team needs to produce the 3 training artifacts (baseline trajectories, SFT JSONL, DPO pairs). Written after Phase 1 wrap-up on 2026-04-22.

### 9.1 Decisions needed from user (blocking)

Before any script runs we need sign-off on these:

| # | Decision | Options | Recommendation |
|---|---|---|---|
| 1 | Teacher LLM | Claude Opus 4.x, Claude Sonnet 4.6, GPT-4o, GPT-4.1 | Claude Sonnet 4.6 — 5× cheaper than Opus, tool-use reliable, JSON discipline strong |
| 2 | Teacher episode count | 200 / 500 / 1000 | 500 (budget ~$30, covers all 8 tasks with spread) |
| 3 | Baseline episode counts | as in plan: 100 naive, 200 clairvoyant | keep — enough statistical power, ~1 hr runtime total |
| 4 | Tasks to cover | 1,2,3 only / all 8 | all 8 — warm-ups (0_1/0_2/0_3) are cheap and make the curves tell a story |
| 5 | HF dataset visibility | public / private / gated | **private** until hackathon morning, then public |
| 6 | HF username/org | — | need your username |

### 9.2 Credentials & accounts (blocking)

Please provide or confirm access to:

1. **`ANTHROPIC_API_KEY`** (if Claude teacher) or **`OPENAI_API_KEY`** (if GPT teacher). Export in shell; do not commit.
2. **HuggingFace account** + a **write-scoped access token** (`HF_TOKEN`). Create at https://huggingface.co/settings/tokens.
3. **`huggingface-cli login`** run once on your machine.
4. **Compute budget confirmation**: ~$50 for teacher rollouts. Below that we should cut episode count.

### 9.3 Scripts to be written (not yet in repo)

All land under `scripts/`. Order matters — each consumes the previous.

| # | Script | Inputs | Outputs | Est. LOC | Runtime |
|---|---|---|---|---|---|
| 1 | `scripts/run_baselines.py` | env, baseline policies | `data/baselines/{naive,clairvoyant}/{task_id}/episodes.jsonl` — one row per step with (obs, action, reward, info) | ~150 | ~1 hr (8 tasks × 300 eps × ~150 steps, single thread) |
| 2 | `scripts/generate_teacher_rollouts.py` | env, teacher API client, prompt template | `data/teacher/{task_id}/episodes.jsonl` + cost ledger JSON | ~250 | ~4 hrs wall-clock with 10-way async; $30–$50 |
| 3 | `scripts/build_sft_dataset.py` | baselines + teacher JSONL | `data/sft/train.jsonl` + `data/sft/eval.jsonl` in ChatML format, filtered to profitable episodes only | ~120 | ~5 min |
| 4 | `scripts/build_dpo_pairs.py` | trajectories + heuristic labeler over 9 failure modes + operator-trust scenarios | `data/dpo/preferences.jsonl` with ≥2000 (prompt, chosen, rejected) triples | ~200 | ~10 min |
| 5 | `scripts/upload_to_hf.py` | built datasets, HF_TOKEN | pushed repos: `{user}/printfarm-sft`, `{user}/printfarm-dpo`, `{user}/printfarm-baselines` | ~60 | ~5 min |

### 9.4 Order of operations

```
┌─────────────────────┐
│ 1. run_baselines.py │  ← no external deps, run first; flushes any lurking env bugs
└──────────┬──────────┘
           │  (produces ground-truth trajectories)
           ▼
┌──────────────────────────────┐
│ 2. generate_teacher_rollouts │  ← API-bound; run overnight
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ 3. build_sft_dataset.py       │  ← merges #1 clairvoyant + #2 teacher, filters to profit>floor
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ 4. build_dpo_pairs.py         │  ← can run in parallel with #3; uses trajectories from #1+#2
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ 5. upload_to_hf.py            │  ← smoke-test that a trainer can `load_dataset(...)` it
└──────────────────────────────┘
```

### 9.5 Cost and time estimate

| Item | Estimate |
|---|---|
| Baseline generation (CPU only) | ~1 hr, $0 |
| Teacher rollouts (Claude Sonnet 4.6, 500 eps × ~150 steps × ~800 input tok + 60 output tok) | ~3 M input + 0.25 M output tokens × blended rate ≈ **$25–$35** |
| SFT + DPO building | ~15 min, $0 |
| HF upload | ~5 min, $0 |
| **Total** | **~5–6 hrs wall-clock, $25–$50** |

Cost cap built into `generate_teacher_rollouts.py`: script aborts if running-total > `--max-cost-usd` (default $50).

### 9.6 Dataset shapes

**SFT ChatML rows:**
```jsonl
{"messages": [
  {"role": "system", "content": "You are the Dispatcher for a 4-printer farm..."},
  {"role": "user", "content": "<serialised FarmObservation>"},
  {"role": "assistant", "content": "<FarmAction as JSON>"}
]}
```

**DPO preference rows:**
```jsonl
{"prompt": "<system+user as ChatML>", "chosen": "<good action JSON>", "rejected": "<bad action JSON>", "label_source": "heuristic:fault_ignored|operator_trust|..."}
```

### 9.7 What I need from you to kick off

Reply with:
1. Teacher LLM choice (default: Claude Sonnet 4.6)
2. Your HuggingFace username/org
3. Confirm the API key is exported in your shell (`echo $ANTHROPIC_API_KEY | wc -c` should be > 40)
4. Go/no-go on $50 budget cap

Once those land I'll write the 5 scripts in one batch, run #1 immediately, and schedule #2 overnight.

---

## 10. Gap-Closure Task List (added 2026-04-22)

Derived from a read of the two organizer docs shared 2026-04-22 ([External] Meta OpenEnv Hackathon Participant Help Guide, Hackathon FAQs). Tasks are ordered by leverage × urgency. Each task is small enough to assign to a single person-day or less.

### 10.1 Gap 1 — GRPO / RLVR training path

> **Why:** Help Guide §10–11 + FAQs #4, #8, #25, #45 frame RL-with-verifiable-rewards (GRPO) as the intended stack. Our original DPO-only path didn't exercise the env's reward during training. This is the largest single gap to the organizer framing.

- [ ] **G1.1** Write `scripts/grpo_rollout.py` — wrap OpenEnv client; given model + task_id return `(prompts, completions, rewards)` for GRPOTrainer
- [ ] **G1.2** Create `notebooks/grpo_dispatcher.ipynb` skeleton (Unsloth + TRL GRPOTrainer, Qwen2.5-3B, Task 1, 50 steps)
- [ ] **G1.3** Smoke-test GRPO loop on Task 1 locally; confirm reward column updates across steps
- [ ] **G1.4** Scale to Tasks 1–3, Qwen2.5-7B, full step budget once smoke-test passes
- [ ] **G1.5** Re-label DPO notebook as "secondary / polish pass"; update Stream B and pitch narrative
- [ ] **G1.6** Pitch line ready: "SFT teaches format, GRPO optimizes against our env's verifier" (Help Guide §11 language)

### 10.2 Gap 2 — Reward-hacking defense artifact

> **Why:** Help Guide §8 + FAQ #13, #43, #57 treat adversarial-reward testing as a first-class judging dimension. Rule #57: *"Do not optimize a reward you have not tried to break yourself first."*

- [ ] **G2.1** Write `tests/test_reward_hacking.py` with ≥5 adversarial policies (spam-OK, sensor-ignore, fake-completion, operator-trust abuse, idle-game)
- [ ] **G2.2** Each adversarial policy asserts `total_reward < honest_baseline_floor`
- [ ] **G2.3** Draft `docs/REWARD_HACKING_AUDIT.md` — table: attempt → component that caught it → invariant test
- [ ] **G2.4** Add "How we stress-tested the reward" slide to pitch deck
- [ ] **G2.5** Add one paragraph citing FAQ #57 to HF blog draft

### 10.3 Gap 3 — Process-aware feedback framing

> **Why:** Help Guide §9 + FAQ #11 distinguish final-only rewards (wasteful) from process-aware supervision. If we already emit per-step rewards, say so; if not, add cheap shaping.

- [ ] **G3.1** Audit `printfarm_env/env.py:step()` — is reward per-action or per-episode?
- [ ] **G3.2** If per-step: add one-liner to README + blog calling it "dense per-decision signal"
- [ ] **G3.3** If per-episode: add lightweight per-step shaping (sensor-ignore −$X, correct-routing +$Y) — keep sparse final P&L dominant
- [ ] **G3.4** Commit updated reward docs to `docs/ROUND2_MANUAL.md` reward section

### 10.4 Gap 4 — Minimum-requirements shipping (parallelizable; start now)

> **Why:** Help Guide top-of-doc lists 5 minimum deliverables; 4 are still ❌ in [§1](#1-organizer-guideline-compliance-minimum-requirements). None depend on training finishing — draft skeletons now.

- [ ] **G4.1** Colab notebook stub: loads env, runs naive-greedy, prints reward (fill SFT/GRPO cells later)
- [ ] **G4.2** HF blog outline — 6 sections (problem / env / reward / training / results / demo), 300-word draft
- [ ] **G4.3** Pitch deck skeleton — 8 slide titles, no content yet
- [ ] **G4.4** Q&A prep doc seeded with 10 anticipated questions (see Stream C)
- [ ] **G4.5** 2-min video script (voiceover) — 4 shots: env → baseline fails → trained succeeds → Oversight narration

### 10.5 Gap 5 — Phase 2 data-generation unblocking

> **Why:** Phase 2 has been blocked on 6 decisions + 4 credential items since 2026-04-22. Every day of slip pushes training toward the wire.

- [ ] **G5.1** Batch-decide: Claude Sonnet 4.6 / 500 eps / all 8 tasks / private HF repo / $50 cap
- [ ] **G5.2** Confirm `ANTHROPIC_API_KEY` and `HF_TOKEN` exported (`echo $VAR | wc -c` > 40)
- [ ] **G5.3** Provide HuggingFace username/org
- [ ] **G5.4** Kick off `scripts/run_baselines.py` (CPU, ~1 hr, no external deps)
- [ ] **G5.5** Kick off `scripts/generate_teacher_rollouts.py` overnight (API-bound)
- [ ] **G5.6** Run `build_sft_dataset.py` + `build_dpo_pairs.py` in parallel next day
- [ ] **G5.7** Run `upload_to_hf.py` — smoke-test `load_dataset(...)` roundtrip

### 10.6 Cross-cutting

- [ ] **X.1** Fill [§7 Ownership table](#7-ownership-template) — one name per stream including new Stream E
- [ ] **X.2** Re-read Help Guide §19 ("what judges find compelling") one week before onsite — use as a *cut* checklist, not an *add* checklist

### 10.7 Suggested 48-hour execution order

```
Tonight          → G5.1–G5.5 (unblock data gen), G4.2–G4.5 (draft deliverable skeletons)
Day +1 AM        → G1.1 + G1.2 (GRPO rollout + notebook), G2.1–G2.3 (reward-hacking audit)
Day +1 PM        → G5.6–G5.7 (datasets built + uploaded), G3.1–G3.3 (process-feedback audit)
Day +2 AM        → SFT run → G1.3 (GRPO smoke test on Task 1)
Day +2 PM        → G1.4 (GRPO scaled), eval harness, first real reward curves
Day +3           → Model-swap verification, blog draft, video script recording
Day +4           → Polish, rehearse, dry run
```

### 10.8 Leverage ranking (if we must cut scope)

1. **G1 — GRPO path** — largest framing mismatch with organizer docs; closes biggest gap
2. **G4 — Minimum-reqs shipping** — required to qualify; zero-risk to start now
3. **G2 — Reward-hacking audit** — cheap, high-signal to judges familiar with FAQs
4. **G5 — Phase 2 unblocking** — gates everything downstream
5. **G3 — Process-feedback framing** — mostly a wording change unless §G3.3 fires
