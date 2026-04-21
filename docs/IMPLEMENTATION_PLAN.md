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
| Use OpenEnv (latest release) | Pin latest in `pyproject.toml`; verify API surface matches | [ ] verify |
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

| Component | File(s) | Definition of done |
|---|---|---|
| Naive-greedy baseline | `baselines/naive_greedy.py` | Plays env FIFO, ignores sensors. Logs trajectories + P&L per episode. |
| Clairvoyant-greedy baseline | `baselines/clairvoyant_greedy.py` | Plays env with ground-truth state access. Logs trajectories. Target: 30 min to write, 200 episodes in ~1 hr. |
| Teacher rollout generator | `scripts/generate_teacher_rollouts.py` | Uses Claude Opus or GPT-4o via API. Produces high-quality SFT trajectories. |
| SFT dataset builder | `scripts/build_sft_dataset.py` | Merges clairvoyant-greedy + teacher trajectories into ChatML JSONL. |
| DPO preference-pair generator | `scripts/build_dpo_pairs.py` | Heuristic labeler: detects (state, chosen_action, rejected_action) tuples from the 9 failure modes + operator trust scenarios. Target: ≥ 2,000 pairs. |
| SFT training notebook | `notebooks/sft_dispatcher.ipynb` | Unsloth-based; one-cell model swap; runs to completion in ≤ 1 hr on A100 / ≤ 3 hr on T4. |
| DPO training notebook | `notebooks/dpo_dispatcher.ipynb` | TRL DPOTrainer; one-cell model swap; ≤ 2 hr on A100. |
| Evaluation harness | `scripts/eval_dispatcher.py` | Runs trained model on Tasks 1–5, computes P&L, SLA rate, false-positive tickets, hallucination catch rate. Outputs JSON + PNG curves. |
| Reward curve plotter | `scripts/plot_rewards.py` | Before/after reward curves + action-distribution shift bar charts. |

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

### Phase 1: Foundation (Days 1–2)
- [ ] Pin OpenEnv latest in `pyproject.toml` + verify API surface
- [ ] Refactor `printfarm_env/env.py` to new action space
- [ ] Write `failure_modes.yaml` (all 9 modes) + `failures.py` injector
- [ ] Write 4 printer profile YAMLs + loader
- [ ] Implement operator NPC policy (`operators.py`)
- [ ] Implement economic layer (`economics.py`) + commit break-even-ratios table
- [ ] Port + expand Round 1 tasks → Tasks 1, 2, 3 in `tasks.py`
- [ ] `pytest` green across all env modules

### Phase 2: Baselines + Data (Days 2–3)
- [ ] Write `naive_greedy.py` + run on Tasks 1–3 (100 eps each)
- [ ] Write `clairvoyant_greedy.py` + run on Tasks 1–3 (200 eps each)
- [ ] Commit baseline P&L numbers — these are your presentation reference points
- [ ] Teacher rollout generation (Claude/GPT-4o, ~500 eps)
- [ ] Build SFT JSONL dataset (clairvoyant + teacher merged)
- [ ] Build DPO preference-pair dataset (target ≥ 2,000 pairs)
- [ ] Upload both datasets to HF datasets hub (private or gated)

### Phase 3: Training (Day 3)
- [ ] SFT notebook runs end-to-end on **Qwen2.5-7B** in Colab
- [ ] DPO notebook runs end-to-end on SFT output
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
