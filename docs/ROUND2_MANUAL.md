# PrintFarmEnv — Round 2 Engineering & Pitch Manual

**Version:** 2.0 (Deployability-First Rewrite)
**Audience:** Hackathon team + anyone extending this for production use
**North Star:** Build something a real print farm would run against their OctoPrint fleet on Monday morning — not a toy that only judges appreciate.

---

## 0. Strategic Positioning

### The thesis (in one sentence)

**PrintFarmEnv is a trust-and-coordination layer for 3D print farm software — an AI dispatcher that orchestrates human operators and refuses to trust hallucinating sensors, mapped 1:1 to OctoPrint/Moonraker APIs.**

### Why this framing wins

1. **Product pitch, not hackathon pitch.** Judges remember "this plugs into real farms" longer than "we hit four themes."
2. **Defensible moat.** Sensor-skepticism + human-operator coordination is real, painful, and under-solved. OctoPrint has 500K+ installs; none have this.
3. **Post-hackathon path.** This is something you can keep building. A hackathon trick dies after the pitch.

### What changed from v1.0 of this manual

| v1.0 Claim | v2.0 Reality Check |
|---|---|
| "We hit themes 1, 2, 3.1, and 5" | Lead with **1 + 5**; themes 2 and 3.1 are load-bearing architecture, not pitch pillars |
| Dispatcher + Technician = two AI agents | Dispatcher = AI; Technician = **simulated human operator** with latency, fatigue, skill variance |
| 5% random JSON corruption = hallucination | **Structured failure-mode library** modeling real thermistor / filament sensor / webcam / MCU failures |
| Normalized reward [0, 1] | Dollar-denominated P&L signal + normalized reward for comparability |
| Homogeneous printer fleet | **Heterogeneous fleet** with per-model profiles (Bambu X1C, Prusa MK4, Creality K1, Voron) |
| DPO is the training method | DPO **or** GRPO — decide after a 2-hour day-1 spike |
| "300% reward improvement" slide | Show the curve. Never pre-commit to a number. |

---

## 1. Theme Alignment (Honest Version)

| Theme | Role in Project | Pitch Weight |
|---|---|---|
| **#1 Multi-Agent Interactions** | Primary pillar — AI↔human coordination with asymmetric information | **Heavy** |
| **#5 Wild Card (Zero-Trust)** | Primary pillar — structured sensor hallucination taxonomy | **Heavy** |
| #2 Long-Horizon Planning | Load-bearing background (inherited from Round 1) | Light mention |
| #3.1 World Modeling | Load-bearing background (partial observability, wear models) | Light mention |

### Bonus prize targeting

**Primary target: Fleet AI "Scalable Oversight" sub-theme** — we add a third **Oversight Agent** that monitors the Dispatcher's decisions and produces natural-language explanations when the Dispatcher overrides human operators or ignores sensor warnings. This is the highest-ROI bonus because we're already 80% there architecturally.

**Secondary (if time): Halluminate "Multi-Actor Environments"** — the human-operator simulation naturally qualifies if we add >2 simulated operators with distinct skill profiles.

---

## 2. Master Problem Statement

Current frontier LLMs excel at static instruction following but fail at three things simultaneously that matter in real manufacturing:

1. **Coordinating with humans** who have partial information, variable response latency, and their own priorities.
2. **Distrusting their own inputs** when sensors degrade in structured, non-obvious ways.
3. **Maintaining economic rationality** over long horizons when the reward signal is "dollars per hour," not task completion.

PrintFarmEnv forces an AI agent to survive all three at once. The agent is the "brain" of a print farm: it dispatches jobs to heterogeneous printers, writes work orders for simulated human technicians, and must detect and route around degrading telemetry to hit weekly SLA targets under a real cost model.

**The deployability bridge:** Every action in the environment maps to a real OctoPrint / Moonraker API call. A trained agent can be pointed at a real farm's REST API and operate it — the simulator is the training ground, not the endgame.

---

## 3. Architecture: Human-in-the-Loop Multi-Agent

### The three actors

#### 3.1 The Dispatcher Agent (AI — this is what you train)

- **Role:** Orchestrates the entire farm. This is the policy we fine-tune.
- **Visibility:** Order queue, deadlines, printer telemetry (subject to hallucination), inventory, operator availability, operator inbox.
- **Blind spots:** Ground-truth physical wear state, operator fatigue levels, true sensor readings.
- **Action space:**
  - `ASSIGN_JOB(printer_id, job_id)` → maps to `POST /api/job`
  - `CANCEL_JOB(job_id)` → maps to `POST /api/job/cancel`
  - `RUN_DIAGNOSTIC(printer_id)` → maps to Moonraker `query_endstops` / `temperature_store`
  - `DISPATCH_TICKET(operator_id, ticket)` → creates work order for human
  - `REQUEST_SPOOL_SWAP(printer_id, material)` → sub-case of DISPATCH_TICKET
  - `REQUEST_MAINTENANCE(printer_id, maintenance_type)` → sub-case of DISPATCH_TICKET
  - `OVERRIDE_OPERATOR(operator_id, ticket_id, reason)` → can reassign / cancel a pending ticket
  - `WAIT`

#### 3.2 Simulated Human Operators (environment-controlled NPCs)

**These are NOT agents the team trains — they are realistic stochastic NPCs the environment simulates. They create the multi-agent surface.**

Each operator has:
- **Skill level** ∈ {junior, senior, lead} — affects action success rate and latency
- **Response latency** — 5–15 minute realistic distribution before acting on a ticket (converted to env steps)
- **Shift window** — only available during their shift; hand off state at shift boundaries
- **Fatigue** — accumulates over the shift; increases latency and error rate
- **Current queue** — finite capacity; the Dispatcher must not overwhelm
- **Skill-gated actions** — juniors can swap spools; only seniors/leads can perform maintenance requiring disassembly

Operator NPC actions (fired stochastically when tickets reach top of their queue):
- `COMPLETE_TICKET(ticket_id)` — succeeds with probability based on skill + fatigue
- `ESCALATE_TICKET(ticket_id)` — juniors escalate complex tickets to seniors
- `REJECT_TICKET(ticket_id, reason)` — e.g., "spool not in inventory", "printer is in use"
- `REPORT_ANOMALY(printer_id, observation)` — human-in-the-loop ground truth; operator tells the Dispatcher what they actually see on the machine

**The critical design point:** Operators have information the Dispatcher doesn't (they see the real printer) but act slowly and imperfectly. The Dispatcher must learn to trust or override them selectively.

#### 3.3 The Oversight Agent (optional but high-ROI)

**Purpose:** Fleet AI bonus track. Also a natural pitch story.

- **Role:** Watches the Dispatcher's action stream and operator outcomes. Produces natural-language explanations of questionable decisions.
- **Not trained in round 2** — uses a stock LLM (Claude / GPT-4o / Llama-70B) with a structured prompt. This is the *scalable oversight* story: one cheap watcher per N expensive dispatcher actions.
- **Output:** Timestamped audit log + alerts when:
  - Dispatcher overrides an operator's `REPORT_ANOMALY`
  - Dispatcher ignores a `REPORT_ANOMALY` warning
  - Dispatcher issues a job against a hallucinated sensor reading
  - Dispatcher's cumulative override rate exceeds threshold
- **Evaluation:** Audit log precision/recall against ground-truth bad decisions (which we can log from env internals).

### Information flow diagram

```
┌─────────────────────────────────────────────────────────┐
│                  GROUND TRUTH (env internal)             │
│  real printer state │ real sensor values │ wear models   │
└──────────────┬──────────────────────┬───────────────────┘
               │ corrupted            │ clean
               ▼                      ▼
        ┌────────────┐         ┌──────────────┐
        │ Dispatcher │ ◄─────► │  Operators   │
        │  (AI, you) │ tickets │ (NPCs, env)  │
        └─────┬──────┘ reports └──────────────┘
              │
              ▼
        ┌────────────┐
        │ Oversight  │   (optional — Fleet AI bonus)
        │ (LLM eval) │
        └────────────┘
```

---

## 4. Zero-Trust: Realistic Sensor Failure Taxonomy

**Throw out the "5% random JSON corruption" model.** Real print farms fail in structured, pattern-based ways. The whole point of Zero-Trust is that the agent must learn *physically plausible* distrust, not random paranoia.

### Failure modes (configurable YAML)

| Mode ID | Real-world Cause | Symptom to Dispatcher | Detection Heuristic |
|---|---|---|---|
| `thermistor_open` | Thermistor wire breaks | `hotend_temp: 0` persistent | Temp drop >100°C/step while heater ON |
| `thermistor_short` | Thermistor shorts | `hotend_temp: 300+` persistent | Temp reads above physical max |
| `filament_sensor_false_runout` | IR sensor drift / dust | `state: PAUSED_RUNOUT` on a full spool | Spool weight still high per last weighing |
| `filament_sensor_missed_runout` | Sensor failure | Printer reports `PRINTING` past end of spool | Cumulative extrusion > last known weight |
| `webcam_freeze` | USB bandwidth / driver | Same image hash for N steps | Hash-based duplicate detection |
| `klipper_mcu_disconnect` | USB/serial drop | All fields frozen to last-known-good | Telemetry timestamp stale |
| `progress_drift` | Slicer time estimate wrong | Progress stuck at X% for N steps while temps nominal | Progress rate deviates from print_time_steps |
| `fan_rpm_ghost` | Tachometer failure | Fan reports 0 RPM but part cooling is fine | Disagrees with print success rate |
| `bed_level_drift` | Probe wear | First-layer fails despite recent ABL | Pattern: repeated early-step failures on one printer |

### Failure injection schedule

- Failures occur **per-printer, per-mode**, with independent Poisson arrival rates.
- Default rates are calibrated against real OctoPrint issue-tracker frequencies (rough: thermistor ~0.1% per print-hour, webcam freezes ~0.5% per print-hour).
- Each failure has a **duration distribution** — some clear themselves, some require `RUN_DIAGNOSTIC`, some require a human `ESCALATE_TICKET`.

### The Dispatcher's skepticism toolkit

1. **Cross-reference operator reports.** If operator says "Printer 3 is fine" but telemetry says `thermistor_open`, believe the human (most of the time — junior operators are wrong 10% of the time).
2. **Temporal consistency.** If a printer was mid-10-step job at step 4 and suddenly reports IDLE at step 6, that's physically impossible.
3. **Cross-sensor consistency.** Hotend temp = 0 but part cooling fan spun up = inconsistent.
4. **Run a diagnostic.** `RUN_DIAGNOSTIC` costs 1 step + small penalty, but returns ground truth on one printer.

### The Mirage Trap (updated)

**Old version:** Random 5% IDLE lie → Dispatcher assigns job → hard crash, -2.0 penalty.

**New version:** Multiple structured traps — the Dispatcher must learn a policy, not memorize one lie.

- **Trap A (Thermistor trap):** Thermistor opens → reports 0°C → naive Dispatcher schedules a maintenance ticket ("cold printer needs attention"). Wastes operator time. Reward: correctly route to `RUN_DIAGNOSTIC` first.
- **Trap B (Filament ghost):** Filament sensor false-positive pauses a healthy print. Naive Dispatcher issues `REQUEST_SPOOL_SWAP`. Wastes operator time + filament. Reward: correctly verify spool weight before dispatching.
- **Trap C (Temporal impossibility — original Mirage):** Mid-print IDLE signal. Naive Dispatcher issues `ASSIGN_JOB` → collision. Reward: temporal reasoning.
- **Trap D (Operator disagreement):** Operator's `REPORT_ANOMALY` contradicts telemetry. Naive Dispatcher picks one based on surface cues. Reward: learn when to trust human vs. machine (ground truth is mixed — operators are right ~85% of the time for mechanical issues, telemetry is right ~95% for electrical issues).

---

## 5. Deployability Bridge: OctoPrint/Moonraker API Mapping

**This is what separates a hackathon project from a product. Do not skip.**

### Action → real API map

| Env Action | OctoPrint REST | Moonraker RPC |
|---|---|---|
| `ASSIGN_JOB` | `POST /api/files/local/{path}` + `POST /api/job` (`start`) | `printer.print.start` |
| `CANCEL_JOB` | `POST /api/job` (`cancel`) | `printer.print.cancel` |
| `RUN_DIAGNOSTIC` | `GET /api/printer` + `GET /api/printer/tool` | `printer.objects.query` |
| `PAUSE_JOB` | `POST /api/job` (`pause`) | `printer.print.pause` |
| `RESUME_JOB` | `POST /api/job` (`pause`, action=resume) | `printer.print.resume` |
| `GET_TELEMETRY` | `GET /api/printer` | `printer.objects.query` |
| `DISPATCH_TICKET` | *(External — e.g., writes to Slack/Discord/internal queue)* | Same |

### Implementation deliverable

Ship an adapter module: `printfarm_env/adapters/octoprint.py` that exposes a class with the same action methods as the env, but hits a real OctoPrint instance. Demonstrate in the pitch: "Here's our trained agent controlling the simulator. Here's the exact same agent — one import line changed — controlling a real Bambu via OctoPrint."

Even a mocked `OctoPrintAdapter` that logs the would-be HTTP calls is enough to prove the design.

---

## 6. Economic Layer (P&L Signal)

Replace the abstract reward [0, 1] with something a farm owner recognizes.

### Cost model inputs (configurable per farm profile)

| Cost Item | Example Value | Source |
|---|---|---|
| Electricity | $0.15/kWh × printer wattage × print time | Utility bill |
| Filament | $20/kg (PLA), $25/kg (PETG), $30/kg (ABS) | Supplier |
| Operator labor | $18/hour (junior), $28/hour (senior), $40/hour (lead) | HR |
| Printer amortization | $3,000 / 10,000 print hours = $0.30/hr | Capex ÷ lifespan |
| SLA penalty | $50 per missed deadline + $5/hour late | Customer contract |
| Rush order premium | 1.5× base price | Pricing |
| Scrap cost | Full filament + labor for failed prints | — |

### Revenue side

- Jobs have a **customer-paid price** in dollars.
- Net profit per job = price − (filament + electricity + labor + amortization + SLA penalty).

### Reward signal

**Primary:** `net_profit_per_step` (dense, per-step dollars earned or lost).

**Secondary (for normalization across tasks):** `normalized_reward ∈ [0, 1]` = (actual_profit − worst_case) / (optimal_profit − worst_case), computed via an oracle solver.

This gives you both a business-readable number for the pitch ("our agent earned $847/day vs. baseline's $312/day") AND a normalized number for comparing across task configurations.

---

## 7. Heterogeneous Printer Fleet

### PrinterProfile schema

```yaml
# profiles/bambu_x1c.yaml
model: "Bambu Lab X1 Carbon"
build_volume_mm: [256, 256, 256]
max_speed_mm_s: 500
materials_supported: [PLA, PETG, ABS, ASA, PA, PC]
multi_material: true  # AMS
reliability_base: 0.98
wear_profile:
  nozzle_hardened_hours: 2000
  belt_cycles_to_service: 500_000
  thermistor_mtbf_hours: 8000
failure_rate_multipliers:
  thermistor_open: 0.7   # higher quality sensors
  filament_sensor_false_runout: 1.2  # AMS is finicky
  webcam_freeze: 0.5
cost:
  capex: 1500
  amortization_per_hour: 0.20
  avg_power_watts: 180
```

Ship profiles for at least: **Bambu X1C, Prusa MK4, Creality K1, Voron 2.4**. These are the four most common production-farm printers and have genuinely different quirks.

### Why this matters for training

Models that learn on homogeneous fleets overfit to one reliability profile. A farm with mixed hardware needs an agent that reasons about per-printer failure rates and material compatibility. This also hits **Theme 3.1 World Modeling** honestly — the agent maintains a belief about each printer's wear state.

---

## 8. Task Curriculum (Revised)

Five tasks, each stress-testing a specific capability. Keep Round 1's tasks as a "warmup" suite (Tasks 0.x) for smoke testing; the new tasks (1–5) are Round 2's evaluation.

### Task 0.1–0.3: Warmup (inherited from Round 1)
Traffic Jam, Spool Runout, Thermal Cooldown. Used for regression testing only.

### Task 1: Human Latency Coordination
**Theme:** Multi-Agent (Theme 1)
**Setup:** 8 printers, 2 junior operators (7-min avg latency), 1 senior (15-min, higher capacity). 6 jobs over 60 steps.
**Challenge:** Dispatcher must batch tickets geographically, avoid overwhelming junior operator queues, and escalate maintenance to the senior.
**Success:** P&L > $X and no operator queue exceeds 3 tickets.

### Task 2: Sensor Trust Calibration
**Theme:** Zero-Trust (Theme 5)
**Setup:** 5 printers. During the episode, 2 printers will experience structured sensor failures (mix of thermistor_open, filament_sensor_false_runout, webcam_freeze).
**Challenge:** Dispatcher must use `RUN_DIAGNOSTIC` selectively, cross-reference operator reports, and not waste operator time on false alarms.
**Success:** Zero operator time spent on false tickets; all real faults caught within 3 steps.

### Task 3: Disagreement Resolution
**Theme:** Multi-Agent + Zero-Trust
**Setup:** Operator sends `REPORT_ANOMALY("Printer 2 is printing fine")` while telemetry shows `thermistor_open`. In another printer, operator is silent and telemetry shows a subtle `progress_drift`. One is sensor failure; the other is operator blindness.
**Challenge:** Dispatcher must evaluate each disagreement on its merits. No single-rule-works-everywhere answer.
**Success:** P&L in top quartile of possible outcomes; no catastrophic failure.

### Task 4: Shift Handoff
**Theme:** Long-Horizon (Theme 2)
**Setup:** 90-step episode spans two shifts. Morning operators go offline at step 45; evening operators come on with no context.
**Challenge:** Dispatcher must maintain state across the handoff, summarize active work for the incoming shift, and not lose track of pending tickets.
**Success:** No job stalls > 10 steps during handoff.

### Task 5: Economic Stress Test
**Theme:** World Modeling + P&L
**Setup:** Realistic order queue with 4 urgent (premium price, tight SLA), 8 normal, 3 low-priority jobs. Limited filament inventory. One printer has degrading nozzle (reliability drops 0.01/step).
**Challenge:** Maximize profit. Correct answer likely involves letting low-priority jobs miss deadlines to protect urgent margins.
**Success:** Net profit > baseline by X%. Evaluated against an oracle LP solver.

---

## 9. Reward Model & Evaluation

### Dense per-step reward (for RL) — **dollar-denominated, unit-consistent**

**Do not mix dollar profit deltas with arbitrary small penalties.** A completed $25 job will dwarf a `-2.0` crash penalty and the agent will learn to crash printers on purpose. Denominate every term in dollars.

```
r_t = Δnet_profit_t                              # dollars gained/lost this step
    - $0.10 × num_pending_tickets_over_capacity  # queue overload cost (operator stress)
    - $0.50 × (unnecessary RUN_DIAGNOSTIC)       # wasted operator/tech time
    + $2.00 × (RUN_DIAGNOSTIC caught real fault) # avoided downstream loss
    - $250   × (catastrophic hardware failure)   # parts replacement + downtime
    - $50    × (SLA deadline missed)             # customer contract penalty
```

**Mandatory sanity-check before training: compute break-even ratios.**

Before you commit numbers, work out what implicit policy they encode:

- WAIT cost ($0.10) vs crash cost ($250) → agent will WAIT up to 2,500 steps to avoid a 10% crash risk. Is that what you want? If the episode is only 90 steps, this is actually reasonable; if longer, tune down.
- Unnecessary diagnostic (−$0.50) vs caught fault (+$2.00) → agent stops diagnosing unless >20% confident a fault is present. This is the implicit detection threshold.
- SLA miss (−$50) vs premium job profit (+$50) → agent is indifferent between shipping late and not shipping at all. Probably wrong — late-with-partial-credit is usually better than cancelling. Bias the late penalty down or add a late-completion partial credit.

Write these ratios into a markdown table in your repo. Review them with the team before kicking off training. They ARE the policy.

### Episode-level summary metrics

- `net_profit_usd`
- `sla_compliance_rate`
- `operator_utilization` (target: 60–80%; over or under is bad)
- `false_positive_ticket_rate`
- `hallucination_caught_rate`
- `catastrophic_failures`

### Baselines (revised — no LP solver)

**Do NOT try to build a true optimal solver.** This problem is Flexible Job Shop Scheduling with sequence-dependent setup times + stochastic arrivals = NP-hard. A correct MIP formulation in PuLP/Gurobi will eat ~12 hours of hackathon time. Don't.

Compute two reference points instead:

1. **Naive-greedy baseline** — assign jobs FIFO to any available printer, ignore sensor warnings, trust everything. (This is what most LLMs do zero-shot. Use it as the floor.)
2. **Clairvoyant-greedy baseline** (aka "prescient policy") — a simple Python script that plays the env but is allowed to cheat: it sees ground-truth printer states, knows exactly when sensors will fail, and knows each operator's true response latency. Still uses a greedy selection rule, so it's not optimal — but it's a legitimate high-water mark. **Targets 30 min to implement.**

Present results as **"trained agent captures X% of the clairvoyant-to-naive gap."** This is honest and fast to compute. If you must say "optimal," say "prescient-information upper bound" — it's what the stochastic scheduling literature actually calls this.

---

## 10. Post-Training Strategy

### Primary recipe: SFT → GRPO (with DPO as optional polish)

> **Updated 2026-04-22:** GRPO promoted to primary. Organizer Help Guide §10–11 frames RL with verifiable rewards as the intended stack, and our env's `step_reward_usd` is already a verifier. DPO remains available as secondary preference-based polish if time permits.

**Recommended sequence (total budget ~3 hours compute):**

1. **SFT (~1 hour).** Run the **clairvoyant-greedy baseline** on ~200 episodes, log the (state, action) pairs, and SFT the student model on them. This grounds the base policy in "roughly competent dispatcher" behavior.
2. **GRPO (~2 hours).** TRL `GRPOTrainer` + Unsloth; rollouts go through the env; reward = dollar P&L from `economics.py`. See `notebooks/grpo_dispatcher.ipynb`.
3. **DPO (optional, ~1 hour if time).** Step-level preference pairs for Zero-Trust sensor skepticism and operator-trust behaviors specifically.

### Honest caveat about GRPO

GRPO optimises return directly via the env reward, but is sensitive to rollout variance from stochastic operators and sensor failures. Mitigations: (a) SFT warm-start ensures the base policy reaches relevant states, (b) use `step_reward_usd` as dense per-step signal rather than sparse episode return.

### Only fall back to DPO-only if...

- GRPO reward curves are flat after 2 hours of training (no visible P&L improvement), AND
- You still have >6 hours of compute left.

Otherwise, stick with SFT + GRPO and accept the tradeoff.

### If DPO: generating preference pairs properly

**Do NOT** do trajectory-level DPO on stochastic episodes — the same prompt rarely reproduces. Instead, **step-level preference pairs** at decision points:

1. During rollout, at each state `s_t`, sample two candidate actions `a_chosen`, `a_rejected` from the policy.
2. Label preference using a programmatic oracle:
   - **Chosen:** uses `RUN_DIAGNOSTIC` when temporal inconsistency is present
   - **Rejected:** issues `ASSIGN_JOB` against a printer with stale telemetry
   - (Define ~10 such heuristics from the failure taxonomy.)
3. Aggregate into a standard DPO dataset.

### Teacher model for rollout generation

- **Prefer:** Claude Opus, GPT-4o, or DeepSeek-V3 via API — high-quality reasoning, cheap enough for ~1000 episodes.
- **Avoid:** Nemotron-70B local — slow, expensive, and its advantage on this task over API models is not worth the time.

### Student model

- **Target:** Llama-3.1-8B or Qwen-2.5-7B (fits on one A100, trains fast in Unsloth).
- **Frame:** ChatML with structured JSON action output.

### Training loop sketch (Unsloth + TRL)

```python
# Day 1 AM  — build clairvoyant-greedy baseline (30 min) + generate SFT trajectories (~200 eps, 1 hr)
# Day 1 PM  — SFT run (~1 hr on 1× A100), then kick off GRPO training
# Day 2 AM  — GRPO training continues; optionally generate DPO preference pairs
# Day 2 PM  — evaluate on Tasks 1, 2, 3 + generate reward curves + record demo trajectory
```

### The winning slide (revised)

**Do not pre-commit to a number.** Instead, commit to the format:

> "Baseline {model} on {task} earned ${X}/episode with {Y}% catastrophic failures.
> After {GRPO/DPO}, same model earned ${X'}/episode with {Y'}% catastrophic failures.
> Here is the reward curve. Here is a before/after trajectory on Task 3, narrated by our Oversight Agent."

The Oversight Agent narration on the demo trajectory is the storytelling kicker.

---

## 11. Pitch Narrative (3-minute script)

### Act 1: The problem (40 seconds)

> "There are 500,000 OctoPrint installations running 3D printers right now. They all face the same three problems: they can't coordinate with human operators intelligently, they blindly trust sensors that fail all the time, and they optimize print completion instead of profit.
>
> We built an environment that forces an AI to solve all three."

### Act 2: The twist (60 seconds)

> "Our environment has one AI agent — the Dispatcher — working with simulated human technicians who are slow, imperfect, and sometimes know things the AI doesn't.
>
> And the sensors lie. Not randomly — they lie the way real sensors lie. Thermistors fail to zero. Filament runout sensors fire on full spools. Webcams freeze. The agent has to learn a taxonomy of distrust.
>
> Then — and this is our Fleet AI bonus submission — a third LLM watches the Dispatcher and tells you in plain English every time it ignored a human warning or overrode a sensor without diagnosing first."

### Act 3: The proof (60 seconds)

> "Here's a 90-step episode. Baseline Llama-8B crashes the factory — trusts a frozen webcam, dispatches a job into a hot collision, $2,100 in losses.
>
> Same model, after SFT + DPO on our environment: handles the same frozen webcam by running a diagnostic first, saves the print, earns $847.
>
> Here's the reward curve. Here's the Oversight Agent's audit log. And here's our OctoPrint adapter — the same trained policy, one import changed, driving a real Bambu X1C."

### Demo recording rules (non-negotiable)

**Conference Wi-Fi will kill a live API-dependent demo.** Pre-record everything that touches the internet.

- ✅ **Pre-bake the Oversight Agent's narration** as a text overlay on the recorded episode video. Run inference once, offline, the night before. Never call Claude/GPT-4o live during the pitch.
- ✅ **Pre-render the reward curves** as PNGs; don't regenerate in a notebook during Q&A.
- ✅ **Pre-record the before/after trajectory comparison** as a side-by-side video.
- 🆗 **OctoPrint adapter stdout** can be shown live — it's a local Python process printing HTTP calls to terminal. Deterministic, no network.
- ❌ **Live model inference in a notebook** — one flaky connection and you're dead on stage.
- ❌ **Live HuggingFace Spaces demo** — same failure mode.

Bring the demo on a laptop. Do not trust the venue's projector to run a remote Colab.

### Act 4: The ask (20 seconds)

> "We're shipping this as a standalone OpenEnv environment AND as an OctoPrint plugin. Any farm running Klipper can pilot it tomorrow."

---

## 12. Risk Register / Anti-Patterns

### Things that will kill this project

| Risk | Mitigation |
|---|---|
| Scope creep — trying to ship all 5 tasks + oversight + OctoPrint adapter + economic layer | **Cut ruthlessly.** Day 1: pick 3 tasks. Day 2: ship or drop. |
| Trying to write a true optimal LP/MIP solver | **Banned.** FJSP-SDST is NP-hard. Use the clairvoyant-greedy baseline only. |
| Reward scaling bugs (dollar profit dwarfs safety penalties, agent learns to crash printers) | Denominate everything in dollars. Compute break-even ratios *before* training. |
| Cold DPO underperforms because base policy doesn't reach target states | SFT warmup on clairvoyant-greedy trajectories first (~1 hr). Non-negotiable. |
| Operator NPC logic becomes its own project | Use a 50-line stochastic policy. Do not over-engineer. |
| OctoPrint adapter becomes a real integration project | Ship a **mocked adapter** that logs HTTP calls. That's enough for the pitch. |
| Demo video tries to explain all 5 tasks | Pick ONE task for the demo trajectory. Task 3 (Disagreement Resolution) tells the best story. |
| Reward curves are flat | Fallback: show step-level action distribution shift ("before: 0% diagnostic use; after: 34% diagnostic use on ambiguous telemetry") |
| Live API calls to Claude/GPT-4o die on conference Wi-Fi | **Pre-bake every LLM inference call** in the demo. Only local Python processes run live. |

### Anti-patterns from v1.0 manual to avoid

1. ❌ "We hit four themes" — reads as unfocused.
2. ❌ Pre-committing to "300% reward improvement" — sets you up for failure.
3. ❌ Beta as a one-action agent — not real multi-agent.
4. ❌ Random 5% sensor corruption — not real Zero-Trust.
5. ❌ Homogeneous printer fleet — not real world.

---

## 13. Day-by-Day Execution Plan

### Pre-onsite (now → day 0)

- [ ] Finalize failure-mode YAML taxonomy
- [ ] Implement `PrinterProfile` loader + 4 profiles
- [ ] Implement operator NPC policy (simple stochastic)
- [ ] Implement 3 of 5 new tasks (1, 2, 3 are highest priority)
- [ ] Mock OctoPrint adapter (log-only)
- [ ] Clairvoyant-greedy Python baseline script (Task 5 high-water mark + SFT trajectory source)
- [ ] Generate teacher rollouts with Claude/GPT-4o (~500 episodes)

### Day 1 onsite

- **AM:** Build clairvoyant-greedy baseline + generate SFT trajectories (~200 episodes). Smoke-test reward scaling with break-even ratio review.
- **PM:** SFT run (~1 hr). Generate step-level DPO preference pairs from preference-labeling heuristics. Kick off DPO training overnight if needed.

### Day 2 onsite

- **AM:** DPO training completes. Evaluate on Tasks 1, 2, 3 vs naive-greedy and clairvoyant-greedy baselines.
- **Midday:** Oversight Agent prompt engineering (offline inference only). Record demo trajectory with pre-baked narration overlay.
- **PM:** Pitch rehearsal. Video recording. Submit.

---

## 14. What Success Looks Like (Post-Hackathon)

After the hackathon, this project should be:

1. A **public OpenEnv environment** on HuggingFace that others use for multi-agent RL research.
2. An **OctoPrint plugin** in alpha with at least one real farm piloting it.
3. A **blog post** on the Zero-Trust sensor taxonomy that print-farm operators cite when describing their real pain.
4. A **training recipe** others can adapt to their own fleet profiles.

If you win the hackathon but none of these four happen by end of month, the project failed its real mission.

If you don't win the hackathon but one of these four happens, the project succeeded.

**Build for deployment, not for judges. The judges will notice.**
