"""
Operator notes generator — injects free-text notes into observations.

Each note is tagged with a hidden ground-truth outcome used only by the
reward function, never visible to the agent.

Categories:
  - predictive: real signal of impending failure (correct action: investigate)
  - benign: distractor, no action needed (correct action: ignore)
  - substitution: material substitution opportunity (correct action: accept-sub)
  - ambiguous: could go either way (correct action depends on context)
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class NoteTemplate:
    text: str
    category: str  # predictive | benign | substitution | ambiguous
    ground_truth_action: str  # investigate | ignore | accept_substitute | context_dependent
    target_printer_slot: Optional[int] = None  # 0-indexed slot in printer list; None = random


# ~30 templates across 4 categories
TEMPLATES: List[NoteTemplate] = [
    # --- Predictive (real signal of impending failure) ---
    NoteTemplate("Printer {pid} sounds rattly on retract — last 3 prints ok but watch it",
                 "predictive", "investigate"),
    NoteTemplate("webcam blur on P{pid}, parts still look fine",
                 "predictive", "investigate"),
    NoteTemplate("spool {spool_id} humidity indicator turned pink",
                 "predictive", "investigate"),
    NoteTemplate("P{pid} bed adhesion seems weak last few prints, might lose one soon",
                 "predictive", "investigate"),
    NoteTemplate("noticed small stringing on P{pid} nozzle, might be partial clog",
                 "predictive", "investigate"),
    NoteTemplate("P{pid} fan sounds different than usual, still spinning but louder",
                 "predictive", "investigate"),
    NoteTemplate("filament path on P{pid} has some dust buildup, could cause friction",
                 "predictive", "investigate"),
    NoteTemplate("P{pid} PTFE tube looks slightly discolored near hotend",
                 "predictive", "investigate"),

    # --- Benign (distractor, no action needed) ---
    NoteTemplate("swept floor around P{pid}",
                 "benign", "ignore"),
    NoteTemplate("lighting flickered briefly near P{pid}, back to normal",
                 "benign", "ignore"),
    NoteTemplate("replaced empty water bottle near the printers",
                 "benign", "ignore"),
    NoteTemplate("organized spare nozzles in drawer",
                 "benign", "ignore"),
    NoteTemplate("P{pid} LCD showing normal temps, all good",
                 "benign", "ignore"),
    NoteTemplate("topped up isopropyl alcohol for bed cleaning",
                 "benign", "ignore"),
    NoteTemplate("door was open, closed it — no temp change on printers",
                 "benign", "ignore"),
    NoteTemplate("ran a test print on P{pid} last shift, came out perfect",
                 "benign", "ignore"),

    # --- Substitution opportunities ---
    NoteTemplate("customer for {job_id} said PETG-CF is fine if we're out of PETG",
                 "substitution", "accept_substitute"),
    NoteTemplate("customer for {job_id} okayed using black instead of charcoal PLA",
                 "substitution", "accept_substitute"),
    NoteTemplate("{job_id} customer said ABS or ASA both work for their part",
                 "substitution", "accept_substitute"),
    NoteTemplate("got word that {job_id} can use any PLA color in stock",
                 "substitution", "accept_substitute"),

    # --- Ambiguous (could go either way) ---
    NoteTemplate("first layer on P{pid} looks slightly off, might be nothing",
                 "ambiguous", "context_dependent"),
    NoteTemplate("P{pid} making a new sound I haven't heard before, print looks ok though",
                 "ambiguous", "context_dependent"),
    NoteTemplate("P{pid} bed seems to need more warmup time than usual",
                 "ambiguous", "context_dependent"),
    NoteTemplate("small vibration on P{pid} at high speed moves, might just be the model geometry",
                 "ambiguous", "context_dependent"),
    NoteTemplate("P{pid} webcam image looks grainy today, maybe it's the lighting",
                 "ambiguous", "context_dependent"),
    NoteTemplate("noticed P{pid} paused briefly mid-layer then resumed — network hiccup?",
                 "ambiguous", "context_dependent"),
    NoteTemplate("P{pid} temps stable but power draw seems slightly higher",
                 "ambiguous", "context_dependent"),
]


def generate_notes(
    step: int,
    printers: list,
    jobs: list,
    rng: random.Random,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Generate 0-2 operator notes for this step.

    Returns:
        (visible_notes, ground_truth_tags)

        visible_notes: list of strings injected into the observation.
        ground_truth_tags: list of dicts with hidden metadata for the reward fn.
            Each dict has: {text, category, ground_truth_action, printer_id, job_id}
    """
    # Probability of any note this step: ~30% for 1 note, ~10% for 2
    roll = rng.random()
    if roll > 0.40:
        return [], []

    n_notes = 2 if roll < 0.10 else 1
    visible: List[str] = []
    tags: List[Dict[str, Any]] = []

    for _ in range(n_notes):
        template = rng.choice(TEMPLATES)

        # Pick a printer for {pid}
        pid = rng.choice(printers).printer_id if printers else 1

        # Pick a job for {job_id}
        pending_jobs = [j for j in jobs if hasattr(j, 'state')
                        and getattr(j.state, 'value', str(j.state)) in ("PENDING", "PRINTING")]
        job_id = rng.choice(pending_jobs).job_id if pending_jobs else "j1"

        # Format the text
        text = template.text.format(
            pid=pid,
            spool_id=f"S{rng.randint(1, 20):02d}",
            job_id=job_id,
        )

        visible.append(text)
        tags.append({
            "text": text,
            "category": template.category,
            "ground_truth_action": template.ground_truth_action,
            "printer_id": pid,
            "job_id": job_id if "job_id" in template.text else None,
        })

    return visible, tags
