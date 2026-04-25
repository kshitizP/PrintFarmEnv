"""
Customer messages generator — injects occasional customer messages with
ambiguous urgency into observations.

Each message is tagged with a hidden ground-truth "correct action" used only
by the reward function.
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MessageTemplate:
    text: str
    category: str  # hard_urgency | soft_urgency | false_urgency | substitution_ask
    ground_truth_action: str  # accept_rush | standard_queue | decline | accept_substitute
    needs_job: bool = True  # whether this message references a specific job


TEMPLATES: List[MessageTemplate] = [
    # --- Hard urgency (genuinely time-sensitive) ---
    MessageTemplate(
        "Trade show tomorrow, will pay 2x for {job_id} to ship today",
        "hard_urgency", "accept_rush"),
    MessageTemplate(
        "Client demo in 6 hours, {job_id} MUST be done — we'll cover rush fee",
        "hard_urgency", "accept_rush"),
    MessageTemplate(
        "Prototype review meeting at 3pm, need {job_id} before that",
        "hard_urgency", "accept_rush"),
    MessageTemplate(
        "Shipping cutoff is end of day for {job_id}, please prioritize",
        "hard_urgency", "accept_rush"),
    MessageTemplate(
        "Airport pickup in 4 hours for {job_id} — any way to expedite?",
        "hard_urgency", "accept_rush"),

    # --- Soft urgency (not critical) ---
    MessageTemplate(
        "Would be nice to have {job_id} by Friday but not critical",
        "soft_urgency", "standard_queue"),
    MessageTemplate(
        "Any update on {job_id}? No rush, just checking in",
        "soft_urgency", "standard_queue"),
    MessageTemplate(
        "Hey, when do you think {job_id} will be ready? Flexible on timing",
        "soft_urgency", "standard_queue"),
    MessageTemplate(
        "Could {job_id} be done this week? If not, next week is fine too",
        "soft_urgency", "standard_queue"),
    MessageTemplate(
        "Just wanted to confirm {job_id} is in the queue. Thanks!",
        "soft_urgency", "standard_queue"),

    # --- False urgency (URGENT label but actually not) ---
    MessageTemplate(
        "URGENT — please prioritize {job_id}!!!",
        "false_urgency", "standard_queue"),
    MessageTemplate(
        "ASAP PLEASE — need {job_id} done immediately!!",
        "false_urgency", "standard_queue"),
    MessageTemplate(
        "This is CRITICAL: {job_id} needs to be first in queue",
        "false_urgency", "standard_queue"),
    MessageTemplate(
        "EMERGENCY: {job_id} is for a very important client",
        "false_urgency", "standard_queue"),

    # --- Substitution asks ---
    MessageTemplate(
        "Can we do black instead of charcoal for {job_id}?",
        "substitution_ask", "accept_substitute"),
    MessageTemplate(
        "Is it possible to use PETG instead of PLA for {job_id}? Either works",
        "substitution_ask", "accept_substitute"),
    MessageTemplate(
        "If you're low on ABS, we can switch {job_id} to PETG",
        "substitution_ask", "accept_substitute"),
    MessageTemplate(
        "Any PLA color is fine for {job_id}, whatever you have loaded",
        "substitution_ask", "accept_substitute"),
    MessageTemplate(
        "{job_id} — customer says natural or white PLA both acceptable",
        "substitution_ask", "accept_substitute"),
]


def generate_messages(
    step: int,
    jobs: list,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate 0-1 customer messages for this step.

    Returns:
        (visible_messages, ground_truth_tags)

        visible_messages: list of {from, text, job_id} dicts for the observation.
        ground_truth_tags: list of dicts with hidden metadata for the reward fn.
    """
    # ~20% chance of a message per step
    if rng.random() > 0.20:
        return [], []

    template = rng.choice(TEMPLATES)

    # Pick a job for this message
    active_jobs = [j for j in jobs if hasattr(j, 'state')
                   and getattr(j.state, 'value', str(j.state)) in ("PENDING", "PRINTING", "PAUSED")]
    if not active_jobs:
        return [], []

    job = rng.choice(active_jobs)
    job_id = job.job_id

    text = template.text.format(job_id=job_id)

    customer_names = ["Alex Chen", "Sam Rivera", "Jordan Kim", "Pat Morgan",
                      "Casey Taylor", "Quinn Lee", "Riley James", "Drew Park"]
    customer = rng.choice(customer_names)

    visible = [{
        "from": customer,
        "text": text,
        "job_id": job_id,
    }]

    tags = [{
        "text": text,
        "category": template.category,
        "ground_truth_action": template.ground_truth_action,
        "job_id": job_id,
        "customer": customer,
        "is_false_urgency": template.category == "false_urgency",
    }]

    return visible, tags
