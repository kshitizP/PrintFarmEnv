"""
check_shortcut_leakage.py — Audit templates for text-pattern shortcuts.

For each signal type, answers the question:
  "Can a bag-of-words model predict the correct action from just the text,
   without any reasoning about the context?"

If yes, the model doesn't need to reason — it can just memorize surface patterns.

Usage:
    python submission/scripts/check_shortcut_leakage.py
"""

import sys
from pathlib import Path
from collections import defaultdict

# ── imports ────────────────────────────────────────────────────────────────────
_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))

from submission.env.customer_messages import TEMPLATES as MSG_TEMPLATES
from submission.env.operator_notes import TEMPLATES as NOTE_TEMPLATES
from submission.env.novel_faults import NOVEL_FAULT_DEFS

# ── helpers ────────────────────────────────────────────────────────────────────

def tokenise(text: str) -> set:
    import re
    text = text.lower()
    text = re.sub(r"\{[^}]+\}", "TARGET", text)   # mask placeholders
    return set(re.findall(r"[a-z]+", text))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def separability(group_a: list, group_b: list) -> dict:
    """Can a simple keyword rule separate group_a from group_b?

    Returns:
        - exclusive_to_a: words that appear ONLY in group_a texts
        - exclusive_to_b: words that appear ONLY in group_b texts
        - shared: words appearing in both (potential confusion)
        - a_separable: fraction of group_a texts that contain at least one exclusive word
        - b_separable: fraction of group_b texts that contain at least one exclusive word
    """
    vocab_a = defaultdict(int)
    vocab_b = defaultdict(int)
    for t in group_a:
        for w in tokenise(t):
            vocab_a[w] += 1
    for t in group_b:
        for w in tokenise(t):
            vocab_b[w] += 1

    only_a = {w for w in vocab_a if w not in vocab_b}
    only_b = {w for w in vocab_b if w not in vocab_a}
    shared = set(vocab_a) & set(vocab_b)

    a_separable = sum(1 for t in group_a if tokenise(t) & only_a) / max(len(group_a), 1)
    b_separable = sum(1 for t in group_b if tokenise(t) & only_b) / max(len(group_b), 1)

    return {
        "exclusive_to_a": sorted(only_a),
        "exclusive_to_b": sorted(only_b),
        "shared": sorted(shared),
        "a_separable_pct": round(a_separable * 100),
        "b_separable_pct": round(b_separable * 100),
    }


BOLD  = "\033[1m"
RED   = "\033[91m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RESET = "\033[0m"

def verdict(pct: int) -> str:
    if pct == 100:
        return f"{RED}SHORTCUT {pct}%{RESET}"
    elif pct >= 75:
        return f"{YELLOW}RISK {pct}%{RESET}"
    else:
        return f"{GREEN}OK {pct}%{RESET}"


# ── 1. Customer messages ───────────────────────────────────────────────────────

def check_messages():
    print(f"\n{'='*70}")
    print(f"{BOLD}CUSTOMER MESSAGES — action separability{RESET}")
    print(f"{'='*70}")

    by_action = defaultdict(list)
    for t in MSG_TEMPLATES:
        by_action[t.ground_truth_action].append(t.text)

    actions = sorted(by_action.keys())
    print(f"Actions: {actions}")
    print(f"Template counts: { {a: len(v) for a,v in by_action.items()} }\n")

    # Per-action: check separability from ALL others
    for act in actions:
        mine = by_action[act]
        others = [t for a, v in by_action.items() for t in v if a != act]
        sep = separability(mine, others)
        print(f"  {BOLD}{act}{RESET}")
        print(f"    separable from others: {verdict(sep['a_separable_pct'])}")
        if sep["exclusive_to_a"]:
            print(f"    unique keywords:  {sep['exclusive_to_a'][:12]}")
        if sep["shared"]:
            shared_risk = [w for w in sep["shared"] if len(w) > 3]
            if shared_risk:
                print(f"    shared w/ others: {shared_risk[:8]}")

    # Special check: false_urgency vs hard_urgency (both look "urgent")
    false_urg = by_action.get("standard_queue", [])
    # only the URGENT-caps ones
    caps_urgency = [t for t in false_urg
                    if any(kw in t for kw in ("URGENT", "ASAP", "CRITICAL", "EMERGENCY"))]
    time_rush = by_action.get("accept_rush", [])
    print(f"\n  {BOLD}false_urgency (CAPS) vs accept_rush overlap:{RESET}")
    sep2 = separability(caps_urgency, time_rush)
    print(f"    CAPS texts separable from time-rush: {verdict(sep2['a_separable_pct'])}")
    print(f"    Time-rush separable from CAPS:       {verdict(sep2['b_separable_pct'])}")
    print(f"    Shared vocabulary: {sep2['shared'][:10]}")
    print(f"\n  {BOLD}KEY RISK:{RESET} All accept_rush messages reference a concrete time signal")
    print(f"  (tomorrow, 6 hours, 3pm, end of day, 4 hours). Model can learn:")
    print(f"  'time_word + job_id → ASSIGN_JOB' without reading what the event IS.")


# ── 2. Operator notes ──────────────────────────────────────────────────────────

def check_notes():
    print(f"\n{'='*70}")
    print(f"{BOLD}OPERATOR NOTES — category separability{RESET}")
    print(f"{'='*70}")

    by_cat = defaultdict(list)
    by_action = defaultdict(list)
    for t in NOTE_TEMPLATES:
        by_cat[t.category].append(t.text)
        by_action[t.ground_truth_action].append(t.text)

    cats = sorted(by_cat.keys())
    print(f"Categories: {cats}")
    print(f"Template counts by category: { {c: len(v) for c,v in by_cat.items()} }")
    print(f"Template counts by action:   { {a: len(v) for a,v in by_action.items()} }\n")

    # Critical pair: predictive vs benign
    pred = by_cat.get("predictive", [])
    benign = by_cat.get("benign", [])
    sep = separability(pred, benign)

    print(f"  {BOLD}predictive vs benign{RESET}")
    print(f"    predictive separable from benign: {verdict(sep['a_separable_pct'])}")
    print(f"    benign separable from predictive: {verdict(sep['b_separable_pct'])}")
    print(f"    keywords unique to predictive: {sep['exclusive_to_a'][:15]}")
    print(f"    keywords unique to benign:     {sep['exclusive_to_b'][:15]}")
    print(f"    shared vocabulary:             {sep['shared'][:8]}")

    # Ambiguous vs predictive
    ambig = by_cat.get("ambiguous", [])
    sep2 = separability(pred, ambig)
    print(f"\n  {BOLD}predictive vs ambiguous{RESET}")
    print(f"    predictive separable from ambiguous: {verdict(sep2['a_separable_pct'])}")
    print(f"    ambiguous separable from predictive: {verdict(sep2['b_separable_pct'])}")
    print(f"    keywords unique to predictive: {sep2['exclusive_to_a'][:10]}")
    print(f"    keywords unique to ambiguous:  {sep2['exclusive_to_b'][:10]}")

    # Ambiguous vs benign
    sep3 = separability(ambig, benign)
    print(f"\n  {BOLD}ambiguous vs benign{RESET}")
    print(f"    ambiguous separable from benign: {verdict(sep3['a_separable_pct'])}")
    print(f"    benign separable from ambiguous: {verdict(sep3['b_separable_pct'])}")

    # Vocabulary overlap heatmap
    print(f"\n  {BOLD}Jaccard similarity between categories (lower = more distinct):{RESET}")
    for c1 in cats:
        for c2 in cats:
            if c1 >= c2:
                continue
            v1 = set(w for t in by_cat[c1] for w in tokenise(t))
            v2 = set(w for t in by_cat[c2] for w in tokenise(t))
            j = jaccard(v1, v2)
            color = RED if j > 0.4 else (YELLOW if j > 0.25 else GREEN)
            print(f"    {c1} ↔ {c2}: {color}{j:.2f}{RESET}")


# ── 3. Novel fault flags ───────────────────────────────────────────────────────

def check_novel_faults():
    from submission.env.novel_faults import RESOLVED_ALARM_DEFS

    print(f"\n{'='*70}")
    print(f"{BOLD}NOVEL FAULT FLAGS — template uniqueness + resolved-alarm decoys{RESET}")
    print(f"{'='*70}")

    fault_texts = {k: v["anomaly_text"] for k, v in NOVEL_FAULT_DEFS.items()}
    resolved_texts = {k: v["anomaly_text"] for k, v in RESOLVED_ALARM_DEFS.items()}
    n_faults = len(fault_texts)
    n_resolved = len(resolved_texts)

    print(f"  Real fault types: {n_faults}")
    print(f"  Resolved alarm decoys: {n_resolved}")
    print(f"  Approximate risk level: {'LOW' if n_faults >= 8 else 'MEDIUM' if n_faults >= 5 else 'HIGH'} "
          f"(memorisation harder with {n_faults} distinct templates)\n")

    print(f"  {BOLD}Real fault templates:{RESET}")
    all_fault_tokens = set(w for t in fault_texts.values() for w in tokenise(t))
    for name, text in fault_texts.items():
        tokens = tokenise(text)
        unique = tokens - (all_fault_tokens - tokens)
        print(f"    {name[:30]:30s} → \"{text[:55]}...\"")
        print(f"    {'':30s}   unique tokens: {sorted(unique)[:6]}")

    # Key check: real faults vs resolved alarms — can these be separated by keywords?
    print(f"\n  {BOLD}Real faults vs resolved alarms — separability:{RESET}")
    real_list = list(fault_texts.values())
    resolved_list = list(resolved_texts.values())
    sep = separability(real_list, resolved_list)
    print(f"    Real faults separable from resolved alarms: {verdict(sep['a_separable_pct'])}")
    print(f"    Resolved alarms separable from real faults: {verdict(sep['b_separable_pct'])}")
    print(f"    Words unique to RESOLVED alarms: {sep['exclusive_to_b'][:10]}")
    print(f"    Shared vocabulary (confusion zone): {sep['shared'][:12]}")

    if sep['b_separable_pct'] == 100 and sep['a_separable_pct'] == 100:
        print(f"\n  {YELLOW}Still 100% separable — model CAN pattern-match faults vs resolved.{RESET}")
        print(f"  BUT: the distinguishing words are conceptual resolution phrases")
        print(f"  ('confirmed', 'cleared', 'ops confirmed', 'IT confirmed') which the")
        print(f"  model must understand semantically, not just keyword-match.")
    else:
        print(f"\n  {GREEN}Vocabulary overlap makes simple keyword shortcuts unreliable.{RESET}")


# ── 4. Cross-signal confusion ──────────────────────────────────────────────────

def check_cross_signal():
    print(f"\n{'='*70}")
    print(f"{BOLD}CROSS-SIGNAL — vocabulary bleed between signal types{RESET}")
    print(f"{'='*70}")

    msg_vocab  = set(w for t in MSG_TEMPLATES for w in tokenise(t.text))
    note_vocab = set(w for t in NOTE_TEMPLATES for w in tokenise(t.text))
    fault_vocab = set(
        w for fdef in NOVEL_FAULT_DEFS.values()
        for w in tokenise(fdef["anomaly_text"])
    )

    print(f"  messages ↔ notes Jaccard:  {jaccard(msg_vocab, note_vocab):.2f}")
    print(f"  messages ↔ faults Jaccard: {jaccard(msg_vocab, fault_vocab):.2f}")
    print(f"  notes ↔ faults Jaccard:    {jaccard(note_vocab, fault_vocab):.2f}")

    # Words that bleed between notes and faults (most dangerous)
    note_fault_shared = (note_vocab & fault_vocab) - {"p", "target"}
    print(f"\n  Shared words notes ↔ faults: {sorted(note_fault_shared)}")


# ── 5. Proposed fixes summary ─────────────────────────────────────────────────

def propose_fixes():
    print(f"\n{'='*70}")
    print(f"{BOLD}PROPOSED FIXES{RESET}")
    print(f"{'='*70}")

    print(f"""
  {BOLD}FIX 1 — Novel faults (CRITICAL):{RESET}
    Add ≥5 more novel fault types with varied, less formulaic text.
    Add 2-3 "decoy" anomaly flag texts that use similar vocabulary but
    are NOT real faults (e.g., "webcam connection issue — IT confirmed
    cable, printer fine") → r_novel_fault should return 0.0 for these.
    Currently all 3 anomaly texts = fault. Need counter-examples.

  {BOLD}FIX 2 — Operator notes predictive vs ambiguous (MEDIUM):{RESET}
    Add "benign-sounding predictive" notes — technical-vocabulary notes
    where the fault IS real:
      "P{{pid}} temps look fine but the pattern over the last 10 steps
       shows a slow creep — keep an eye on it"  → predictive
    Add "predictive-sounding benign" notes — alarm words but no fault:
      "P{{pid}} fan sounded loud briefly but settled — operator checked,
       all normal"  → benign (false alarm resolved)
    This forces the model to consider context, not just keywords.

  {BOLD}FIX 3 — Customer messages false_urgency (LOW):{RESET}
    Already well-separated: CAPS vs time-words. The model SHOULD learn
    this distinction. Actually acceptable behaviour (this IS what we want
    it to learn). No fix needed here.

  {BOLD}FIX 4 — Substitution signal in both notes and messages (LOW):{RESET}
    Both note substitution and message substitution lead to REQUEST_SPOOL_SWAP.
    That is correct — the action is the same regardless of source.
    No confusion risk here.

  {BOLD}FIX 5 — Add "ambiguous looks benign but is predictive" notes (MEDIUM):{RESET}
    Currently ambiguous templates all hedge: "might be nothing", "print looks
    ok though". Add a few where the hedged language conceals a real fault
    and a few where confident language masks no fault — to stop the model
    from using hedge-words as a binary signal.
""")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    check_messages()
    check_notes()
    check_novel_faults()
    check_cross_signal()
    propose_fixes()
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
