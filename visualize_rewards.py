"""
Visualize the PrintFarmEnv reward/scoring design across all 3 tasks.

Generates reward_design.png showing:
  - Priority weighting
  - Credit by job state per task
  - Step penalty accumulation
  - Latency decay curve
  - Task-specific score breakdowns
  - Thermal cooldown timeline (Task 3)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Colour palette ──────────────────────────────────────────────────
C_COMPLETED   = "#2ecc71"
C_LATE        = "#f39c12"
C_PRINTING    = "#3498db"
C_FAILED      = "#e74c3c"
C_PENDING     = "#bdc3c7"
C_BONUS       = "#9b59b6"
C_BG          = "#fafafa"

fig = plt.figure(figsize=(20, 14), facecolor="white")
fig.suptitle("PrintFarmEnv — Reward Design Overview", fontsize=18, fontweight="bold", y=0.98)

# =====================================================================
# Panel 1 — Priority Weights
# =====================================================================
ax1 = fig.add_subplot(2, 3, 1)
priorities = ["Low (1)", "Normal (2)", "Urgent (3)"]
weights = [0.5, 1.0, 2.0]
colors = ["#95a5a6", "#3498db", "#e74c3c"]
bars = ax1.bar(priorities, weights, color=colors, edgecolor="white", linewidth=2, width=0.6)
for bar, w in zip(bars, weights):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
             f"{w}x", ha="center", va="bottom", fontsize=14, fontweight="bold")
ax1.set_ylabel("Weight multiplier", fontsize=12)
ax1.set_title("Priority Weighting", fontsize=14, fontweight="bold")
ax1.set_ylim(0, 2.5)
ax1.spines[["top", "right"]].set_visible(False)

# =====================================================================
# Panel 2 — Job State → Credit (per task)
# =====================================================================
ax2 = fig.add_subplot(2, 3, 2)
tasks = ["Task 1\nTraffic Jam", "Task 2\nSpool Runout", "Task 3\nThermal Cooldown"]
# Rows: Completed On-Time, Completed Late, In-Progress (50%), Failed w/ Progress
states = ["Completed\n(on time)", "Completed\n(late)", "In-Progress\n(50%)", "Failed\n(w/ progress)"]

# Credit fractions: [task1, task2, task3] for each state
data = np.array([
    [1.0,  1.0,  1.0 ],  # Completed on-time
    [0.55, 0.55, 0.55],  # Completed 9 steps late: max(0.1, 1-0.05*9) = 0.55
    [0.2,  0.25, 0.2 ],  # In-progress at 50% (0.4*0.5, 0.5*0.5, 0.4*0.5)
    [0.0,  0.15, 0.1 ],  # Failed with progress
])

x = np.arange(len(tasks))
width = 0.18
state_colors = [C_COMPLETED, C_LATE, C_PRINTING, C_FAILED]

for i, (state, color) in enumerate(zip(states, state_colors)):
    offset = (i - 1.5) * width
    rects = ax2.bar(x + offset, data[i], width, label=state, color=color,
                    edgecolor="white", linewidth=1.5)
    for rect, val in zip(rects, data[i]):
        if val > 0:
            ax2.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.02,
                     f"{val:.0%}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels(tasks, fontsize=11)
ax2.set_ylabel("Credit (fraction of weight)", fontsize=12)
ax2.set_title("Credit by Job State", fontsize=14, fontweight="bold")
ax2.set_ylim(0, 1.2)
ax2.legend(loc="upper right", fontsize=9, ncol=2)
ax2.spines[["top", "right"]].set_visible(False)

# =====================================================================
# Panel 3 — Latency Decay Curve
# =====================================================================
ax3 = fig.add_subplot(2, 3, 3)
steps_late = np.arange(0, 20)
decay = np.maximum(0.1, 1.0 - 0.05 * steps_late)

ax3.plot(steps_late, decay, "o-", color="#e74c3c", linewidth=2.5, markersize=5)
ax3.fill_between(steps_late, decay, alpha=0.15, color="#e74c3c")
ax3.axhline(y=0.1, color="gray", linestyle="--", alpha=0.5, label="Floor (10%)")
ax3.set_xlabel("Steps past deadline", fontsize=12)
ax3.set_ylabel("Late multiplier", fontsize=12)
ax3.set_title("Deadline Latency Decay", fontsize=14, fontweight="bold")
ax3.set_ylim(0, 1.15)
ax3.set_xlim(0, 19)
ax3.legend(fontsize=10)
ax3.spines[["top", "right"]].set_visible(False)
ax3.annotate("5% per step", xy=(6, 0.7), fontsize=11, color="#e74c3c", fontweight="bold")

# =====================================================================
# Panel 4 — Task 1 "Traffic Jam" Timeline
# =====================================================================
ax4 = fig.add_subplot(2, 3, 4)

# Path A: PLA(1)+PLA(1)+PLA(1)+Swap(2)+PETG(1)+PETG(1) = 7 steps
path_a_colors = [C_COMPLETED, C_COMPLETED, C_COMPLETED, "#95a5a6", "#95a5a6", C_LATE, C_LATE]
path_a_labels = ["PLA", "PLA", "PLA", "SWAP", "SWAP", "PETG", "PETG"]

# Path B: Swap(2)+PETG(1)+Swap(2)+PLA(1)+PLA(1)+PLA(1) = 8 steps
path_b_colors = ["#95a5a6", "#95a5a6", C_COMPLETED, "#95a5a6", "#95a5a6", C_COMPLETED, C_COMPLETED, C_COMPLETED]
path_b_labels = ["SWAP", "SWAP", "PETG*", "SWAP", "SWAP", "PLA", "PLA", "PLA"]

y_a, y_b = 1.5, 0.5
for i, (c, l) in enumerate(zip(path_a_colors, path_a_labels)):
    ax4.barh(y_a, 1, left=i, height=0.6, color=c, edgecolor="white", linewidth=1.5)
    ax4.text(i + 0.5, y_a, l, ha="center", va="center", fontsize=8, fontweight="bold", color="white")

for i, (c, l) in enumerate(zip(path_b_colors, path_b_labels)):
    ax4.barh(y_b, 1, left=i, height=0.6, color=c, edgecolor="white", linewidth=1.5)
    ax4.text(i + 0.5, y_b, l, ha="center", va="center", fontsize=8, fontweight="bold", color="white")

ax4.axvline(x=12, color="#e74c3c", linestyle="--", linewidth=2, alpha=0.7)
ax4.text(12.1, 2.0, "Deadline\n(step 12)", fontsize=9, color="#e74c3c", fontweight="bold")
ax4.set_yticks([0.5, 1.5])
ax4.set_yticklabels(["Path B\n(Greedy)", "Path A\n(Batch)"], fontsize=11)
ax4.set_xlabel("Time step", fontsize=12)
ax4.set_title("Task 1 — Traffic Jam Strategies", fontsize=14, fontweight="bold")
ax4.set_xlim(0, 14)
ax4.set_ylim(0, 2.3)
ax4.spines[["top", "right"]].set_visible(False)

# =====================================================================
# Panel 5 — Task 2 "Spool Runout" Flow
# =====================================================================
ax5 = fig.add_subplot(2, 3, 5)

phases = ["Assign &\nPrint", "Spool\nRunout!", "Swap\nFilament", "Resume\nJob", "Continue\nPrinting"]
phase_x = [0, 1, 2, 3, 4]
phase_colors = [C_COMPLETED, C_FAILED, "#95a5a6", C_PRINTING, C_COMPLETED]
phase_widths = [4, 0, 2, 0, 6]  # Rough step counts for display

ax5.bar(phase_x, [1]*5, color=phase_colors, edgecolor="white", linewidth=2, width=0.7)

annotations = [
    "~4 steps\n(300g used)",
    "PAUSED_\nRUNOUT",
    "2 steps\n+50g purge",
    "1 step",
    "~6 steps\n(remaining)"
]
for i, (px, ann) in enumerate(zip(phase_x, annotations)):
    ax5.text(px, 1.05, ann, ha="center", va="bottom", fontsize=9, fontweight="bold")

# Show the wrong path
ax5.annotate("X  CANCEL = lose all progress\n(0.15 x weight credit only)",
             xy=(1, 0.5), xytext=(2.5, 0.35),
             fontsize=9, color="#e74c3c", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5),
             ha="center")

ax5.set_xticks(phase_x)
ax5.set_xticklabels(phases, fontsize=10)
ax5.set_ylabel("", fontsize=12)
ax5.set_title("Task 2 — Spool Runout Recovery", fontsize=14, fontweight="bold")
ax5.set_ylim(0, 1.6)
ax5.set_yticks([])
ax5.spines[["top", "right", "left"]].set_visible(False)

# =====================================================================
# Panel 6 — Task 3 "Thermal Cooldown" Timeline
# =====================================================================
ax6 = fig.add_subplot(2, 3, 6)

# Timeline: 3 WAIT (cooldown) + 3 MAINTENANCE + 5 PRINTING = 11 steps
timeline_colors = (
    ["#f39c12"] * 3 +   # Cooldown (WAIT)
    ["#9b59b6"] * 3 +   # Maintenance
    [C_COMPLETED] * 5    # Printing
)
timeline_labels = (
    ["W"] * 3 +
    ["M"] * 3 +
    ["P"] * 5
)

for i, (c, l) in enumerate(zip(timeline_colors, timeline_labels)):
    ax6.barh(1.5, 1, left=i, height=0.6, color=c, edgecolor="white", linewidth=1.5)
    ax6.text(i + 0.5, 1.5, l, ha="center", va="center", fontsize=10,
             fontweight="bold", color="white")

# Annotations
ax6.annotate("Cooldown\n(3 idle steps)", xy=(1.5, 1.85), fontsize=10,
             ha="center", fontweight="bold", color="#f39c12")
ax6.annotate("Maintenance\n(resets fatigue)", xy=(4.5, 1.85), fontsize=10,
             ha="center", fontweight="bold", color="#9b59b6")
ax6.annotate("Print job_critical\n(5 steps, safe)", xy=(8.5, 1.85), fontsize=10,
             ha="center", fontweight="bold", color=C_COMPLETED)

# Show fatigue level
fatigue_steps = list(range(12))
fatigue_vals = [7, 7, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0]  # Stays 7 during cooldown+maint, drops to 0
ax6.plot([x + 0.5 for x in fatigue_steps], [f / 10.0 * 0.8 + 0.2 for f in fatigue_vals],
         "D-", color="#e74c3c", linewidth=2, markersize=5, label="Fatigue / 10")
ax6.axhline(y=0.2 + 10/10*0.8, color="#e74c3c", linestyle="--", alpha=0.3)
ax6.text(10.5, 0.2 + 10/10*0.8 + 0.02, "Fatal (10)", fontsize=8, color="#e74c3c", alpha=0.6)

ax6.axvline(x=18, color="#e74c3c", linestyle="--", linewidth=2, alpha=0.5)
ax6.text(15, 0.05, "Deadline\n(step 18)", fontsize=9, color="#e74c3c", fontweight="bold")

ax6.set_xlabel("Time step", fontsize=12)
ax6.set_title("Task 3 — Thermal Cooldown Timeline", fontsize=14, fontweight="bold")
ax6.set_xlim(0, 20)
ax6.set_ylim(0, 2.2)
ax6.set_yticks([])
ax6.legend(loc="lower right", fontsize=9)
ax6.spines[["top", "right", "left"]].set_visible(False)

# ── Final layout ────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0.02, 1, 0.95])

fig.text(0.5, 0.005,
         "Scores clamped to (0.001, 0.999) per OpenEnv spec.  "
         "Formula: score = clamp(earned / total_weight − step_penalty)",
         ha="center", fontsize=10, style="italic", color="#666")

plt.savefig("reward_design.png", dpi=150, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print("Saved reward_design.png")
print("Saved: reward_design.png")
plt.close()
