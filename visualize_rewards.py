"""
Visualize the PrintFarmEnv reward/scoring design across all 3 tasks.

Generates reward_design.png showing:
  - How each job state maps to score credit
  - Priority weighting
  - Deadline impact
  - Step penalty effects
  - Score ranges per task
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Colour palette ──────────────────────────────────────────────────
C_COMPLETED   = "#2ecc71"
C_LATE        = "#f39c12"
C_PRINTING    = "#3498db"
C_FAILED      = "#e74c3c"
C_PENDING     = "#bdc3c7"
C_BONUS       = "#9b59b6"
C_PENALTY     = "#e74c3c"
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
tasks = ["Task 1\n(Easy)", "Task 2\n(Medium)", "Task 3\n(Hard)"]
# Rows: Completed On-Time, Completed Late, In-Progress (50%), Failed w/ Progress
states = ["Completed\n(on time)", "Completed\n(late)", "In-Progress\n(50%)", "Failed\n(w/ progress)"]

# Credit fractions: [task1, task2, task3] for each state
data = np.array([
    [1.0,  1.0,  1.0 ],  # Completed on-time
    [0.6,  0.5,  0.4 ],  # Completed late
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
# Panel 3 — Step Penalties
# =====================================================================
ax3 = fig.add_subplot(2, 3, 3)
steps = np.arange(0, 21)
wait_penalty = steps * 0.01
fail_penalty = steps * 0.02

ax3.plot(steps, wait_penalty, "o-", color="#f39c12", label="WAIT penalties", linewidth=2, markersize=4)
ax3.plot(steps, fail_penalty, "s-", color="#e74c3c", label="Failed action penalties", linewidth=2, markersize=4)
ax3.fill_between(steps, wait_penalty, alpha=0.1, color="#f39c12")
ax3.fill_between(steps, fail_penalty, alpha=0.1, color="#e74c3c")
ax3.set_xlabel("Number of penalty actions", fontsize=12)
ax3.set_ylabel("Total penalty subtracted", fontsize=12)
ax3.set_title("Cumulative Step Penalties", fontsize=14, fontweight="bold")
ax3.legend(fontsize=10)
ax3.spines[["top", "right"]].set_visible(False)
ax3.set_xlim(0, 20)
ax3.set_ylim(0, 0.45)

# =====================================================================
# Panel 4 — Task 1 Score Breakdown (stacked bar)
# =====================================================================
ax4 = fig.add_subplot(2, 3, 4)

# Task 1 jobs: job_1(p3,dl10), job_2(p2), job_3(p2), job_4(p1), job_5(p2,dl15)
job_labels = ["job_1\n(Urgent)", "job_2\n(Normal)", "job_3\n(Normal)", "job_4\n(Low)", "job_5\n(Normal)"]
job_weights = [2.0, 1.0, 1.0, 0.5, 1.0]
total_w = sum(job_weights)
normalized = [w / total_w for w in job_weights]

# Show what fraction of total score each job contributes when completed on time
bars4 = ax4.bar(job_labels, normalized, color=[C_COMPLETED if i != 3 else "#95a5a6" for i in range(5)],
                edgecolor="white", linewidth=2)
for bar, w, nw in zip(bars4, job_weights, normalized):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{w}x\n({nw:.0%})", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax4.set_ylabel("Share of total score", fontsize=12)
ax4.set_title("Task 1 — Job Score Shares", fontsize=14, fontweight="bold")
ax4.set_ylim(0, 0.5)
ax4.axhline(y=1/5, color="gray", linestyle="--", alpha=0.3, label="Equal share")
ax4.spines[["top", "right"]].set_visible(False)

# =====================================================================
# Panel 5 — Task 3 Score Formula (visual breakdown)
# =====================================================================
ax5 = fig.add_subplot(2, 3, 5)

# Task 3: total_weight = 2+2+0.5+1 = 5.5, divisor = 5.5+0.5 = 6.0
components = [
    "job_critical\n(ABS, P3, dl20)",
    "job_petg_rush\n(PETG, P3, dl24)",
    "job_bulk\n(ABS, P1)",
    "job_filler\n(PETG, P2, dl30)",
    "Urgent Deadline\nBonus",
]
max_credits = [2.0, 2.0, 0.5, 1.0, 0.5]
divisor = 6.0
normalized_max = [c / divisor for c in max_credits]

bar_colors = ["#e74c3c", "#e74c3c", "#95a5a6", "#3498db", C_BONUS]
bars5 = ax5.barh(components[::-1], normalized_max[::-1], color=bar_colors[::-1],
                 edgecolor="white", linewidth=2, height=0.6)

for bar, val in zip(bars5, normalized_max[::-1]):
    ax5.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
             f"{val:.0%}", ha="left", va="center", fontsize=10, fontweight="bold")

ax5.set_xlabel("Max contribution to score", fontsize=12)
ax5.set_title("Task 3 — Score Components (divisor=6.0)", fontsize=14, fontweight="bold")
ax5.set_xlim(0, 0.45)
ax5.spines[["top", "right"]].set_visible(False)

# =====================================================================
# Panel 6 — Benchmark Scores Heatmap
# =====================================================================
ax6 = fig.add_subplot(2, 3, 6)

models = ["GPT-4", "GPT-5.2", "GPT-5.4", "GPT-4.1", "GPT-5.1", "GPT-4o", "GPT-3.5"]
scores_data = np.array([
    [0.990, 0.850, 0.920],
    [0.990, 0.820, 0.800],
    [0.980, 0.940, 0.567],
    [0.990, 0.840, 0.557],
    [0.990, 0.900, 0.473],
    [1.000, 0.737, 0.400],
    [0.260, 0.497, 0.000],
])

im = ax6.imshow(scores_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
ax6.set_xticks([0, 1, 2])
ax6.set_xticklabels(["Task 1\n(Easy)", "Task 2\n(Medium)", "Task 3\n(Hard)"], fontsize=10)
ax6.set_yticks(range(len(models)))
ax6.set_yticklabels(models, fontsize=10)

for i in range(len(models)):
    for j in range(3):
        color = "white" if scores_data[i, j] < 0.4 or scores_data[i, j] > 0.8 else "black"
        ax6.text(j, i, f"{scores_data[i, j]:.3f}", ha="center", va="center",
                 fontsize=10, fontweight="bold", color=color)

ax6.set_title("Baseline Benchmark Scores", fontsize=14, fontweight="bold")
cbar = plt.colorbar(im, ax=ax6, shrink=0.8)
cbar.set_label("Score", fontsize=10)

# ── Final layout ────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0.02, 1, 0.95])

# Add a footer with the clamping note
fig.text(0.5, 0.005,
         "Scores clamped to (0.001, 0.999) — strictly between 0 and 1 per OpenEnv spec.  "
         "Formula: score = clamp(earned / total_weight - step_penalty)",
         ha="center", fontsize=10, style="italic", color="#666")

plt.savefig("/Users/home-pc/Projects/PrintFarmEnv/reward_design.png", dpi=150, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print("Saved: reward_design.png")
plt.close()
