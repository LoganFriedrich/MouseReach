"""Generate a single-page PI-friendly validation dashboard."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- DATA ---

# Reach detection funnel
funnel_labels = [
    "Human-annotated reaches",
    "Algorithm found",
    "Start frame correct",
    "End frame correct",
    "Both boundaries correct",
]
funnel_values = [2608, 2495, 2487, 2470, 2465]

# Outcome classification
outcome_labels = ["Displaced SA", "Untouched", "Retrieved", "Displaced Outside"]
outcome_correct = [169, 176, 48, 1]
outcome_total = [170, 177, 52, 1]
outcome_pcts = [c/t*100 for c, t in zip(outcome_correct, outcome_total)]

# Error rates
error_types = [
    "Missed reaches",
    "Spurious detections",
    "Boundary errors",
    "Wrong outcome",
    "Interaction frame >5 fr off",
]
error_counts = [113, 494, 30, 6, 25]
error_bases = [2608, 2608, 2608, 400, 212]
error_rates = [c/b*100 for c, b in zip(error_counts, error_bases)]

# --- FIGURE ---

fig, axes = plt.subplots(3, 1, figsize=(14, 18), facecolor='white',
                          gridspec_kw={'height_ratios': [1.2, 0.8, 0.8], 'hspace': 0.45})

fig.suptitle(
    "MouseReach Algorithm Validation Dashboard",
    fontsize=24, fontweight='bold', y=0.96
)
fig.text(
    0.5, 0.935,
    "Reach v5.3.0  |  Outcome v2.4.4  |  Segmentation v2.1.0  |  Evaluated 2026-02-16",
    ha='center', fontsize=13, color='#555555'
)

# ============================================================
# PANEL 1: Reach Detection Funnel
# ============================================================
ax1 = axes[0]
ax1.set_title("Reach Detection: How many real reaches survive each quality gate?",
              fontsize=16, fontweight='bold', pad=20, loc='left')

colors_funnel = ['#3B7DD8', '#4CAF50', '#4CAF50', '#4CAF50', '#2E7D32']

# Ghost bars (full 2608) behind
for i in range(1, len(funnel_values)):
    ax1.barh(i, 2608, color='#EEEEEE', height=0.6, zorder=0)

# Actual bars
ax1.barh(range(len(funnel_labels)), funnel_values,
         color=colors_funnel, height=0.6, edgecolor='white', linewidth=2, zorder=1)

ax1.set_yticks(range(len(funnel_labels)))
ax1.set_yticklabels(funnel_labels, fontsize=14)
ax1.invert_yaxis()
ax1.set_xlim(0, 3200)
ax1.set_xlabel("Number of reaches", fontsize=13)
ax1.tick_params(axis='x', labelsize=12)

# Labels inside the bars for contrast
for i, val in enumerate(funnel_values):
    pct = val / 2608 * 100
    ax1.text(val - 60, i, f"{val:,}", va='center', ha='right',
             fontsize=15, fontweight='bold', color='white', zorder=2)
    ax1.text(val + 40, i, f"{pct:.1f}%", va='center', ha='left',
             fontsize=15, fontweight='bold', color='#333333', zorder=2)

# Note about tolerance
ax1.text(0.99, 0.02, '"Correct" = within 2 frames (10 ms) of human annotation',
         transform=ax1.transAxes, ha='right', fontsize=11, color='#777777', style='italic')

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ============================================================
# PANEL 2: Outcome Classification
# ============================================================
ax2 = axes[1]
ax2.set_title("Pellet Outcome: Algorithm vs. human classification",
              fontsize=16, fontweight='bold', pad=20, loc='left')

y_pos = range(len(outcome_labels))
# Correct
ax2.barh(y_pos, outcome_pcts, color='#4CAF50', height=0.55, label='Correct', zorder=1)
# Wrong (stacked)
error_pcts_out = [100 - p for p in outcome_pcts]
ax2.barh(y_pos, error_pcts_out, left=outcome_pcts, color='#E53935', height=0.55, label='Wrong', zorder=1)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(outcome_labels, fontsize=14)
ax2.invert_yaxis()
ax2.set_xlim(0, 115)
ax2.set_xlabel("% correct", fontsize=13)
ax2.tick_params(axis='x', labelsize=12)

for i, (pct, corr, tot) in enumerate(zip(outcome_pcts, outcome_correct, outcome_total)):
    ax2.text(102, i, f"{pct:.1f}%   ({corr}/{tot})",
             va='center', fontsize=14, fontweight='bold', color='#333333')

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(loc='lower right', fontsize=12, framealpha=0.9)

# Overall note
ax2.text(0.99, 0.02, 'Overall: 394/400 correct (98.5%)  |  Interaction frame mean error: 12.7 frames',
         transform=ax2.transAxes, ha='right', fontsize=11, color='#777777', style='italic')

# ============================================================
# PANEL 3: Remaining Error Rates
# ============================================================
ax3 = axes[2]
ax3.set_title("What's still wrong: remaining error rates",
              fontsize=16, fontweight='bold', pad=20, loc='left')

y_pos_err = range(len(error_types))
bar_colors = ['#E53935' if r > 10 else '#FB8C00' if r > 3 else '#FDD835' for r in error_rates]

ax3.barh(y_pos_err, error_rates, color=bar_colors, height=0.55,
         edgecolor='white', linewidth=1, zorder=1)

ax3.set_yticks(y_pos_err)
ax3.set_yticklabels(error_types, fontsize=14)
ax3.invert_yaxis()
ax3.set_xlim(0, max(error_rates) * 1.5)
ax3.set_xlabel("Error rate (%)", fontsize=13)
ax3.tick_params(axis='x', labelsize=12)

for i, (rate, count, base) in enumerate(zip(error_rates, error_counts, error_bases)):
    ax3.text(rate + 0.5, i, f"{rate:.1f}%   ({count} of {base:,})",
             va='center', fontsize=14, fontweight='bold', color='#333333')

ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Caveat
ax3.text(0.99, 0.02,
         'Measured on training videos. Error rates on new videos may be higher.',
         transform=ax3.transAxes, ha='right', fontsize=11, color='#777777', style='italic')

# ============================================================
# SAVE
# ============================================================
output_path = Path(r"y:\2_Connectome\Behavior\MouseReach\ALGORITHM_VALIDATION_DASHBOARD.png")
fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved to: {output_path}")
plt.close()
