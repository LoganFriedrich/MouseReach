"""Quick graph of reach features for presentation."""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load data
df = pd.read_excel(Path(__file__).parent / 'unified_reaches_for_presentation.xlsx', sheet_name='All_Reaches')
print(f"Loaded {len(df)} reaches")
print(f"Columns: {list(df.columns)}")

# Create figure with reach feature comparisons
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Color map for outcomes
colors = {'retrieved': '#2ecc71', 'displaced_sa': '#f39c12', 'untouched': '#e74c3c', 'displaced_outside': '#9b59b6'}

# 1. Reach Extent Distribution
ax1 = axes[0, 0]
for outcome in ['retrieved', 'displaced_sa', 'untouched']:
    data = df[df['outcome'] == outcome]['max_extent_mm'].dropna()
    ax1.hist(data, bins=30, alpha=0.6, label=outcome, color=colors[outcome])
ax1.set_title('Reach Extent Distribution', fontsize=12, fontweight='bold')
ax1.set_xlabel('Max Extent (mm)')
ax1.set_ylabel('Count')
ax1.legend()

# 2. Reach Duration Distribution
ax2 = axes[0, 1]
for outcome in ['retrieved', 'displaced_sa', 'untouched']:
    data = df[df['outcome'] == outcome]['duration_sec'].dropna()
    ax2.hist(data, bins=30, alpha=0.6, label=outcome, color=colors[outcome])
ax2.set_title('Reach Duration Distribution', fontsize=12, fontweight='bold')
ax2.set_xlabel('Duration (seconds)')
ax2.set_ylabel('Count')
ax2.legend()

# 3. Extent vs Duration scatter
ax3 = axes[0, 2]
for outcome in ['retrieved', 'displaced_sa', 'untouched']:
    subset = df[df['outcome'] == outcome]
    ax3.scatter(subset['duration_sec'], subset['max_extent_mm'],
                alpha=0.3, label=outcome, color=colors[outcome], s=20)
ax3.set_title('Extent vs Duration', fontsize=12, fontweight='bold')
ax3.set_xlabel('Duration (seconds)')
ax3.set_ylabel('Max Extent (mm)')
ax3.legend()

# 4. Mean extent by outcome (bar chart with error bars)
ax4 = axes[1, 0]
extent_stats = df.groupby('outcome')['max_extent_mm'].agg(['mean', 'std', 'count'])
extent_stats = extent_stats.loc[['retrieved', 'displaced_sa', 'untouched']]
bars = ax4.bar(extent_stats.index, extent_stats['mean'],
               yerr=extent_stats['std']/np.sqrt(extent_stats['count']),
               color=[colors[o] for o in extent_stats.index], capsize=5)
ax4.set_title('Mean Reach Extent by Outcome', fontsize=12, fontweight='bold')
ax4.set_ylabel('Extent (mm) ± SEM')
ax4.tick_params(axis='x', rotation=15)

# 5. Mean duration by outcome
ax5 = axes[1, 1]
dur_stats = df.groupby('outcome')['duration_sec'].agg(['mean', 'std', 'count'])
dur_stats = dur_stats.loc[['retrieved', 'displaced_sa', 'untouched']]
bars = ax5.bar(dur_stats.index, dur_stats['mean'],
               yerr=dur_stats['std']/np.sqrt(dur_stats['count']),
               color=[colors[o] for o in dur_stats.index], capsize=5)
ax5.set_title('Mean Reach Duration by Outcome', fontsize=12, fontweight='bold')
ax5.set_ylabel('Duration (sec) ± SEM')
ax5.tick_params(axis='x', rotation=15)

# 6. Reaches per segment by outcome
ax6 = axes[1, 2]
# Count how many reaches per segment lead to each outcome (using is_causal_reach if available)
outcome_counts = df['outcome'].value_counts()
outcome_counts = outcome_counts[['retrieved', 'displaced_sa', 'untouched']]
wedges, texts, autotexts = ax6.pie(outcome_counts.values, labels=outcome_counts.index,
                                    colors=[colors[o] for o in outcome_counts.index],
                                    autopct='%1.1f%%', startangle=90)
ax6.set_title('Overall Outcome Distribution', fontsize=12, fontweight='bold')

plt.suptitle(f'Reach Feature Analysis (n={len(df):,} reaches, {df["animal"].nunique()} mice)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

# Save
out_path = Path(__file__).parent / 'reach_features.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_path}")

plt.show()
