"""
Cross-group disharmony scatter: lateral (X) vs extension (Y) deviation
at 1wk Post-Injury for all groups on the same axes.

Uses pre-computed all_groups_disharmony_scores.csv (from all_groups_disharmony.py).
No recalculation -- just plotting.

Each group gets a different marker. Suspects highlighted.
Tests whether all groups share a common trend line (severity continuum).
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_connectome_root = SCRIPT_DIR
while os.path.basename(_connectome_root) != '2_Connectome' and os.path.dirname(_connectome_root) != _connectome_root:
    _connectome_root = os.path.dirname(_connectome_root)
OUTPUT_BASE = os.path.join(_connectome_root, 'Databases', 'figures', 'bodypart_tracking_quality')

DISHARMONY_CSV = os.path.join(OUTPUT_BASE, 'all_groups_disharmony_scores.csv')
SUSPECT_CSV = os.path.join(OUTPUT_BASE, 'suspect_animals.csv')

GROUP_STYLES = {
    'D': {'color': '#1f77b4', 'marker': 'o', 'label': 'D (Pyramidotomy)'},
    'G': {'color': '#2ca02c', 'marker': 's', 'label': 'G (Transection)'},
    'H': {'color': '#17becf', 'marker': 'D', 'label': 'H (Transection)'},
    'K': {'color': '#CC44CC', 'marker': '^', 'label': 'K (Contusion 70kD)'},
    'L': {'color': '#9467bd', 'marker': 'v', 'label': 'L (Contusion 50kD)'},
    'M': {'color': '#5588AA', 'marker': 'P', 'label': 'M (Contusion 60kD)'},
}


def main():
    print('Loading disharmony scores...')
    df = pd.read_csv(DISHARMONY_CSV)
    print('Total reaches: %d' % len(df))

    # Load suspects
    suspects = set()
    if os.path.exists(SUSPECT_CSV):
        suspects = set(pd.read_csv(SUSPECT_CSV)['animal'].values)

    # Compute per-animal median lateral and extension at 1wk Post
    phase = '1wk Post'
    phase_df = df[df['phase'] == phase]

    animal_stats = phase_df.groupby(['group', 'animal']).agg(
        lateral=('lateral', 'median'),
        extension=('extension', 'median'),
        disharmony=('disharmony', 'median'),
        n_reaches=('disharmony', 'count'),
    ).reset_index()

    # Drop animals with too few reaches at this phase
    animal_stats = animal_stats[animal_stats['n_reaches'] >= 5]
    animal_stats['suspect'] = animal_stats['animal'].isin(suspects)

    print('Animals with data at %s: %d' % (phase, len(animal_stats)))
    print('Groups represented: %s' % sorted(animal_stats['group'].unique()))

    # --- Figure 1: All groups on one scatter ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.patch.set_facecolor('white')

    # Panel 1: All groups, color by group
    ax = axes[0]
    all_lat, all_ext = [], []
    for group in sorted(animal_stats['group'].unique()):
        gdf = animal_stats[animal_stats['group'] == group]
        style = GROUP_STYLES.get(group, {'color': 'gray', 'marker': 'o', 'label': group})

        normals = gdf[~gdf['suspect']]
        sups = gdf[gdf['suspect']]

        if len(normals) > 0:
            ax.scatter(normals['lateral'], normals['extension'],
                       c=style['color'], marker=style['marker'], s=80,
                       label=style['label'], edgecolors='black', linewidth=0.5, zorder=5)
        if len(sups) > 0:
            ax.scatter(sups['lateral'], sups['extension'],
                       c=style['color'], marker=style['marker'], s=160,
                       edgecolors='red', linewidth=2.5, zorder=6)
            for _, row in sups.iterrows():
                ax.annotate(row['animal'], (row['lateral'], row['extension']),
                            fontsize=7, fontweight='bold', color='red',
                            xytext=(5, 5), textcoords='offset points')

        all_lat.extend(gdf['lateral'].values)
        all_ext.extend(gdf['extension'].values)

    # Trend line across all groups
    if len(all_lat) > 5:
        slope, intercept, r, p, se = stats.linregress(all_lat, all_ext)
        x_range = np.linspace(min(all_lat) - 0.2, max(all_lat) + 0.2, 100)
        ax.plot(x_range, slope * x_range + intercept, '--', color='gray', alpha=0.6,
                label='All-group trend (r=%.2f, p=%.3f)' % (r, p))

    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Lateral disharmony (z-scored, + = wider)', fontsize=11)
    ax.set_ylabel('Extension disharmony (z-scored, + = further)', fontsize=11)
    ax.set_title('%s: Lateral vs Extension Deviation\nAll Groups (red border = suspect)' % phase,
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')

    # Panel 2: Per-group boxplots of overall disharmony
    ax2 = axes[1]
    groups_ordered = ['D', 'G', 'H', 'K', 'L', 'M']
    groups_present = [g for g in groups_ordered if g in animal_stats['group'].values]
    box_data = []
    box_colors = []
    box_labels = []
    for group in groups_present:
        gdf = animal_stats[animal_stats['group'] == group]
        box_data.append(gdf['disharmony'].values)
        box_colors.append(GROUP_STYLES.get(group, {}).get('color', 'gray'))
        box_labels.append(GROUP_STYLES.get(group, {}).get('label', group))

    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay individual points
    for i, group in enumerate(groups_present):
        gdf = animal_stats[animal_stats['group'] == group]
        x_jitter = np.random.normal(i + 1, 0.08, len(gdf))
        normals = gdf[~gdf['suspect']]
        sups = gdf[gdf['suspect']]
        if len(normals) > 0:
            ax2.scatter(x_jitter[:len(normals)], normals['disharmony'].values,
                        c='black', s=20, zorder=5, alpha=0.7)
        if len(sups) > 0:
            ax2.scatter(x_jitter[len(normals):], sups['disharmony'].values,
                        c='red', s=40, zorder=6, edgecolors='black', linewidth=0.5)
            for j, (_, row) in enumerate(sups.iterrows()):
                ax2.annotate(row['animal'],
                             (x_jitter[len(normals) + j], row['disharmony']),
                             fontsize=7, fontweight='bold', color='red',
                             xytext=(5, 3), textcoords='offset points')

    ax2.set_ylabel('Overall disharmony (z-scored)', fontsize=11)
    ax2.set_title('%s: Disharmony by Group\n(red = suspect)' % phase,
                  fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    out = os.path.join(OUTPUT_BASE, 'all_groups_disharmony_scatter.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print('Saved: %s' % out)

    # --- Figure 2: Per-phase scatter for M only (matches existing disharmony_scores.png bottom panel) ---
    # This extends it to show all phases side by side
    m_df = df[df['group'] == 'M']
    phases_ordered = ['Pre-Injury', '1wk Post', '2-4wk Post', 'Post-Rehab']
    phases_present = [p for p in phases_ordered if p in m_df['phase'].unique()]

    fig2, axes2 = plt.subplots(1, len(phases_present), figsize=(6 * len(phases_present), 6))
    fig2.patch.set_facecolor('white')
    if len(phases_present) == 1:
        axes2 = [axes2]

    for i, phase_name in enumerate(phases_present):
        ax = axes2[i]
        pdf = m_df[m_df['phase'] == phase_name]
        astats = pdf.groupby('animal').agg(
            lateral=('lateral', 'median'),
            extension=('extension', 'median'),
        ).reset_index()
        astats['suspect'] = astats['animal'].isin(suspects)

        normals = astats[~astats['suspect']]
        sups = astats[astats['suspect']]

        ax.scatter(normals['lateral'], normals['extension'],
                   c='#4444FF', s=80, edgecolors='black', linewidth=0.5, label='Normal', zorder=5)
        ax.scatter(sups['lateral'], sups['extension'],
                   c='#FF4444', s=120, edgecolors='black', linewidth=0.5, label='Suspect', zorder=6)

        for _, row in astats.iterrows():
            ax.annotate(row['animal'], (row['lateral'], row['extension']),
                        fontsize=7,
                        fontweight='bold' if row['suspect'] else 'normal',
                        color='red' if row['suspect'] else 'blue',
                        xytext=(5, 5), textcoords='offset points')

        # Trend line
        if len(astats) > 3:
            slope, intercept, r, p, se = stats.linregress(astats['lateral'], astats['extension'])
            x_range = np.linspace(astats['lateral'].min() - 0.2, astats['lateral'].max() + 0.2, 100)
            ax.plot(x_range, slope * x_range + intercept, '--', color='gray', alpha=0.6)
            ax.set_title('%s\nr=%.2f, p=%.3f' % (phase_name, r, p), fontsize=11, fontweight='bold')
        else:
            ax.set_title(phase_name, fontsize=11, fontweight='bold')

        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Lateral (+ = wider)', fontsize=10)
        ax.set_ylabel('Extension (+ = further)', fontsize=10)
        if i == 0:
            ax.legend(fontsize=8)

    fig2.suptitle('Group M: Lateral vs Extension Disharmony Across Phases',
                  fontsize=13, fontweight='bold')
    plt.tight_layout()
    out2 = os.path.join(OUTPUT_BASE, 'M_disharmony_scatter_by_phase.png')
    plt.savefig(out2, dpi=200, bbox_inches='tight')
    plt.close()
    print('Saved: %s' % out2)


if __name__ == '__main__':
    main()
