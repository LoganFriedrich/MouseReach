"""
DLC Tracking Quality vs ASPA Scoring Discrepancy — Statistical Report

Produces publication-quality diagnostic figures examining the relationship between
DLC pellet tracking metrics and ASPA-vs-Manual scoring discrepancy.

NOTE: Ceiling analysis (pellet_ceiling_analysis.py) showed ALL animals achieve
0.999 pellet likelihood at the 95th percentile — including flagged animals.
This means DLC tracks the pellet well when present. Low mean pellet likelihood
reflects genuine pellet absence (eaten, displaced, or occluded), NOT tracking
failure. The correlation between tracking metrics and scoring discrepancy
is real but the mechanism is NOT "poor tracking causes misclassification."

Figures produced:
  1. Group-level pellet tracking quality (K vs L vs M) — violin + box
  2. Within-Group-L: flagged vs unflagged animals — paired metrics
  3. Per-animal tracking quality vs scoring discrepancy — regression
  4. ASPA classification mechanism diagram — how pellet disappearance → "retrieved"
  5. Per-animal heatmap of session-level pellet dropout rates
  6. Effect size summary — forest plot of all comparisons

Statistical methods:
  - Mann-Whitney U (non-parametric group comparisons)
  - Spearman correlation (tracking quality vs discrepancy)
  - Cohen's d effect sizes
  - Bootstrap 95% CIs
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import os
import re
import warnings
from pathlib import Path
from scipy import stats
from collections import defaultdict

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_connectome_root = SCRIPT_DIR
while os.path.basename(_connectome_root) != '2_Connectome' and os.path.dirname(_connectome_root) != _connectome_root:
    _connectome_root = os.path.dirname(_connectome_root)

MOUSEDB_DIR = os.path.join(_connectome_root, 'MouseDB')
OUTPUT_DIR = os.path.join(MOUSEDB_DIR, 'figures', 'dlc_tracking_quality')
os.makedirs(OUTPUT_DIR, exist_ok=True)
EXPORT_DIR = os.path.join(MOUSEDB_DIR, 'exports')
os.makedirs(EXPORT_DIR, exist_ok=True)

DLC_BASE = Path(r"X:\! DLC Output\Analyzed")
UCSF_DATA_PATH = os.path.join(
    r'C:\Users\friedrichl\OneDrive - Marquette University\Blackmore Lab Notes - Sharepoint',
    r'3 Lab Projects\Automated single pellet apparatus\!UCSF_Collab',
    r'May2025_Uploads\ODC Uploads\Session_Data.csv'
)

CONTUSION_GROUPS = ['K', 'L', 'M']
FLAGGED_ANIMALS = {'L02', 'L10', 'L12', 'L13'}

# Style
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.spines.top': False, 'axes.spines.right': False,
})

GROUP_COLORS = {'K': '#e377c2', 'L': '#2ca02c', 'M': '#ff7f0e'}
GROUP_LABELS = {'K': 'K (70kd)', 'L': 'L (50kd)', 'M': 'M (60kd)'}

# ============================================================================
# DATA LOADING
# ============================================================================

def extract_animal_id(filename):
    match = re.search(r'\d{8}_([A-Z]\d{2})_', filename)
    return match.group(1) if match else None

def extract_date(filename):
    match = re.search(r'(\d{8})_', filename)
    return match.group(1) if match else None

def load_dlc_csv(csv_path):
    """Load DLC CSV, return pellet and pillar likelihood arrays."""
    try:
        df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
        scorer = df.columns.get_level_values(0)[0]
        pellet_lik = df[(scorer, 'Pellet', 'likelihood')].values.astype(float)
        pillar_lik = df[(scorer, 'Pillar', 'likelihood')].values.astype(float)
        return pellet_lik, pillar_lik
    except Exception:
        return None, None

def compute_tracking_metrics(lik):
    """Compute per-session tracking quality metrics."""
    if lik is None or len(lik) == 0:
        return None
    below = lik < 0.5
    # Dropout events: consecutive runs of frames < 0.5
    dropout_lens = []
    in_drop = False
    start = 0
    for i, b in enumerate(below):
        if b and not in_drop:
            in_drop = True; start = i
        elif not b and in_drop:
            in_drop = False; dropout_lens.append(i - start)
    if in_drop:
        dropout_lens.append(len(below) - start)

    return {
        'mean_lik': np.mean(lik),
        'median_lik': np.median(lik),
        'pct_below_05': 100 * np.mean(below),
        'pct_below_03': 100 * np.mean(lik < 0.3),
        'pct_above_095': 100 * np.mean(lik > 0.95),
        'n_dropouts': len(dropout_lens),
        'mean_dropout_len': np.mean(dropout_lens) if dropout_lens else 0,
        'max_dropout_len': np.max(dropout_lens) if dropout_lens else 0,
        'total_frames': len(lik),
    }

def load_all_tracking_data(max_per_group=200):
    """Load DLC tracking metrics for all contusion groups."""
    print("Loading DLC tracking data...")
    rows = []
    for grp in CONTUSION_GROUPS:
        grp_path = DLC_BASE / grp / 'Post-Processing'
        if not grp_path.exists():
            grp_path = DLC_BASE / grp / 'Multi-Animal'
        csvs = sorted(grp_path.glob('*DLC*.csv'))
        print(f"  Group {grp}: {len(csvs)} CSVs found at {grp_path}")

        # Ensure all flagged animals fully included
        flagged_csvs = [c for c in csvs if extract_animal_id(c.name) in FLAGGED_ANIMALS]
        other_csvs = [c for c in csvs if extract_animal_id(c.name) not in FLAGGED_ANIMALS]
        if len(other_csvs) > max_per_group:
            np.random.seed(42)
            other_csvs = list(np.random.choice(other_csvs, max_per_group, replace=False))
        to_process = flagged_csvs + other_csvs

        for csv_file in to_process:
            aid = extract_animal_id(csv_file.name)
            date = extract_date(csv_file.name)
            if not aid:
                continue
            pel, pil = load_dlc_csv(csv_file)
            if pel is None:
                continue
            pm = compute_tracking_metrics(pel)
            pim = compute_tracking_metrics(pil)
            if pm is None:
                continue
            rows.append({
                'group': grp, 'animal_id': aid, 'date': date,
                'filename': csv_file.name,
                'is_flagged': aid in FLAGGED_ANIMALS,
                **{f'pellet_{k}': v for k, v in pm.items()},
                'pillar_mean_lik': pim['mean_lik'] if pim else np.nan,
                'pillar_pct_below_05': pim['pct_below_05'] if pim else np.nan,
            })

    df = pd.DataFrame(rows)
    print(f"  Total sessions loaded: {len(df)}")
    return df

def load_ucsf_scoring_discrepancy():
    """Load UCSF session data and compute per-animal Manual-Video discrepancy."""
    df = pd.read_csv(UCSF_DATA_PATH)
    for col in ['Video_Retrieved', 'Manual_Retrieved', 'Video_Contacted', 'Manual_Contacted',
                 'Video_Displaced', 'Manual_Displaced']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[df['Tray_Type'] == 'Pillar'].copy()

    # Per-animal totals
    animal_totals = df.groupby(['SubjectID', 'Group']).agg(
        manual_retrieved=('Manual_Retrieved', 'sum'),
        video_retrieved=('Video_Retrieved', lambda x: x.dropna().sum()),
        manual_contacted=('Manual_Contacted', 'sum'),
        video_contacted=('Video_Contacted', lambda x: x.dropna().sum()),
        manual_displaced=('Manual_Displaced', 'sum'),
        video_displaced=('Video_Displaced', lambda x: x.dropna().sum()),
        n_trays=('Tray_ID', 'count'),
    ).reset_index()

    # Discrepancy: (Video - Manual) / Manual * 100
    for metric in ['retrieved', 'contacted', 'displaced']:
        m = animal_totals[f'manual_{metric}']
        v = animal_totals[f'video_{metric}']
        animal_totals[f'{metric}_discrepancy_pct'] = np.where(m > 0, (v - m) / m * 100, np.nan)
        animal_totals[f'{metric}_abs_diff'] = v - m

    return animal_totals

# ============================================================================
# STATISTICAL HELPERS
# ============================================================================

def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * np.std(a, ddof=1)**2 + (nb - 1) * np.std(b, ddof=1)**2) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0

def bootstrap_ci(arr, n_boot=5000, ci=95):
    """Bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(42)
    means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(means, (100 - ci) / 2)
    hi = np.percentile(means, 100 - (100 - ci) / 2)
    return lo, hi

def mw_test(a, b):
    """Mann-Whitney U test, returns (U, p, effect_size_r)."""
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan
    u_stat, p_val = stats.mannwhitneyu(a, b, alternative='two-sided')
    # rank-biserial r = 1 - 2U / (n1*n2)
    r_effect = 1 - (2 * u_stat) / (len(a) * len(b))
    return u_stat, p_val, r_effect

# ============================================================================
# FIGURES
# ============================================================================

def fig1_group_comparison(df_tracking):
    """Figure 1: Pellet tracking quality across contusion groups — violin + box."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Pellet Tracking Quality by Injury Group\n(DLC Pellet Likelihood)', fontsize=14, fontweight='bold')

    metrics = [
        ('pellet_mean_lik', 'Mean Pellet Likelihood', False),
        ('pellet_pct_below_05', '% Frames with Pellet Likelihood < 0.5\n(Invisible to ASPA)', True),
        ('pellet_n_dropouts', 'Dropout Events per Session\n(Consecutive frames < 0.5)', True),
    ]

    for ax_i, (col, ylabel, higher_is_worse) in enumerate(metrics):
        ax = axes[ax_i]
        data_by_group = []
        positions = []
        for gi, grp in enumerate(CONTUSION_GROUPS):
            vals = df_tracking[df_tracking['group'] == grp][col].dropna().values
            data_by_group.append(vals)
            positions.append(gi)

        # Violin
        parts = ax.violinplot(data_by_group, positions=positions, showmeans=False,
                              showmedians=False, showextrema=False)
        for gi, pc in enumerate(parts['bodies']):
            pc.set_facecolor(GROUP_COLORS[CONTUSION_GROUPS[gi]])
            pc.set_alpha(0.3)

        # Box
        bp = ax.boxplot(data_by_group, positions=positions, widths=0.3, patch_artist=True,
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5),
                        flierprops=dict(markersize=3, alpha=0.3))
        for gi, patch in enumerate(bp['boxes']):
            patch.set_facecolor(GROUP_COLORS[CONTUSION_GROUPS[gi]])
            patch.set_alpha(0.6)

        # Stats: pairwise Mann-Whitney
        comparisons = [(0, 1, 'K vs L'), (1, 2, 'L vs M'), (0, 2, 'K vs M')]
        y_max = max(np.percentile(d, 95) for d in data_by_group if len(d) > 0)
        y_range = y_max - min(np.percentile(d, 5) for d in data_by_group if len(d) > 0)

        for ci, (i, j, label) in enumerate(comparisons):
            u, p, r_eff = mw_test(data_by_group[i], data_by_group[j])
            if np.isnan(p):
                continue
            stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            y_bar = y_max + y_range * (0.08 + ci * 0.12)
            ax.plot([i, j], [y_bar, y_bar], 'k-', linewidth=1.2)
            ax.text((i + j) / 2, y_bar + y_range * 0.01, f'{stars}\np={p:.4f}',
                    ha='center', va='bottom', fontsize=8)

        ax.set_xticks(positions)
        ax.set_xticklabels([f'{GROUP_LABELS[g]}\n(n={len(data_by_group[i])})' for i, g in enumerate(CONTUSION_GROUPS)])
        ax.set_ylabel(ylabel)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig1_group_tracking_quality.png')
    plt.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out}")


def fig2_flagged_vs_unflagged(df_tracking):
    """Figure 2: Within Group L — flagged vs unflagged animals."""
    df_l = df_tracking[df_tracking['group'] == 'L'].copy()
    flagged = df_l[df_l['is_flagged']]
    unflagged = df_l[~df_l['is_flagged']]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Group L (50kd): Flagged Animals (L02, L10, L12, L13) vs Rest\n'
                 'Flagged = animals with >25% Video-Manual scoring discrepancy',
                 fontsize=13, fontweight='bold')

    metrics = [
        ('pellet_mean_lik', 'Mean Pellet Likelihood'),
        ('pellet_pct_below_05', '% Frames < 0.5 Likelihood'),
        ('pellet_pct_below_03', '% Frames < 0.3 Likelihood'),
        ('pellet_n_dropouts', 'Dropout Events / Session'),
        ('pellet_mean_dropout_len', 'Mean Dropout Length (frames)'),
        ('pellet_max_dropout_len', 'Max Dropout Length (frames)'),
    ]

    flag_color = '#d62728'
    ok_color = '#2ca02c'

    for ax_i, (col, ylabel) in enumerate(metrics):
        ax = axes[ax_i // 3, ax_i % 3]
        f_vals = flagged[col].dropna().values
        u_vals = unflagged[col].dropna().values

        # Paired violin + box
        data = [f_vals, u_vals]
        parts = ax.violinplot(data, positions=[0, 1], showmeans=False,
                              showmedians=False, showextrema=False)
        for pi, (pc, color) in enumerate(zip(parts['bodies'], [flag_color, ok_color])):
            pc.set_facecolor(color)
            pc.set_alpha(0.25)

        bp = ax.boxplot(data, positions=[0, 1], widths=0.4, patch_artist=True,
                        medianprops=dict(color='black', linewidth=2))
        bp['boxes'][0].set_facecolor(flag_color); bp['boxes'][0].set_alpha(0.5)
        bp['boxes'][1].set_facecolor(ok_color); bp['boxes'][1].set_alpha(0.5)

        # Stats
        u, p, r_eff = mw_test(f_vals, u_vals)
        d = cohens_d(f_vals, u_vals)
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.set_title(f'{ylabel}\nU={u:.0f}, p={p:.4f} {stars}, d={d:.2f}', fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f'Flagged\n(n={len(f_vals)})', f'Unflagged\n(n={len(u_vals)})'])

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig2_flagged_vs_unflagged.png')
    plt.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out}")


def fig3_tracking_vs_discrepancy(df_tracking, df_scoring):
    """Figure 3: Per-animal tracking quality vs scoring discrepancy — scatter + regression."""
    # Average tracking quality per animal
    animal_tracking = df_tracking.groupby('animal_id').agg(
        mean_pellet_lik=('pellet_mean_lik', 'mean'),
        mean_pct_below_05=('pellet_pct_below_05', 'mean'),
        n_sessions=('filename', 'count'),
        group=('group', 'first'),
    ).reset_index()

    # Merge with scoring discrepancy
    merged = animal_tracking.merge(
        df_scoring[['SubjectID', 'Group', 'retrieved_discrepancy_pct', 'contacted_discrepancy_pct',
                     'manual_retrieved', 'video_retrieved']],
        left_on='animal_id', right_on='SubjectID', how='inner'
    )

    if len(merged) < 3:
        print("  WARNING: Not enough merged data for fig3")
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Pellet Visibility Metrics vs Scoring Discrepancy\n'
                 'Each point = one animal (averaged across all sessions)',
                 fontsize=14, fontweight='bold')

    # Panel A: pellet dropout rate vs retrieved discrepancy
    ax = axes[0]
    for _, row in merged.iterrows():
        color = GROUP_COLORS.get(row['group'], '#888')
        marker = 'D' if row['animal_id'] in FLAGGED_ANIMALS else 'o'
        size = 120 if row['animal_id'] in FLAGGED_ANIMALS else 60
        ax.scatter(row['mean_pct_below_05'], row['retrieved_discrepancy_pct'],
                  color=color, marker=marker, s=size, edgecolor='black', linewidth=0.5,
                  zorder=5 if marker == 'D' else 3, alpha=0.8)
        if row['animal_id'] in FLAGGED_ANIMALS:
            ax.annotate(row['animal_id'], (row['mean_pct_below_05'], row['retrieved_discrepancy_pct']),
                       fontsize=7, ha='left', va='bottom', xytext=(4, 4), textcoords='offset points')

    # Spearman correlation
    x = merged['mean_pct_below_05'].values
    y = merged['retrieved_discrepancy_pct'].dropna().values
    mask = ~np.isnan(y) & ~np.isnan(x[:len(y)])
    if mask.sum() > 3:
        rho, p_rho = stats.spearmanr(x[:len(y)][mask], y[mask])
        # Regression line
        z = np.polyfit(x[:len(y)][mask], y[mask], 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), 'k--', linewidth=1.5, alpha=0.6)
        ax.text(0.05, 0.95, f'Spearman rho={rho:.3f}\np={p_rho:.4f}',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('Mean % Frames with Pellet Likelihood < 0.5')
    ax.set_ylabel('Retrieved Discrepancy (Video - Manual) / Manual %')
    ax.set_title('A) Pellet Invisibility Rate vs Retrieved Over-counting')
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='-', alpha=0.5)

    # Panel B: pellet dropout vs contacted discrepancy
    ax = axes[1]
    for _, row in merged.iterrows():
        color = GROUP_COLORS.get(row['group'], '#888')
        marker = 'D' if row['animal_id'] in FLAGGED_ANIMALS else 'o'
        size = 120 if row['animal_id'] in FLAGGED_ANIMALS else 60
        ax.scatter(row['mean_pct_below_05'], row['contacted_discrepancy_pct'],
                  color=color, marker=marker, s=size, edgecolor='black', linewidth=0.5,
                  zorder=5 if marker == 'D' else 3, alpha=0.8)

    y2 = merged['contacted_discrepancy_pct'].values
    mask2 = ~np.isnan(y2) & ~np.isnan(x[:len(y2)])
    if mask2.sum() > 3:
        rho2, p_rho2 = stats.spearmanr(x[:len(y2)][mask2], y2[mask2])
        z2 = np.polyfit(x[:len(y2)][mask2], y2[mask2], 1)
        ax.plot(x_line, np.polyval(z2, x_line), 'k--', linewidth=1.5, alpha=0.6)
        ax.text(0.05, 0.95, f'Spearman rho={rho2:.3f}\np={p_rho2:.4f}',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('Mean % Frames with Pellet Likelihood < 0.5')
    ax.set_ylabel('Contacted Discrepancy (Video - Manual) / Manual %')
    ax.set_title('B) Pellet Invisibility Rate vs Contacted Discrepancy')
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='-', alpha=0.5)

    # Panel C: Absolute counts — Manual vs Video retrieved
    ax = axes[2]
    for _, row in merged.iterrows():
        color = GROUP_COLORS.get(row['group'], '#888')
        marker = 'D' if row['animal_id'] in FLAGGED_ANIMALS else 'o'
        size = 120 if row['animal_id'] in FLAGGED_ANIMALS else 60
        ax.scatter(row['manual_retrieved'], row['video_retrieved'],
                  color=color, marker=marker, s=size, edgecolor='black', linewidth=0.5,
                  zorder=5 if marker == 'D' else 3, alpha=0.8)
        if row['animal_id'] in FLAGGED_ANIMALS:
            ax.annotate(row['animal_id'], (row['manual_retrieved'], row['video_retrieved']),
                       fontsize=7, ha='left', va='bottom', xytext=(4, 4), textcoords='offset points')

    max_val = max(merged['manual_retrieved'].max(), merged['video_retrieved'].max()) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5, label='Perfect agreement')
    ax.set_xlabel('Manual Retrieved (total pellets)')
    ax.set_ylabel('Video/ASPA Retrieved (total count)')
    ax.set_title('C) Manual vs ASPA Retrieved Counts')
    ax.legend(fontsize=8)

    # Shared legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=GROUP_COLORS['K'],
               markersize=8, markeredgecolor='black', label='K (70kd)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=GROUP_COLORS['L'],
               markersize=8, markeredgecolor='black', label='L (50kd)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=GROUP_COLORS['M'],
               markersize=8, markeredgecolor='black', label='M (60kd)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
               markersize=10, markeredgecolor='black', label='Flagged (>25% discrepancy)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=9,
              bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    out = os.path.join(OUTPUT_DIR, 'fig3_tracking_vs_discrepancy.png')
    plt.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out}")


def fig4_aspa_mechanism(df_tracking):
    """Figure 4: ASPA classification mechanism — how pellet dropout causes false 'retrieved'."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("ASPA Classification Rule & Pellet Likelihood Dynamics\n"
                 "How pellet disappearance (real or apparent) drives 'Retrieved' classification",
                 fontsize=13, fontweight='bold')

    # Panel A: Simulated decision diagram
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('A) ASPA Classification Decision Tree', fontsize=12, fontweight='bold')

    # Decision boxes
    boxes = [
        (5, 9.2, 'Swipe detected\n(hand crosses threshold)', '#ddd', 10),
        (5, 7.5, 'Pellet visible at start?\n(likelihood > 0.5)', '#ffe0b2', 10),
        (2, 5.8, 'NO → "No Pellet"\n(excluded)', '#bbdefb', 4),
        (8, 5.8, 'YES → Check end...', '#c8e6c9', 4),
        (5.5, 4.0, 'Pellet visible at end?\n(likelihood > 0.5)', '#ffe0b2', 10),
        (2, 2.2, 'NO → "RETRIEVED"\n(pellet disappeared)', '#ef9a9a', 4),
        (8, 2.2, 'YES + moved off pillar\n→ "DISPLACED"', '#c8e6c9', 4),
        (8, 0.5, 'YES + still on pillar\n→ "MISSED"', '#bbdefb', 4),
    ]

    for x, y, text, color, width_mult in boxes:
        w = 0.18 * width_mult
        h = 0.6
        rect = plt.Rectangle((x - w, y - h/2), 2*w, h, facecolor=color,
                              edgecolor='black', linewidth=1.5, transform=ax.transData)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=7.5, fontweight='bold')

    # Arrows
    arrows = [
        (5, 8.9, 5, 8.05),    # start → visible?
        (3.2, 7.2, 2, 6.4),   # no pellet
        (6.8, 7.2, 8, 6.4),   # yes
        (8, 5.5, 5.5, 4.6),   # → end check
        (3.7, 3.7, 2, 2.8),   # no end → retrieved
        (7.3, 3.7, 8, 2.8),   # yes end → displaced
        (8, 1.9, 8, 1.1),     # still on pillar → missed
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Highlight the key classification path
    ax.annotate('KEY PATH:\nPellet invisible at end\n→ classified "RETRIEVED"\n(real or occluded)',
                xy=(2, 2.2), xytext=(0.3, 0.5),
                fontsize=8, fontweight='bold', color='#1565C0',
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2),
                bbox=dict(boxstyle='round', facecolor='#BBDEFB', alpha=0.9))

    # Panel B: Distribution of end-of-swipe pellet likelihood by group
    ax = axes[1]
    ax.set_title('B) Pellet Visibility Distribution\n'
                 '(Frames where pellet likelihood < 0.5 — invisible to ASPA)',
                 fontsize=11, fontweight='bold')

    # Histogram of per-session "% frames below 0.5" for each group
    for grp in CONTUSION_GROUPS:
        vals = df_tracking[df_tracking['group'] == grp]['pellet_pct_below_05'].dropna().values
        ax.hist(vals, bins=30, alpha=0.4, color=GROUP_COLORS[grp],
                label=f'{GROUP_LABELS[grp]} (n={len(vals)}, mean={np.mean(vals):.1f}%)',
                density=True, edgecolor='none')
        # KDE
        if len(vals) > 5:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(vals, bw_method=0.3)
            x_kde = np.linspace(0, 100, 200)
            ax.plot(x_kde, kde(x_kde), color=GROUP_COLORS[grp], linewidth=2)

    ax.axvline(x=50, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
              label='50% threshold (majority invisible)')
    ax.set_xlabel('% Frames with Pellet Likelihood < 0.5 (per session)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig4_aspa_mechanism.png')
    plt.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out}")


def fig5_per_animal_heatmap(df_tracking):
    """Figure 5: Per-animal per-session pellet dropout heatmap for Group L."""
    df_l = df_tracking[df_tracking['group'] == 'L'].copy()
    df_l['date_str'] = df_l['date']

    # Pivot: animal × date → pellet_pct_below_05
    # Average across positions within same date
    pivot_data = df_l.groupby(['animal_id', 'date_str'])['pellet_pct_below_05'].mean().reset_index()
    pivot = pivot_data.pivot(index='animal_id', columns='date_str', values='pellet_pct_below_05')

    # Sort: flagged animals first, then by mean dropout rate
    animal_order = []
    for aid in sorted(FLAGGED_ANIMALS):
        if aid in pivot.index:
            animal_order.append(aid)
    remaining = [a for a in pivot.index if a not in FLAGGED_ANIMALS]
    remaining.sort(key=lambda a: pivot.loc[a].mean(), reverse=True)
    animal_order.extend(remaining)
    pivot = pivot.loc[[a for a in animal_order if a in pivot.index]]

    fig, ax = plt.subplots(figsize=(20, max(6, len(pivot) * 0.4 + 2)))
    fig.suptitle('Group L: Per-Animal Per-Session Pellet Dropout Rate\n'
                 '(% frames with pellet likelihood < 0.5 — higher = worse tracking)',
                 fontsize=13, fontweight='bold')

    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=80,
                   interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label='% Frames Below 0.5')

    ax.set_yticks(range(len(pivot.index)))
    ylabels = []
    for a in pivot.index:
        flag = ' ***' if a in FLAGGED_ANIMALS else ''
        ylabels.append(f'{a}{flag}')
    ax.set_yticklabels(ylabels, fontsize=9)

    # Highlight flagged rows
    for i, a in enumerate(pivot.index):
        if a in FLAGGED_ANIMALS:
            ax.get_yticklabels()[i].set_fontweight('bold')
            ax.get_yticklabels()[i].set_color('#d32f2f')

    ax.set_xticks(range(len(pivot.columns)))
    date_labels = [f'{d[4:6]}/{d[6:8]}' for d in pivot.columns]
    ax.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=7)
    ax.set_xlabel('Session Date (MM/DD)')
    ax.set_ylabel('Animal ID')

    # Add separator between flagged and unflagged
    n_flagged = sum(1 for a in pivot.index if a in FLAGGED_ANIMALS)
    if n_flagged > 0 and n_flagged < len(pivot.index):
        ax.axhline(y=n_flagged - 0.5, color='red', linewidth=2, linestyle='--')
        ax.text(-0.5, n_flagged - 0.5, 'Flagged ↑', fontsize=8, color='red',
                fontweight='bold', va='center', ha='right')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig5_animal_session_heatmap.png')
    plt.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out}")


def fig6_effect_size_forest(df_tracking, df_scoring):
    """Figure 6: Forest plot of all statistical comparisons."""
    # Compute all comparisons
    comparisons = []

    # Group L vs K
    lvals = df_tracking[df_tracking['group'] == 'L']['pellet_pct_below_05'].dropna().values
    kvals = df_tracking[df_tracking['group'] == 'K']['pellet_pct_below_05'].dropna().values
    mvals = df_tracking[df_tracking['group'] == 'M']['pellet_pct_below_05'].dropna().values
    u, p, r = mw_test(lvals, kvals)
    d = cohens_d(lvals, kvals)
    comparisons.append(('L vs K: Pellet dropout rate', d, p, len(lvals), len(kvals)))

    u, p, r = mw_test(lvals, mvals)
    d = cohens_d(lvals, mvals)
    comparisons.append(('L vs M: Pellet dropout rate', d, p, len(lvals), len(mvals)))

    # Flagged vs unflagged within L
    df_l = df_tracking[df_tracking['group'] == 'L']
    fvals = df_l[df_l['is_flagged']]['pellet_pct_below_05'].dropna().values
    uvals = df_l[~df_l['is_flagged']]['pellet_pct_below_05'].dropna().values
    u, p, r = mw_test(fvals, uvals)
    d = cohens_d(fvals, uvals)
    comparisons.append(('Flagged vs Unflagged (L): Dropout rate', d, p, len(fvals), len(uvals)))

    # Mean likelihood comparisons
    for metric, label in [('pellet_mean_lik', 'Mean pellet likelihood'),
                          ('pellet_n_dropouts', 'Dropout events'),
                          ('pellet_mean_dropout_len', 'Mean dropout length')]:
        fv = df_l[df_l['is_flagged']][metric].dropna().values
        uv = df_l[~df_l['is_flagged']][metric].dropna().values
        u, p, r = mw_test(fv, uv)
        d = cohens_d(fv, uv)
        comparisons.append((f'Flagged vs Unflagged (L): {label}', d, p, len(fv), len(uv)))

    # Scoring discrepancy for L vs other groups
    l_scoring = df_scoring[df_scoring['Group'] == 'L']['retrieved_discrepancy_pct'].dropna().values
    k_scoring = df_scoring[df_scoring['Group'] == 'K']['retrieved_discrepancy_pct'].dropna().values
    m_scoring = df_scoring[df_scoring['Group'] == 'M']['retrieved_discrepancy_pct'].dropna().values
    if len(l_scoring) > 1 and len(k_scoring) > 1:
        u, p, r = mw_test(l_scoring, k_scoring)
        d = cohens_d(l_scoring, k_scoring)
        comparisons.append(('L vs K: Retrieved scoring discrepancy', d, p, len(l_scoring), len(k_scoring)))
    if len(l_scoring) > 1 and len(m_scoring) > 1:
        u, p, r = mw_test(l_scoring, m_scoring)
        d = cohens_d(l_scoring, m_scoring)
        comparisons.append(('L vs M: Retrieved scoring discrepancy', d, p, len(l_scoring), len(m_scoring)))

    # --- Forest plot ---
    fig, ax = plt.subplots(figsize=(12, max(5, len(comparisons) * 0.6 + 2)))
    fig.suptitle("Effect Size Summary: All Comparisons\n(Cohen's d with 95% CI from bootstrap)",
                 fontsize=13, fontweight='bold')

    y_positions = list(range(len(comparisons)))
    for yi, (label, d_val, p_val, n1, n2) in enumerate(comparisons):
        # Bootstrap CI for Cohen's d
        # Approximate: SE(d) ~ sqrt(1/n1 + 1/n2 + d^2/(2*(n1+n2)))
        se_d = np.sqrt(1/n1 + 1/n2 + d_val**2 / (2*(n1+n2)))
        ci_lo = d_val - 1.96 * se_d
        ci_hi = d_val + 1.96 * se_d

        color = '#d32f2f' if p_val < 0.05 else '#888888'
        stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'

        ax.plot([ci_lo, ci_hi], [yi, yi], color=color, linewidth=2.5, solid_capstyle='round')
        ax.scatter(d_val, yi, color=color, s=100, zorder=5, edgecolor='black', linewidth=0.5)
        ax.text(max(ci_hi, d_val) + 0.1, yi, f'd={d_val:.2f} {stars} (p={p_val:.4f})',
                va='center', fontsize=9, color=color)

    ax.axvline(x=0, color='black', linewidth=1, linestyle='-')
    ax.axvline(x=0.2, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
    ax.axvline(x=0.5, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
    ax.axvline(x=0.8, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([c[0] for c in comparisons], fontsize=9)
    ax.set_xlabel("Cohen's d (effect size)")

    # Effect size guidelines
    ax.text(0.2, -0.8, 'Small', ha='center', fontsize=7, color='gray', style='italic')
    ax.text(0.5, -0.8, 'Medium', ha='center', fontsize=7, color='gray', style='italic')
    ax.text(0.8, -0.8, 'Large', ha='center', fontsize=7, color='gray', style='italic')

    ax.invert_yaxis()
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig6_effect_size_forest.png')
    plt.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out}")


def export_statistics(df_tracking, df_scoring):
    """Export summary statistics CSV for downstream use."""
    # Per-animal summary
    animal_summary = df_tracking.groupby(['group', 'animal_id', 'is_flagged']).agg(
        n_sessions=('filename', 'count'),
        mean_pellet_lik=('pellet_mean_lik', 'mean'),
        sd_pellet_lik=('pellet_mean_lik', 'std'),
        mean_pct_below_05=('pellet_pct_below_05', 'mean'),
        sd_pct_below_05=('pellet_pct_below_05', 'std'),
        mean_n_dropouts=('pellet_n_dropouts', 'mean'),
        mean_dropout_len=('pellet_mean_dropout_len', 'mean'),
        mean_pillar_lik=('pillar_mean_lik', 'mean'),
    ).reset_index()

    # Merge scoring
    merged = animal_summary.merge(
        df_scoring[['SubjectID', 'retrieved_discrepancy_pct', 'contacted_discrepancy_pct',
                     'manual_retrieved', 'video_retrieved', 'manual_contacted', 'video_contacted']],
        left_on='animal_id', right_on='SubjectID', how='left'
    )

    out = os.path.join(EXPORT_DIR, 'dlc_tracking_quality_summary.csv')
    merged.to_csv(out, index=False)
    print(f"  Exported: {out}")

    # Session-level export
    session_out = os.path.join(EXPORT_DIR, 'dlc_tracking_quality_sessions.csv')
    df_tracking.to_csv(session_out, index=False)
    print(f"  Exported: {session_out}")

    return merged


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("DLC TRACKING QUALITY vs ASPA SCORING DISCREPANCY -- STATISTICAL REPORT")
    print("=" * 80)

    # Load data
    df_tracking = load_all_tracking_data(max_per_group=200)
    df_scoring = load_ucsf_scoring_discrepancy()

    print(f"\nTracking data: {len(df_tracking)} sessions, "
          f"{df_tracking['animal_id'].nunique()} animals across {len(CONTUSION_GROUPS)} groups")
    print(f"Scoring data: {len(df_scoring)} animals")

    # Print key stats
    print(f"\n{'='*80}")
    print("GROUP-LEVEL PELLET TRACKING SUMMARY")
    print(f"{'='*80}")
    for grp in CONTUSION_GROUPS:
        g = df_tracking[df_tracking['group'] == grp]
        pct = g['pellet_pct_below_05']
        lik = g['pellet_mean_lik']
        print(f"  Group {GROUP_LABELS[grp]}:")
        print(f"    Sessions: {len(g)}, Animals: {g['animal_id'].nunique()}")
        print(f"    Mean pellet likelihood: {lik.mean():.3f} (SD={lik.std():.3f})")
        print(f"    % frames below 0.5: {pct.mean():.1f}% (SD={pct.std():.1f}%)")
        ci = bootstrap_ci(pct.values)
        print(f"    95% CI for mean dropout rate: [{ci[0]:.1f}%, {ci[1]:.1f}%]")

    # Generate figures
    print(f"\nGenerating figures...")
    fig1_group_comparison(df_tracking)
    fig2_flagged_vs_unflagged(df_tracking)
    fig3_tracking_vs_discrepancy(df_tracking, df_scoring)
    fig4_aspa_mechanism(df_tracking)
    fig5_per_animal_heatmap(df_tracking)
    fig6_effect_size_forest(df_tracking, df_scoring)

    # Export
    print(f"\nExporting statistics...")
    merged = export_statistics(df_tracking, df_scoring)

    # Print merged summary for data scientist
    print(f"\n{'='*80}")
    print("PER-ANIMAL TRACKING QUALITY vs SCORING DISCREPANCY (Contusion Groups Only)")
    print(f"{'='*80}")
    print(f"{'Animal':6s} {'Group':5s} {'Flag':5s} {'Sessions':8s} "
          f"{'PelletLik':10s} {'%<0.5':7s} {'Retr.Disc%':11s} {'Cont.Disc%':11s}")
    print("-" * 80)
    for _, r in merged.sort_values(['group', 'animal_id']).iterrows():
        flag = '***' if r['is_flagged'] else ''
        rd = f"{r['retrieved_discrepancy_pct']:+.1f}" if pd.notna(r.get('retrieved_discrepancy_pct')) else 'N/A'
        cd = f"{r['contacted_discrepancy_pct']:+.1f}" if pd.notna(r.get('contacted_discrepancy_pct')) else 'N/A'
        print(f"{r['animal_id']:6s} {r['group']:5s} {flag:5s} {r['n_sessions']:8.0f} "
              f"{r['mean_pellet_lik']:10.3f} {r['mean_pct_below_05']:7.1f} {rd:>11s} {cd:>11s}")

    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    print(f"Statistics exported to: {EXPORT_DIR}")


if __name__ == '__main__':
    main()
