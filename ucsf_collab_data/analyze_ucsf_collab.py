"""
UCSF Collaboration Data Analysis — Algorithmic (DLC) + Manual Scoring

Analyzes session-level behavioral data from the Automated Single Pellet apparatus
with dual scoring: DeepLabCut algorithmic (Video_*) and operator manual (Manual_*).

Groups: D (Pyramidotomy), G/H (Transection), K (Contusion 70kd),
        L (Contusion 50kd), M (Contusion 60kd)

Time windows (mapped from Test_Type_Grouped_1):
  - Final 3: 2_Pre-injury_1/2/3
  - Immediate Post: 3_1wk_Post-injury (+ _First_Test for Group D)
  - 2-4 Post: 3_2wk/3wk/4wk_Post-injury
  - Post-Rehab: 5_Post-rehab_Test_1/2

Metrics:
  - Manual: pellet counts out of 20 per tray → % per day
  - Video (DLC): reach outcome counts per tray (can exceed 20)
  - Both: Retrieved (eaten) and Contacted (displaced + retrieved)

Figures produced:
  1. Per-group behavior bars (4 windows × 2 metrics × 2 scoring methods)
  2. Per-group recovery trajectories (4-point, nadir-based classification)
  3. Per-group trajectory + waterfall (nadir-based)
  4. Mega-cohort normalized analysis
  5. Scoring method comparison (Manual vs DLC head-to-head)
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import warnings
from collections import defaultdict
from scipy import stats

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Output to MouseDB/figures for finalized data products
_connectome_root = SCRIPT_DIR
while os.path.basename(_connectome_root) != '2_Connectome' and os.path.dirname(_connectome_root) != _connectome_root:
    _connectome_root = os.path.dirname(_connectome_root)
MOUSEDB_DIR = os.path.join(_connectome_root, 'MouseDB')
OUTPUT_DIR = os.path.join(MOUSEDB_DIR, 'figures', 'behavior_ucsf_collab')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Source data path
DATA_PATH = os.path.join(
    r'C:\Users\friedrichl\OneDrive - Marquette University\Blackmore Lab Notes - Sharepoint',
    r'3 Lab Projects\Automated single pellet apparatus\!UCSF_Collab',
    r'May2025_Uploads\ODC Uploads\Session_Data.csv'
)

# Injury type display names (matching historical analysis)
INJURY_NAMES = {
    'Pyramidotomy': 'Pyramidotomy',
    'Cervical_Transection': 'Transection',
    'Contusion-70kD': 'Contusion 70kd',
    'Contusion-50kD': 'Contusion 50kd',
    'Contusion-60kD': 'Contusion 60kd',
}

# Injury type color map (matching historical mega-cohort)
INJURY_COLORS = {
    'Pyramidotomy': '#1f77b4',
    'Transection': '#d62728',
    'Contusion 50kd': '#2ca02c',
    'Contusion 60kd': '#ff7f0e',
    'Contusion 70kd': '#e377c2',
}

# Learner criterion: Final 3 avg manual retrieved % > 5%
LEARNER_THRESHOLD = 5.0

# Time window mapping from Test_Type_Grouped_1
WINDOW_MAP = {
    'final_3': ['2_Pre-injury_1', '2_Pre-injury_2', '2_Pre-injury_3'],
    'immediate_post': ['3_1wk_Post-injury', '3_1wk_Post-injury_First_Test'],
    '2_4_post': ['3_2wk_Post-injury', '3_3wk_Post-injury', '3_4wk_Post-injury'],
    'last_2': ['5_Post-rehab_Test_1', '5_Post-rehab_Test_2'],
}


def load_and_prepare_data():
    """Load session CSV and compute per-animal per-day metrics."""
    df = pd.read_csv(DATA_PATH)

    # Coerce numeric columns
    num_cols = ['Video_Displaced', 'Video_Retrieved', 'Video_Contacted',
                'Manual_Displaced', 'Manual_Retrieved', 'Manual_Contacted',
                'Total_Swipes_AI', 'Attention_AI',
                'Contacted_Match', 'Displaced_Match', 'Retrieved_Match']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Filter to Pillar trays only (consistent with historical analysis)
    df = df[df['Tray_Type'] == 'Pillar'].copy()

    # Clean injury names
    df['Injury_display'] = df['Injury_type'].map(INJURY_NAMES)

    return df


def aggregate_daily(df):
    """Aggregate tray-level data to per-animal per-test-day.

    Manual scores: sum pellets across trays, divide by total pellets (n_trays × 20)
    Video scores: sum reach outcomes across trays, then cap at total pellets for % calc
    """
    grouped = df.groupby(
        ['SubjectID', 'Group', 'Injury_type', 'Injury_display',
         'Test_Type', 'Test_Type_Grouped_1', 'Test_Type_Grouped_2', 'Test_Type_Grouped_3']
    ).agg(
        n_trays=('Tray_ID', 'count'),
        manual_contacted=('Manual_Contacted', 'sum'),
        manual_retrieved=('Manual_Retrieved', 'sum'),
        video_contacted=('Video_Contacted', lambda x: x.dropna().sum()),
        video_retrieved=('Video_Retrieved', lambda x: x.dropna().sum()),
        total_swipes=('Total_Swipes_AI', lambda x: x.dropna().sum()),
        attention=('Attention_AI', 'mean'),
    ).reset_index()

    # Total pellets available per day = n_trays × 20
    grouped['total_pellets'] = grouped['n_trays'] * 20

    # Manual percentages (pellet-based, max 100%)
    grouped['manual_contacted_pct'] = (grouped['manual_contacted'] / grouped['total_pellets'] * 100).clip(0, 100)
    grouped['manual_retrieved_pct'] = (grouped['manual_retrieved'] / grouped['total_pellets'] * 100).clip(0, 100)

    # Video percentages: cap reach counts at total pellets, then compute %
    # (A reach count > 20 means multiple reaches contacted same pellet;
    #  capping gives pellet-equivalent %)
    grouped['video_contacted_pct'] = (grouped['video_contacted'].clip(upper=grouped['total_pellets']) / grouped['total_pellets'] * 100)
    grouped['video_retrieved_pct'] = (grouped['video_retrieved'].clip(upper=grouped['total_pellets']) / grouped['total_pellets'] * 100)

    return grouped


def assign_windows(daily):
    """Assign time window labels based on Test_Type_Grouped_1."""
    def map_window(g1):
        for window_name, g1_values in WINDOW_MAP.items():
            if g1 in g1_values:
                return window_name
        return None

    daily['window'] = daily['Test_Type_Grouped_1'].apply(map_window)
    return daily


def get_learners(daily):
    """Return set of (SubjectID, Group) that meet learner criterion.

    Criterion: average manual_retrieved_pct across Final 3 > LEARNER_THRESHOLD.
    """
    f3 = daily[daily['window'] == 'final_3']
    animal_means = f3.groupby(['SubjectID', 'Group'])['manual_retrieved_pct'].mean()
    learners = set(animal_means[animal_means > LEARNER_THRESHOLD].index)
    return learners


def compute_window_data(daily, group, learners, scoring='manual'):
    """Compute per-animal window averages for a group.

    Returns dict of window_name -> {
        'animals': [list], 'eaten': [list], 'contacted': [list],
        'n': int, 'eaten_mean': float, 'eaten_sem': float, ...
    }
    """
    sub = daily[(daily['Group'] == group) &
                (daily[['SubjectID', 'Group']].apply(tuple, axis=1).isin(learners))]

    prefix = scoring  # 'manual' or 'video'
    eaten_col = f'{prefix}_retrieved_pct'
    contacted_col = f'{prefix}_contacted_pct'

    window_data = {}
    for wname in ['final_3', 'immediate_post', '2_4_post', 'last_2']:
        wsub = sub[sub['window'] == wname]
        if wsub.empty:
            window_data[wname] = {
                'animals': [], 'eaten': [], 'contacted': [],
                'n': 0, 'eaten_mean': None, 'eaten_sem': None,
                'contacted_mean': None, 'contacted_sem': None,
            }
            continue

        # Per-animal averages across test days in window
        animal_avgs = wsub.groupby('SubjectID').agg(
            eaten=(eaten_col, 'mean'),
            contacted=(contacted_col, 'mean'),
        ).reset_index()

        animals = list(animal_avgs['SubjectID'])
        eaten_vals = list(animal_avgs['eaten'])
        contacted_vals = list(animal_avgs['contacted'])
        n = len(animals)

        window_data[wname] = {
            'animals': animals,
            'eaten': eaten_vals,
            'contacted': contacted_vals,
            'n': n,
            'eaten_mean': np.mean(eaten_vals),
            'eaten_sem': stats.sem(eaten_vals) if n > 1 else 0,
            'contacted_mean': np.mean(contacted_vals),
            'contacted_sem': stats.sem(contacted_vals) if n > 1 else 0,
        }

    return window_data


def plot_group_behavior(group, injury_display, window_data_manual, window_data_video,
                        n_learners, output_dir):
    """2-row bar panel: top=Manual scoring, bottom=Video (DLC) scoring.

    Each row has 2 panels: Retrieved and Contacted across 4 windows.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f'Behavior Performance: {group} - {injury_display} (N={n_learners} learners)\n'
        f'Top: Manually Scored  |  Bottom: Algorithmically Scored (DLC)',
        fontsize=14, fontweight='bold'
    )

    window_labels = ['Pre-Injury\n(Final 3)', '1 Wk\nPost-Injury',
                     '2-4 Wk\nPost-Injury', 'Post-Rehab\n(Last 2)']
    window_keys = ['final_3', 'immediate_post', '2_4_post', 'last_2']

    for row, (wd, scoring_label) in enumerate([
        (window_data_manual, 'Manually Scored'),
        (window_data_video, 'Algorithmically Scored (DLC)'),
    ]):
        for col, (metric_key, metric_name, ylabel) in enumerate([
            ('eaten', 'Retrieved', '% Pellets Retrieved'),
            ('contacted', 'Contacted', '% Pellets Contacted'),
        ]):
            ax = axes[row, col]
            means = []
            sems = []
            ns = []

            for wk in window_keys:
                w = wd[wk]
                m = w[f'{metric_key}_mean']
                s = w[f'{metric_key}_sem']
                means.append(m if m is not None else 0)
                sems.append(s if s is not None else 0)
                ns.append(w['n'])

            x = np.arange(len(window_keys))
            bars = ax.bar(x, means, yerr=sems, capsize=5,
                         color='#7f7f7f', edgecolor='black', linewidth=0.5, alpha=0.7)

            # Overlay individual animal points
            for i, wk in enumerate(window_keys):
                w = wd[wk]
                vals = w[metric_key]
                if vals:
                    jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
                    ax.scatter(np.full(len(vals), i) + jitter, vals,
                              color='#1f1f1f', s=20, alpha=0.5, zorder=5, edgecolor='none')

            # N labels
            for i, n in enumerate(ns):
                ax.text(i, -3, f'N={n}', ha='center', fontsize=8, color='gray')

            ax.set_xticks(x)
            ax.set_xticklabels(window_labels, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f'{scoring_label}: {metric_name}', fontsize=11, fontweight='bold')
            ax.set_ylim(bottom=-5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(output_dir, f'behavior_{group}.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out}")


def plot_recovery(group, injury_display, window_data, scoring_label, output_dir, suffix=''):
    """Per-animal 4-point recovery trajectories with nadir-based classification."""
    from matplotlib.lines import Line2D

    f3 = window_data['final_3']
    ip = window_data['immediate_post']
    p24 = window_data['2_4_post']
    l2 = window_data['last_2']

    f3_idx = {a: i for i, a in enumerate(f3['animals'])}
    ip_idx = {a: i for i, a in enumerate(ip['animals'])}
    p24_idx = {a: i for i, a in enumerate(p24['animals'])}
    l2_idx = {a: i for i, a in enumerate(l2['animals'])}

    paired_animals = sorted(set(f3['animals']) & set(l2['animals']))
    has_ip = set(ip['animals'])
    has_24 = set(p24['animals'])

    if len(paired_animals) < 2:
        return

    n = len(paired_animals)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        f'Per-Animal Recovery ({scoring_label}): {group} - {injury_display} (N={n})\n'
        f'Full Trajectory: Pre-Injury through Post-Rehab',
        fontsize=13, fontweight='bold'
    )

    for col, (metric_key, metric_name, ylabel) in enumerate([
        ('eaten', 'Retrieved', '% Pellets Retrieved'),
        ('contacted', 'Contacted', '% Pellets Contacted'),
    ]):
        ax = axes[col]
        mean_vals = {0: [], 1: [], 2: [], 3: []}

        for animal in paired_animals:
            pre = f3[metric_key][f3_idx[animal]]
            rehab = l2[metric_key][l2_idx[animal]]
            post_1wk = ip[metric_key][ip_idx[animal]] if animal in has_ip else None
            post_24wk = p24[metric_key][p24_idx[animal]] if animal in has_24 else None

            post_vals = [v for v in [post_1wk, post_24wk] if v is not None]
            nadir = min(post_vals) if post_vals else None

            if nadir is not None:
                if rehab > pre * 0.8:
                    color = '#2ca02c'; alpha = 0.7
                elif rehab > nadir:
                    color = '#ff7f0e'; alpha = 0.6
                else:
                    color = '#d62728'; alpha = 0.5
            else:
                color = '#888888'; alpha = 0.5

            xs, ys = [0], [pre]
            mean_vals[0].append(pre)
            if post_1wk is not None:
                xs.append(1); ys.append(post_1wk)
                mean_vals[1].append(post_1wk)
            if post_24wk is not None:
                xs.append(2); ys.append(post_24wk)
                mean_vals[2].append(post_24wk)
            xs.append(3); ys.append(rehab)
            mean_vals[3].append(rehab)

            if post_1wk is not None and post_24wk is not None:
                ax.plot(xs, ys, 'o-', color=color, alpha=alpha,
                        markersize=5, linewidth=1.2, zorder=3)
            elif post_1wk is not None:
                ax.plot([0, 1], [pre, post_1wk], 'o-', color=color, alpha=alpha,
                        markersize=5, linewidth=1.2, zorder=3)
                ax.plot([1, 3], [post_1wk, rehab], 'o--', color=color, alpha=alpha * 0.7,
                        markersize=5, linewidth=0.8, zorder=3)
            elif post_24wk is not None:
                ax.plot([0, 2], [pre, post_24wk], 'o--', color=color, alpha=alpha * 0.7,
                        markersize=5, linewidth=0.8, zorder=3)
                ax.plot([2, 3], [post_24wk, rehab], 'o-', color=color, alpha=alpha,
                        markersize=5, linewidth=1.2, zorder=3)
            else:
                ax.plot([0, 3], [pre, rehab], 'o--', color=color, alpha=alpha * 0.5,
                        markersize=5, linewidth=0.8, zorder=3)

        # Group mean
        mean_x, mean_y = [], []
        for xi in [0, 1, 2, 3]:
            if mean_vals[xi]:
                mean_x.append(xi)
                mean_y.append(np.mean(mean_vals[xi]))
        ax.plot(mean_x, mean_y, 's-', color='black', markersize=10, linewidth=3, zorder=5)

        # Stats
        pre_arr = np.array(mean_vals[0])
        rehab_arr = np.array(mean_vals[3])
        if len(pre_arr) >= 2:
            t_stat, p_val = stats.ttest_rel(pre_arr, rehab_arr)
            diff = rehab_arr - pre_arr
            mean_diff = np.mean(diff)
            sem_diff = stats.sem(diff)

            n_recovered = n_improved = n_none = 0
            for animal in paired_animals:
                pre = f3[metric_key][f3_idx[animal]]
                rehab = l2[metric_key][l2_idx[animal]]
                pv = []
                if animal in has_ip: pv.append(ip[metric_key][ip_idx[animal]])
                if animal in has_24: pv.append(p24[metric_key][p24_idx[animal]])
                nad = min(pv) if pv else rehab
                if rehab > pre * 0.8: n_recovered += 1
                elif rehab > nad: n_improved += 1
                else: n_none += 1

            stats_text = (
                f'Pre vs Post-Rehab: p={p_val:.4f}\n'
                f'Mean diff: {mean_diff:+.1f} +/- {sem_diff:.1f}\n'
                f'Recovered (>80% pre): {n_recovered}/{n}\n'
                f'Improved from nadir: {n_improved}/{n}\n'
                f'No improvement: {n_none}/{n}'
            )
            ax.text(0.5, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
                    va='top', ha='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    family='monospace')

        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(['Pre-Injury\n(Final 3)', '1 Wk\nPost-Injury',
                            '2-4 Wk\nPost-Injury', 'Post-Rehab\n(Last 2)'], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlim(-0.3, 3.3)
        ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        legend_elements = [
            Line2D([0], [0], color='#2ca02c', marker='o', label='Recovered (>80% of pre)'),
            Line2D([0], [0], color='#ff7f0e', marker='o', label='Improved from nadir'),
            Line2D([0], [0], color='#d62728', marker='o', label='No improvement'),
            Line2D([0], [0], color='black', marker='s', linewidth=3, label='Group Mean'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=7)

    plt.tight_layout()
    out = os.path.join(output_dir, f'recovery_{group}{suffix}.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out}")


def plot_trajectory_waterfall(group, injury_display, window_data, scoring_label, output_dir, suffix=''):
    """Four-point trajectory + recovery waterfall with per-animal nadir."""
    from matplotlib.lines import Line2D

    f3 = window_data['final_3']
    ip = window_data['immediate_post']
    p24 = window_data['2_4_post']
    l2 = window_data['last_2']

    f3_idx = {a: i for i, a in enumerate(f3['animals'])}
    ip_idx = {a: i for i, a in enumerate(ip['animals'])}
    p24_idx = {a: i for i, a in enumerate(p24['animals'])}
    l2_idx = {a: i for i, a in enumerate(l2['animals'])}

    core_animals = set(f3['animals']) & set(l2['animals'])
    has_ip = set(ip['animals'])
    has_24 = set(p24['animals'])
    traj_animals = sorted(core_animals & (has_ip | has_24))

    wf_animals = sorted(set(l2['animals']) & (has_ip | has_24))

    if len(traj_animals) < 2 and len(wf_animals) < 2:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f'Recovery Analysis ({scoring_label}): {group} - {injury_display}\n'
        f'Full Trajectory: Pre-Injury through Post-Rehab',
        fontsize=14, fontweight='bold'
    )

    for row, (metric_key, metric_name, ylabel) in enumerate([
        ('eaten', 'Retrieved', '% Pellets Retrieved'),
        ('contacted', 'Contacted', '% Pellets Contacted'),
    ]):
        # === LEFT: Trajectory ===
        ax_traj = axes[row, 0]
        mean_vals = {0: [], 1: [], 2: [], 3: []}

        for animal in traj_animals:
            pre = f3[metric_key][f3_idx[animal]]
            rehab = l2[metric_key][l2_idx[animal]]

            post_1wk = ip[metric_key][ip_idx[animal]] if animal in has_ip else None
            post_24wk = p24[metric_key][p24_idx[animal]] if animal in has_24 else None

            pv = [v for v in [post_1wk, post_24wk] if v is not None]
            nadir = min(pv) if pv else 0

            if rehab > pre * 0.8:
                color = '#2ca02c'; alpha = 0.7
            elif rehab > nadir:
                color = '#ff7f0e'; alpha = 0.6
            else:
                color = '#d62728'; alpha = 0.5

            xs, ys = [0], [pre]
            mean_vals[0].append(pre)
            if post_1wk is not None:
                xs.append(1); ys.append(post_1wk)
                mean_vals[1].append(post_1wk)
            if post_24wk is not None:
                xs.append(2); ys.append(post_24wk)
                mean_vals[2].append(post_24wk)
            xs.append(3); ys.append(rehab)
            mean_vals[3].append(rehab)

            if len(xs) == 4:
                ax_traj.plot(xs, ys, 'o-', color=color, alpha=alpha,
                            markersize=5, linewidth=1.2, zorder=3)
            else:
                # Connect available points, dashed for gaps
                for j in range(len(xs) - 1):
                    style = 'o-' if (xs[j+1] - xs[j]) == 1 else 'o--'
                    lw = 1.2 if style == 'o-' else 0.8
                    a = alpha if style == 'o-' else alpha * 0.7
                    ax_traj.plot(xs[j:j+2], ys[j:j+2], style, color=color,
                                alpha=a, markersize=5, linewidth=lw, zorder=3)

        # Group mean
        mx, my = [], []
        for xi in [0, 1, 2, 3]:
            if mean_vals[xi]:
                mx.append(xi)
                my.append(np.mean(mean_vals[xi]))
        ax_traj.plot(mx, my, 's-', color='black', markersize=10, linewidth=3, zorder=5)

        ax_traj.set_xticks([0, 1, 2, 3])
        ax_traj.set_xticklabels(['Pre-Injury\n(Final 3)', '1 Wk\nPost-Injury',
                                  '2-4 Wk\nPost-Injury', 'Post-Rehab\n(Last 2)'], fontsize=9)
        ax_traj.set_ylabel(ylabel, fontsize=11)
        ax_traj.set_title(f'{metric_name}: Individual Trajectories (N={len(traj_animals)})',
                         fontsize=11, fontweight='bold')
        ax_traj.spines['top'].set_visible(False)
        ax_traj.spines['right'].set_visible(False)
        ax_traj.set_xlim(-0.3, 3.3)

        traj_legend = [
            Line2D([0], [0], color='#2ca02c', marker='o', label='Recovered (>80% of pre)'),
            Line2D([0], [0], color='#ff7f0e', marker='o', label='Improved from nadir'),
            Line2D([0], [0], color='#d62728', marker='o', label='No improvement'),
            Line2D([0], [0], color='black', marker='s', linewidth=3, label='Group Mean'),
        ]
        ax_traj.legend(handles=traj_legend, loc='upper right', fontsize=7)

        # === RIGHT: Waterfall ===
        ax_wf = axes[row, 1]
        deltas = []
        for animal in wf_animals:
            if animal not in l2_idx:
                continue
            rehab_val = l2[metric_key][l2_idx[animal]]
            pv = []
            if animal in ip_idx: pv.append(ip[metric_key][ip_idx[animal]])
            if animal in p24_idx: pv.append(p24[metric_key][p24_idx[animal]])
            if not pv:
                continue
            nadir = min(pv)
            deltas.append(rehab_val - nadir)

        if not deltas:
            ax_wf.text(0.5, 0.5, 'Insufficient data', transform=ax_wf.transAxes,
                      ha='center', va='center', fontsize=12, color='gray')
            continue

        sorted_idx = np.argsort(deltas)[::-1]
        sorted_deltas = [deltas[i] for i in sorted_idx]
        bar_colors = ['#2ca02c' if d > 0 else '#d62728' if d < 0 else '#888888'
                      for d in sorted_deltas]

        ax_wf.bar(np.arange(len(sorted_deltas)), sorted_deltas,
                  color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax_wf.axhline(y=0, color='black', linewidth=1)

        n_total = len(sorted_deltas)
        n_improved = sum(1 for d in sorted_deltas if d > 0)
        n_declined = sum(1 for d in sorted_deltas if d < 0)
        mean_all = np.mean(sorted_deltas)
        pos_deltas = [d for d in sorted_deltas if d > 0]
        mean_resp = np.mean(pos_deltas) if pos_deltas else 0

        p_str = 'N/A'
        try:
            non_zero = [d for d in sorted_deltas if d != 0]
            if len(non_zero) >= 5:
                _, wil_p = stats.wilcoxon(non_zero)
                p_str = f'p={wil_p:.4f}' if wil_p >= 0.0001 else 'p<0.0001'
        except Exception:
            pass

        stats_text = (
            f'All animals (N={n_total}):\n'
            f'  Improved: {n_improved}  |  Declined: {n_declined}\n'
            f'  Mean delta: {mean_all:+.1f}%  |  Wilcoxon {p_str}\n'
            f'Responders (N={len(pos_deltas)}): +{mean_resp:.1f}%'
        )
        ax_wf.text(0.98, 0.98, stats_text, transform=ax_wf.transAxes, fontsize=8,
                  va='top', ha='right',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                  family='monospace')

        ax_wf.set_xlabel('Animals (sorted by recovery)', fontsize=10)
        ax_wf.set_ylabel(f'Change in {ylabel}\n(Post-Rehab - Nadir)', fontsize=10)
        ax_wf.set_title(f'{metric_name}: Recovery Waterfall (from nadir)',
                       fontsize=11, fontweight='bold')
        ax_wf.spines['top'].set_visible(False)
        ax_wf.spines['right'].set_visible(False)
        ax_wf.set_xticks([])

    plt.tight_layout()
    out = os.path.join(output_dir, f'trajectory_{group}{suffix}.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out}")


def plot_scoring_comparison(daily, learners, output_dir):
    """Head-to-head comparison of Manual vs Video (DLC) scoring methods.

    Panel layout:
      Top-left: Scatter Manual vs Video contacted (per-session)
      Top-right: Scatter Manual vs Video retrieved (per-session)
      Bottom-left: Bland-Altman for contacted
      Bottom-right: Per-group mean comparison across windows
    """
    from matplotlib.lines import Line2D

    sub = daily[daily[['SubjectID', 'Group']].apply(tuple, axis=1).isin(learners)].copy()
    sub = sub.dropna(subset=['manual_contacted_pct', 'video_contacted_pct'])

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(
        'Scoring Method Comparison: Manual vs Algorithmic (DLC)\n'
        f'Pillar Trays Only, Learners Only (N={sub["SubjectID"].nunique()} animals)',
        fontsize=14, fontweight='bold'
    )

    group_colors = {g: plt.cm.tab10(i) for i, g in enumerate(sorted(sub['Group'].unique()))}

    # === TOP-LEFT: Scatter Contacted ===
    ax = axes[0, 0]
    for g in sorted(sub['Group'].unique()):
        gs = sub[sub['Group'] == g]
        ax.scatter(gs['manual_contacted_pct'], gs['video_contacted_pct'],
                  color=group_colors[g], alpha=0.3, s=15, label=g, edgecolor='none')
    r = sub['manual_contacted_pct'].corr(sub['video_contacted_pct'])
    ax.plot([0, 100], [0, 100], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Manual Contacted %', fontsize=11)
    ax.set_ylabel('DLC Contacted %', fontsize=11)
    ax.set_title(f'Contacted: r={r:.3f}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # === TOP-RIGHT: Scatter Retrieved ===
    ax = axes[0, 1]
    for g in sorted(sub['Group'].unique()):
        gs = sub[sub['Group'] == g]
        ax.scatter(gs['manual_retrieved_pct'], gs['video_retrieved_pct'],
                  color=group_colors[g], alpha=0.3, s=15, label=g, edgecolor='none')
    r = sub['manual_retrieved_pct'].corr(sub['video_retrieved_pct'])
    ax.plot([0, 100], [0, 100], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Manual Retrieved %', fontsize=11)
    ax.set_ylabel('DLC Retrieved %', fontsize=11)
    ax.set_title(f'Retrieved: r={r:.3f}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # === BOTTOM-LEFT: Bland-Altman Contacted ===
    ax = axes[1, 0]
    mean_contacted = (sub['manual_contacted_pct'] + sub['video_contacted_pct']) / 2
    diff_contacted = sub['video_contacted_pct'] - sub['manual_contacted_pct']
    for g in sorted(sub['Group'].unique()):
        mask = sub['Group'] == g
        ax.scatter(mean_contacted[mask], diff_contacted[mask],
                  color=group_colors[g], alpha=0.3, s=15, edgecolor='none')

    md = diff_contacted.mean()
    sd = diff_contacted.std()
    ax.axhline(md, color='blue', linewidth=1.5, label=f'Mean diff: {md:+.1f}%')
    ax.axhline(md + 1.96*sd, color='red', linewidth=1, linestyle='--',
              label=f'+1.96 SD: {md+1.96*sd:+.1f}%')
    ax.axhline(md - 1.96*sd, color='red', linewidth=1, linestyle='--',
              label=f'-1.96 SD: {md-1.96*sd:+.1f}%')
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Mean of Manual & DLC Contacted %', fontsize=11)
    ax.set_ylabel('DLC - Manual (Contacted %)', fontsize=11)
    ax.set_title('Bland-Altman: Contacted', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # === BOTTOM-RIGHT: Per-group mean comparison ===
    ax = axes[1, 1]
    window_labels = ['Final 3', '1 Wk Post', '2-4 Wk Post', 'Post-Rehab']
    window_keys = ['final_3', 'immediate_post', '2_4_post', 'last_2']
    groups = sorted(sub['Group'].unique())

    x_base = np.arange(len(window_keys))
    width = 0.12
    n_groups = len(groups)

    for gi, g in enumerate(groups):
        gsub = sub[sub['Group'] == g]
        manual_means = []
        video_means = []
        for wk in window_keys:
            wsub = gsub[gsub['window'] == wk]
            if wsub.empty:
                manual_means.append(0)
                video_means.append(0)
            else:
                animal_m = wsub.groupby('SubjectID')['manual_contacted_pct'].mean()
                animal_v = wsub.groupby('SubjectID')['video_contacted_pct'].mean()
                manual_means.append(animal_m.mean())
                video_means.append(animal_v.mean())

        offset = (gi - n_groups/2 + 0.5) * width
        ax.bar(x_base + offset - width/4, manual_means, width/2,
               color=group_colors[g], alpha=0.5, edgecolor='black', linewidth=0.3)
        ax.bar(x_base + offset + width/4, video_means, width/2,
               color=group_colors[g], alpha=0.9, edgecolor='black', linewidth=0.3,
               hatch='///')

    ax.set_xticks(x_base)
    ax.set_xticklabels(window_labels, fontsize=9)
    ax.set_ylabel('Contacted %', fontsize=11)
    ax.set_title('Group Means: Solid=Manual, Hatched=DLC', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Custom legend
    legend_items = []
    for g in groups:
        legend_items.append(Line2D([0], [0], color=group_colors[g], marker='s',
                                   linestyle='', markersize=8,
                                   label=f'{g} ({sub[sub["Group"]==g]["Injury_display"].iloc[0]})'))
    ax.legend(handles=legend_items, fontsize=7, loc='upper right')

    plt.tight_layout()
    out = os.path.join(output_dir, 'scoring_comparison.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out}")


def plot_mega_cohort(all_group_data, scoring, scoring_label, output_dir):
    """Normalized mega-cohort analysis across all groups."""
    from matplotlib.lines import Line2D

    records = []
    for group, gd in all_group_data.items():
        wd = gd[f'window_data_{scoring}']
        injury = gd['injury_display']

        f3 = wd['final_3']
        ip = wd['immediate_post']
        p24 = wd['2_4_post']
        l2 = wd['last_2']

        f3_idx = {a: i for i, a in enumerate(f3['animals'])}
        ip_idx = {a: i for i, a in enumerate(ip['animals'])}
        p24_idx = {a: i for i, a in enumerate(p24['animals'])}
        l2_idx = {a: i for i, a in enumerate(l2['animals'])}

        for animal in f3['animals']:
            if animal not in l2_idx:
                continue
            pre_e = f3['eaten'][f3_idx[animal]]
            rehab_e = l2['eaten'][l2_idx[animal]]
            p1_e = ip['eaten'][ip_idx[animal]] if animal in ip_idx else None
            p24_e = p24['eaten'][p24_idx[animal]] if animal in p24_idx else None

            pre_c = f3['contacted'][f3_idx[animal]]
            rehab_c = l2['contacted'][l2_idx[animal]]
            p1_c = ip['contacted'][ip_idx[animal]] if animal in ip_idx else None
            p24_c = p24['contacted'][p24_idx[animal]] if animal in p24_idx else None

            if p1_e is None and p24_e is None:
                continue

            records.append({
                'group': group, 'injury': injury, 'animal': animal,
                'pre': pre_e, 'post_1wk': p1_e, 'post_24wk': p24_e, 'rehab': rehab_e,
                'pre_c': pre_c, 'post_1wk_c': p1_c, 'post_24wk_c': p24_c, 'rehab_c': rehab_c,
            })

    if not records:
        return

    for metric_suffix, metric_label, ylabel_norm in [
        ('', 'Retrieved', '% of Pre-Injury Baseline'),
        ('_c', 'Contacted', '% of Pre-Injury Baseline'),
    ]:
        pre_key = 'pre' + metric_suffix
        p1_key = 'post_1wk' + metric_suffix
        p24_key = 'post_24wk' + metric_suffix
        rehab_key = 'rehab' + metric_suffix

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle(
            f'Mega-Cohort Analysis ({scoring_label}): {metric_label}\n'
            f'Normalized to Pre-Injury Baseline (N={len(records)} animals)',
            fontsize=14, fontweight='bold'
        )

        # Normalize
        norm_records = []
        for r in records:
            pv = r[pre_key]
            if pv <= 0:
                continue
            nr = {'group': r['group'], 'injury': r['injury'], 'animal': r['animal'],
                  'pre': 100.0}
            for tkey, nkey in [(p1_key, 'post_1wk'), (p24_key, 'post_24wk'), (rehab_key, 'rehab')]:
                raw = r[tkey]
                nr[nkey] = (raw / pv) * 100 if raw is not None else None
            pvs = [nr[k] for k in ['post_1wk', 'post_24wk'] if nr[k] is not None]
            nr['nadir'] = min(pvs) if pvs else None
            norm_records.append(nr)

        if not norm_records:
            plt.close()
            continue

        # === TOP LEFT: All animals pooled ===
        ax = axes[0, 0]
        mean_vals = {0: [], 1: [], 2: [], 3: []}
        for nr in norm_records:
            xs, ys = [0], [nr['pre']]
            mean_vals[0].append(nr['pre'])
            if nr['post_1wk'] is not None:
                xs.append(1); ys.append(nr['post_1wk'])
                mean_vals[1].append(nr['post_1wk'])
            if nr['post_24wk'] is not None:
                xs.append(2); ys.append(nr['post_24wk'])
                mean_vals[2].append(nr['post_24wk'])
            if nr['rehab'] is not None:
                xs.append(3); ys.append(nr['rehab'])
                mean_vals[3].append(nr['rehab'])

            if nr['rehab'] is not None and nr['rehab'] > 80:
                color = '#2ca02c'; alpha = 0.25
            elif nr['rehab'] is not None and nr['nadir'] is not None and nr['rehab'] > nr['nadir']:
                color = '#ff7f0e'; alpha = 0.2
            else:
                color = '#d62728'; alpha = 0.2
            ax.plot(xs, ys, 'o-', color=color, alpha=alpha, markersize=3, linewidth=0.8, zorder=2)

        gmx, gmy = [], []
        for xi in [0, 1, 2, 3]:
            if mean_vals[xi]:
                gmx.append(xi)
                gmy.append(np.mean(mean_vals[xi]))
        ax.plot(gmx, gmy, 's-', color='black', markersize=12, linewidth=3.5, zorder=5)
        ax.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(['Pre-Injury', '1 Wk Post', '2-4 Wk Post', 'Post-Rehab'], fontsize=9)
        ax.set_ylabel(ylabel_norm, fontsize=11)
        ax.set_title(f'All Animals Pooled (N={len(norm_records)})', fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(-0.3, 3.3)

        # === TOP RIGHT: Recovery waterfall ===
        ax = axes[0, 1]
        deltas, delta_injuries = [], []
        for nr in norm_records:
            if nr['nadir'] is not None and nr['rehab'] is not None:
                deltas.append(nr['rehab'] - nr['nadir'])
                delta_injuries.append(nr['injury'])

        if deltas:
            sorted_idx = np.argsort(deltas)[::-1]
            sd = [deltas[i] for i in sorted_idx]
            si = [delta_injuries[i] for i in sorted_idx]
            bc = [INJURY_COLORS.get(inj, '#888888') for inj in si]

            ax.bar(np.arange(len(sd)), sd, color=bc, alpha=0.8, edgecolor='none')
            ax.axhline(y=0, color='black', linewidth=1)

            n_imp = sum(1 for d in sd if d > 0)
            n_dec = sum(1 for d in sd if d < 0)
            mean_a = np.mean(sd)
            pd_list = [d for d in sd if d > 0]
            mean_r = np.mean(pd_list) if pd_list else 0

            p_str = 'N/A'
            try:
                nz = [d for d in sd if d != 0]
                if len(nz) >= 5:
                    _, wp = stats.wilcoxon(nz)
                    p_str = f'p={wp:.4f}' if wp >= 0.0001 else 'p<0.0001'
            except: pass

            ax.text(0.98, 0.98,
                    f'N={len(sd)}\nImproved: {n_imp} | Declined: {n_dec}\n'
                    f'Mean: {mean_a:+.1f}%\nWilcoxon: {p_str}\n'
                    f'Responders ({len(pd_list)}): +{mean_r:.1f}%',
                    transform=ax.transAxes, fontsize=8, va='top', ha='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    family='monospace')

            seen = []
            for inj in si:
                if inj not in seen:
                    seen.append(inj)
            ax.legend(handles=[Line2D([0],[0],color=INJURY_COLORS.get(i,'#888'),marker='s',
                               linestyle='',markersize=8,label=i) for i in seen],
                     loc='lower right', fontsize=7, title='Injury Type', title_fontsize=8)

        ax.set_xlabel('Animals (sorted)', fontsize=10)
        ax.set_ylabel(f'Recovery from Nadir\n({ylabel_norm})', fontsize=10)
        ax.set_title('Recovery Waterfall', fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([])

        # === BOTTOM LEFT: Group mean by injury type ===
        ax = axes[1, 0]
        injury_groups = {}
        for nr in norm_records:
            injury_groups.setdefault(nr['injury'], []).append(nr)

        for inj, inj_recs in injury_groups.items():
            color = INJURY_COLORS.get(inj, '#888888')
            for nr in inj_recs:
                xs, ys = [0], [nr['pre']]
                if nr['post_1wk'] is not None: xs.append(1); ys.append(nr['post_1wk'])
                if nr['post_24wk'] is not None: xs.append(2); ys.append(nr['post_24wk'])
                if nr['rehab'] is not None: xs.append(3); ys.append(nr['rehab'])
                ax.plot(xs, ys, '-', color=color, alpha=0.1, linewidth=0.5, zorder=2)

            gm = {0: [], 1: [], 2: [], 3: []}
            for nr in inj_recs:
                gm[0].append(nr['pre'])
                if nr['post_1wk'] is not None: gm[1].append(nr['post_1wk'])
                if nr['post_24wk'] is not None: gm[2].append(nr['post_24wk'])
                if nr['rehab'] is not None: gm[3].append(nr['rehab'])
            mx, my = [], []
            for xi in [0,1,2,3]:
                if gm[xi]: mx.append(xi); my.append(np.mean(gm[xi]))
            ax.plot(mx, my, 'o-', color=color, markersize=8, linewidth=2.5, zorder=5,
                    label=f'{inj} (N={len(inj_recs)})')
            for xi in mx:
                if len(gm[xi]) > 1:
                    ax.errorbar(xi, np.mean(gm[xi]), yerr=stats.sem(gm[xi]),
                               color=color, capsize=4, linewidth=1.5, zorder=4)

        ax.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xticks([0,1,2,3])
        ax.set_xticklabels(['Pre-Injury', '1 Wk Post', '2-4 Wk Post', 'Post-Rehab'], fontsize=9)
        ax.set_ylabel(ylabel_norm, fontsize=11)
        ax.set_title('Group Trajectories by Injury Type', fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(-0.3, 3.3)

        # === BOTTOM RIGHT: Box plots ===
        ax = axes[1, 1]
        inj_sorted = sorted(injury_groups.keys(),
                           key=lambda k: np.mean([nr['rehab'] for nr in injury_groups[k]
                                                  if nr['rehab'] is not None]))
        box_data, box_labels, box_colors = [], [], []
        for inj in inj_sorted:
            vals = [nr['rehab'] for nr in injury_groups[inj] if nr['rehab'] is not None]
            if vals:
                box_data.append(vals)
                box_labels.append(f'{inj}\n(N={len(vals)})')
                box_colors.append(INJURY_COLORS.get(inj, '#888'))

        if box_data:
            bp = ax.boxplot(box_data, patch_artist=True, widths=0.6,
                           medianprops=dict(color='black', linewidth=2))
            for patch, c in zip(bp['boxes'], box_colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.6)
            np.random.seed(42)
            for i, vals in enumerate(box_data):
                jitter = np.random.uniform(-0.15, 0.15, len(vals))
                ax.scatter(np.full(len(vals), i+1) + jitter, vals,
                          color=box_colors[i], s=25, zorder=5, alpha=0.7,
                          edgecolor='white', linewidth=0.3)
            ax.set_xticklabels(box_labels, fontsize=8)
            ax.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.5)

            if len(box_data) >= 2:
                try:
                    kw_stat, kw_p = stats.kruskal(*box_data)
                    kw_str = f'Kruskal-Wallis: H={kw_stat:.1f}, p={kw_p:.4f}' if kw_p >= 0.0001 else f'H={kw_stat:.1f}, p<0.0001'
                    ax.set_xlabel(kw_str, fontsize=9)
                except: pass

        ax.set_ylabel(ylabel_norm, fontsize=11)
        ax.set_title('Post-Rehab by Injury Type', fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        sfx = 'eaten' if metric_suffix == '' else 'contacted'
        out = os.path.join(output_dir, f'mega_cohort_{scoring}_{sfx}.png')
        plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {out}")


def main():
    print("=" * 100)
    print("UCSF COLLABORATION DATA ANALYSIS")
    print("  Dual Scoring: Manual (operator) + Algorithmic (DLC)")
    print(f"  Learner criterion: Final 3 avg manual retrieved > {LEARNER_THRESHOLD}%")
    print(f"  Data source: {DATA_PATH}")
    print("=" * 100)

    # Load and prepare
    print("\nLoading data...")
    df = load_and_prepare_data()
    print(f"  Pillar-tray rows: {len(df)}")

    daily = aggregate_daily(df)
    daily = assign_windows(daily)
    print(f"  Aggregated daily records: {len(daily)}")
    print(f"  Records with window assignment: {daily['window'].notna().sum()}")

    learners = get_learners(daily)
    all_animals = set(daily[['SubjectID', 'Group']].apply(tuple, axis=1))
    n_excluded = len(all_animals) - len(learners)
    print(f"  Animals: {len(all_animals)} total, {len(learners)} learners, {n_excluded} excluded")

    all_group_data = {}

    for group in sorted(daily['Group'].unique()):
        gsub = daily[daily['Group'] == group]
        injury = gsub['Injury_display'].iloc[0]
        injury_raw = gsub['Injury_type'].iloc[0]

        group_learners = {(s, g) for s, g in learners if g == group}
        n_learners = len(group_learners)
        n_total = gsub['SubjectID'].nunique()

        print(f"\n--- {group} ({injury}) ---")
        print(f"  Animals: {n_total} total, {n_learners} learners")

        # Show window mapping
        for wname, wlabel in [('final_3', 'Final 3'), ('immediate_post', 'Immediate Post'),
                               ('2_4_post', '2-4 Post'), ('last_2', 'Post-Rehab')]:
            wsub = gsub[(gsub['window'] == wname) &
                        gsub[['SubjectID', 'Group']].apply(tuple, axis=1).isin(learners)]
            tts = sorted(wsub['Test_Type'].unique())
            n_anim = wsub['SubjectID'].nunique()
            print(f"    {wlabel:16s}: {n_anim:2d} animals | {', '.join(tts)}")

        # Compute window data for both scoring methods
        wd_manual = compute_window_data(daily, group, learners, scoring='manual')
        wd_video = compute_window_data(daily, group, learners, scoring='video')

        # Generate per-group figures
        plot_group_behavior(group, injury, wd_manual, wd_video, n_learners, OUTPUT_DIR)

        for scoring, wd, label, sfx in [
            ('manual', wd_manual, 'Manually Scored', '_manual'),
            ('video', wd_video, 'Algorithmically Scored (DLC)', '_dlc'),
        ]:
            plot_recovery(group, injury, wd, label, OUTPUT_DIR, suffix=sfx)
            plot_trajectory_waterfall(group, injury, wd, label, OUTPUT_DIR, suffix=sfx)

        all_group_data[group] = {
            'injury_type': injury_raw,
            'injury_display': injury,
            'window_data_manual': wd_manual,
            'window_data_video': wd_video,
        }

    # Scoring comparison
    print(f"\n--- Scoring Method Comparison ---")
    plot_scoring_comparison(daily, learners, OUTPUT_DIR)

    # Mega-cohort for both scoring methods
    for scoring, label in [('manual', 'Manually Scored'), ('video', 'Algorithmically Scored (DLC)')]:
        print(f"\n--- Mega-Cohort: {label} ---")
        plot_mega_cohort(all_group_data, scoring, label, OUTPUT_DIR)

    # Summary CSV
    print(f"\n--- Summary Table ---")
    results = []
    for group, gd in all_group_data.items():
        for scoring in ['manual', 'video']:
            wd = gd[f'window_data_{scoring}']
            for wname, wlabel in [('final_3', 'Final 3'), ('immediate_post', 'Immediate Post'),
                                   ('2_4_post', '2-4 Post'), ('last_2', 'Post-Rehab')]:
                w = wd[wname]
                results.append({
                    'Group': group,
                    'Injury': gd['injury_display'],
                    'Scoring': scoring,
                    'Window': wlabel,
                    'Eaten_Mean': f"{w['eaten_mean']:.2f}" if w['eaten_mean'] is not None else '',
                    'Eaten_SEM': f"{w['eaten_sem']:.2f}" if w['eaten_sem'] is not None else '',
                    'Contacted_Mean': f"{w['contacted_mean']:.2f}" if w['contacted_mean'] is not None else '',
                    'Contacted_SEM': f"{w['contacted_sem']:.2f}" if w['contacted_sem'] is not None else '',
                    'N': w['n'],
                })

    results_df = pd.DataFrame(results)
    csv_path = os.path.join(MOUSEDB_DIR, 'exports', 'ucsf_collab_analysis.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Print summary
    print(f"\n{'='*100}")
    print("SUMMARY: MANUAL SCORING")
    print(f"{'Group':6s} {'Injury':15s} {'Window':16s} {'Eaten % (M+/-SEM)':22s} {'Contacted % (M+/-SEM)':26s} {'N':4s}")
    print(f"{'-'*6} {'-'*15} {'-'*16} {'-'*22} {'-'*26} {'-'*4}")
    for r in results:
        if r['Scoring'] == 'manual' and r['Eaten_Mean']:
            print(f"{r['Group']:6s} {r['Injury']:15s} {r['Window']:16s} "
                  f"{float(r['Eaten_Mean']):5.1f} +/- {float(r['Eaten_SEM']):4.1f}      "
                  f"{float(r['Contacted_Mean']):5.1f} +/- {float(r['Contacted_SEM']):4.1f}       "
                  f"{r['N']:4d}")

    print(f"\n{'='*100}")
    print("SUMMARY: DLC ALGORITHMIC SCORING")
    print(f"{'Group':6s} {'Injury':15s} {'Window':16s} {'Eaten % (M+/-SEM)':22s} {'Contacted % (M+/-SEM)':26s} {'N':4s}")
    print(f"{'-'*6} {'-'*15} {'-'*16} {'-'*22} {'-'*26} {'-'*4}")
    for r in results:
        if r['Scoring'] == 'video' and r['Eaten_Mean']:
            print(f"{r['Group']:6s} {r['Injury']:15s} {r['Window']:16s} "
                  f"{float(r['Eaten_Mean']):5.1f} +/- {float(r['Eaten_SEM']):4.1f}      "
                  f"{float(r['Contacted_Mean']):5.1f} +/- {float(r['Contacted_SEM']):4.1f}       "
                  f"{r['N']:4d}")

    print(f"\nAll figures saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
