"""
Cross-Group Bodypart POSITIONAL Analysis

Checks whether M (60kD) has different spatial patterns in DLC tracking
compared to K/L, beyond just confidence scores.

What to look for:
  1. RightHand position distributions (x,y) — does M reach to different places?
  2. Inter-bodypart distances (nose-hand, ear-ear, nose-pillar) — spatial consistency
  3. Frame-to-frame jitter/jumps in RightHand — noisy tracking masquerading as confident
  4. Phase-specific positional shifts — does M's spatial pattern change differently post-injury?
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
import warnings
from pathlib import Path
from scipy import stats

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_connectome_root = SCRIPT_DIR
while os.path.basename(_connectome_root) != '2_Connectome' and os.path.dirname(_connectome_root) != _connectome_root:
    _connectome_root = os.path.dirname(_connectome_root)

OUTPUT_DIR = os.path.join(_connectome_root, 'Databases', 'figures', 'bodypart_tracking_quality')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DLC_BASE = Path(r"X:\! DLC Output\Analyzed")
UCSF_DATA_PATH = os.path.join(
    r'C:\Users\friedrichl\OneDrive - Marquette University\Blackmore Lab Notes - Sharepoint',
    r'3 Lab Projects\Automated single pellet apparatus\!UCSF_Collab',
    r'May2025_Uploads\ODC Uploads\Session_Data.csv'
)

GROUPS = ['K', 'L', 'M']
GROUP_COLORS = {'K': '#e377c2', 'L': '#2ca02c', 'M': '#ff7f0e'}
GROUP_LABELS = {'K': 'K (70kD)', 'L': 'L (50kD)', 'M': 'M (60kD)'}

PHASE_MAP = {
    'Pre-Injury': ['2_Pre-injury_1', '2_Pre-injury_2', '2_Pre-injury_3'],
    'Immediate Post': ['3_1wk_Post-injury', '3_1wk_Post-injury_First_Test'],
    '2-4wk Post': ['3_2wk_Post-injury', '3_3wk_Post-injury', '3_4wk_Post-injury'],
    'Post-Rehab': ['5_Post-rehab_Test_1', '5_Post-rehab_Test_2'],
}

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.spines.top': False, 'axes.spines.right': False,
})


def extract_animal_id(filename):
    match = re.search(r'\d{8}_([A-Z]\d{2})_', filename)
    return match.group(1) if match else None


def extract_date(filename):
    match = re.search(r'(\d{8})_', filename)
    return match.group(1) if match else None


def build_phase_lookup(ucsf_path):
    df = pd.read_csv(ucsf_path)
    test_to_phase = {}
    for phase, tests in PHASE_MAP.items():
        for t in tests:
            test_to_phase[t] = phase

    lookup = {}
    for _, row in df.iterrows():
        sid = row['SubjectID']
        try:
            dt = pd.to_datetime(row['Test_Date'], format='%m/%d/%Y')
            date_str = dt.strftime('%Y%m%d')
        except Exception:
            continue
        phase = test_to_phase.get(row['Test_Type_Grouped_1'], None)
        if phase:
            lookup[(sid, date_str)] = phase
    return lookup


def load_dlc_full(csv_path):
    """Load DLC CSV, return full dataframe with all bodypart x, y, likelihood."""
    try:
        df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
        return df
    except Exception:
        return None


def compute_positional_metrics(df):
    """Compute positional summary metrics from a full DLC dataframe."""
    scorer = df.columns.get_level_values(0)[0]
    metrics = {}

    def get_col(bp, coord):
        try:
            return df[(scorer, bp, coord)].values.astype(float)
        except KeyError:
            return None

    # RightHand position stats
    rh_x = get_col('RightHand', 'x')
    rh_y = get_col('RightHand', 'y')
    rh_lik = get_col('RightHand', 'likelihood')

    if rh_x is None or rh_y is None:
        return None

    # Only use high-confidence frames for position analysis
    if rh_lik is not None:
        high_conf = rh_lik > 0.5
        rh_x_hc = rh_x[high_conf] if high_conf.sum() > 10 else rh_x
        rh_y_hc = rh_y[high_conf] if high_conf.sum() > 10 else rh_y
    else:
        rh_x_hc = rh_x
        rh_y_hc = rh_y

    metrics['rh_mean_x'] = np.mean(rh_x_hc)
    metrics['rh_mean_y'] = np.mean(rh_y_hc)
    metrics['rh_std_x'] = np.std(rh_x_hc)
    metrics['rh_std_y'] = np.std(rh_y_hc)
    metrics['rh_range_x'] = np.ptp(rh_x_hc)
    metrics['rh_range_y'] = np.ptp(rh_y_hc)
    metrics['rh_n_highconf'] = int(high_conf.sum()) if rh_lik is not None else len(rh_x)
    metrics['rh_pct_highconf'] = 100 * metrics['rh_n_highconf'] / len(rh_x)

    # RightHand max extension (minimum y in image coords = highest reach)
    metrics['rh_max_extent_y'] = np.min(rh_y_hc) if len(rh_y_hc) > 0 else np.nan
    # Percentiles of RH y position
    if len(rh_y_hc) > 0:
        metrics['rh_y_p05'] = np.percentile(rh_y_hc, 5)
        metrics['rh_y_p25'] = np.percentile(rh_y_hc, 25)
        metrics['rh_y_p50'] = np.percentile(rh_y_hc, 50)
        metrics['rh_y_p75'] = np.percentile(rh_y_hc, 75)
        metrics['rh_y_p95'] = np.percentile(rh_y_hc, 95)

    # Frame-to-frame jitter (displacement between consecutive frames)
    # This catches noisy tracking even when likelihood is high
    dx = np.diff(rh_x)
    dy = np.diff(rh_y)
    displacements = np.sqrt(dx**2 + dy**2)
    metrics['rh_mean_jitter'] = np.mean(displacements)
    metrics['rh_median_jitter'] = np.median(displacements)
    metrics['rh_p95_jitter'] = np.percentile(displacements, 95)
    metrics['rh_p99_jitter'] = np.percentile(displacements, 99)
    metrics['rh_max_jitter'] = np.max(displacements)

    # Large jumps (> 50 pixels in one frame = likely tracking error)
    metrics['rh_n_large_jumps'] = int(np.sum(displacements > 50))
    metrics['rh_pct_large_jumps'] = 100 * metrics['rh_n_large_jumps'] / len(displacements)

    # High-confidence jitter (jitter only when BOTH frames are high confidence)
    if rh_lik is not None:
        both_hc = high_conf[:-1] & high_conf[1:]
        if both_hc.sum() > 10:
            hc_displacements = displacements[both_hc]
            metrics['rh_hc_mean_jitter'] = np.mean(hc_displacements)
            metrics['rh_hc_median_jitter'] = np.median(hc_displacements)
            metrics['rh_hc_p95_jitter'] = np.percentile(hc_displacements, 95)
            metrics['rh_hc_n_large_jumps'] = int(np.sum(hc_displacements > 50))

    # Nose position
    nose_x = get_col('Nose', 'x')
    nose_y = get_col('Nose', 'y')
    nose_lik = get_col('Nose', 'likelihood')
    if nose_x is not None and nose_y is not None:
        nose_hc = nose_lik > 0.5 if nose_lik is not None else np.ones(len(nose_x), dtype=bool)
        if nose_hc.sum() > 10:
            metrics['nose_mean_x'] = np.mean(nose_x[nose_hc])
            metrics['nose_mean_y'] = np.mean(nose_y[nose_hc])
            metrics['nose_std_x'] = np.std(nose_x[nose_hc])
            metrics['nose_std_y'] = np.std(nose_y[nose_hc])

    # Inter-bodypart distances (high-conf frames only)
    # Nose-to-RightHand distance (proxy for reach extension)
    if nose_x is not None:
        both_hc_nr = (rh_lik > 0.5) & (nose_lik > 0.5) if (rh_lik is not None and nose_lik is not None) else np.ones(len(rh_x), dtype=bool)
        if both_hc_nr.sum() > 10:
            nr_dist = np.sqrt((rh_x[both_hc_nr] - nose_x[both_hc_nr])**2 +
                              (rh_y[both_hc_nr] - nose_y[both_hc_nr])**2)
            metrics['nose_hand_mean_dist'] = np.mean(nr_dist)
            metrics['nose_hand_std_dist'] = np.std(nr_dist)
            metrics['nose_hand_max_dist'] = np.max(nr_dist)
            metrics['nose_hand_p05_dist'] = np.percentile(nr_dist, 5)

    # Ear-to-ear distance (head width proxy)
    re_x = get_col('RightEar', 'x')
    re_y = get_col('RightEar', 'y')
    le_x = get_col('LeftEar', 'x')
    le_y = get_col('LeftEar', 'y')
    re_lik = get_col('RightEar', 'likelihood')
    le_lik = get_col('LeftEar', 'likelihood')
    if re_x is not None and le_x is not None:
        both_ears = np.ones(len(re_x), dtype=bool)
        if re_lik is not None and le_lik is not None:
            both_ears = (re_lik > 0.5) & (le_lik > 0.5)
        if both_ears.sum() > 10:
            ear_dist = np.sqrt((re_x[both_ears] - le_x[both_ears])**2 +
                               (re_y[both_ears] - le_y[both_ears])**2)
            metrics['ear_ear_mean_dist'] = np.mean(ear_dist)
            metrics['ear_ear_std_dist'] = np.std(ear_dist)

    # Reference-to-Pillar distance (should be constant across all groups/phases)
    ref_x = get_col('Reference', 'x')
    ref_y = get_col('Reference', 'y')
    pil_x = get_col('Pillar', 'x')
    pil_y = get_col('Pillar', 'y')
    if ref_x is not None and pil_x is not None:
        ref_lik = get_col('Reference', 'likelihood')
        pil_lik = get_col('Pillar', 'likelihood')
        both_rp = np.ones(len(ref_x), dtype=bool)
        if ref_lik is not None and pil_lik is not None:
            both_rp = (ref_lik > 0.9) & (pil_lik > 0.5)
        if both_rp.sum() > 10:
            rp_dist = np.sqrt((ref_x[both_rp] - pil_x[both_rp])**2 +
                              (ref_y[both_rp] - pil_y[both_rp])**2)
            metrics['ref_pillar_mean_dist'] = np.mean(rp_dist)
            metrics['ref_pillar_std_dist'] = np.std(rp_dist)

    # Nose-to-Pillar distance (approach distance)
    if nose_x is not None and pil_x is not None:
        both_np = np.ones(len(nose_x), dtype=bool)
        if nose_lik is not None and pil_lik is not None:
            both_np = (nose_lik > 0.5) & (pil_lik > 0.5)
        if both_np.sum() > 10:
            np_dist = np.sqrt((nose_x[both_np] - pil_x[both_np])**2 +
                              (nose_y[both_np] - pil_y[both_np])**2)
            metrics['nose_pillar_mean_dist'] = np.mean(np_dist)
            metrics['nose_pillar_std_dist'] = np.std(np_dist)

    metrics['total_frames'] = len(rh_x)
    return metrics


def load_all_positional_data(phase_lookup):
    """Load positional metrics for all groups."""
    print("Loading positional data from DLC CSVs...")
    rows = []

    for grp in GROUPS:
        grp_path = DLC_BASE / grp / 'Post-Processing'
        if not grp_path.exists():
            grp_path = DLC_BASE / grp / 'Multi-Animal'

        csvs = sorted(grp_path.glob('*DLC*.csv'))
        print(f"  Group {grp}: processing {len(csvs)} CSVs")

        for ci, csv_file in enumerate(csvs):
            if ci % 100 == 0 and ci > 0:
                print(f"    ... {ci}/{len(csvs)}")

            aid = extract_animal_id(csv_file.name)
            date = extract_date(csv_file.name)
            if not aid:
                continue

            phase = phase_lookup.get((aid, date), 'Unknown')

            dlc_df = load_dlc_full(csv_file)
            if dlc_df is None:
                continue

            metrics = compute_positional_metrics(dlc_df)
            if metrics is None:
                continue

            metrics['group'] = grp
            metrics['animal_id'] = aid
            metrics['date'] = date
            metrics['phase'] = phase
            metrics['filename'] = csv_file.name
            rows.append(metrics)

    df = pd.DataFrame(rows)
    print(f"  Total sessions: {len(df)}")
    return df


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_jitter(df):
    """Compare frame-to-frame RightHand jitter across groups."""
    print("\n" + "=" * 80)
    print("RIGHTHAND FRAME-TO-FRAME JITTER (tracking noise)")
    print("=" * 80)

    df_known = df[df['phase'] != 'Unknown'].copy()

    for metric, label in [
        ('rh_mean_jitter', 'Mean frame-to-frame displacement (all frames)'),
        ('rh_hc_mean_jitter', 'Mean frame-to-frame displacement (high-conf only)'),
        ('rh_p95_jitter', '95th pctl displacement (all frames)'),
        ('rh_pct_large_jumps', '% frames with >50px jumps'),
    ]:
        if metric not in df_known.columns:
            continue

        print(f"\n  {label}:")
        for grp in GROUPS:
            vals = df_known[df_known['group'] == grp][metric].dropna()
            print(f"    {GROUP_LABELS[grp]}: mean={vals.mean():.2f} (SD={vals.std():.2f}, n={len(vals)})")

        kl_vals = df_known[df_known['group'].isin(['K', 'L'])][metric].dropna()
        m_vals = df_known[df_known['group'] == 'M'][metric].dropna()
        if len(kl_vals) > 2 and len(m_vals) > 2:
            u, p = stats.mannwhitneyu(kl_vals, m_vals, alternative='two-sided')
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            print(f"    M vs K+L: p={p:.6f} {sig}")


def analyze_spatial_relationships(df):
    """Compare inter-bodypart distances across groups by phase."""
    print("\n" + "=" * 80)
    print("SPATIAL RELATIONSHIPS BY GROUP AND PHASE")
    print("=" * 80)

    df_known = df[df['phase'] != 'Unknown'].copy()
    phases_ordered = ['Pre-Injury', 'Immediate Post', '2-4wk Post', 'Post-Rehab']

    for metric, label in [
        ('nose_hand_mean_dist', 'Nose-to-Hand mean distance (reach proxy)'),
        ('nose_hand_max_dist', 'Nose-to-Hand max distance (max extension)'),
        ('ear_ear_mean_dist', 'Ear-to-Ear mean distance (head width)'),
        ('nose_pillar_mean_dist', 'Nose-to-Pillar mean distance (approach)'),
    ]:
        if metric not in df_known.columns:
            continue

        print(f"\n  {label}:")
        for phase in phases_ordered:
            phase_data = df_known[df_known['phase'] == phase]
            vals_by_grp = {}
            for grp in GROUPS:
                vals = phase_data[phase_data['group'] == grp][metric].dropna()
                if len(vals) > 0:
                    vals_by_grp[grp] = vals

            if len(vals_by_grp) < 2:
                continue

            line = f"    {phase:18s}:"
            for grp in GROUPS:
                if grp in vals_by_grp:
                    line += f"  {grp}={vals_by_grp[grp].mean():.1f}"
            print(line)

            # M vs K+L test
            if 'M' in vals_by_grp and any(g in vals_by_grp for g in ['K', 'L']):
                kl_vals = pd.concat([vals_by_grp[g] for g in ['K', 'L'] if g in vals_by_grp])
                m_vals = vals_by_grp['M']
                if len(kl_vals) > 2 and len(m_vals) > 2:
                    u, p = stats.mannwhitneyu(kl_vals, m_vals, alternative='two-sided')
                    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                    diff = m_vals.mean() - kl_vals.mean()
                    print(f"                      M vs K+L: diff={diff:+.1f} p={p:.4f} {sig}")


def analyze_positional_shifts(df):
    """Check if M's spatial patterns change differently pre-to-post injury."""
    print("\n" + "=" * 80)
    print("POSITIONAL SHIFTS: Pre-Injury -> Post-Injury changes by group")
    print("=" * 80)

    df_known = df[df['phase'] != 'Unknown'].copy()

    for metric, label in [
        ('rh_mean_y', 'RightHand mean Y position'),
        ('rh_max_extent_y', 'RightHand max extension (min Y)'),
        ('rh_mean_jitter', 'RightHand mean jitter'),
        ('nose_hand_mean_dist', 'Nose-Hand distance'),
        ('nose_hand_max_dist', 'Nose-Hand max distance'),
        ('rh_range_y', 'RightHand Y range (spatial spread)'),
    ]:
        if metric not in df_known.columns:
            continue

        print(f"\n  {label}:")
        for grp in GROUPS:
            grp_data = df_known[df_known['group'] == grp]
            pre = grp_data[grp_data['phase'] == 'Pre-Injury'].groupby('animal_id')[metric].mean()
            post = grp_data[grp_data['phase'].isin(['Immediate Post', '2-4wk Post'])].groupby('animal_id')[metric].mean()
            common = pre.index.intersection(post.index)
            if len(common) < 2:
                print(f"    {GROUP_LABELS[grp]}: insufficient paired data (n={len(common)})")
                continue
            shifts = post.loc[common] - pre.loc[common]
            pct_shift = (shifts / pre.loc[common].abs().replace(0, np.nan) * 100).dropna()
            print(f"    {GROUP_LABELS[grp]}: shift={shifts.mean():+.1f} ({pct_shift.mean():+.1f}%) "
                  f"SD={shifts.std():.1f} n={len(common)}")

        # M vs K+L shift comparison
        m_data = df_known[df_known['group'] == 'M']
        kl_data = df_known[df_known['group'].isin(['K', 'L'])]

        m_pre = m_data[m_data['phase'] == 'Pre-Injury'].groupby('animal_id')[metric].mean()
        m_post = m_data[m_data['phase'].isin(['Immediate Post', '2-4wk Post'])].groupby('animal_id')[metric].mean()
        m_common = m_pre.index.intersection(m_post.index)
        m_shifts = (m_post.loc[m_common] - m_pre.loc[m_common]).values if len(m_common) > 1 else np.array([])

        kl_pre = kl_data[kl_data['phase'] == 'Pre-Injury'].groupby('animal_id')[metric].mean()
        kl_post = kl_data[kl_data['phase'].isin(['Immediate Post', '2-4wk Post'])].groupby('animal_id')[metric].mean()
        kl_common = kl_pre.index.intersection(kl_post.index)
        kl_shifts = (kl_post.loc[kl_common] - kl_pre.loc[kl_common]).values if len(kl_common) > 1 else np.array([])

        if len(m_shifts) > 1 and len(kl_shifts) > 1:
            u, p = stats.mannwhitneyu(m_shifts, kl_shifts, alternative='two-sided')
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            print(f"    M vs K+L shift comparison: p={p:.4f} {sig}")


# ============================================================================
# FIGURES
# ============================================================================

def fig_jitter_comparison(df):
    """Violin plots of RightHand jitter metrics by group."""
    df_known = df[df['phase'] != 'Unknown'].copy()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('RightHand Frame-to-Frame Jitter by Group\n'
                 'Higher jitter = noisier tracking (even if likelihood is high)',
                 fontsize=14, fontweight='bold')

    metrics = [
        ('rh_mean_jitter', 'Mean Jitter (px/frame)'),
        ('rh_hc_mean_jitter', 'Mean Jitter (high-conf frames only)'),
        ('rh_p95_jitter', '95th Pctl Jitter (px/frame)'),
        ('rh_pct_large_jumps', '% Frames with >50px Jumps'),
    ]

    for ax_i, (col, ylabel) in enumerate(metrics):
        ax = axes[ax_i // 2, ax_i % 2]
        if col not in df_known.columns:
            ax.set_title(f'{ylabel}\n(not available)')
            continue

        data_by_group = []
        for grp in GROUPS:
            vals = df_known[df_known['group'] == grp][col].dropna().values
            data_by_group.append(vals)

        parts = ax.violinplot(data_by_group, positions=range(len(GROUPS)),
                              showmeans=False, showmedians=False, showextrema=False)
        for gi, pc in enumerate(parts['bodies']):
            pc.set_facecolor(GROUP_COLORS[GROUPS[gi]])
            pc.set_alpha(0.3)

        bp = ax.boxplot(data_by_group, positions=range(len(GROUPS)), widths=0.3,
                        patch_artist=True, medianprops=dict(color='black', linewidth=2),
                        flierprops=dict(markersize=3, alpha=0.3))
        for gi, patch in enumerate(bp['boxes']):
            patch.set_facecolor(GROUP_COLORS[GROUPS[gi]])
            patch.set_alpha(0.6)

        # M vs K+L stat
        kl = np.concatenate([data_by_group[0], data_by_group[1]])
        m = data_by_group[2]
        if len(kl) > 1 and len(m) > 1:
            u, p = stats.mannwhitneyu(kl, m, alternative='two-sided')
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            ax.text(0.95, 0.95, f'M vs K+L: p={p:.4f} {sig}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels([f'{GROUP_LABELS[g]}\n(n={len(data_by_group[i])})'
                            for i, g in enumerate(GROUPS)])
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontsize=12, fontweight='bold')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig5_righthand_jitter.png')
    plt.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out}")


def fig_spatial_relationships_by_phase(df):
    """Line plots: inter-bodypart distances across phases by group."""
    df_known = df[df['phase'] != 'Unknown'].copy()
    phases_ordered = ['Pre-Injury', 'Immediate Post', '2-4wk Post', 'Post-Rehab']

    metrics = [
        ('nose_hand_mean_dist', 'Nose-Hand Mean Distance\n(reach workspace size)'),
        ('nose_hand_max_dist', 'Nose-Hand Max Distance\n(maximum extension)'),
        ('ear_ear_mean_dist', 'Ear-Ear Distance\n(head width consistency)'),
        ('rh_range_y', 'RightHand Y Range\n(vertical reach spread)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Spatial Relationships Across Phases\n'
                 'Do M animals show different positional patterns post-injury?',
                 fontsize=14, fontweight='bold')

    for ax_i, (col, ylabel) in enumerate(metrics):
        ax = axes[ax_i // 2, ax_i % 2]
        if col not in df_known.columns:
            ax.set_title(f'{ylabel}\n(not available)')
            continue

        for grp in GROUPS:
            means = []
            sems = []
            valid_phases = []
            for phase in phases_ordered:
                vals = df_known[(df_known['group'] == grp) & (df_known['phase'] == phase)][col].dropna()
                if len(vals) >= 3:
                    means.append(vals.mean())
                    sems.append(vals.std() / np.sqrt(len(vals)))
                    valid_phases.append(phase)

            if means:
                x = [phases_ordered.index(p) for p in valid_phases]
                ax.errorbar(x, means, yerr=sems, marker='o', linewidth=2,
                           color=GROUP_COLORS[grp], label=GROUP_LABELS[grp],
                           capsize=4, markersize=6)

        ax.set_xticks(range(len(phases_ordered)))
        ax.set_xticklabels(phases_ordered, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel(ylabel.split('\n')[0])
        ax.set_title(ylabel, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig6_spatial_relationships_by_phase.png')
    plt.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out}")


def fig_positional_shifts(df):
    """Bar chart: pre-to-post shifts in key spatial metrics by group."""
    df_known = df[df['phase'] != 'Unknown'].copy()

    metrics = [
        ('rh_mean_y', 'RH Mean Y'),
        ('rh_max_extent_y', 'RH Max Extent'),
        ('rh_mean_jitter', 'RH Mean Jitter'),
        ('nose_hand_mean_dist', 'Nose-Hand Dist'),
        ('nose_hand_max_dist', 'Nose-Hand Max'),
        ('rh_range_y', 'RH Y Range'),
    ]

    fig, ax = plt.subplots(figsize=(18, 8))
    fig.suptitle('Pre-Injury to Post-Injury Shifts in Spatial Metrics\n'
                 'Does M change differently than K and L?',
                 fontsize=14, fontweight='bold')

    bar_width = 0.25
    x_base = np.arange(len(metrics))

    for gi, grp in enumerate(GROUPS):
        grp_data = df_known[df_known['group'] == grp]
        means = []
        errs = []
        for col, _ in metrics:
            if col not in df_known.columns:
                means.append(0)
                errs.append(0)
                continue
            pre = grp_data[grp_data['phase'] == 'Pre-Injury'].groupby('animal_id')[col].mean()
            post = grp_data[grp_data['phase'].isin(['Immediate Post', '2-4wk Post'])].groupby('animal_id')[col].mean()
            common = pre.index.intersection(post.index)
            if len(common) < 2:
                means.append(0)
                errs.append(0)
                continue
            shifts = post.loc[common] - pre.loc[common]
            # Normalize: percent change from pre
            pct_shifts = (shifts / pre.loc[common].abs().replace(0, np.nan) * 100).dropna()
            means.append(pct_shifts.mean())
            errs.append(pct_shifts.std() / np.sqrt(len(pct_shifts)))

        ax.bar(x_base + gi * bar_width, means, bar_width, yerr=errs,
               color=GROUP_COLORS[grp], label=GROUP_LABELS[grp], alpha=0.8,
               edgecolor='black', linewidth=0.5, capsize=3)

    ax.set_xticks(x_base + bar_width)
    ax.set_xticklabels([label for _, label in metrics], rotation=30, ha='right')
    ax.set_ylabel('% Change (Post - Pre)')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.legend(fontsize=10)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig7_positional_shifts.png')
    plt.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("BODYPART POSITIONAL ANALYSIS -- CROSS-GROUP")
    print("Does M (60kD) violate positional trends seen in K and L?")
    print("=" * 80)

    phase_lookup = build_phase_lookup(UCSF_DATA_PATH)
    df = load_all_positional_data(phase_lookup)

    # Save raw
    raw_out = os.path.join(OUTPUT_DIR, 'positional_metrics_raw.csv')
    df.to_csv(raw_out, index=False)
    print(f"\nRaw data saved: {raw_out}")

    # Analysis
    analyze_jitter(df)
    analyze_spatial_relationships(df)
    analyze_positional_shifts(df)

    # Figures
    print("\nGenerating figures...")
    fig_jitter_comparison(df)
    fig_spatial_relationships_by_phase(df)
    fig_positional_shifts(df)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
