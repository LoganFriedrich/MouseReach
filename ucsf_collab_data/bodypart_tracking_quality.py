"""
Cross-Group Bodypart DLC Tracking Quality Analysis

Hypothesis: If DLC tracks M (60kD) bodyparts worse than K/L, that noise
could flatten kinematic profiles by adding variance that washes out real deficits.

Approach:
  1. Find bodyparts with high confidence in K and L across all phases
  2. Check if those same bodyparts have lower confidence in M
  3. Look for outlier sessions/animals in M
  4. Break down by experimental phase to see if tracking degrades differentially

Bodyparts tracked: Reference, Pellet, Pillar, RightHand, Nose, RightEar, LeftEar, LeftFoot, TailBase
Kinematic-critical: RightHand (trajectory/velocity), Nose (posture), RightEar/LeftEar (head angle)
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
from collections import defaultdict

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Paths ---
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
BODYPARTS = ['Reference', 'Pellet', 'Pillar', 'RightHand', 'Nose', 'RightEar', 'LeftEar', 'LeftFoot', 'TailBase']
KINEMATIC_BODYPARTS = ['RightHand', 'Nose', 'RightEar', 'LeftEar']

# Phase grouping for analysis
PHASE_MAP = {
    'Pre-Injury': ['2_Pre-injury_1', '2_Pre-injury_2', '2_Pre-injury_3'],
    'Immediate Post': ['3_1wk_Post-injury', '3_1wk_Post-injury_First_Test'],
    '2-4wk Post': ['3_2wk_Post-injury', '3_3wk_Post-injury', '3_4wk_Post-injury'],
    '5+ wk Post': ['3_5wk_Post-injury', '3_6wk_Post-injury', '3_7wk_Post-injury',
                    '3_8wk_Post-injury', '3_9wk_Post-injury'],
    'Rehab': ['4_Rehab_1_easy', '4_Rehab_2_easy', '4_Rehab_3_easy', '4_Rehab_4_easy',
              '4_Rehab_5_easy', '4_Rehab_6_flat', '4_Rehab_7_flat', '4_Rehab_8_flat'],
    'Post-Rehab': ['5_Post-rehab_Test_1', '5_Post-rehab_Test_2'],
}

GROUP_COLORS = {'K': '#e377c2', 'L': '#2ca02c', 'M': '#ff7f0e'}
GROUP_LABELS = {'K': 'K (70kD)', 'L': 'L (50kD)', 'M': 'M (60kD)'}

# Style
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.spines.top': False, 'axes.spines.right': False,
})


# ============================================================================
# DATA LOADING
# ============================================================================

def extract_animal_id(filename):
    match = re.search(r'\d{8}_([A-Z]\d{2})_', filename)
    return match.group(1) if match else None


def extract_date(filename):
    match = re.search(r'(\d{8})_', filename)
    return match.group(1) if match else None


def build_phase_lookup(ucsf_path):
    """Build (SubjectID, date_YYYYMMDD) -> phase lookup from Session_Data.csv."""
    df = pd.read_csv(ucsf_path)

    # Reverse the PHASE_MAP for fast lookup
    test_to_phase = {}
    for phase, tests in PHASE_MAP.items():
        for t in tests:
            test_to_phase[t] = phase

    # Parse Test_Date (MM/DD/YYYY) to YYYYMMDD
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

    print(f"  Phase lookup built: {len(lookup)} (subject, date) -> phase mappings")
    return lookup


def load_bodypart_likelihoods(csv_path):
    """Load DLC CSV, return dict of bodypart -> likelihood array."""
    try:
        df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
        scorer = df.columns.get_level_values(0)[0]

        result = {}
        for bp in BODYPARTS:
            try:
                lik = df[(scorer, bp, 'likelihood')].values.astype(float)
                result[bp] = lik
            except KeyError:
                pass
        return result
    except Exception as e:
        return None


def load_all_bodypart_data(phase_lookup, max_per_group=None):
    """Load per-bodypart likelihood stats for all groups, mapped to phases."""
    print("Loading bodypart tracking data from DLC CSVs...")
    rows = []

    for grp in GROUPS:
        grp_path = DLC_BASE / grp / 'Post-Processing'
        if not grp_path.exists():
            grp_path = DLC_BASE / grp / 'Multi-Animal'

        csvs = sorted(grp_path.glob('*DLC*.csv'))
        if max_per_group and len(csvs) > max_per_group:
            np.random.seed(42)
            csvs = list(np.random.choice(csvs, max_per_group, replace=False))

        print(f"  Group {grp}: processing {len(csvs)} CSVs from {grp_path}")

        for ci, csv_file in enumerate(csvs):
            if ci % 100 == 0 and ci > 0:
                print(f"    ... {ci}/{len(csvs)}")

            aid = extract_animal_id(csv_file.name)
            date = extract_date(csv_file.name)
            if not aid:
                continue

            phase = phase_lookup.get((aid, date), 'Unknown')

            bp_data = load_bodypart_likelihoods(csv_file)
            if bp_data is None:
                continue

            row = {
                'group': grp,
                'animal_id': aid,
                'date': date,
                'phase': phase,
                'filename': csv_file.name,
                'total_frames': len(next(iter(bp_data.values()))),
            }

            for bp, lik in bp_data.items():
                row[f'{bp}_mean_lik'] = np.mean(lik)
                row[f'{bp}_median_lik'] = np.median(lik)
                row[f'{bp}_pct_above_09'] = 100 * np.mean(lik > 0.9)
                row[f'{bp}_pct_below_05'] = 100 * np.mean(lik < 0.5)
                row[f'{bp}_pct_below_03'] = 100 * np.mean(lik < 0.3)
                row[f'{bp}_std_lik'] = np.std(lik)

            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  Total sessions loaded: {len(df)}")
    print(f"  Per group: {dict(df.groupby('group').size())}")
    print(f"  Phase distribution:")
    for phase in list(PHASE_MAP.keys()) + ['Unknown']:
        n = len(df[df['phase'] == phase])
        if n > 0:
            print(f"    {phase}: {n}")
    return df


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_high_confidence_bodyparts(df):
    """Find bodyparts that are consistently high-confidence in K and L,
    then check if M deviates."""
    print("\n" + "=" * 80)
    print("BODYPART CONFIDENCE: K & L BASELINE vs M COMPARISON")
    print("=" * 80)

    # Only use known phases (exclude Unknown)
    df_known = df[df['phase'] != 'Unknown'].copy()

    results = []
    for bp in BODYPARTS:
        col = f'{bp}_mean_lik'
        if col not in df_known.columns:
            continue

        kl_vals = df_known[df_known['group'].isin(['K', 'L'])][col].dropna()
        m_vals = df_known[df_known['group'] == 'M'][col].dropna()

        if len(kl_vals) < 5 or len(m_vals) < 5:
            continue

        kl_mean = kl_vals.mean()
        m_mean = m_vals.mean()
        diff = m_mean - kl_mean

        # Mann-Whitney test
        u_stat, p_val = stats.mannwhitneyu(kl_vals, m_vals, alternative='two-sided')

        # Cohen's d
        pooled_std = np.sqrt(((len(kl_vals)-1)*kl_vals.std()**2 + (len(m_vals)-1)*m_vals.std()**2) /
                             (len(kl_vals) + len(m_vals) - 2))
        d = diff / pooled_std if pooled_std > 0 else 0

        is_kinematic = bp in KINEMATIC_BODYPARTS
        results.append({
            'bodypart': bp,
            'kinematic': is_kinematic,
            'KL_mean': kl_mean,
            'KL_std': kl_vals.std(),
            'M_mean': m_mean,
            'M_std': m_vals.std(),
            'diff': diff,
            'cohens_d': d,
            'p_value': p_val,
            'KL_n': len(kl_vals),
            'M_n': len(m_vals),
        })

        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        kin_marker = ' [KINEMATIC]' if is_kinematic else ''
        print(f"\n  {bp}{kin_marker}:")
        print(f"    K+L mean: {kl_mean:.4f} (SD={kl_vals.std():.4f}, n={len(kl_vals)})")
        print(f"    M   mean: {m_mean:.4f} (SD={m_vals.std():.4f}, n={len(m_vals)})")
        print(f"    Diff: {diff:+.4f}, Cohen's d={d:.3f}, p={p_val:.6f} {sig}")

    return pd.DataFrame(results)


def analyze_by_phase(df):
    """Compare M vs K+L for kinematic bodyparts broken down by phase."""
    print("\n" + "=" * 80)
    print("PHASE-BY-PHASE COMPARISON: KINEMATIC BODYPARTS")
    print("=" * 80)

    df_known = df[df['phase'] != 'Unknown'].copy()
    phases_ordered = ['Pre-Injury', 'Immediate Post', '2-4wk Post', '5+ wk Post', 'Post-Rehab']

    results = []
    for bp in KINEMATIC_BODYPARTS:
        col = f'{bp}_mean_lik'
        if col not in df_known.columns:
            continue

        print(f"\n  {bp}:")
        for phase in phases_ordered:
            phase_data = df_known[df_known['phase'] == phase]
            kl_vals = phase_data[phase_data['group'].isin(['K', 'L'])][col].dropna()
            m_vals = phase_data[phase_data['group'] == 'M'][col].dropna()

            if len(kl_vals) < 3 or len(m_vals) < 3:
                print(f"    {phase}: insufficient data (KL={len(kl_vals)}, M={len(m_vals)})")
                continue

            diff = m_vals.mean() - kl_vals.mean()
            u_stat, p_val = stats.mannwhitneyu(kl_vals, m_vals, alternative='two-sided')
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'

            results.append({
                'bodypart': bp, 'phase': phase,
                'KL_mean': kl_vals.mean(), 'M_mean': m_vals.mean(),
                'diff': diff, 'p_value': p_val,
                'KL_n': len(kl_vals), 'M_n': len(m_vals),
            })

            print(f"    {phase:18s}: KL={kl_vals.mean():.4f} M={m_vals.mean():.4f} "
                  f"diff={diff:+.4f} p={p_val:.4f} {sig} (KL n={len(kl_vals)}, M n={len(m_vals)})")

    return pd.DataFrame(results)


def find_outlier_animals(df):
    """Identify M animals with unusually low bodypart tracking quality."""
    print("\n" + "=" * 80)
    print("OUTLIER DETECTION: M ANIMALS WITH LOW TRACKING ON KINEMATIC BODYPARTS")
    print("=" * 80)

    df_known = df[df['phase'] != 'Unknown'].copy()

    for bp in KINEMATIC_BODYPARTS:
        col = f'{bp}_mean_lik'
        if col not in df_known.columns:
            continue

        # Get K+L distribution as reference
        kl_vals = df_known[df_known['group'].isin(['K', 'L'])][col].dropna()
        threshold_low = kl_vals.quantile(0.05)  # Below 5th percentile of K+L

        # Check each M animal
        m_data = df_known[df_known['group'] == 'M'].groupby('animal_id')[col].agg(['mean', 'std', 'count'])
        m_outliers = m_data[m_data['mean'] < threshold_low]

        if len(m_outliers) > 0:
            print(f"\n  {bp} (K+L 5th pctl threshold: {threshold_low:.4f}):")
            for aid, row in m_outliers.iterrows():
                print(f"    {aid}: mean={row['mean']:.4f} (SD={row['std']:.4f}, n={row['count']:.0f}) ** OUTLIER **")
        else:
            print(f"\n  {bp}: No M outliers below K+L 5th percentile ({threshold_low:.4f})")

    # Also: per-animal summary across ALL kinematic bodyparts
    print("\n  --- Per-Animal Composite Kinematic Tracking Score ---")
    kin_cols = [f'{bp}_mean_lik' for bp in KINEMATIC_BODYPARTS if f'{bp}_mean_lik' in df_known.columns]
    df_known['composite_kin_lik'] = df_known[kin_cols].mean(axis=1)

    animal_composite = df_known.groupby(['group', 'animal_id'])['composite_kin_lik'].agg(['mean', 'std', 'count'])
    animal_composite = animal_composite.reset_index()

    # Print M animals sorted by composite score
    m_animals = animal_composite[animal_composite['group'] == 'M'].sort_values('mean')
    kl_mean = animal_composite[animal_composite['group'].isin(['K', 'L'])]['mean'].mean()
    kl_std = animal_composite[animal_composite['group'].isin(['K', 'L'])]['mean'].std()

    print(f"\n  K+L composite mean: {kl_mean:.4f} (SD={kl_std:.4f})")
    print(f"  M animals (sorted by composite kinematic tracking score):")
    for _, row in m_animals.iterrows():
        z = (row['mean'] - kl_mean) / kl_std if kl_std > 0 else 0
        flag = ' ** LOW **' if z < -2 else ' * below avg *' if z < -1 else ''
        print(f"    {row['animal_id']}: {row['mean']:.4f} (z={z:+.2f}, n={row['count']:.0f}){flag}")


def analyze_phase_shift_pattern(df):
    """Check if M's tracking quality CHANGES differently across phases compared to K+L.

    This is the key test: if M's tracking degrades more post-injury than K+L's,
    that differential degradation could explain flat kinematics.
    """
    print("\n" + "=" * 80)
    print("DIFFERENTIAL PHASE SHIFT: Does M tracking degrade MORE post-injury than K+L?")
    print("=" * 80)

    df_known = df[df['phase'] != 'Unknown'].copy()

    for bp in KINEMATIC_BODYPARTS:
        col = f'{bp}_mean_lik'
        if col not in df_known.columns:
            continue

        print(f"\n  {bp}:")

        # Per-animal: compute pre-injury mean, then post-injury mean, then shift
        for grp in GROUPS:
            grp_data = df_known[df_known['group'] == grp]

            pre = grp_data[grp_data['phase'] == 'Pre-Injury'].groupby('animal_id')[col].mean()
            post = grp_data[grp_data['phase'].isin(['Immediate Post', '2-4wk Post'])].groupby('animal_id')[col].mean()

            # Animals with both pre and post
            common = pre.index.intersection(post.index)
            if len(common) < 2:
                print(f"    {GROUP_LABELS[grp]}: insufficient paired data (n={len(common)})")
                continue

            shifts = post.loc[common] - pre.loc[common]
            print(f"    {GROUP_LABELS[grp]}: pre-to-post shift = {shifts.mean():+.4f} "
                  f"(SD={shifts.std():.4f}, n={len(common)})")

        # Statistical test: M shift vs K+L shift
        m_data = df_known[df_known['group'] == 'M']
        kl_data = df_known[df_known['group'].isin(['K', 'L'])]

        m_pre = m_data[m_data['phase'] == 'Pre-Injury'].groupby('animal_id')[col].mean()
        m_post = m_data[m_data['phase'].isin(['Immediate Post', '2-4wk Post'])].groupby('animal_id')[col].mean()
        m_common = m_pre.index.intersection(m_post.index)
        m_shifts = (m_post.loc[m_common] - m_pre.loc[m_common]).values if len(m_common) > 1 else np.array([])

        kl_pre = kl_data[kl_data['phase'] == 'Pre-Injury'].groupby('animal_id')[col].mean()
        kl_post = kl_data[kl_data['phase'].isin(['Immediate Post', '2-4wk Post'])].groupby('animal_id')[col].mean()
        kl_common = kl_pre.index.intersection(kl_post.index)
        kl_shifts = (kl_post.loc[kl_common] - kl_pre.loc[kl_common]).values if len(kl_common) > 1 else np.array([])

        if len(m_shifts) > 1 and len(kl_shifts) > 1:
            u, p = stats.mannwhitneyu(m_shifts, kl_shifts, alternative='two-sided')
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            print(f"    M vs K+L shift comparison: p={p:.4f} {sig}")


# ============================================================================
# FIGURES
# ============================================================================

def fig_bodypart_comparison(df):
    """Heatmap: mean likelihood per bodypart per group, plus per-phase breakdown."""
    df_known = df[df['phase'] != 'Unknown'].copy()

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Bodypart DLC Tracking Quality: K+L vs M\n'
                 'Do kinematic bodyparts track worse in M (60kD)?',
                 fontsize=14, fontweight='bold')

    # Panel A: Overall bodypart means by group
    ax = axes[0]
    data_matrix = []
    for grp in GROUPS:
        row = []
        for bp in BODYPARTS:
            col = f'{bp}_mean_lik'
            if col in df_known.columns:
                row.append(df_known[df_known['group'] == grp][col].mean())
            else:
                row.append(np.nan)
        data_matrix.append(row)

    data_matrix = np.array(data_matrix)
    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8, label='Mean Likelihood')

    ax.set_xticks(range(len(BODYPARTS)))
    ax.set_xticklabels(BODYPARTS, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(GROUPS)))
    ax.set_yticklabels([GROUP_LABELS[g] for g in GROUPS], fontsize=11)
    ax.set_title('A) Mean Likelihood by Bodypart & Group', fontsize=12)

    # Annotate values
    for i in range(len(GROUPS)):
        for j in range(len(BODYPARTS)):
            val = data_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.5 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8, color=color)

    # Highlight kinematic bodyparts
    for j, bp in enumerate(BODYPARTS):
        if bp in KINEMATIC_BODYPARTS:
            ax.get_xticklabels()[j].set_fontweight('bold')
            ax.get_xticklabels()[j].set_color('#d32f2f')

    # Panel B: M minus K+L difference for kinematic bodyparts by phase
    ax = axes[1]
    phases_ordered = ['Pre-Injury', 'Immediate Post', '2-4wk Post', '5+ wk Post', 'Post-Rehab']
    diff_matrix = []
    for bp in KINEMATIC_BODYPARTS:
        col = f'{bp}_mean_lik'
        if col not in df_known.columns:
            diff_matrix.append([np.nan] * len(phases_ordered))
            continue
        row = []
        for phase in phases_ordered:
            phase_data = df_known[df_known['phase'] == phase]
            kl_mean = phase_data[phase_data['group'].isin(['K', 'L'])][col].mean()
            m_mean = phase_data[phase_data['group'] == 'M'][col].mean()
            row.append(m_mean - kl_mean)
        diff_matrix.append(row)

    diff_matrix = np.array(diff_matrix)
    max_abs = max(0.02, np.nanmax(np.abs(diff_matrix)))
    im2 = ax.imshow(diff_matrix, cmap='RdBu', aspect='auto', vmin=-max_abs, vmax=max_abs)
    plt.colorbar(im2, ax=ax, shrink=0.8, label='M minus K+L (likelihood diff)')

    ax.set_xticks(range(len(phases_ordered)))
    ax.set_xticklabels(phases_ordered, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(KINEMATIC_BODYPARTS)))
    ax.set_yticklabels(KINEMATIC_BODYPARTS, fontsize=11)
    ax.set_title('B) M vs K+L Difference by Phase (kinematic bodyparts)\n'
                 'Red = M worse, Blue = M better', fontsize=11)

    for i in range(len(KINEMATIC_BODYPARTS)):
        for j in range(len(phases_ordered)):
            val = diff_matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:+.4f}', ha='center', va='center', fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig1_bodypart_tracking_comparison.png')
    plt.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out}")


def fig_kinematic_bodypart_distributions(df):
    """Violin plots: per-session mean likelihood for each kinematic bodypart, by group."""
    df_known = df[df['phase'] != 'Unknown'].copy()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Kinematic Bodypart Tracking Quality Distributions\n'
                 'Per-session mean likelihood by group',
                 fontsize=14, fontweight='bold')

    for ax_i, bp in enumerate(KINEMATIC_BODYPARTS):
        ax = axes[ax_i // 2, ax_i % 2]
        col = f'{bp}_mean_lik'
        if col not in df_known.columns:
            continue

        data_by_group = []
        for gi, grp in enumerate(GROUPS):
            vals = df_known[df_known['group'] == grp][col].dropna().values
            data_by_group.append(vals)

        # Violin
        parts = ax.violinplot(data_by_group, positions=range(len(GROUPS)),
                              showmeans=False, showmedians=False, showextrema=False)
        for gi, pc in enumerate(parts['bodies']):
            pc.set_facecolor(GROUP_COLORS[GROUPS[gi]])
            pc.set_alpha(0.3)

        # Box
        bp_plot = ax.boxplot(data_by_group, positions=range(len(GROUPS)), widths=0.3,
                             patch_artist=True,
                             medianprops=dict(color='black', linewidth=2),
                             flierprops=dict(markersize=3, alpha=0.3))
        for gi, patch in enumerate(bp_plot['boxes']):
            patch.set_facecolor(GROUP_COLORS[GROUPS[gi]])
            patch.set_alpha(0.6)

        # Stats
        kl_vals = np.concatenate([data_by_group[0], data_by_group[1]])  # K + L
        m_vals = data_by_group[2]  # M
        if len(kl_vals) > 1 and len(m_vals) > 1:
            u, p = stats.mannwhitneyu(kl_vals, m_vals, alternative='two-sided')
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            ax.text(0.95, 0.95, f'M vs K+L: p={p:.4f} {sig}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels([f'{GROUP_LABELS[g]}\n(n={len(data_by_group[i])})' for i, g in enumerate(GROUPS)])
        ax.set_ylabel('Mean Likelihood')
        ax.set_title(f'{bp}', fontsize=13, fontweight='bold')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig2_kinematic_bodypart_distributions.png')
    plt.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out}")


def fig_phase_trajectories(df):
    """Line plots: per-phase mean likelihood trajectory for kinematic bodyparts."""
    df_known = df[df['phase'] != 'Unknown'].copy()
    phases_ordered = ['Pre-Injury', 'Immediate Post', '2-4wk Post', '5+ wk Post', 'Post-Rehab']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Kinematic Bodypart Tracking Quality Across Phases\n'
                 'Does M tracking degrade differently post-injury?',
                 fontsize=14, fontweight='bold')

    for ax_i, bp in enumerate(KINEMATIC_BODYPARTS):
        ax = axes[ax_i // 2, ax_i % 2]
        col = f'{bp}_mean_lik'
        if col not in df_known.columns:
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
        ax.set_ylabel('Mean Likelihood (+/- SEM)')
        ax.set_title(f'{bp}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig3_phase_trajectories.png')
    plt.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out}")


def fig_per_animal_heatmap(df):
    """Heatmap: per M animal, composite kinematic tracking score across phases."""
    df_known = df[(df['phase'] != 'Unknown') & (df['group'] == 'M')].copy()
    phases_ordered = ['Pre-Injury', 'Immediate Post', '2-4wk Post', '5+ wk Post', 'Post-Rehab']

    kin_cols = [f'{bp}_mean_lik' for bp in KINEMATIC_BODYPARTS if f'{bp}_mean_lik' in df_known.columns]
    df_known['composite'] = df_known[kin_cols].mean(axis=1)

    pivot = df_known.groupby(['animal_id', 'phase'])['composite'].mean().reset_index()
    pivot = pivot.pivot(index='animal_id', columns='phase', values='composite')

    # Reorder columns
    pivot = pivot[[p for p in phases_ordered if p in pivot.columns]]

    # Sort by overall mean
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.35 + 2)))
    fig.suptitle('Group M: Per-Animal Composite Kinematic Tracking Score\n'
                 '(Mean of RightHand, Nose, RightEar, LeftEar likelihoods)',
                 fontsize=13, fontweight='bold')

    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0.3, vmax=1.0)
    plt.colorbar(im, ax=ax, shrink=0.8, label='Composite Mean Likelihood')

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha='right', fontsize=9)
    ax.set_xlabel('Phase')
    ax.set_ylabel('Animal ID')

    # Annotate
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.5 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=7, color=color)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig4_M_animal_tracking_heatmap.png')
    plt.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("BODYPART DLC TRACKING QUALITY -- CROSS-GROUP ANALYSIS")
    print("Question: Does M (60kD) have worse DLC tracking on kinematic bodyparts?")
    print("=" * 80)

    # Build phase lookup
    phase_lookup = build_phase_lookup(UCSF_DATA_PATH)

    # Load all bodypart data
    df = load_all_bodypart_data(phase_lookup)

    # Save raw data
    raw_out = os.path.join(OUTPUT_DIR, 'bodypart_tracking_raw.csv')
    df.to_csv(raw_out, index=False)
    print(f"\nRaw data saved: {raw_out}")

    # Analysis
    summary_df = analyze_high_confidence_bodyparts(df)
    phase_df = analyze_by_phase(df)
    find_outlier_animals(df)
    analyze_phase_shift_pattern(df)

    # Save summary
    summary_out = os.path.join(OUTPUT_DIR, 'bodypart_comparison_summary.csv')
    summary_df.to_csv(summary_out, index=False)
    print(f"\nSummary saved: {summary_out}")

    if len(phase_df) > 0:
        phase_out = os.path.join(OUTPUT_DIR, 'phase_comparison.csv')
        phase_df.to_csv(phase_out, index=False)
        print(f"Phase comparison saved: {phase_out}")

    # Figures
    print("\nGenerating figures...")
    fig_bodypart_comparison(df)
    fig_kinematic_bodypart_distributions(df)
    fig_phase_trajectories(df)
    fig_per_animal_heatmap(df)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    main()
