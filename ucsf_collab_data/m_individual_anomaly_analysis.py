"""
M (60kD) Individual Mouse Anomaly Analysis

Three hypotheses for why M mice show flat kinematics despite visible impairment:
1. CANCELLATION: Some mice worsen, others improve, averaging to flat
2. SELECTION BIAS: ASPA drops weak post-injury attempts, survivors look normal
3. RATIO ARTIFACT: Path_length and Duration both drop, keeping Path/Frames flat

All analyses are WITHIN-ANIMAL (each mouse vs its own baseline).
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = os.path.join(
    r'C:\Users\friedrichl\OneDrive - Marquette University\Blackmore Lab Notes - Sharepoint',
    r'3 Lab Projects\Automated single pellet apparatus\!UCSF_Collab',
    r'May2025_Uploads\ODC Uploads'
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_connectome_root = SCRIPT_DIR
while os.path.basename(_connectome_root) != '2_Connectome' and os.path.dirname(_connectome_root) != _connectome_root:
    _connectome_root = os.path.dirname(_connectome_root)
OUTPUT_DIR = os.path.join(_connectome_root, 'Databases', 'figures', 'bodypart_tracking_quality')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PHASE_MAP = {
    'Pre-Injury': ['2_Pre-injury_1', '2_Pre-injury_2', '2_Pre-injury_3'],
    'Post-Injury': ['3_1wk_Post-injury', '3_1wk_Post-injury_First_Test',
                     '3_2wk_Post-injury', '3_3wk_Post-injury', '3_4wk_Post-injury'],
    'Post-Rehab': ['5_Post-rehab_Test_1', '5_Post-rehab_Test_2'],
}

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.spines.top': False, 'axes.spines.right': False,
})


def load_m_data():
    """Load M group data only."""
    df1 = pd.read_csv(os.path.join(base, 'Swipe_Contusion_Data.csv'))
    df2 = pd.read_csv(os.path.join(base, 'Swipe_Contusion_Data_2.csv'))
    df = pd.concat([df1, df2], ignore_index=True)
    df['group'] = df['SubjectID'].str[0]
    df = df[df['group'] == 'M'].copy()

    for col in ['Path_length', 'Path_over_Frames', 'Swipe_Duration_Frames',
                'Swipe_length', 'Swipe_area', 'Swipe_breadth']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    test_to_phase = {}
    for phase, tests in PHASE_MAP.items():
        for t in tests:
            test_to_phase[t] = phase
    df['phase'] = df['Test_Type_Grouped_1'].map(test_to_phase)

    return df[df['phase'].notna()].copy()


def hypothesis_1_cancellation(df):
    """H1: Do some M mice get worse while others get better?"""
    print("\n" + "=" * 80)
    print("HYPOTHESIS 1: CANCELLATION")
    print("Do some M mice worsen while others improve, averaging to flat?")
    print("=" * 80)

    features = ['Path_over_Frames', 'Swipe_length', 'Swipe_area']
    animals = sorted(df['SubjectID'].unique())

    shift_data = {f: {} for f in features}

    for feature in features:
        print("\n  %s -- per-animal pre-to-post shift:" % feature)
        worse = 0
        better = 0
        for animal in animals:
            a = df[df['SubjectID'] == animal]
            pre_val = a[a['phase'] == 'Pre-Injury'][feature].mean()
            post_val = a[a['phase'] == 'Post-Injury'][feature].mean()

            if pd.isna(pre_val) or pd.isna(post_val) or pre_val == 0:
                continue

            pct = (post_val - pre_val) / abs(pre_val) * 100
            shift_data[feature][animal] = pct
            direction = "WORSE" if pct < 0 else "BETTER"
            if pct < 0:
                worse += 1
            else:
                better += 1
            print("    %s: %+6.1f%% (%s)" % (animal, pct, direction))

        print("\n    Summary: %d got worse, %d got better" % (worse, better))
        if better > 0:
            status = "CONFIRMED" if better >= 3 else "PARTIAL"
            print("    --> CANCELLATION %s: %d/%d mice improved post-injury"
                  % (status, better, worse + better))

    return shift_data


def hypothesis_2_selection_bias(df):
    """H2: Does ASPA detect fewer reaches post-injury for each M mouse?"""
    print("\n" + "=" * 80)
    print("HYPOTHESIS 2: SELECTION BIAS")
    print("Does ASPA detect fewer reaches post-injury? (survivors = best attempts)")
    print("=" * 80)

    animals = sorted(df['SubjectID'].unique())

    print("\n  %-12s %8s %8s %8s %8s" % ('Animal', 'Pre #', 'Post #', 'Ratio', 'Drop?'))
    print("  " + "-" * 50)

    ratios = []
    for animal in animals:
        a = df[df['SubjectID'] == animal]
        pre = a[a['phase'] == 'Pre-Injury']
        post = a[a['phase'] == 'Post-Injury']

        pre_sessions = pre['Session_ID'].nunique()
        post_sessions = post['Session_ID'].nunique()

        if pre_sessions == 0 or post_sessions == 0:
            continue

        pre_rate = len(pre) / pre_sessions
        post_rate = len(post) / post_sessions

        ratio = post_rate / pre_rate
        ratios.append(ratio)
        drop = "YES" if ratio < 0.7 else "no"
        print("  %-12s %6.0f/ses %6.0f/ses %8.2fx %8s"
              % (animal, pre_rate, post_rate, ratio, drop))

    print("\n  Mean ratio: %.2fx" % np.mean(ratios))
    print("  Mice with >30%% drop: %d/%d" % (sum(1 for r in ratios if r < 0.7), len(ratios)))

    if np.mean(ratios) < 0.7:
        print("  --> SELECTION BIAS SUPPORTED: substantial reach count drop post-injury")
    else:
        print("  --> SELECTION BIAS WEAK: reach counts did not drop dramatically")

    return ratios


def hypothesis_3_ratio_artifact(df):
    """H3: Do Path_length and Duration both drop, keeping the ratio flat?"""
    print("\n" + "=" * 80)
    print("HYPOTHESIS 3: RATIO ARTIFACT")
    print("Path_over_Frames = Path_length / Duration_Frames")
    print("If BOTH drop proportionally, the ratio stays flat despite real changes")
    print("=" * 80)

    animals = sorted(df['SubjectID'].unique())

    print("\n  %-12s %10s %10s %8s %9s %9s %8s %8s %6s" % (
        'Animal', 'Path Pre', 'Path Post', 'Path %',
        'Dur Pre', 'Dur Post', 'Dur %', 'PoF %', 'Mask?'))
    print("  " + "-" * 90)

    masked_count = 0
    total = 0
    component_data = []

    for animal in animals:
        a = df[df['SubjectID'] == animal]
        pre = a[a['phase'] == 'Pre-Injury']
        post = a[a['phase'] == 'Post-Injury']

        if len(pre) < 5 or len(post) < 5:
            continue

        path_pre = pre['Path_length'].mean()
        path_post = post['Path_length'].mean()
        dur_pre = pre['Swipe_Duration_Frames'].mean()
        dur_post = post['Swipe_Duration_Frames'].mean()
        pof_pre = pre['Path_over_Frames'].mean()
        pof_post = post['Path_over_Frames'].mean()

        if path_pre == 0 or dur_pre == 0 or pof_pre == 0:
            continue

        path_pct = (path_post - path_pre) / abs(path_pre) * 100
        dur_pct = (dur_post - dur_pre) / abs(dur_pre) * 100
        pof_pct = (pof_post - pof_pre) / abs(pof_pre) * 100

        # Masked = both components changed substantially but ratio didn't
        both_changed = abs(path_pct) > 10 and abs(dur_pct) > 10
        ratio_flat = abs(pof_pct) < 15
        masked = both_changed and ratio_flat
        if masked:
            masked_count += 1
        total += 1

        mask_flag = "YES" if masked else ""
        print("  %-12s %10.1f %10.1f %+7.1f%% %9.1f %9.1f %+7.1f%% %+7.1f%% %6s" % (
            animal, path_pre, path_post, path_pct,
            dur_pre, dur_post, dur_pct, pof_pct, mask_flag))

        component_data.append({
            'animal': animal, 'path_pct': path_pct,
            'dur_pct': dur_pct, 'pof_pct': pof_pct
        })

    print("\n  Ratio masking detected: %d/%d mice" % (masked_count, total))
    if masked_count >= total * 0.4:
        print("  --> RATIO ARTIFACT CONFIRMED: Path and Duration co-vary, masking real changes")
    else:
        print("  --> RATIO ARTIFACT WEAK: Components do not consistently co-vary")

    return component_data


def fig_individual_shifts(shift_data):
    """Waterfall chart of per-animal shifts for each feature."""
    features = list(shift_data.keys())
    fig, axes = plt.subplots(1, len(features), figsize=(6 * len(features), 8))
    fig.suptitle('H1: Per-Animal Pre-to-Post Kinematic Shift (M group)\n'
                 'Do improvements and deficits cancel out?',
                 fontsize=14, fontweight='bold')

    for ax, feature in zip(axes, features):
        animals = sorted(shift_data[feature].keys(),
                        key=lambda a: shift_data[feature][a])
        vals = [shift_data[feature][a] for a in animals]
        colors = ['#d62728' if v < 0 else '#2ca02c' for v in vals]

        ax.barh(range(len(animals)), vals, color=colors, alpha=0.8,
                edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(animals)))
        ax.set_yticklabels(animals, fontsize=8)
        ax.axvline(0, color='black', linewidth=1)
        ax.axvline(np.mean(vals), color='orange', linewidth=2, linestyle='--',
                   label='Mean: %+.1f%%' % np.mean(vals))
        ax.set_xlabel('% Change from Pre-Injury')
        ax.set_title(feature, fontweight='bold')
        ax.legend(fontsize=9)

        for i, (v, a) in enumerate(zip(vals, animals)):
            ax.text(v + (2 if v >= 0 else -2), i, '%+.0f%%' % v,
                    va='center', ha='left' if v >= 0 else 'right', fontsize=7)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig10_M_individual_shifts.png')
    plt.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close()
    print("\n  Saved: %s" % out)


def fig_ratio_components(component_data):
    """Scatter: Path_length change vs Duration change per animal."""
    fig, ax = plt.subplots(figsize=(10, 8))

    path_pcts = [d['path_pct'] for d in component_data]
    dur_pcts = [d['dur_pct'] for d in component_data]
    animals = [d['animal'] for d in component_data]

    ax.scatter(dur_pcts, path_pcts, s=100, c='#ff7f0e', edgecolor='black', zorder=5)

    for a, dp, pp in zip(animals, dur_pcts, path_pcts):
        ax.annotate(a, (dp, pp), fontsize=8, textcoords='offset points',
                   xytext=(5, 5))

    # Identity line (where ratio would be perfectly preserved)
    all_vals = path_pcts + dur_pcts
    lim = max(abs(min(all_vals)) + 10, abs(max(all_vals)) + 10)
    ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.5,
            label='Ratio preserved (Path % = Duration %)')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)

    ax.set_xlabel('Duration Change (%)', fontsize=12)
    ax.set_ylabel('Path Length Change (%)', fontsize=12)
    ax.set_title('H3: Do Path_length and Duration Co-vary?\n'
                 'Points near dashed line = ratio artifact (Path/Frames stays flat)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig11_M_ratio_components.png')
    plt.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: %s" % out)


def main():
    print("=" * 80)
    print("M (60kD) INDIVIDUAL MOUSE ANOMALY ANALYSIS")
    print("Why do visibly impaired M mice show flat ASPA kinematics?")
    print("=" * 80)

    print("\nLoading M data...")
    df = load_m_data()
    print("  %d phased swipes, %d animals" % (len(df), df['SubjectID'].nunique()))

    shift_data = hypothesis_1_cancellation(df)
    hypothesis_2_selection_bias(df)
    component_data = hypothesis_3_ratio_artifact(df)

    print("\nGenerating figures...")
    fig_individual_shifts(shift_data)
    fig_ratio_components(component_data)

    print("\nAll outputs: %s" % OUTPUT_DIR)


if __name__ == '__main__':
    main()
