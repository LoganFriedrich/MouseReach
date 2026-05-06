"""
All-Group Kinematic Comparison

Compare M (60kD) pre-to-post kinematic shifts against ALL injury groups
(D, G, H, K, L) — not just contusion groups.

Question: Is M's -12% path/frames shift an outlier, or do other groups
show similar-magnitude effects?
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

# Phase mapping
PHASE_MAP = {
    'Pre-Injury': ['2_Pre-injury_1', '2_Pre-injury_2', '2_Pre-injury_3'],
    'Immediate Post': ['3_1wk_Post-injury', '3_1wk_Post-injury_First_Test'],
    '2-4wk Post': ['3_2wk_Post-injury', '3_3wk_Post-injury', '3_4wk_Post-injury'],
    '5+ wk Post': ['3_5wk_Post-injury', '3_6wk_Post-injury', '3_7wk_Post-injury',
                    '3_8wk_Post-injury', '3_9wk_Post-injury'],
    'Post-Rehab': ['5_Post-rehab_Test_1', '5_Post-rehab_Test_2'],
}

GROUP_INFO = {
    'D': {'label': 'D (Pyramidotomy)', 'color': '#1f77b4', 'injury': 'Pyramidotomy'},
    'G': {'label': 'G (Transection)', 'color': '#d62728', 'injury': 'Transection'},
    'H': {'label': 'H (Transection)', 'color': '#9467bd', 'injury': 'Transection'},
    'K': {'label': 'K (Contusion 70kD)', 'color': '#e377c2', 'injury': 'Contusion'},
    'L': {'label': 'L (Contusion 50kD)', 'color': '#2ca02c', 'injury': 'Contusion'},
    'M': {'label': 'M (Contusion 60kD)', 'color': '#ff7f0e', 'injury': 'Contusion'},
}

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.spines.top': False, 'axes.spines.right': False,
})


def load_all_swipe_data():
    """Load swipe data from all injury type files."""
    frames = []

    # Contusion (K, L, M)
    df1 = pd.read_csv(os.path.join(base, 'Swipe_Contusion_Data.csv'))
    df2 = pd.read_csv(os.path.join(base, 'Swipe_Contusion_Data_2.csv'))
    frames.append(pd.concat([df1, df2], ignore_index=True))
    print(f"  Contusion: {len(df1)+len(df2)} rows")

    # Pyramidotomy (D)
    pyra = pd.read_csv(os.path.join(base, 'Swipe_Pyramidotomy_Data.csv'))
    frames.append(pyra)
    print(f"  Pyramidotomy: {len(pyra)} rows")

    # Transection (G, H)
    t1 = pd.read_csv(os.path.join(base, 'Swipe_Transection_Data.csv'))
    t2_path = os.path.join(base, 'Swipe_Transection_Data_2.csv')
    if os.path.exists(t2_path):
        t2 = pd.read_csv(t2_path)
        frames.append(pd.concat([t1, t2], ignore_index=True))
        print(f"  Transection: {len(t1)+len(t2)} rows")
    else:
        frames.append(t1)
        print(f"  Transection: {len(t1)} rows")

    df_all = pd.concat(frames, ignore_index=True)
    df_all['group'] = df_all['SubjectID'].str[0]

    # Parse kinematic columns
    for col in ['Path_over_Frames', 'Swipe_speed', 'Swipe_length', 'Swipe_area',
                'Swipe_breadth', 'Swipe_Duration_Frames']:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

    # Map phases
    test_to_phase = {}
    for phase, tests in PHASE_MAP.items():
        for t in tests:
            test_to_phase[t] = phase
    df_all['phase'] = df_all['Test_Type_Grouped_1'].map(test_to_phase)

    return df_all


def analyze_all_groups(df_all):
    """Compare pre-to-post kinematic shifts across all groups."""
    df_phased = df_all[df_all['phase'].notna()].copy()

    kin_features = ['Path_over_Frames', 'Swipe_length', 'Swipe_area', 'Swipe_breadth']
    groups_ordered = ['D', 'G', 'H', 'K', 'L', 'M']

    # Basic stats
    print("\n" + "=" * 80)
    print("ALL-GROUP OVERVIEW")
    print("=" * 80)
    for grp in groups_ordered:
        g = df_phased[df_phased['group'] == grp]
        if len(g) == 0:
            continue
        n_animals = g['SubjectID'].nunique()
        per_animal = len(g) / n_animals
        info = GROUP_INFO.get(grp, {})
        print(f"  {info.get('label', grp)}: {len(g)} swipes, {n_animals} animals, "
              f"{per_animal:.0f}/animal")

    # Per-animal mean by phase
    print("\n" + "=" * 80)
    print("PER-ANIMAL MEAN KINEMATICS BY PHASE")
    print("=" * 80)
    for feature in kin_features:
        print(f"\n  {feature}:")
        header = f"    {'Phase':18s}"
        for grp in groups_ordered:
            header += f"  {grp:>8s}"
        print(header)
        print("    " + "-" * (18 + 10 * len(groups_ordered)))

        for phase in ['Pre-Injury', 'Immediate Post', '2-4wk Post', 'Post-Rehab']:
            line = f"    {phase:18s}"
            for grp in groups_ordered:
                g = df_phased[(df_phased['group'] == grp) & (df_phased['phase'] == phase)]
                if len(g) < 5:
                    line += f"  {'n/a':>8s}"
                    continue
                animal_means = g.groupby('SubjectID')[feature].mean().dropna()
                if len(animal_means) < 2:
                    line += f"  {'n/a':>8s}"
                else:
                    line += f"  {animal_means.mean():>8.2f}"
            print(line)

    # Pre-to-post shifts (THE KEY TABLE)
    print("\n" + "=" * 80)
    print("PRE-TO-POST KINEMATIC SHIFT (% change from pre-injury baseline)")
    print("This is the comparison that matters: all groups on equal footing")
    print("=" * 80)

    shift_data = {}  # For plotting
    for feature in kin_features:
        print(f"\n  {feature}:")
        shift_data[feature] = {}

        for grp in groups_ordered:
            g = df_phased[df_phased['group'] == grp]
            pre = g[g['phase'] == 'Pre-Injury'].groupby('SubjectID')[feature].mean()
            post = g[g['phase'].isin(['Immediate Post', '2-4wk Post'])].groupby('SubjectID')[feature].mean()
            common = pre.index.intersection(post.index)

            if len(common) < 2:
                info = GROUP_INFO.get(grp, {})
                print(f"    {info.get('label', grp):30s}: insufficient paired data (n={len(common)})")
                continue

            shifts = ((post.loc[common] - pre.loc[common]) / pre.loc[common].abs().replace(0, np.nan) * 100).dropna()
            info = GROUP_INFO.get(grp, {})
            shift_data[feature][grp] = shifts.values

            marker = " <<<" if grp == 'M' else ""
            print(f"    {info.get('label', grp):30s}: {shifts.mean():+6.1f}% "
                  f"(SD={shifts.std():.1f}%, n={len(shifts)}){marker}")

    # Post-rehab recovery
    print("\n" + "=" * 80)
    print("POST-REHAB RECOVERY (% change from pre-injury)")
    print("=" * 80)
    for feature in kin_features:
        print(f"\n  {feature}:")
        for grp in groups_ordered:
            g = df_phased[df_phased['group'] == grp]
            pre = g[g['phase'] == 'Pre-Injury'].groupby('SubjectID')[feature].mean()
            post = g[g['phase'] == 'Post-Rehab'].groupby('SubjectID')[feature].mean()
            common = pre.index.intersection(post.index)

            if len(common) < 2:
                continue

            shifts = ((post.loc[common] - pre.loc[common]) / pre.loc[common].abs().replace(0, np.nan) * 100).dropna()
            info = GROUP_INFO.get(grp, {})
            marker = " <<<" if grp == 'M' else ""
            print(f"    {info.get('label', grp):30s}: {shifts.mean():+6.1f}% "
                  f"(SD={shifts.std():.1f}%, n={len(shifts)}){marker}")

    # Swipe counts per group
    print("\n" + "=" * 80)
    print("SWIPES PER ANIMAL BY PHASE (detecting ASPA version differences)")
    print("=" * 80)
    for phase in ['Pre-Injury', 'Immediate Post', '2-4wk Post', 'Post-Rehab']:
        print(f"\n  {phase}:")
        for grp in groups_ordered:
            g = df_phased[(df_phased['group'] == grp) & (df_phased['phase'] == phase)]
            if len(g) == 0:
                continue
            per_animal = g.groupby('SubjectID').size()
            info = GROUP_INFO.get(grp, {})
            print(f"    {info.get('label', grp):30s}: {per_animal.mean():>6.0f}/animal "
                  f"(range {per_animal.min()}-{per_animal.max()}, {g['SubjectID'].nunique()} animals)")

    # Outcome distributions
    print("\n" + "=" * 80)
    print("REACH OUTCOME DISTRIBUTION BY GROUP (phased data)")
    print("=" * 80)
    for grp in groups_ordered:
        g = df_phased[df_phased['group'] == grp]
        if len(g) == 0:
            continue
        info = GROUP_INFO.get(grp, {})
        print(f"\n  {info.get('label', grp)} ({len(g)} swipes):")
        for val, cnt in g['Reach_outcome'].value_counts().head(6).items():
            print(f"    {val}: {cnt} ({100*cnt/len(g):.1f}%)")

    return shift_data


def fig_all_groups_shifts(shift_data):
    """Bar chart: pre-to-post kinematic shift for ALL groups."""
    kin_features = ['Path_over_Frames', 'Swipe_length', 'Swipe_area', 'Swipe_breadth']
    groups_ordered = ['D', 'G', 'H', 'K', 'L', 'M']

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle('Pre-Injury to Post-Injury Kinematic Shift: ALL Groups\n'
                 'Where does M (60kD) fall relative to every other injury model?',
                 fontsize=14, fontweight='bold')

    for ax_i, feature in enumerate(kin_features):
        ax = axes[ax_i // 2, ax_i % 2]

        means = []
        sems = []
        colors = []
        labels = []
        edge_colors = []

        for grp in groups_ordered:
            if grp not in shift_data[feature]:
                continue
            vals = shift_data[feature][grp]
            means.append(np.mean(vals))
            sems.append(np.std(vals) / np.sqrt(len(vals)))
            info = GROUP_INFO.get(grp, {})
            colors.append(info.get('color', '#888'))
            labels.append(f"{info.get('label', grp)}\n(n={len(vals)})")
            edge_colors.append('red' if grp == 'M' else 'black')

        x = np.arange(len(means))
        bars = ax.bar(x, means, yerr=sems, capsize=4, color=colors, alpha=0.8,
                      edgecolor=edge_colors, linewidth=[3 if ec == 'red' else 1 for ec in edge_colors])

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('% Change (Post - Pre)')
        ax.set_title(feature, fontsize=13, fontweight='bold')
        ax.axhline(0, color='black', linewidth=0.8)

        # Add value labels
        for i, (m, s) in enumerate(zip(means, sems)):
            ax.text(i, m - s - 3 if m < 0 else m + s + 1, f'{m:+.1f}%',
                    ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig8_all_groups_kinematic_shifts.png')
    plt.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  Saved: {out}")


def fig_phase_trajectories_all(df_all):
    """Line plots: per-phase kinematic trajectories for ALL groups."""
    df_phased = df_all[df_all['phase'].notna()].copy()
    kin_features = ['Path_over_Frames', 'Swipe_length', 'Swipe_area', 'Swipe_breadth']
    groups_ordered = ['D', 'G', 'H', 'K', 'L', 'M']
    phases_ordered = ['Pre-Injury', 'Immediate Post', '2-4wk Post', 'Post-Rehab']

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle('Kinematic Feature Trajectories Across Phases: ALL Groups\n'
                 'M (60kD, orange) highlighted',
                 fontsize=14, fontweight='bold')

    for ax_i, feature in enumerate(kin_features):
        ax = axes[ax_i // 2, ax_i % 2]

        for grp in groups_ordered:
            g = df_phased[df_phased['group'] == grp]
            means = []
            sems = []
            valid_phases = []

            for phase in phases_ordered:
                phase_data = g[g['phase'] == phase]
                animal_means = phase_data.groupby('SubjectID')[feature].mean().dropna()
                if len(animal_means) >= 2:
                    means.append(animal_means.mean())
                    sems.append(animal_means.std() / np.sqrt(len(animal_means)))
                    valid_phases.append(phase)

            if len(means) < 2:
                continue

            info = GROUP_INFO.get(grp, {})
            x = [phases_ordered.index(p) for p in valid_phases]
            lw = 3 if grp == 'M' else 1.5
            ms = 10 if grp == 'M' else 6
            zorder = 10 if grp == 'M' else 3
            ax.errorbar(x, means, yerr=sems, marker='o', linewidth=lw,
                       color=info.get('color', '#888'), label=info.get('label', grp),
                       capsize=4, markersize=ms, zorder=zorder)

        ax.set_xticks(range(len(phases_ordered)))
        ax.set_xticklabels(phases_ordered, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel(feature)
        ax.set_title(feature, fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, loc='best')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig9_all_groups_phase_trajectories.png')
    plt.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out}")


def main():
    print("=" * 80)
    print("ALL-GROUP KINEMATIC COMPARISON")
    print("Where does M (60kD) fall relative to ALL injury models?")
    print("=" * 80)

    print("\nLoading all swipe data...")
    df_all = load_all_swipe_data()
    print(f"Total: {len(df_all)} swipes across {df_all['group'].nunique()} groups")

    shift_data = analyze_all_groups(df_all)

    print("\nGenerating figures...")
    fig_all_groups_shifts(shift_data)
    fig_phase_trajectories_all(df_all)

    print(f"\nAll outputs: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
