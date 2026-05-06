"""
M (60kD) Per-Session Diagnostic for Suspect Animals

Identifies specific sessions where post-injury kinematics EXCEED
the animal's own pre-injury baseline. These sessions need human
video inspection -- kinematic improvement after contusion injury
is biologically implausible and suggests artifact.

Suspect animals (from LOO analysis):
  M05, M06, M13, M14 = PRIMARY DRIVERS of flat group profile
  M07 = only 13 pre-injury swipes (unreliable baseline)
"""

import pandas as pd
import numpy as np
import os
import json
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

SUSPECTS = ['M05', 'M06', 'M07', 'M13', 'M14']
NORMALS = ['M01', 'M10', 'M12']
FEATURES = ['Path_over_Frames', 'Swipe_length', 'Swipe_area']

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.spines.top': False, 'axes.spines.right': False,
})


def load_data():
    df1 = pd.read_csv(os.path.join(base, 'Swipe_Contusion_Data.csv'))
    df2 = pd.read_csv(os.path.join(base, 'Swipe_Contusion_Data_2.csv'))
    df = pd.concat([df1, df2], ignore_index=True)
    df['group'] = df['SubjectID'].str[0]
    df = df[df['group'] == 'M'].copy()

    for col in ['Path_over_Frames', 'Swipe_length', 'Swipe_area', 'Path_length',
                'Swipe_Duration_Frames', 'Swipe_breadth', 'Swipe_speed']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['date_serial'] = df['Session_ID'].str.extract(r'-(\d{5})-')[0].astype(float)

    test_to_phase = {}
    for phase, tests in PHASE_MAP.items():
        for t in tests:
            test_to_phase[t] = phase
    df['phase'] = df['Test_Type_Grouped_1'].map(test_to_phase)

    return df


def session_diagnostic(df):
    """Per-session analysis flagging anomalous post-injury sessions."""
    flagged_sessions = []

    print('PER-SESSION DIAGNOSTIC FOR SUSPECT M ANIMALS')
    print('=' * 120)
    print('Goal: identify specific sessions where post-injury kinematics exceed pre-injury baseline')
    print()

    for animal in SUSPECTS + NORMALS:
        ad = df[df['SubjectID'] == animal].copy()

        pre = ad[ad['phase'] == 'Pre-Injury']
        if len(pre) == 0:
            print('%s: NO PRE-INJURY DATA -- skip' % animal)
            continue

        baselines = {}
        for f in FEATURES:
            vals = pre[f].dropna()
            baselines[f] = {
                'mean': vals.mean(),
                'std': vals.std(),
                'p75': vals.quantile(0.75),
                'p90': vals.quantile(0.90),
            }

        is_suspect = animal in SUSPECTS
        tag = '*** SUSPECT ***' if is_suspect else '(normal control)'
        print('\n%s %s' % (animal, tag))
        print('  Pre-injury baseline: PoF=%.2f, Length=%.1f, Area=%.1f  (N=%d swipes, %d sessions)' % (
            baselines['Path_over_Frames']['mean'],
            baselines['Swipe_length']['mean'],
            baselines['Swipe_area']['mean'],
            len(pre),
            pre['Session_ID'].nunique()))
        print()

        sessions = ad.groupby('Session_ID').agg(
            phase=('phase', 'first'),
            test_type=('Test_Type_Grouped_1', 'first'),
            date_serial=('date_serial', 'first'),
            n_swipes=('Swipe_ID', 'count'),
            pof_mean=('Path_over_Frames', 'mean'),
            pof_median=('Path_over_Frames', 'median'),
            len_mean=('Swipe_length', 'mean'),
            area_mean=('Swipe_area', 'mean'),
            dur_mean=('Swipe_Duration_Frames', 'mean'),
        ).sort_values('date_serial')

        sessions = sessions[sessions['phase'].notna()]

        pre_rate = len(pre) / pre['Session_ID'].nunique() if pre['Session_ID'].nunique() > 0 else 0

        print('  %-28s %-8s %5s %7s %7s %7s %7s %s' % (
            'Session', 'Phase', 'N', 'PoF', 'Length', 'Area', 'Dur', 'FLAGS'))
        print('  ' + '-' * 105)

        for sid, row in sessions.iterrows():
            flags = []

            if row['phase'] in ['Post-Injury', 'Post-Rehab']:
                if row['pof_mean'] > baselines['Path_over_Frames']['p75']:
                    flags.append('PoF>pre_p75')
                if row['area_mean'] > baselines['Swipe_area']['p75']:
                    flags.append('Area>pre_p75')
                if row['len_mean'] > baselines['Swipe_length']['p75']:
                    flags.append('Len>pre_p75')
                if row['n_swipes'] < 5:
                    flags.append('LOW_N')
                if pre_rate > 0 and row['n_swipes'] > pre_rate * 1.5:
                    flags.append('SWIPE_SURGE(%.1fx)' % (row['n_swipes'] / pre_rate))

            flag_str = ' | '.join(flags) if flags else ''
            phase_short = row['phase'][:4]

            print('  %-28s %-8s %5d %7.2f %7.1f %7.1f %7.1f %s' % (
                sid, phase_short, row['n_swipes'], row['pof_mean'],
                row['len_mean'], row['area_mean'], row['dur_mean'], flag_str))

            if flags and is_suspect:
                flagged_sessions.append({
                    'animal': animal,
                    'session_id': sid,
                    'phase': row['phase'],
                    'test_type': row['test_type'],
                    'n_swipes': int(row['n_swipes']),
                    'pof': round(float(row['pof_mean']), 2),
                    'length': round(float(row['len_mean']), 1),
                    'area': round(float(row['area_mean']), 1),
                    'duration': round(float(row['dur_mean']), 1),
                    'flags': flags,
                })

    return flagged_sessions


def print_flagged_summary(flagged_sessions):
    """Print the flagged sessions requiring human review."""
    print('\n\n')
    print('=' * 120)
    print('FLAGGED SESSIONS REQUIRING HUMAN INSPECTION')
    print('=' * 120)
    print('Post-injury sessions where kinematics EXCEED the animal own pre-injury 75th percentile.')
    print('Kinematic improvement after contusion is biologically implausible -- check video for:')
    print('  - Camera angle/position change between pre and post')
    print('  - ASPA detecting non-reach movements as swipes')
    print('  - Grooming, cage exploration, or other non-reaching movements')
    print('  - DLC tracking jumps or identity swaps')
    print()

    for animal in SUSPECTS:
        af = [s for s in flagged_sessions if s['animal'] == animal]
        if not af:
            print('%s: No flagged sessions (OK)' % animal)
            continue
        print('%s: %d flagged sessions' % (animal, len(af)))
        for s in af:
            print('  -> %-28s (%s) N=%d, PoF=%.2f, Area=%.1f -- %s' % (
                s['session_id'], s['phase'], s['n_swipes'], s['pof'], s['area'],
                ' | '.join(s['flags'])))
        print()

    # Summary
    print('\nSUMMARY')
    print('=' * 60)
    print('Total flagged sessions: %d' % len(flagged_sessions))
    for animal in SUSPECTS:
        af = [s for s in flagged_sessions if s['animal'] == animal]
        print('  %s: %d flagged' % (animal, len(af)))

    flag_types = {}
    for s in flagged_sessions:
        for f in s['flags']:
            key = f.split('(')[0]
            flag_types[key] = flag_types.get(key, 0) + 1
    print('\nFlag distribution:')
    for k, v in sorted(flag_types.items(), key=lambda x: -x[1]):
        print('  %s: %d sessions' % (k, v))


def fig_session_timelines(df, flagged_sessions):
    """Per-animal session timeline showing where flags land."""
    fig, axes = plt.subplots(len(SUSPECTS), 1, figsize=(18, 4 * len(SUSPECTS)))
    fig.suptitle('M Group: Per-Session Kinematic Timelines\n'
                 'Red markers = sessions exceeding pre-injury 75th percentile (need video review)',
                 fontsize=14, fontweight='bold')

    flagged_sids = {s['session_id'] for s in flagged_sessions}

    for ax, animal in zip(axes, SUSPECTS):
        ad = df[(df['SubjectID'] == animal) & (df['phase'].notna())].copy()
        pre = ad[ad['phase'] == 'Pre-Injury']

        if len(pre) == 0:
            ax.text(0.5, 0.5, '%s: No pre-injury data' % animal,
                    transform=ax.transAxes, ha='center', va='center')
            continue

        pre_p75 = pre['Path_over_Frames'].dropna().quantile(0.75)

        sessions = ad.groupby('Session_ID').agg(
            phase=('phase', 'first'),
            date_serial=('date_serial', 'first'),
            n_swipes=('Swipe_ID', 'count'),
            pof_mean=('Path_over_Frames', 'mean'),
        ).sort_values('date_serial').reset_index()

        phase_colors = {
            'Pre-Injury': '#2196F3',
            'Post-Injury': '#FF9800',
            'Post-Rehab': '#4CAF50',
        }

        x = range(len(sessions))
        colors = []
        edge_colors = []
        sizes = []
        for _, row in sessions.iterrows():
            is_flagged = row['Session_ID'] in flagged_sids
            colors.append('#d62728' if is_flagged else phase_colors.get(row['phase'], '#999'))
            edge_colors.append('red' if is_flagged else 'black')
            sizes.append(120 if is_flagged else 40)

        ax.scatter(x, sessions['pof_mean'], c=colors, edgecolors=edge_colors,
                   s=sizes, zorder=5, linewidths=[2 if s > 40 else 0.5 for s in sizes])
        ax.axhline(pre_p75, color='blue', linestyle='--', alpha=0.5,
                   label='Pre-injury 75th pctl (%.2f)' % pre_p75)

        # Shade phases
        for phase, color in phase_colors.items():
            mask = sessions['phase'] == phase
            if mask.any():
                indices = sessions.index[mask]
                ax.axvspan(indices[0] - 0.5, indices[-1] + 0.5, alpha=0.08, color=color)

        ax.set_ylabel('Path/Frames')
        ax.set_title('%s  (Pre: %d swipes, %d sessions)' % (
            animal, len(pre), pre['Session_ID'].nunique()), fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')

        # Label flagged sessions
        for i, (_, row) in enumerate(sessions.iterrows()):
            if row['Session_ID'] in flagged_sids:
                ax.annotate(row['Session_ID'].split('-')[1],
                           (i, row['pof_mean']),
                           fontsize=7, rotation=45,
                           textcoords='offset points', xytext=(5, 5))

    axes[-1].set_xlabel('Session (chronological order)')
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig12_M_session_timelines.png')
    plt.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close()
    print('\nSaved: %s' % out)


def fig_swipe_distributions(df):
    """Compare swipe-level kinematic distributions pre vs post for suspects."""
    fig, axes = plt.subplots(len(SUSPECTS), 3, figsize=(18, 4 * len(SUSPECTS)))
    fig.suptitle('M Suspect Animals: Swipe-Level Distributions Pre vs Post\n'
                 'Look for bimodality or shifted distributions post-injury',
                 fontsize=14, fontweight='bold')

    for row_idx, animal in enumerate(SUSPECTS):
        ad = df[df['SubjectID'] == animal]
        pre = ad[ad['phase'] == 'Pre-Injury']
        post = ad[ad['phase'] == 'Post-Injury']

        for col_idx, feature in enumerate(FEATURES):
            ax = axes[row_idx, col_idx]
            pre_vals = pre[feature].dropna()
            post_vals = post[feature].dropna()

            if len(pre_vals) > 0:
                ax.hist(pre_vals, bins=30, alpha=0.5, color='#2196F3',
                        label='Pre (N=%d)' % len(pre_vals), density=True)
            if len(post_vals) > 0:
                ax.hist(post_vals, bins=30, alpha=0.5, color='#FF9800',
                        label='Post (N=%d)' % len(post_vals), density=True)

            ax.legend(fontsize=8)
            if row_idx == 0:
                ax.set_title(feature, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(animal, fontweight='bold', fontsize=12)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig13_M_suspect_distributions.png')
    plt.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: %s' % out)


def main():
    print('Loading M data...')
    df = load_data()
    print('  %d total swipes, %d animals' % (len(df), df['SubjectID'].nunique()))

    flagged_sessions = session_diagnostic(df)
    print_flagged_summary(flagged_sessions)

    # Save flagged list
    out_path = os.path.join(OUTPUT_DIR, 'M_flagged_sessions_for_review.json')
    with open(out_path, 'w') as f:
        json.dump(flagged_sessions, f, indent=2)
    print('\nSaved flagged sessions list: %s' % out_path)

    print('\nGenerating figures...')
    fig_session_timelines(df, flagged_sessions)
    fig_swipe_distributions(df)

    print('\nAll outputs: %s' % OUTPUT_DIR)


if __name__ == '__main__':
    main()
