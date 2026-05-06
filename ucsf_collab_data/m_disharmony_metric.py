"""
Disharmony Metric: quantify how much each reach deviates from the animal's
pre-injury ideal trajectory.

For each animal:
1. Build a mean trajectory from pre-injury successful retrievals (nose-centered, time-normalized)
2. For every reach in every phase, compute point-by-point distance from the mean
3. Disharmony score = mean distance across all time points

Output: per-animal, per-phase disharmony distributions + group comparison.
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

ASPA_BASE = r'Y:\2_Connectome\Behavior\MouseReach_Pipeline\Analyzed\ASPA\M\Post-Processing'
UCSF_BASE = os.path.join(
    r'C:\Users\friedrichl\OneDrive - Marquette University\Blackmore Lab Notes - Sharepoint',
    r'3 Lab Projects\Automated single pellet apparatus\!UCSF_Collab',
    r'May2025_Uploads\ODC Uploads'
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_connectome_root = SCRIPT_DIR
while os.path.basename(_connectome_root) != '2_Connectome' and os.path.dirname(_connectome_root) != _connectome_root:
    _connectome_root = os.path.dirname(_connectome_root)
OUTPUT_BASE = os.path.join(_connectome_root, 'Databases', 'figures', 'bodypart_tracking_quality')

N_NORM_POINTS = 30  # time-normalize all trajectories to this many points
MIN_RETRIEVALS = 10  # minimum successful retrievals to build model


def parse_frame_range(s):
    parts = str(s).strip().split('-')
    if len(parts) == 2:
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            pass
    return None, None


def session_to_stem(sid):
    parts = sid.split('-')
    date_str = (datetime(1899, 12, 30) + timedelta(days=int(parts[1]))).strftime('%Y%m%d')
    return '%s_%s_%s' % (date_str, parts[0], parts[2])


def extract_trajectory(dlc, s, e):
    """Extract nose-centered RH trajectory for one swipe."""
    rh_x = dlc['RightHand']['x'].iloc[s:e+1].values
    rh_y = dlc['RightHand']['y'].iloc[s:e+1].values
    rh_lk = dlc['RightHand']['likelihood'].iloc[s:e+1].values
    nose_x = dlc['Nose']['x'].iloc[s:e+1].values
    nose_y = dlc['Nose']['y'].iloc[s:e+1].values

    cx = rh_x - nose_x[0]
    cy = rh_y - nose_y[0]

    if np.mean(rh_lk) > 0.3 and len(cx) >= 3:
        return cx, cy
    return None, None


def time_normalize(x, y, n_pts):
    """Time-normalize a trajectory to n_pts points."""
    t_orig = np.linspace(0, 1, len(x))
    t_new = np.linspace(0, 1, n_pts)
    return np.interp(t_new, t_orig, x), np.interp(t_new, t_orig, y)


def compute_disharmony(traj_x, traj_y, mean_x, mean_y, std_x=None, std_y=None):
    """Compute variability-weighted disharmony scores.

    Returns: (overall, lateral_signed, extension_signed)
    - overall: mean z-scored Euclidean deviation (always positive)
    - lateral_signed: mean signed z-score in X (positive = wider, negative = narrower)
    - extension_signed: mean signed z-score in Y (positive = reaching further, negative = shorter)
    """
    dx = traj_x - mean_x
    dy = traj_y - mean_y

    if std_x is not None and std_y is not None:
        zx = dx / np.maximum(std_x, 0.5)
        zy = dy / np.maximum(std_y, 0.5)
        z_dist = np.sqrt(zx**2 + zy**2)
        return np.mean(z_dist), np.mean(zx), np.mean(zy)
    else:
        distances = np.sqrt(dx**2 + dy**2)
        return np.mean(distances), np.mean(dx), np.mean(dy)


def main():
    print('Loading UCSF data...')
    df1 = pd.read_csv(os.path.join(UCSF_BASE, 'Swipe_Contusion_Data.csv'), low_memory=False)
    df2 = pd.read_csv(os.path.join(UCSF_BASE, 'Swipe_Contusion_Data_2.csv'), low_memory=False)
    df = pd.concat([df1, df2], ignore_index=True)
    df['group'] = df['SubjectID'].str[0]
    df = df[df['group'] == 'M'].copy()
    df['sd_start'], df['sd_end'] = zip(*df['Swipe_Duration'].apply(parse_frame_range))

    phases = {
        'Pre-Injury': ['2_Pre-injury_1', '2_Pre-injury_2', '2_Pre-injury_3'],
        '1wk Post': ['3_1wk_Post-injury'],
        '2-4wk Post': ['3_2wk_Post-injury', '3_3wk_Post-injury', '3_4wk_Post-injury'],
        'Post-Rehab': ['5_Post-rehab_Test_1', '5_Post-rehab_Test_2'],
    }

    animals = sorted([a for a in df['SubjectID'].unique() if a != 'M07'])  # exclude M07 (no baseline)

    dlc_cache = {}

    def get_dlc(stem):
        if stem not in dlc_cache:
            h5 = glob.glob(os.path.join(ASPA_BASE, stem + 'DLC*.h5'))
            if not h5:
                dlc_cache[stem] = None
            else:
                d = pd.read_hdf(h5[0])
                sc = d.columns.get_level_values(0)[0]
                dlc_cache[stem] = d[sc]
        return dlc_cache[stem]

    # Step 1: Build per-animal mean trajectory from pre-injury retrievals
    print('\nBuilding per-animal ideal trajectory models...')
    animal_models = {}

    for animal in animals:
        pre = df[(df['SubjectID'] == animal) &
                 (df['Test_Type_Grouped_1'].isin(phases['Pre-Injury'])) &
                 (df['sd_start'].notna())]

        # Try retrievals first, fall back to displaced+retrieved
        retrievals = pre[pre['Reach_outcome'] == 'swipe successful']
        if len(retrievals) < MIN_RETRIEVALS:
            retrievals = pre[pre['Reach_outcome'].isin(['swipe successful', 'pellet displaced'])]

        trajs = []
        for _, row in retrievals.iterrows():
            s, e = int(row['sd_start']), int(row['sd_end'])
            stem = session_to_stem(row['Session_ID'])
            dlc = get_dlc(stem)
            if dlc is None or e + 1 > len(dlc):
                continue
            cx, cy = extract_trajectory(dlc, s, e)
            if cx is not None:
                nx, ny = time_normalize(cx, cy, N_NORM_POINTS)
                trajs.append((nx, ny))

        if len(trajs) >= 5:
            mean_x = np.mean([t[0] for t in trajs], axis=0)
            mean_y = np.mean([t[1] for t in trajs], axis=0)
            std_x = np.std([t[0] for t in trajs], axis=0)
            std_y = np.std([t[1] for t in trajs], axis=0)
            animal_models[animal] = (mean_x, mean_y, std_x, std_y, len(trajs))
            print('  %s: model from %d retrieval trajectories' % (animal, len(trajs)))
        else:
            print('  %s: SKIP (only %d trajectories)' % (animal, len(trajs)))

    # Step 2: Score every reach against its animal's model
    print('\nScoring all reaches...')
    results = []

    for animal in animals:
        if animal not in animal_models:
            continue
        mean_x, mean_y, std_x, std_y, _ = animal_models[animal]

        for phase_name, phase_labels in phases.items():
            phase_df = df[(df['SubjectID'] == animal) &
                          (df['Test_Type_Grouped_1'].isin(phase_labels)) &
                          (df['sd_start'].notna())]

            for _, row in phase_df.iterrows():
                s, e = int(row['sd_start']), int(row['sd_end'])
                stem = session_to_stem(row['Session_ID'])
                dlc = get_dlc(stem)
                if dlc is None or e + 1 > len(dlc):
                    continue
                cx, cy = extract_trajectory(dlc, s, e)
                if cx is None:
                    continue
                nx, ny = time_normalize(cx, cy, N_NORM_POINTS)
                score, lateral, extension = compute_disharmony(nx, ny, mean_x, mean_y, std_x, std_y)

                results.append({
                    'animal': animal,
                    'phase': phase_name,
                    'disharmony': score,
                    'lateral': lateral,
                    'extension': extension,
                    'outcome': row['Reach_outcome'],
                    'session': row['Session_ID'],
                })

    rdf = pd.DataFrame(results)
    print('  Total scored reaches: %d' % len(rdf))

    # Step 3: Output summary
    print('\nDISHARMONY SCORES BY ANIMAL AND PHASE (overall)')
    print('=' * 90)
    print('%-6s %8s %8s %8s %8s %8s' % ('Animal', 'Pre', '1wk', '2-4wk', 'PostRe', 'Post/Pre'))
    print('-' * 90)

    suspect_animals = ['M05', 'M06', 'M13', 'M14']

    for animal in animals:
        if animal not in animal_models:
            continue
        vals = {}
        for phase_name in phases.keys():
            phase_scores = rdf[(rdf['animal'] == animal) & (rdf['phase'] == phase_name)]['disharmony']
            vals[phase_name] = phase_scores.median() if len(phase_scores) > 0 else np.nan

        ratio = vals.get('1wk Post', np.nan) / vals.get('Pre-Injury', 1) if vals.get('Pre-Injury', 0) > 0 else np.nan
        tag = '***' if animal in suspect_animals else ''

        print('%-6s %8.1f %8.1f %8.1f %8.1f %8.2fx %s' % (
            animal,
            vals.get('Pre-Injury', np.nan),
            vals.get('1wk Post', np.nan),
            vals.get('2-4wk Post', np.nan),
            vals.get('Post-Rehab', np.nan),
            ratio, tag))

    print('\nLATERAL COMPONENT (positive = wider than ideal)')
    print('=' * 90)
    print('%-6s %8s %8s %8s %8s' % ('Animal', 'Pre', '1wk', '2-4wk', 'PostRe'))
    print('-' * 90)

    for animal in animals:
        if animal not in animal_models:
            continue
        vals = {}
        for phase_name in phases.keys():
            phase_scores = rdf[(rdf['animal'] == animal) & (rdf['phase'] == phase_name)]['lateral']
            vals[phase_name] = phase_scores.median() if len(phase_scores) > 0 else np.nan

        tag = '***' if animal in suspect_animals else ''
        print('%-6s %+8.2f %+8.2f %+8.2f %+8.2f %s' % (
            animal,
            vals.get('Pre-Injury', np.nan),
            vals.get('1wk Post', np.nan),
            vals.get('2-4wk Post', np.nan),
            vals.get('Post-Rehab', np.nan),
            tag))

    print('\nEXTENSION COMPONENT (positive = reaching further than ideal)')
    print('=' * 90)
    print('%-6s %8s %8s %8s %8s' % ('Animal', 'Pre', '1wk', '2-4wk', 'PostRe'))
    print('-' * 90)

    for animal in animals:
        if animal not in animal_models:
            continue
        vals = {}
        for phase_name in phases.keys():
            phase_scores = rdf[(rdf['animal'] == animal) & (rdf['phase'] == phase_name)]['extension']
            vals[phase_name] = phase_scores.median() if len(phase_scores) > 0 else np.nan

        tag = '***' if animal in suspect_animals else ''
        print('%-6s %+8.2f %+8.2f %+8.2f %+8.2f %s' % (
            animal,
            vals.get('Pre-Injury', np.nan),
            vals.get('1wk Post', np.nan),
            vals.get('2-4wk Post', np.nan),
            vals.get('Post-Rehab', np.nan),
            tag))

    # Step 4: Plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 18))

    # Panel 1: Box plots per phase for suspects vs normals
    ax = axes[0]
    suspect_data = {p: rdf[(rdf['animal'].isin(suspect_animals)) & (rdf['phase'] == p)]['disharmony'].values
                    for p in phases.keys()}
    normal_data = {p: rdf[(~rdf['animal'].isin(suspect_animals)) & (rdf['phase'] == p)]['disharmony'].values
                   for p in phases.keys()}

    positions_s = [1, 4, 7, 10]
    positions_n = [2, 5, 8, 11]
    bp_s = ax.boxplot([suspect_data[p] for p in phases.keys()], positions=positions_s,
                       widths=0.8, patch_artist=True, showfliers=False)
    bp_n = ax.boxplot([normal_data[p] for p in phases.keys()], positions=positions_n,
                       widths=0.8, patch_artist=True, showfliers=False)

    for patch in bp_s['boxes']:
        patch.set_facecolor('#FF6666')
    for patch in bp_n['boxes']:
        patch.set_facecolor('#6666FF')

    ax.set_xticks([1.5, 4.5, 7.5, 10.5])
    ax.set_xticklabels(list(phases.keys()))
    ax.set_ylabel('Disharmony (mean z-scored deviation from ideal)')
    ax.set_title('Disharmony Score: Suspects (red) vs Normals (blue)')
    ax.set_ylim(0, 6)
    ax.legend([bp_s['boxes'][0], bp_n['boxes'][0]], ['Suspects (M05,06,13,14)', 'Normals'], fontsize=9)

    # Panel 2: Per-animal line plot across phases
    ax = axes[1]
    phase_order = list(phases.keys())
    for animal in animals:
        if animal not in animal_models:
            continue
        medians = []
        for p in phase_order:
            scores = rdf[(rdf['animal'] == animal) & (rdf['phase'] == p)]['disharmony']
            medians.append(scores.median() if len(scores) > 0 else np.nan)

        color = '#FF4444' if animal in suspect_animals else '#4444FF'
        alpha = 0.8 if animal in suspect_animals else 0.3
        lw = 2 if animal in suspect_animals else 1
        ax.plot(range(len(phase_order)), medians, 'o-', color=color, alpha=alpha,
                linewidth=lw, label=animal if animal in suspect_animals else None)

    ax.set_xticks(range(len(phase_order)))
    ax.set_xticklabels(phase_order)
    ax.set_ylabel('Median Disharmony (z-scored)')
    ax.set_title('Per-Animal Disharmony Trajectory')
    ax.set_ylim(0, 5)  # cap to exclude M10 outlier
    ax.legend(fontsize=9)

    # Panel 3: Per-animal lateral vs extension at 1wk post (scatter)
    ax = axes[2]
    for animal in animals:
        if animal not in animal_models:
            continue
        post1 = rdf[(rdf['animal'] == animal) & (rdf['phase'] == '1wk Post')]
        if len(post1) == 0:
            continue
        lat_med = post1['lateral'].median()
        ext_med = post1['extension'].median()
        color = '#FF4444' if animal in suspect_animals else '#4444FF'
        size = 100 if animal in suspect_animals else 50
        ax.scatter(lat_med, ext_med, c=color, s=size, edgecolors='black', linewidths=0.5, zorder=5)
        ax.annotate(animal, (lat_med, ext_med), fontsize=8, textcoords='offset points',
                    xytext=(5, 5))

    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Lateral disharmony (z-scored, + = wider)')
    ax.set_ylabel('Extension disharmony (z-scored, + = further)')
    ax.set_title('1wk Post-Injury: Lateral vs Extension Deviation from Pre-Injury Ideal\n'
                 'Red = suspects, Blue = normals')

    plt.tight_layout()
    out = os.path.join(OUTPUT_BASE, 'disharmony_scores.png')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print('\nSaved: %s' % out)

    # Save raw scores
    out_csv = os.path.join(OUTPUT_BASE, 'disharmony_scores.csv')
    rdf.to_csv(out_csv, index=False)
    print('Saved: %s' % out_csv)


if __name__ == '__main__':
    main()
