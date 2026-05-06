"""
Disharmony metric for ALL injury groups (D, G, H, K, L, M).
Same method as m_disharmony_metric.py but generalized.

For each animal in each group:
1. Build a mean trajectory from pre-injury successful retrievals (nose-centered, time-normalized)
2. Score every reach in every phase against that model
3. Disharmony = variability-weighted z-scored deviation
4. Signed lateral (X) and extension (Y) components preserved

Usage:
    python all_groups_disharmony.py              # All groups
    python all_groups_disharmony.py --group K    # Single group
    python all_groups_disharmony.py --quick      # Quick: M only
"""

import pandas as pd
import numpy as np
import os
import sys
import glob
from datetime import datetime, timedelta

# --- Paths ---
ASPA_ROOT = r'Y:\2_Connectome\Behavior\MouseReach_Pipeline\Analyzed\ASPA'
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

N_NORM_POINTS = 30
MIN_RETRIEVALS = 10

# DLC path mapping (same as all_groups_spaghetti.py)
DLC_PATHS = {
    'D': (os.path.join(ASPA_ROOT, 'OptD', 'Post-Processing', 'D', 'All D Analyzed'), 'mixed'),
    'G': (os.path.join(ASPA_ROOT, 'OptG', 'Post-Processing'), 'single'),
    'H': (os.path.join(ASPA_ROOT, 'H', 'Post-Processing'), 'single'),
    'K': (os.path.join(ASPA_ROOT, 'K', 'Post-Processing'), 'single'),
    'L': (os.path.join(ASPA_ROOT, 'L', 'Post-Processing'), 'single'),
    'M': (os.path.join(ASPA_ROOT, 'M', 'Post-Processing'), 'single'),
}

GROUP_FILES = {
    'D': ['Swipe_Pyramidotomy_Data.csv'],
    'G': ['Swipe_Transection_Data.csv', 'Swipe_Transection_Data_2.csv'],
    'H': ['Swipe_Transection_Data.csv', 'Swipe_Transection_Data_2.csv'],
    'K': ['Swipe_Contusion_Data.csv', 'Swipe_Contusion_Data_2.csv'],
    'L': ['Swipe_Contusion_Data.csv', 'Swipe_Contusion_Data_2.csv'],
    'M': ['Swipe_Contusion_Data.csv', 'Swipe_Contusion_Data_2.csv'],
}

PHASES = {
    'Pre-Injury': ['2_Pre-injury_1', '2_Pre-injury_2', '2_Pre-injury_3'],
    '1wk Post': ['3_1wk_Post-injury', '3_1wk_Post-injury_First_Test'],
    '2-4wk Post': ['3_2wk_Post-injury', '3_3wk_Post-injury', '3_4wk_Post-injury'],
    'Post-Rehab': ['5_Post-rehab_Test_1', '5_Post-rehab_Test_2'],
}


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


def find_dlc_h5(stem, dlc_dir, style, collage_cache):
    """Find DLC h5 file. Handles single and collage naming."""
    h5 = glob.glob(os.path.join(dlc_dir, stem + 'DLC*.h5'))
    if h5:
        return h5[0]

    if style in ('collage', 'mixed'):
        parts = stem.split('_')
        if len(parts) < 3:
            return None
        date, animal, pillar = parts[0], parts[1], parts[2]

        if dlc_dir not in collage_cache:
            idx = {}
            for f in glob.glob(os.path.join(dlc_dir, '*DLC*.h5')):
                bn = os.path.basename(f)
                if ',' in bn:
                    try:
                        prefix = bn.split('DLC')[0]
                        mkv_parts = prefix.split('.mkv_')
                        if len(mkv_parts) == 2:
                            position = int(mkv_parts[1].rstrip('_'))
                            before_mkv = mkv_parts[0]
                            first_underscore = before_mkv.index('_')
                            last_underscore = before_mkv.rindex('_')
                            file_date = before_mkv[:first_underscore]
                            file_pillar = before_mkv[last_underscore+1:]
                            animal_str = before_mkv[first_underscore+1:last_underscore]
                            animal_list = animal_str.split(',')
                            if position <= len(animal_list):
                                target_animal = animal_list[position - 1]
                                key = (file_date, target_animal, file_pillar)
                                idx[key] = f
                    except (ValueError, IndexError):
                        continue
            collage_cache[dlc_dir] = idx

        idx = collage_cache[dlc_dir]
        key = (date, animal, pillar)
        if key in idx:
            return idx[key]

    return None


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
    t_orig = np.linspace(0, 1, len(x))
    t_new = np.linspace(0, 1, n_pts)
    return np.interp(t_new, t_orig, x), np.interp(t_new, t_orig, y)


def compute_disharmony(traj_x, traj_y, mean_x, mean_y, std_x, std_y):
    """Variability-weighted disharmony. Returns (overall, lateral_signed, extension_signed)."""
    dx = traj_x - mean_x
    dy = traj_y - mean_y
    zx = dx / np.maximum(std_x, 0.5)
    zy = dy / np.maximum(std_y, 0.5)
    z_dist = np.sqrt(zx**2 + zy**2)
    return np.mean(z_dist), np.mean(zx), np.mean(zy)


def load_group_data(group):
    dfs = []
    for fname in GROUP_FILES[group]:
        dfs.append(pd.read_csv(os.path.join(UCSF_BASE, fname), low_memory=False))
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['SubjectID'].str.startswith(group, na=False)].copy()
    df['sd_start'], df['sd_end'] = zip(*df['Swipe_Duration'].apply(parse_frame_range))
    return df


def run_group(group):
    """Compute disharmony for all animals in one group."""
    print('\n=== Group %s ===' % group)

    df = load_group_data(group)
    animals = sorted(df['SubjectID'].unique())

    dlc_dir, dlc_style = DLC_PATHS.get(group, (None, None))
    if dlc_dir is None or not os.path.isdir(dlc_dir):
        print('  No DLC directory, skipping.')
        return []

    dlc_cache = {}
    collage_cache = {}

    def get_dlc(stem):
        if stem not in dlc_cache:
            h5_path = find_dlc_h5(stem, dlc_dir, dlc_style, collage_cache)
            if h5_path is None:
                dlc_cache[stem] = None
            else:
                d = pd.read_hdf(h5_path)
                sc = d.columns.get_level_values(0)[0]
                dlc_cache[stem] = d[sc]
        return dlc_cache[stem]

    # Step 1: Build per-animal model from pre-injury retrievals
    print('  Building models...')
    animal_models = {}
    for animal in animals:
        pre = df[(df['SubjectID'] == animal) &
                 (df['Test_Type_Grouped_1'].isin(PHASES['Pre-Injury'])) &
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
            print('    %s: %d retrieval trajectories' % (animal, len(trajs)))
        else:
            print('    %s: SKIP (%d trajectories)' % (animal, len(trajs)))

    # Step 2: Score every reach
    print('  Scoring...')
    results = []
    for animal in animals:
        if animal not in animal_models:
            continue
        mean_x, mean_y, std_x, std_y, _ = animal_models[animal]

        for phase_name, phase_labels in PHASES.items():
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
                    'group': group,
                    'animal': animal,
                    'phase': phase_name,
                    'disharmony': score,
                    'lateral': lateral,
                    'extension': extension,
                    'outcome': row['Reach_outcome'],
                    'session': row['Session_ID'],
                })

    print('  %d reaches scored, %d animals with models' % (len(results), len(animal_models)))
    return results


def main():
    quick = '--quick' in sys.argv

    target_group = None
    for i, arg in enumerate(sys.argv):
        if arg == '--group' and i + 1 < len(sys.argv):
            target_group = sys.argv[i + 1].upper()

    groups = [target_group] if target_group else ['D', 'G', 'H', 'K', 'L', 'M']

    all_results = []
    for group in groups:
        if group not in DLC_PATHS:
            print('Unknown group: %s' % group)
            continue
        results = run_group(group)
        all_results.extend(results)

    if not all_results:
        print('No results.')
        return

    rdf = pd.DataFrame(all_results)
    print('\n=== SUMMARY ===')
    print('Total reaches scored: %d' % len(rdf))
    print('Groups: %s' % sorted(rdf['group'].unique()))

    # Summary table: median disharmony per animal per phase
    print('\n%-8s %-6s %8s %8s %8s %8s %8s' % (
        'Group', 'Animal', 'Pre', '1wk', '2-4wk', 'PostRe', 'Post/Pre'))
    print('-' * 70)

    # Load suspect list for tagging
    suspect_csv = os.path.join(OUTPUT_BASE, 'suspect_animals.csv')
    suspects = set()
    if os.path.exists(suspect_csv):
        suspects = set(pd.read_csv(suspect_csv)['animal'].values)

    for group in sorted(rdf['group'].unique()):
        gdf = rdf[rdf['group'] == group]
        for animal in sorted(gdf['animal'].unique()):
            adf = gdf[gdf['animal'] == animal]
            vals = {}
            for phase_name in PHASES.keys():
                ps = adf[adf['phase'] == phase_name]['disharmony']
                vals[phase_name] = ps.median() if len(ps) > 0 else np.nan

            pre = vals.get('Pre-Injury', np.nan)
            post1 = vals.get('1wk Post', np.nan)
            ratio = post1 / pre if pre > 0 and not np.isnan(post1) else np.nan
            tag = ' ***' if animal in suspects else ''

            print('%-8s %-6s %8.1f %8.1f %8.1f %8.1f %8.2fx%s' % (
                group, animal,
                vals.get('Pre-Injury', np.nan),
                vals.get('1wk Post', np.nan),
                vals.get('2-4wk Post', np.nan),
                vals.get('Post-Rehab', np.nan),
                ratio, tag))
        print()

    # Save
    out = os.path.join(OUTPUT_BASE, 'all_groups_disharmony_scores.csv')
    rdf.to_csv(out, index=False)
    print('Saved: %s' % out)


if __name__ == '__main__':
    main()
