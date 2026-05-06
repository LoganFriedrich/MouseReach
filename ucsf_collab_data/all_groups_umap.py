"""
UMAP visualization of reach trajectories across all injury groups.

Four panels, same reaches, same coloring (by group), different feature spaces:
1. Raw trajectory shape (60D: 30 x + 30 y, nose-centered, time-normalized)
2. Kinematic features (6D: area, length, breadth, speed, path/frames, duration)
3. Combined (66D: trajectory + kinematics)
4. Per-quintile trajectory (20D: mean x,y per quintile segment)

Color by group. Suspect animals marked with edge highlight.

Usage:
    python all_groups_umap.py              # All groups, all phases
    python all_groups_umap.py --phase "1wk Post"   # Specific phase
    python all_groups_umap.py --quick      # M only, Pre + 1wk Post
"""

import pandas as pd
import numpy as np
import os
import sys
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import umap

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
SUSPECT_CSV = os.path.join(OUTPUT_BASE, 'suspect_animals.csv')

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

GROUP_COLORS = {
    'D': '#1f77b4',   # blue
    'G': '#2ca02c',   # green
    'H': '#17becf',   # cyan
    'K': '#CC44CC',   # magenta
    'L': '#9467bd',   # purple
    'M': '#5588AA',   # slate blue
}

GROUP_LABELS = {
    'D': 'D (Pyramidotomy)',
    'G': 'G (Transection)',
    'H': 'H (Transection)',
    'K': 'K (Contusion 70kD)',
    'L': 'L (Contusion 50kD)',
    'M': 'M (Contusion 60kD)',
}

PHASES = {
    'Pre-Injury': ['2_Pre-injury_1', '2_Pre-injury_2', '2_Pre-injury_3'],
    '1wk Post': ['3_1wk_Post-injury', '3_1wk_Post-injury_First_Test'],
    '2-4wk Post': ['3_2wk_Post-injury', '3_3wk_Post-injury', '3_4wk_Post-injury'],
    'Post-Rehab': ['5_Post-rehab_Test_1', '5_Post-rehab_Test_2'],
}

N_NORM_POINTS = 30
N_QUINTILE_POINTS = 5


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
                                idx[(file_date, target_animal, file_pillar)] = f
                    except (ValueError, IndexError):
                        continue
            collage_cache[dlc_dir] = idx
        idx = collage_cache[dlc_dir]
        key = (date, animal, pillar)
        if key in idx:
            return idx[key]
    return None


def extract_reach_features(swipes_df, dlc_dir, dlc_style, dlc_cache, collage_cache):
    """Extract per-reach feature vectors for all 4 feature spaces.

    Returns list of dicts, each with:
        group, animal, phase, suspect,
        traj_features (60D), kin_features (6D), combined (66D), quintile (20D)
    """
    results = []
    for _, row in swipes_df.iterrows():
        s = row['sd_start']
        e = row['sd_end']
        if pd.isna(s):
            continue
        s, e = int(s), int(e)
        stem = session_to_stem(row['Session_ID'])
        if stem not in dlc_cache:
            h5_path = find_dlc_h5(stem, dlc_dir, dlc_style, collage_cache)
            if h5_path is None:
                dlc_cache[stem] = None
                continue
            d = pd.read_hdf(h5_path)
            sc = d.columns.get_level_values(0)[0]
            dlc_cache[stem] = d[sc]
        dlc = dlc_cache[stem]
        if dlc is None or e + 1 > len(dlc):
            continue

        rh_x = dlc['RightHand']['x'].iloc[s:e+1].values
        rh_y = dlc['RightHand']['y'].iloc[s:e+1].values
        rh_lk = dlc['RightHand']['likelihood'].iloc[s:e+1].values
        nose_x = dlc['Nose']['x'].iloc[s:e+1].values
        nose_y = dlc['Nose']['y'].iloc[s:e+1].values

        if np.mean(rh_lk) < 0.3 or len(rh_x) < 3:
            continue

        n0x, n0y = nose_x[0], nose_y[0]
        cx = rh_x - n0x
        cy = rh_y - n0y

        # Feature space 1: Time-normalized trajectory (60D)
        t_orig = np.linspace(0, 1, len(cx))
        t_new = np.linspace(0, 1, N_NORM_POINTS)
        norm_x = np.interp(t_new, t_orig, cx)
        norm_y = np.interp(t_new, t_orig, cy)
        traj_feat = np.concatenate([norm_x, norm_y])

        # Feature space 2: Kinematic features (6D)
        n_frames = len(cx)
        path_length = np.sum(np.sqrt(np.diff(cx)**2 + np.diff(cy)**2))
        area = 0.5 * np.abs(np.dot(cx, np.roll(cy, 1)) - np.dot(cy, np.roll(cx, 1)))
        length = np.max(cy) - nose_y[0] + n0y  # max RH_y - nose_y at peak
        # Recalculate length as max extension from nose
        peak_idx = np.argmax(cy)
        length = cy[peak_idx]  # already nose-centered
        breadth = np.max(cx) - np.min(cx)
        speed = (path_length / 4.0) / ((n_frames - 1) / 60.0) if n_frames > 1 else 0
        pof = path_length / n_frames if n_frames > 0 else 0
        kin_feat = np.array([area, length, breadth, speed, pof, n_frames])

        # Feature space 5: All available UCSF numeric features
        def safe_float(val):
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0.0

        all_csv_feat = np.array([
            safe_float(row.get('Swipe_breadth', 0)),
            safe_float(row.get('Swipe_length', 0)),
            safe_float(row.get('Swipe_area', 0)),
            safe_float(row.get('Swipe_speed', 0)),
            safe_float(row.get('Path_length', 0)),
            safe_float(row.get('Swipe_Duration_Frames', 0)),
            safe_float(row.get('Path_over_Frames', 0)),
            safe_float(row.get('Attention_AI', 0)),
            safe_float(row.get('Total_Swipes_AI', 0)),
            safe_float(row.get('Video_Displaced', 0)),
            safe_float(row.get('Video_Retrieved', 0)),
            safe_float(row.get('Video_Contacted', 0)),
            safe_float(row.get('Pellet_#', 0)),
        ])

        # Feature space 3: Combined (66D)
        combined_feat = np.concatenate([traj_feat, kin_feat])

        # Feature space 4: Per-quintile means (20D)
        q_size = N_NORM_POINTS // N_QUINTILE_POINTS
        quintile_feat = []
        for qi in range(N_QUINTILE_POINTS):
            start = qi * q_size
            end = start + q_size
            quintile_feat.extend([
                np.mean(norm_x[start:end]),
                np.mean(norm_y[start:end]),
                np.std(norm_x[start:end]),
                np.std(norm_y[start:end]),
            ])
        quintile_feat = np.array(quintile_feat)

        results.append({
            'group': row['group'],
            'animal': row['SubjectID'],
            'phase': row['phase'],
            'traj': traj_feat,
            'kin': kin_feat,
            'combined': combined_feat,
            'quintile': quintile_feat,
            'all_csv': all_csv_feat,
        })

    return results


SUSPECT_COLOR = '#FFFF00'  # yellow for suspects, matching spaghetti convention

PANEL_DESCRIPTIONS = {
    'Trajectory Shape (60D)': 'Trajectory Shape\n30 time-normalized (x,y) points per reach\nPure spatial path geometry',
    'Kinematics (6D)': 'Kinematic Features\nArea, Length, Breadth, Speed,\nPath/Frames, Duration',
    'Combined (66D)': 'Combined\nTrajectory shape + kinematic features\n(66 dimensions)',
    'Per-Quintile (20D)': 'Per-Quintile Trajectory\nMean + SD of (x,y) in 5 reach segments\nCaptures WHERE deviation occurs',
    'All CSV Features (13D)': 'All UCSF Features\nAll per-swipe + per-video metrics\nincl. Attention, pellet counts, outcomes',
}


def plot_umap_panel(ax, embedding, groups, suspects, title):
    """Plot one UMAP panel colored by group. Suspects in yellow."""
    unique_groups = sorted(set(groups))

    # Draw normals first (behind)
    for g in unique_groups:
        mask = np.array([gi == g for gi in groups])
        normal_mask = ~np.array(suspects) & mask
        color = GROUP_COLORS.get(g, 'gray')
        if normal_mask.any():
            ax.scatter(embedding[normal_mask, 0], embedding[normal_mask, 1],
                       c=color, s=8, alpha=0.4, label=GROUP_LABELS.get(g, g),
                       edgecolors='none', rasterized=True)

    # Draw suspects on top in yellow
    suspect_mask = np.array(suspects)
    if suspect_mask.any():
        ax.scatter(embedding[suspect_mask, 0], embedding[suspect_mask, 1],
                   c=SUSPECT_COLOR, s=12, alpha=0.6, edgecolors='black', linewidth=0.3,
                   label='Suspects', rasterized=True)

    desc = PANEL_DESCRIPTIONS.get(title, title)
    ax.set_title(desc, fontsize=10, fontweight='bold')
    ax.set_xlabel('UMAP 1', fontsize=9)
    ax.set_ylabel('UMAP 2', fontsize=9)
    ax.tick_params(labelsize=7)


def main():
    quick = '--quick' in sys.argv

    # Parse --phase
    target_phase = None
    for i, arg in enumerate(sys.argv):
        if arg == '--phase' and i + 1 < len(sys.argv):
            target_phase = sys.argv[i + 1]

    # Load suspects
    suspects_set = set()
    if os.path.exists(SUSPECT_CSV):
        suspects_set = set(pd.read_csv(SUSPECT_CSV)['animal'].values)

    # Load and merge all UCSF data
    print('Loading UCSF data...')
    all_dfs = []
    for group, files in GROUP_FILES.items():
        if quick and group != 'M':
            continue
        for fname in files:
            df = pd.read_csv(os.path.join(UCSF_BASE, fname), low_memory=False)
            df = df[df['SubjectID'].str.startswith(group, na=False)]
            all_dfs.append(df)

    df = pd.concat(all_dfs, ignore_index=True)
    df['group'] = df['SubjectID'].str[0]
    df['sd_start'], df['sd_end'] = zip(*df['Swipe_Duration'].apply(parse_frame_range))

    # Assign phases
    test_to_phase = {}
    for phase, tests in PHASES.items():
        for t in tests:
            test_to_phase[t] = phase
    df['phase'] = df['Test_Type_Grouped_1'].map(test_to_phase)
    df = df[df['phase'].notna()].copy()

    if target_phase:
        df = df[df['phase'] == target_phase]
        print('Filtering to phase: %s' % target_phase)

    if quick:
        df = df[df['phase'].isin(['Pre-Injury', '1wk Post'])]
        print('QUICK MODE: M only, Pre + 1wk Post')

    print('Total swipes to process: %d' % len(df))

    # Extract features per group
    all_features = []
    for group in sorted(df['group'].unique()):
        gdf = df[df['group'] == group]
        dlc_dir, dlc_style = DLC_PATHS.get(group, (None, None))
        if dlc_dir is None or not os.path.isdir(dlc_dir):
            print('  %s: no DLC dir, skipping' % group)
            continue

        print('  Extracting %s (%d swipes)...' % (group, len(gdf)))
        dlc_cache = {}
        collage_cache = {}
        feats = extract_reach_features(gdf, dlc_dir, dlc_style, dlc_cache, collage_cache)
        all_features.extend(feats)
        print('    -> %d reaches extracted' % len(feats))

    print('Total reaches with features: %d' % len(all_features))

    if len(all_features) < 100:
        print('Too few reaches for UMAP. Exiting.')
        return

    # Build matrices
    groups = [f['group'] for f in all_features]
    animals = [f['animal'] for f in all_features]
    phases = [f['phase'] for f in all_features]
    is_suspect = [f['animal'] in suspects_set for f in all_features]

    feature_spaces = {
        'Trajectory Shape (60D)': np.array([f['traj'] for f in all_features]),
        'Kinematics (6D)': np.array([f['kin'] for f in all_features]),
        'Combined (66D)': np.array([f['combined'] for f in all_features]),
        'Per-Quintile (20D)': np.array([f['quintile'] for f in all_features]),
        'All CSV Features (13D)': np.array([f['all_csv'] for f in all_features]),
    }

    # Standardize each feature space
    from sklearn.preprocessing import StandardScaler
    for name in feature_spaces:
        scaler = StandardScaler()
        feature_spaces[name] = scaler.fit_transform(feature_spaces[name])
        # Replace any NaN/Inf with 0
        feature_spaces[name] = np.nan_to_num(feature_spaces[name], nan=0.0, posinf=0.0, neginf=0.0)

    # Run UMAP on each
    print('\nRunning UMAP...')
    embeddings = {}
    for name, X in feature_spaces.items():
        print('  %s (%d x %d)...' % (name, X.shape[0], X.shape[1]))
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric='euclidean', random_state=42)
        embeddings[name] = reducer.fit_transform(X)

    # Plot: 5 panels (3x2 grid, last cell empty)
    fig, axes = plt.subplots(2, 3, figsize=(27, 16))
    fig.patch.set_facecolor('white')

    for ax, (name, emb) in zip(axes.flat, embeddings.items()):
        plot_umap_panel(ax, emb, groups, is_suspect, name)

    # Hide the 6th (empty) panel
    if len(embeddings) < 6:
        axes.flat[5].set_visible(False)

    # Single legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for g in sorted(set(groups)):
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=GROUP_COLORS.get(g, 'gray'),
                   markersize=8, label=GROUP_LABELS.get(g, g)))
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor=SUSPECT_COLOR,
               markersize=10, markeredgecolor='black', markeredgewidth=1.5,
               label='Suspect animals (yellow)'))

    fig.legend(handles=legend_elements, loc='upper center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 0.98))

    phase_str = target_phase if target_phase else ('Pre + 1wk (quick)' if quick else 'All phases')
    fig.suptitle('UMAP: Reach Trajectories Across Injury Groups\n'
                 'Phase: %s | n=%d reaches | 4 feature spaces'
                 % (phase_str, len(all_features)),
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    suffix = '_quick' if quick else ''
    if target_phase:
        suffix = '_' + target_phase.replace(' ', '_').replace('-', '')
    out = os.path.join(OUTPUT_BASE, 'all_groups_umap%s.png' % suffix)
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print('Saved: %s' % out)

    # Save embeddings for reuse
    emb_df = pd.DataFrame({
        'group': groups,
        'animal': animals,
        'phase': phases,
        'suspect': is_suspect,
    })
    for name, emb in embeddings.items():
        short = name.split('(')[0].strip().replace(' ', '_').lower()
        emb_df['%s_umap1' % short] = emb[:, 0]
        emb_df['%s_umap2' % short] = emb[:, 1]

    emb_csv = os.path.join(OUTPUT_BASE, 'all_groups_umap_embeddings%s.csv' % suffix)
    emb_df.to_csv(emb_csv, index=False)
    print('Saved embeddings: %s' % emb_csv)


if __name__ == '__main__':
    main()
