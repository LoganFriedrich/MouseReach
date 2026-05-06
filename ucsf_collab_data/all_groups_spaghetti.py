"""
All-groups spaghetti: one figure per injury group, all mice on one plot.
Suspects (from all_groups_suspect_identification.py) in yellow, normals in magenta.
Pellet heatmap in cyan with median centroid marker.
4 phases x 2 rows (all reaches / D+R).

Usage:
    python all_groups_spaghetti.py              # All 6 groups
    python all_groups_spaghetti.py --group M    # Single group
    python all_groups_spaghetti.py --quick      # Quick: M only, Pre-Injury only
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
OUTPUT_BASE = os.path.join(_connectome_root, 'Databases', 'figures', 'bodypart_tracking_quality', 'labeled_frames')
SUSPECT_CSV = os.path.join(_connectome_root, 'Databases', 'figures', 'bodypart_tracking_quality', 'suspect_animals.csv')

# --- DLC path mapping per group ---
# Maps group letter to (DLC directory, naming style)
# 'single': files named YYYYMMDD_AnimalID_PX_DLC*.h5
# 'collage': files named YYYYMMDD_A1,A2,...,AN_PX.mkv_NDLC*.h5
DLC_PATHS = {
    'D': (os.path.join(ASPA_ROOT, 'OptD', 'Post-Processing', 'D', 'All D Analyzed'), 'mixed'),
    'G': (os.path.join(ASPA_ROOT, 'OptG', 'Post-Processing'), 'single'),
    'H': (os.path.join(ASPA_ROOT, 'H', 'Post-Processing'), 'single'),
    'K': (os.path.join(ASPA_ROOT, 'K', 'Post-Processing'), 'single'),
    'L': (os.path.join(ASPA_ROOT, 'L', 'Post-Processing'), 'single'),
    'M': (os.path.join(ASPA_ROOT, 'M', 'Post-Processing'), 'single'),
}

# --- UCSF data files per group ---
GROUP_FILES = {
    'D': ['Swipe_Pyramidotomy_Data.csv'],
    'G': ['Swipe_Transection_Data.csv', 'Swipe_Transection_Data_2.csv'],
    'H': ['Swipe_Transection_Data.csv', 'Swipe_Transection_Data_2.csv'],
    'K': ['Swipe_Contusion_Data.csv', 'Swipe_Contusion_Data_2.csv'],
    'L': ['Swipe_Contusion_Data.csv', 'Swipe_Contusion_Data_2.csv'],
    'M': ['Swipe_Contusion_Data.csv', 'Swipe_Contusion_Data_2.csv'],
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

# CYM: Cyan=pellet, Magenta=normal, Yellow=suspect
COLOR_NORMAL = '#FF00FF'
COLOR_NORMAL_LIGHT = '#993399'
COLOR_SUSPECT = '#FFFF00'
COLOR_SUSPECT_LIGHT = '#999933'


def parse_frame_range(s):
    parts = str(s).strip().split('-')
    if len(parts) == 2:
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            pass
    return None, None


def session_to_stem(sid):
    """Convert UCSF Session_ID to filename stem. e.g. D01-44609-P1 -> 20220217_D01_P1"""
    parts = sid.split('-')
    date_str = (datetime(1899, 12, 30) + timedelta(days=int(parts[1]))).strftime('%Y%m%d')
    return '%s_%s_%s' % (date_str, parts[0], parts[2])


def find_dlc_h5(stem, dlc_dir, style, collage_index_cache):
    """Find DLC h5 file for a given stem. Handles single and collage naming."""
    # Try single-animal match first (works for all groups)
    h5 = glob.glob(os.path.join(dlc_dir, stem + 'DLC*.h5'))
    if h5:
        return h5[0]

    if style in ('collage', 'mixed'):
        # Parse stem: YYYYMMDD_AnimalID_PX
        parts = stem.split('_')
        if len(parts) < 3:
            return None
        date = parts[0]
        animal = parts[1]
        pillar = parts[2]  # e.g. P1

        # Build collage index for this directory if not cached
        if dlc_dir not in collage_index_cache:
            idx = {}
            for f in glob.glob(os.path.join(dlc_dir, '*DLC*.h5')):
                bn = os.path.basename(f)
                if ',' in bn:
                    # Collage file: YYYYMMDD_A1,A2,...,AN_PX.mkv_NDLC*.h5
                    # Extract date, animal list, pillar, position
                    try:
                        # Split off the DLC suffix to get the prefix
                        prefix = bn.split('DLC')[0]
                        # prefix like: 20220217_D01,D02,...,D08_P1.mkv_4
                        # Find the position number (last _ before DLC)
                        mkv_parts = prefix.split('.mkv_')
                        if len(mkv_parts) == 2:
                            position = int(mkv_parts[1].rstrip('_'))
                            before_mkv = mkv_parts[0]
                            # before_mkv: 20220217_D01,D02,...,D08_P1
                            first_underscore = before_mkv.index('_')
                            last_underscore = before_mkv.rindex('_')
                            file_date = before_mkv[:first_underscore]
                            file_pillar = before_mkv[last_underscore+1:]
                            animal_str = before_mkv[first_underscore+1:last_underscore]
                            animal_list = animal_str.split(',')
                            # position is 1-based index into animal_list
                            if position <= len(animal_list):
                                target_animal = animal_list[position - 1]
                                key = (file_date, target_animal, file_pillar)
                                idx[key] = f
                    except (ValueError, IndexError):
                        continue
            collage_index_cache[dlc_dir] = idx

        idx = collage_index_cache[dlc_dir]
        key = (date, animal, pillar)
        if key in idx:
            return idx[key]

    return None


def extract_trajectories(swipes_df, dlc_dir, dlc_style, dlc_cache, collage_cache):
    """Extract nose-centered RH trajectories + pellet positions."""
    trajectories = []
    pellet_positions = []
    miss_count = 0
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
                miss_count += 1
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
        n0x, n0y = nose_x[0], nose_y[0]
        cx = rh_x - n0x
        cy = rh_y - n0y
        if np.mean(rh_lk) > 0.3 and len(cx) >= 3:
            trajectories.append((cx, cy))
        pel_x = dlc['Pellet']['x'].iloc[s:e+1].values
        pel_y = dlc['Pellet']['y'].iloc[s:e+1].values
        pel_lk = dlc['Pellet']['likelihood'].iloc[s:e+1].values
        good = pel_lk > 0.5
        if good.any():
            pellet_positions.extend(zip(pel_x[good] - n0x, pel_y[good] - n0y))
    if miss_count > 0:
        print('    (%d sessions with no DLC match)' % miss_count)
    return trajectories, pellet_positions


def compute_pellet_centroid(pellet_positions):
    if not pellet_positions or len(pellet_positions) < 10:
        return None
    px = np.array([p[0] for p in pellet_positions])
    py = np.array([p[1] for p in pellet_positions])
    return np.median(px), np.median(py)


def compute_mean_traj(trajectories, n_pts=30):
    if not trajectories:
        return None
    norm_x, norm_y = [], []
    for x, y in trajectories:
        t_orig = np.linspace(0, 1, len(x))
        t_new = np.linspace(0, 1, n_pts)
        norm_x.append(np.interp(t_new, t_orig, x))
        norm_y.append(np.interp(t_new, t_orig, y))
    mean_x = np.mean(norm_x, axis=0)
    mean_y = np.mean(norm_y, axis=0)
    return mean_x, mean_y, norm_x, norm_y


def draw_pellet_heatmap(ax, pellet_positions):
    if not pellet_positions:
        return
    px = np.array([p[0] for p in pellet_positions])
    py = np.array([p[1] for p in pellet_positions])
    h, xedges, yedges = np.histogram2d(px, py, bins=50)
    h = h.T
    rgba = np.zeros((*h.shape, 4))
    if h.max() > 0:
        norm_h = h / h.max()
        rgba[:, :, 1] = norm_h
        rgba[:, :, 2] = norm_h
        rgba[:, :, 3] = norm_h * 0.8
    extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
    ax.imshow(rgba, extent=extent, aspect='auto', interpolation='bilinear', zorder=1)


def draw_pellet_centroid(ax, centroid):
    if centroid is None:
        return
    ax.plot(centroid[0], centroid[1], '*', color='#00FFFF', markersize=14,
            markeredgecolor='white', markeredgewidth=0.8, zorder=20)


def draw_trajectories(ax, trajectories, color, color_light):
    n = len(trajectories)
    if n == 0:
        return
    alpha = max(0.04, min(0.20, 30.0 / max(n, 1)))
    lw = max(0.3, min(0.5, 25.0 / max(n, 1)))
    for x, y in trajectories:
        ax.plot(x, y, color=color_light, alpha=alpha, linewidth=lw, zorder=2)


def draw_mean_with_band(ax, trajectories, color, lw=2.5, zorder=10):
    result = compute_mean_traj(trajectories)
    if result is None:
        return
    mean_x, mean_y, norm_x, norm_y = result
    n_pts = len(mean_x)
    n_trajs = len(trajectories)

    dx = np.gradient(mean_x)
    dy = np.gradient(mean_y)
    perp_x = -dy
    perp_y = dx
    mag = np.sqrt(perp_x**2 + perp_y**2)
    mag[mag == 0] = 1
    perp_x /= mag
    perp_y /= mag

    perp_dists = np.zeros((n_trajs, n_pts))
    for i in range(n_trajs):
        diff_x = norm_x[i] - mean_x
        diff_y = norm_y[i] - mean_y
        perp_dists[i, :] = diff_x * perp_x + diff_y * perp_y

    # IQR band
    q25 = np.percentile(perp_dists, 25, axis=0)
    q75 = np.percentile(perp_dists, 75, axis=0)
    upper_x = mean_x + perp_x * q75
    upper_y = mean_y + perp_y * q75
    lower_x = mean_x + perp_x * q25
    lower_y = mean_y + perp_y * q25

    poly_x = np.concatenate([upper_x, lower_x[::-1]])
    poly_y = np.concatenate([upper_y, lower_y[::-1]])
    ax.fill(poly_x, poly_y, color=color, alpha=0.15, zorder=zorder - 1)

    ax.plot(mean_x, mean_y, color='white', linewidth=lw + 1.5, alpha=0.4, zorder=zorder)
    ax.plot(mean_x, mean_y, color=color, linewidth=lw, alpha=1.0, zorder=zorder + 1)


def plot_panel(ax, normal_trajs, suspect_trajs, pellet_positions, title, pellet_centroid=None):
    n_total = len(normal_trajs) + len(suspect_trajs)

    draw_pellet_heatmap(ax, pellet_positions)
    draw_trajectories(ax, normal_trajs, COLOR_NORMAL, COLOR_NORMAL_LIGHT)
    draw_trajectories(ax, suspect_trajs, COLOR_SUSPECT, COLOR_SUSPECT_LIGHT)

    if normal_trajs:
        draw_mean_with_band(ax, normal_trajs, COLOR_NORMAL, lw=2.5, zorder=10)
    if suspect_trajs:
        draw_mean_with_band(ax, suspect_trajs, COLOR_SUSPECT, lw=2.5, zorder=12)

    ax.plot(0, 0, 'o', color='white', markersize=5, markeredgecolor='black',
            markeredgewidth=1, zorder=15)
    draw_pellet_centroid(ax, pellet_centroid)

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title('%s (n=%d)' % (title, n_total), fontsize=10, fontweight='bold', color='white')
    ax.set_xlabel('Lateral from nose (px)', fontsize=8)
    ax.set_ylabel('Extension from nose (px)', fontsize=8)
    ax.set_facecolor('black')
    ax.tick_params(colors='white', labelsize=7)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')


def load_group_data(group):
    """Load UCSF swipe data for one group."""
    dfs = []
    for fname in GROUP_FILES[group]:
        dfs.append(pd.read_csv(os.path.join(UCSF_BASE, fname), low_memory=False))
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['SubjectID'].str.startswith(group, na=False)].copy()
    df['sd_start'], df['sd_end'] = zip(*df['Swipe_Duration'].apply(parse_frame_range))
    return df


def run_group(group, suspects, quick=False):
    """Generate spaghetti figure for one group."""
    print('\n=== Group %s ===' % GROUP_LABELS.get(group, group))

    df = load_group_data(group)
    animals = sorted(df['SubjectID'].unique())
    group_suspects = [a for a in animals if a in suspects]
    normal_animals = [a for a in animals if a not in suspects]
    print('Animals: %d total, %d suspects %s' % (len(animals), len(group_suspects), group_suspects))

    dlc_dir, dlc_style = DLC_PATHS.get(group, (None, None))
    if dlc_dir is None or not os.path.isdir(dlc_dir):
        print('  No DLC directory found, skipping.')
        return

    if quick:
        phases_to_run = {'Pre-Injury': PHASES['Pre-Injury']}
    else:
        phases_to_run = PHASES

    dlc_cache = {}
    collage_cache = {}

    phase_data = {}
    for phase_name, phase_labels in phases_to_run.items():
        phase_df = df[df['Test_Type_Grouped_1'].isin(phase_labels)]
        contact_df = phase_df[phase_df['Reach_outcome'].isin(['pellet displaced', 'swipe successful'])]

        normal_df = phase_df[phase_df['SubjectID'].isin(normal_animals)]
        suspect_df = phase_df[phase_df['SubjectID'].isin(group_suspects)]
        normal_contact = contact_df[contact_df['SubjectID'].isin(normal_animals)]
        suspect_contact = contact_df[contact_df['SubjectID'].isin(group_suspects)]

        print('  %s...' % phase_name)
        n_all, n_all_pel = extract_trajectories(normal_df, dlc_dir, dlc_style, dlc_cache, collage_cache)
        s_all, s_all_pel = extract_trajectories(suspect_df, dlc_dir, dlc_style, dlc_cache, collage_cache)
        n_contact, n_contact_pel = extract_trajectories(normal_contact, dlc_dir, dlc_style, dlc_cache, collage_cache)
        s_contact, s_contact_pel = extract_trajectories(suspect_contact, dlc_dir, dlc_style, dlc_cache, collage_cache)
        print('    normal: %d all, %d contact | suspect: %d all, %d contact' % (
            len(n_all), len(n_contact), len(s_all), len(s_contact)))

        phase_data[phase_name] = {
            'normal_all': n_all, 'suspect_all': s_all,
            'normal_contact': n_contact, 'suspect_contact': s_contact,
            'all_pellets': n_all_pel + s_all_pel,
            'contact_pellets': n_contact_pel + s_contact_pel,
        }

    phase_order = [p for p in ['Pre-Injury', '1wk Post', '2-4wk Post', 'Post-Rehab'] if p in phases_to_run]
    n_cols = len(phase_order)

    fig, axes = plt.subplots(2, n_cols, figsize=(8 * n_cols, 16))
    fig.patch.set_facecolor('black')
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    # Consistent axis limits across panels
    all_trajs = []
    for pn in phase_order:
        all_trajs.extend(phase_data[pn]['normal_all'])
        all_trajs.extend(phase_data[pn]['suspect_all'])

    xlim, ylim = None, None
    if all_trajs:
        all_x = np.concatenate([t[0] for t in all_trajs])
        all_y = np.concatenate([t[1] for t in all_trajs])
        pad = 10
        xlim = (np.percentile(all_x, 1) - pad, np.percentile(all_x, 99) + pad)
        ylim = (np.percentile(all_y, 1) - pad, np.percentile(all_y, 99) + pad)

    for col, phase_name in enumerate(phase_order):
        pd_item = phase_data[phase_name]
        centroid = compute_pellet_centroid(pd_item['all_pellets'])

        plot_panel(axes[0, col], pd_item['normal_all'], pd_item['suspect_all'],
                   pd_item['all_pellets'], '%s (all)' % phase_name,
                   pellet_centroid=centroid)
        plot_panel(axes[1, col], pd_item['normal_contact'], pd_item['suspect_contact'],
                   pd_item['contact_pellets'], '%s (D+R)' % phase_name,
                   pellet_centroid=centroid)

        if xlim:
            for row in range(2):
                axes[row, col].set_xlim(xlim)
                axes[row, col].set_ylim(ylim)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLOR_NORMAL, lw=2.5, label='Normal (n=%d)' % len(normal_animals)),
    ]
    if group_suspects:
        legend_elements.append(
            Line2D([0], [0], color=COLOR_SUSPECT, lw=2.5,
                   label='Suspect (n=%d: %s)' % (len(group_suspects), ', '.join(group_suspects))))
    legend_elements.append(
        Line2D([0], [0], color='#00CCCC', lw=2.5, label='Pellet position heatmap'))
    fig.legend(handles=legend_elements, loc='upper right', fontsize=11,
               facecolor='#222222', edgecolor='white', labelcolor='white')

    suspect_str = ', '.join(group_suspects) if group_suspects else 'none'
    fig.suptitle('%s: All Mice Reach Trajectories\n'
                 'Magenta = normal, Yellow = suspect (%s), Cyan = pellet\n'
                 'Row 1: all reaches. Row 2: displaced+retrieved. Origin = nose.'
                 % (GROUP_LABELS.get(group, group), suspect_str),
                 fontsize=14, fontweight='bold', color='white')
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    suffix = '_quick' if quick else ''
    out = os.path.join(OUTPUT_BASE, '%s_all_mice_spaghetti%s.png' % (group, suffix))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='black')
    plt.close()
    print('  Saved: %s' % out)


def main():
    quick = '--quick' in sys.argv

    # Parse --group argument
    target_group = None
    for i, arg in enumerate(sys.argv):
        if arg == '--group' and i + 1 < len(sys.argv):
            target_group = sys.argv[i + 1].upper()

    # Load suspect list
    suspects = set()
    if os.path.exists(SUSPECT_CSV):
        sdf = pd.read_csv(SUSPECT_CSV)
        suspects = set(sdf['animal'].values)
        print('Loaded %d suspects from %s' % (len(suspects), SUSPECT_CSV))
    else:
        print('WARNING: No suspect list found at %s' % SUSPECT_CSV)

    groups = [target_group] if target_group else ['D', 'G', 'H', 'K', 'L', 'M']

    for group in groups:
        if group not in DLC_PATHS:
            print('Unknown group: %s, skipping' % group)
            continue
        run_group(group, suspects, quick=quick)

    print('\nDone.')


if __name__ == '__main__':
    main()
