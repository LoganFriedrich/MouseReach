"""
All-M spaghetti: all mice on one plot, suspects visually distinct.
4 phases x 2 rows (all reaches / D+R).
Suspects (M05, M06, M13, M14) in warm amber, normals in cool blue.
Each subpopulation gets its own mean trajectory.

Usage:
    python m_spaghetti_all_mice.py              # Full run, all mice, all phases
    python m_spaghetti_all_mice.py --quick      # Quick sample: 1 suspect + 1 normal, Pre only
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

SUSPECTS = ['M05', 'M06', 'M13', 'M14']

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
OUTPUT_BASE = os.path.join(_connectome_root, 'Databases', 'figures', 'bodypart_tracking_quality', 'labeled_frames')

PHASES = {
    'Pre-Injury': ['2_Pre-injury_1', '2_Pre-injury_2', '2_Pre-injury_3'],
    '1wk Post': ['3_1wk_Post-injury'],
    '2-4wk Post': ['3_2wk_Post-injury', '3_3wk_Post-injury', '3_4wk_Post-injury'],
    'Post-Rehab': ['5_Post-rehab_Test_1', '5_Post-rehab_Test_2'],
}

# CYM scheme on black: Cyan=pellet, Magenta=normal, Yellow=suspect
COLOR_NORMAL = '#FF00FF'       # Magenta
COLOR_NORMAL_LIGHT = '#993399'
COLOR_SUSPECT = '#FFFF00'      # Yellow
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
    parts = sid.split('-')
    date_str = (datetime(1899, 12, 30) + timedelta(days=int(parts[1]))).strftime('%Y%m%d')
    return '%s_%s_%s' % (date_str, parts[0], parts[2])


def extract_trajectories(swipes_df, dlc_cache):
    """Extract nose-centered RH trajectories + pellet positions.
    Returns (trajectories, pellet_positions) where trajectories is list of (cx, cy)
    and pellet_positions is list of (px, py) tuples."""
    trajectories = []
    pellet_positions = []
    for _, row in swipes_df.iterrows():
        s = row['sd_start']
        e = row['sd_end']
        if pd.isna(s):
            continue
        s, e = int(s), int(e)
        stem = session_to_stem(row['Session_ID'])
        if stem not in dlc_cache:
            h5 = glob.glob(os.path.join(ASPA_BASE, stem + 'DLC*.h5'))
            if not h5:
                dlc_cache[stem] = None
                continue
            d = pd.read_hdf(h5[0])
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
        # Pellet positions (nose-centered)
        pel_x = dlc['Pellet']['x'].iloc[s:e+1].values
        pel_y = dlc['Pellet']['y'].iloc[s:e+1].values
        pel_lk = dlc['Pellet']['likelihood'].iloc[s:e+1].values
        good = pel_lk > 0.5
        if good.any():
            pellet_positions.extend(zip(pel_x[good] - n0x, pel_y[good] - n0y))
    return trajectories, pellet_positions


def compute_mean_traj(trajectories, n_pts=30):
    """Time-normalize and compute mean + std."""
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
    std_x = np.std(norm_x, axis=0)
    std_y = np.std(norm_y, axis=0)
    return mean_x, mean_y, std_x, std_y, norm_x, norm_y


def compute_pellet_centroid(pellet_positions):
    """Compute median pellet position. Returns (x, y) or None."""
    if not pellet_positions or len(pellet_positions) < 10:
        return None
    px = np.array([p[0] for p in pellet_positions])
    py = np.array([p[1] for p in pellet_positions])
    return np.median(px), np.median(py)


def draw_pellet_heatmap(ax, pellet_positions):
    """Draw cyan pellet position heatmap."""
    if not pellet_positions:
        return
    px = np.array([p[0] for p in pellet_positions])
    py = np.array([p[1] for p in pellet_positions])
    h, xedges, yedges = np.histogram2d(px, py, bins=50)
    h = h.T
    rgba = np.zeros((*h.shape, 4))
    if h.max() > 0:
        norm_h = h / h.max()
        rgba[:, :, 1] = norm_h  # green channel
        rgba[:, :, 2] = norm_h  # blue channel
        rgba[:, :, 3] = norm_h * 0.8
    extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
    ax.imshow(rgba, extent=extent, aspect='auto', interpolation='bilinear', zorder=1)


def draw_pellet_centroid(ax, centroid):
    """Draw pellet centroid as a prominent cyan marker on top of everything."""
    if centroid is None:
        return
    ax.plot(centroid[0], centroid[1], '*', color='#00FFFF', markersize=14,
            markeredgecolor='white', markeredgewidth=0.8, zorder=20)


def draw_trajectories(ax, trajectories, color, color_light, alpha_base=None):
    """Draw individual reach traces."""
    n = len(trajectories)
    if n == 0:
        return
    alpha = alpha_base if alpha_base else max(0.04, min(0.20, 30.0 / max(n, 1)))
    lw = max(0.3, min(0.5, 25.0 / max(n, 1)))
    for x, y in trajectories:
        ax.plot(x, y, color=color_light, alpha=alpha, linewidth=lw, zorder=2)


def draw_mean_with_band(ax, trajectories, color, sigma=0.5, lw=3, zorder=10):
    """Draw mean trajectory with perpendicular SEM band."""
    result = compute_mean_traj(trajectories)
    if result is None:
        return
    mean_x, mean_y, std_x, std_y, norm_x, norm_y = result
    n_pts = len(mean_x)
    n_trajs = len(trajectories)

    # Perpendicular band
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

    # IQR band — shows where the middle 50% of reaches fall
    q25 = np.percentile(perp_dists, 25, axis=0)
    q75 = np.percentile(perp_dists, 75, axis=0)
    upper_x = mean_x + perp_x * q75
    upper_y = mean_y + perp_y * q75
    lower_x = mean_x + perp_x * q25
    lower_y = mean_y + perp_y * q25

    poly_x = np.concatenate([upper_x, lower_x[::-1]])
    poly_y = np.concatenate([upper_y, lower_y[::-1]])
    ax.fill(poly_x, poly_y, color=color, alpha=0.15, zorder=zorder - 1)

    # Mean line with glow
    ax.plot(mean_x, mean_y, color='white', linewidth=lw + 1.5, alpha=0.4, zorder=zorder)
    ax.plot(mean_x, mean_y, color=color, linewidth=lw, alpha=1.0, zorder=zorder + 1)


def plot_panel(ax, normal_trajs, suspect_trajs, pellet_positions, title, pellet_centroid=None):
    """Plot one panel with both populations overlaid."""
    n_total = len(normal_trajs) + len(suspect_trajs)

    # Pellet heatmap first (background)
    draw_pellet_heatmap(ax, pellet_positions)

    # Draw individual traces: normals first (behind), suspects on top
    draw_trajectories(ax, normal_trajs, COLOR_NORMAL, COLOR_NORMAL_LIGHT)
    draw_trajectories(ax, suspect_trajs, COLOR_SUSPECT, COLOR_SUSPECT_LIGHT)

    # Mean trajectories for each subpopulation
    if normal_trajs:
        draw_mean_with_band(ax, normal_trajs, COLOR_NORMAL, sigma=0.5, lw=2.5, zorder=10)
    if suspect_trajs:
        draw_mean_with_band(ax, suspect_trajs, COLOR_SUSPECT, sigma=0.5, lw=2.5, zorder=12)

    # Nose origin
    ax.plot(0, 0, 'o', color='white', markersize=5, markeredgecolor='black',
            markeredgewidth=1, zorder=15)

    # Pellet centroid (on top of everything)
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


def main():
    quick = '--quick' in sys.argv

    print('Loading UCSF data...')
    df1 = pd.read_csv(os.path.join(UCSF_BASE, 'Swipe_Contusion_Data.csv'), low_memory=False)
    df2 = pd.read_csv(os.path.join(UCSF_BASE, 'Swipe_Contusion_Data_2.csv'), low_memory=False)
    df = pd.concat([df1, df2], ignore_index=True)
    df = df[df['SubjectID'].str.startswith('M', na=False)].copy()
    df['sd_start'], df['sd_end'] = zip(*df['Swipe_Duration'].apply(parse_frame_range))

    if quick:
        # Quick mode: just M01 (normal) and M05 (suspect), Pre-Injury only
        df = df[df['SubjectID'].isin(['M01', 'M05'])]
        phases_to_run = {'Pre-Injury': PHASES['Pre-Injury']}
        print('QUICK MODE: M01 + M05, Pre-Injury only')
    else:
        phases_to_run = PHASES

    animals = sorted(df['SubjectID'].unique())
    normal_animals = [a for a in animals if a not in SUSPECTS]
    suspect_animals = [a for a in animals if a in SUSPECTS]
    print('Normal: %s' % normal_animals)
    print('Suspect: %s' % suspect_animals)

    dlc_cache = {}

    # Extract trajectories per phase, split by normal/suspect
    phase_data = {}
    for phase_name, phase_labels in phases_to_run.items():
        phase_df = df[df['Test_Type_Grouped_1'].isin(phase_labels)]
        contact_df = phase_df[phase_df['Reach_outcome'].isin(['pellet displaced', 'swipe successful'])]

        normal_df = phase_df[phase_df['SubjectID'].isin(normal_animals)]
        suspect_df = phase_df[phase_df['SubjectID'].isin(suspect_animals)]
        normal_contact = contact_df[contact_df['SubjectID'].isin(normal_animals)]
        suspect_contact = contact_df[contact_df['SubjectID'].isin(suspect_animals)]

        print('Extracting %s...' % phase_name)
        n_all, n_all_pel = extract_trajectories(normal_df, dlc_cache)
        s_all, s_all_pel = extract_trajectories(suspect_df, dlc_cache)
        n_contact, n_contact_pel = extract_trajectories(normal_contact, dlc_cache)
        s_contact, s_contact_pel = extract_trajectories(suspect_contact, dlc_cache)
        print('  normal: %d all, %d contact' % (len(n_all), len(n_contact)))
        print('  suspect: %d all, %d contact' % (len(s_all), len(s_contact)))

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

    # Collect all trajectories for consistent axis limits
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

        # Compute pellet centroid from "all reaches" — use for both rows
        # (D+R has biased pellet positions since successful reaches cluster)
        centroid = compute_pellet_centroid(pd_item['all_pellets'])

        # Row 1: all reaches
        plot_panel(axes[0, col], pd_item['normal_all'], pd_item['suspect_all'],
                   pd_item['all_pellets'], '%s (all)' % phase_name,
                   pellet_centroid=centroid)

        # Row 2: contact only — same centroid from all reaches
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
        Line2D([0], [0], color=COLOR_SUSPECT, lw=2.5, label='Suspect (n=%d: %s)' % (len(suspect_animals), ', '.join(suspect_animals))),
        Line2D([0], [0], color='#00CCCC', lw=2.5, label='Pellet position heatmap'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=11,
               facecolor='#222222', edgecolor='white', labelcolor='white')

    suffix = '_quick' if quick else ''
    fig.suptitle('Group M: All Mice Reach Trajectories\n'
                 'Magenta = normal, Yellow = suspect (%s), Cyan = pellet\n'
                 'Row 1: all reaches. Row 2: displaced+retrieved. Origin = nose.' % ', '.join(suspect_animals),
                 fontsize=14, fontweight='bold', color='white')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(OUTPUT_BASE, 'M_all_mice_spaghetti%s.png' % suffix)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='black')
    plt.close()
    print('Saved: %s' % out)


if __name__ == '__main__':
    main()
