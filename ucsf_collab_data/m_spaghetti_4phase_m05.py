"""
M05 spaghetti: 2 rows x 4 timepoints.
Row 1: all reaches. Row 2: displaced+retrieved only.
Top 5% by area shown in white (subtle).
Pellet heatmap in cyan.
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
OUTPUT_BASE = os.path.join(_connectome_root, 'Databases', 'figures', 'bodypart_tracking_quality', 'labeled_frames')


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


def extract_trajectories_and_pellet(swipes_df, dlc_cache):
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
        pel_x = dlc['Pellet']['x'].iloc[s:e+1].values
        pel_y = dlc['Pellet']['y'].iloc[s:e+1].values
        pel_lk = dlc['Pellet']['likelihood'].iloc[s:e+1].values
        n0x = nose_x[0]
        n0y = nose_y[0]
        cx = rh_x - n0x
        cy = rh_y - n0y
        if np.mean(rh_lk) > 0.3 and len(cx) >= 3:
            trajectories.append((cx, cy))
        good_pel = pel_lk > 0.5
        if good_pel.any():
            px = pel_x[good_pel] - n0x
            py = pel_y[good_pel] - n0y
            pellet_positions.extend(zip(px, py))
    return trajectories, pellet_positions


def compute_mean_traj(trajectories, n_pts=30):
    norm_x = []
    norm_y = []
    for x, y in trajectories:
        t_orig = np.linspace(0, 1, len(x))
        t_new = np.linspace(0, 1, n_pts)
        norm_x.append(np.interp(t_new, t_orig, x))
        norm_y.append(np.interp(t_new, t_orig, y))
    if norm_x:
        mean_x = np.mean(norm_x, axis=0)
        mean_y = np.mean(norm_y, axis=0)
        std_x = np.std(norm_x, axis=0)
        std_y = np.std(norm_y, axis=0)
        return mean_x, mean_y, std_x, std_y, norm_x, norm_y
    return None, None, None, None, None, None


def traj_area(traj):
    x, y = traj
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def plot_panel(ax, trajectories, pellet_positions, color, color_light, title, show_top5=True):
    n = len(trajectories)

    # Pellet heatmap
    if pellet_positions:
        px = np.array([p[0] for p in pellet_positions])
        py = np.array([p[1] for p in pellet_positions])
        bins = 50
        h, xedges, yedges = np.histogram2d(px, py, bins=bins)
        h = h.T
        rgba = np.zeros((*h.shape, 4))
        if h.max() > 0:
            norm_h = h / h.max()
            rgba[:, :, 1] = norm_h
            rgba[:, :, 2] = norm_h
            rgba[:, :, 3] = norm_h * 0.8
        extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
        ax.imshow(rgba, extent=extent, aspect='auto', interpolation='bilinear', zorder=1)

    # Individual traces — #1 outlier bold, #2-5 fade rapidly
    N_FADE = 5
    if n > 0:
        all_areas_list = [traj_area(t) for t in trajectories]
        ranked = np.argsort(all_areas_list)[::-1]  # highest first
        rank_map = {idx: rank for rank, idx in enumerate(ranked)}

        base_alpha = max(0.08, min(0.25, 30.0 / max(n, 1)))
        for i, (x, y) in enumerate(trajectories):
            rank = rank_map[i]
            if rank == 0:
                # #1 outlier: max prominence
                ax.plot(x, y, color=color, linewidth=2.5, alpha=0.7, zorder=9)
            elif rank < N_FADE:
                # #2-5: rapid exponential falloff but higher floor
                fade = np.exp(-rank * 0.8)  # e^-0.8=0.45, e^-1.6=0.20, e^-2.4=0.09, e^-3.2=0.04
                lw = 0.6 + fade * 2.0
                a = base_alpha + fade * 0.5
                ax.plot(x, y, color=color, linewidth=lw, alpha=a, zorder=8)
            else:
                ax.plot(x, y, color=color_light, alpha=base_alpha, linewidth=0.6, zorder=2)

    # Mean trajectory with perpendicular SEM band
    result = compute_mean_traj(trajectories)
    if result[0] is not None:
        mean_x, mean_y, std_x, std_y, norm_trajs_x, norm_trajs_y = result
        n_pts = len(mean_x)
        n_trajs = len(trajectories)

        dx = np.gradient(mean_x)
        dy = np.gradient(mean_y)
        perp_x = -dy
        perp_y = dx
        mag = np.sqrt(perp_x**2 + perp_y**2)
        mag[mag == 0] = 1
        perp_x = perp_x / mag
        perp_y = perp_y / mag

        perp_dists = np.zeros((n_trajs, n_pts))
        for i in range(n_trajs):
            diff_x = norm_trajs_x[i] - mean_x
            diff_y = norm_trajs_y[i] - mean_y
            perp_dists[i, :] = diff_x * perp_x + diff_y * perp_y

        SIGMA = 0.5
        band_width = np.std(perp_dists, axis=0) * SIGMA

        upper_x = mean_x + perp_x * band_width
        upper_y = mean_y + perp_y * band_width
        lower_x = mean_x - perp_x * band_width
        lower_y = mean_y - perp_y * band_width

        poly_x = np.concatenate([upper_x, lower_x[::-1]])
        poly_y = np.concatenate([upper_y, lower_y[::-1]])
        ax.fill(poly_x, poly_y, color=color, alpha=0.15, zorder=8)

        ax.plot(mean_x, mean_y, color='white', linewidth=4.5, alpha=0.5, zorder=10)  # white glow
        ax.plot(mean_x, mean_y, color=color, linewidth=3, alpha=1.0, zorder=11)

    # (outlier scaling is now integrated into individual traces above)

    ax.plot(0, 0, 'o', color='white', markersize=6, markeredgecolor='black',
            markeredgewidth=1, zorder=11)

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title('%s (n=%d)' % (title, n), fontsize=10, fontweight='bold', color='white')
    ax.set_xlabel('Lateral from nose (px)', fontsize=8)
    ax.set_ylabel('Extension from nose (px)', fontsize=8)
    ax.set_facecolor('black')
    ax.tick_params(colors='white', labelsize=7)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')


def main():
    print('Loading UCSF data...')
    df1 = pd.read_csv(os.path.join(UCSF_BASE, 'Swipe_Contusion_Data.csv'), low_memory=False)
    df2 = pd.read_csv(os.path.join(UCSF_BASE, 'Swipe_Contusion_Data_2.csv'), low_memory=False)
    df = pd.concat([df1, df2], ignore_index=True)
    df['sd_start'], df['sd_end'] = zip(*df['Swipe_Duration'].apply(parse_frame_range))

    m05 = df[df['SubjectID'] == 'M05'].copy()

    # Define 4 phase groups
    phases = {
        'Pre-Injury': ['2_Pre-injury_1', '2_Pre-injury_2', '2_Pre-injury_3'],
        '1wk Post': ['3_1wk_Post-injury'],
        '2-4wk Post': ['3_2wk_Post-injury', '3_3wk_Post-injury', '3_4wk_Post-injury'],
        'Post-Rehab': ['5_Post-rehab_Test_1', '5_Post-rehab_Test_2'],
    }

    phase_colors = {
        'Pre-Injury': ('#FF00FF', '#993399'),
        '1wk Post': ('#FFFF00', '#999933'),
        '2-4wk Post': ('#FF8800', '#995500'),
        'Post-Rehab': ('#00FF88', '#009955'),
    }

    dlc_cache = {}

    # Extract trajectories for each phase
    phase_data = {}
    for phase_name, phase_labels in phases.items():
        phase_df = m05[m05['Test_Type_Grouped_1'].isin(phase_labels)]
        contact_df = phase_df[phase_df['Reach_outcome'].isin(['pellet displaced', 'swipe successful'])]

        print('Extracting %s...' % phase_name)
        all_trajs, all_pellets = extract_trajectories_and_pellet(phase_df, dlc_cache)
        contact_trajs, contact_pellets = extract_trajectories_and_pellet(contact_df, dlc_cache)
        print('  all: %d, contact: %d' % (len(all_trajs), len(contact_trajs)))

        phase_data[phase_name] = {
            'all_trajs': all_trajs, 'all_pellets': all_pellets,
            'contact_trajs': contact_trajs, 'contact_pellets': contact_pellets,
        }

    fig, axes = plt.subplots(2, 4, figsize=(32, 16))
    fig.patch.set_facecolor('black')

    phase_order = ['Pre-Injury', '1wk Post', '2-4wk Post', 'Post-Rehab']

    # Collect all trajectories for axis limits
    all_trajs_combined = []
    for pn in phase_order:
        all_trajs_combined.extend(phase_data[pn]['all_trajs'])

    if all_trajs_combined:
        all_x = np.concatenate([t[0] for t in all_trajs_combined])
        all_y = np.concatenate([t[1] for t in all_trajs_combined])
        pad = 10
        xlim = (np.percentile(all_x, 1) - pad, np.percentile(all_x, 99) + pad)
        ylim = (np.percentile(all_y, 1) - pad, np.percentile(all_y, 99) + pad)

    for col, phase_name in enumerate(phase_order):
        color, color_light = phase_colors[phase_name]
        pd_item = phase_data[phase_name]

        # Row 1: all reaches
        plot_panel(axes[0, col], pd_item['all_trajs'], pd_item['all_pellets'],
                   color, color_light, '%s (all)' % phase_name)

        # Row 2: contact only
        plot_panel(axes[1, col], pd_item['contact_trajs'], pd_item['contact_pellets'],
                   color, color_light, '%s (D+R)' % phase_name)

        # Set same limits
        if all_trajs_combined:
            axes[0, col].set_xlim(xlim)
            axes[0, col].set_ylim(ylim)
            axes[1, col].set_xlim(xlim)
            axes[1, col].set_ylim(ylim)

    fig.suptitle('M05: Reach Trajectories Across Recovery\n'
                 'Row 1: all reaches. Row 2: displaced+retrieved. White = top 5%% area.\n'
                 'Cyan = pellet. Origin = nose.',
                 fontsize=14, fontweight='bold', color='white')
    plt.tight_layout()
    out = os.path.join(OUTPUT_BASE, 'M05_spaghetti_4phase.png')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='black')
    plt.close()
    print('Saved: %s' % out)


if __name__ == '__main__':
    main()
