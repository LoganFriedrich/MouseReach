"""
Generate behavior bar plots matching fig5_behavior_layout4.png style
for all historical (non-CNT, non-ENCR) injury groups.

Creates per-group figures:
  - 2x2 bar panels (asterisks / p-values x Retrieved / Contacted)
  - Per-animal recovery panel (paired lines: baseline vs end-of-rehab)

Data is read directly from each group's "1 - ENTER DATA HERE" sheet,
computing Eaten% and Contacted% from raw pellet scores (0/1/2).

Stats: Dunnett's test (post-injury windows as controls vs pre-injury and rehab).
Last 2: Uses last 2 Pillar tray days in continuous rehab (before any >5-day gap).
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import os
import re
from collections import defaultdict
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    'bar': '#7f7f7f',
    'point': '#1f1f1f',
    'sig_bracket': '#333333',
}

GROUP_FILES = {
    'ABS3': ('ABS3.xlsx', 'Early Study'),
    'G': ('G - Transection.xlsx', 'Transection'),
    'H': ('H - Transection.xlsx', 'Transection'),
    'K': ('K - Contusion 70kd.xlsx', 'Contusion 70kd'),
    'L': ('L - Contusion - 50kd.xlsx', 'Contusion 50kd'),
    'M': ('M - Contusion - 60kd.xlsx', 'Contusion 60kd'),
    'OptD': ('OptD - Rehab 1 - pyramidotomy.xlsx', 'Pyramidotomy'),
}

TRAY_OFFSETS_4 = [(7, 27), (31, 51), (55, 75), (79, 99)]
TRAY_OFFSETS_2 = [(7, 27), (31, 51)]

WINDOW_LABELS = ['Last 3', 'Post\nInjury 1', 'Post\nInjury 2-4', 'Rehab\nPillar']
WINDOW_KEYS = ['final_3', 'immediate_post', '2_4_post', 'last_2']

# Learner criterion: exclude animals with Final 3 avg eaten% <= this threshold
LEARNER_EATEN_THRESHOLD = 5.0

# Bracket drawing order (shortest span first, for clean stacking)
BRACKET_ORDER = [
    (0, 1),  # Final 3 vs Post Injury 1 (Dunnett test 1, span=1)
    (2, 3),  # Post Injury 2-4 vs Rehab Pillar (Dunnett test 2, span=1)
    (0, 2),  # Final 3 vs Post Injury 2-4 (Dunnett test 2, span=2)
    (1, 3),  # Post Injury 1 vs Rehab Pillar (Dunnett test 1, span=2)
]


def get_significance_stars(p_value):
    if p_value < 0.0001:
        return '****'
    elif p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'


def add_stat_bracket(ax, x1, x2, y, h, p_val, show_pval=False):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=COLORS['sig_bracket'])
    if show_pval:
        text = '<0.0001' if p_val < 0.0001 else f'{p_val:.4f}'
    else:
        text = get_significance_stars(p_val)
    ax.text((x1+x2)/2, y+h, text, ha='center', va='bottom', fontsize=10, fontweight='bold')


def run_dunnett_tests(data_by_phase):
    """Run Dunnett's tests with post-injury windows as controls.

    Test 1: Control = Immediate Post (idx 1), vs Final 3 (idx 0) and Rehab Pillar (idx 3)
    Test 2: Control = 2-4 Post (idx 2), vs Final 3 (idx 0) and Rehab Pillar (idx 3)

    Returns dict: {(bar_i, bar_j): p_value}
    """
    results = {}
    final3 = data_by_phase[0]
    imm_post = data_by_phase[1]
    post_2_4 = data_by_phase[2]
    rehab = data_by_phase[3]

    # Test 1: control = immediate_post
    if len(imm_post) > 1 and len(final3) > 1 and len(rehab) > 1:
        res = stats.dunnett(final3, rehab, control=imm_post)
        results[(0, 1)] = res.pvalue[0]  # Final 3 vs Immediate Post
        results[(1, 3)] = res.pvalue[1]  # Rehab Pillar vs Immediate Post

    # Test 2: control = 2_4_post
    if len(post_2_4) > 1 and len(final3) > 1 and len(rehab) > 1:
        res2 = stats.dunnett(final3, rehab, control=post_2_4)
        results[(0, 2)] = res2.pvalue[0]  # Final 3 vs 2-4 Post
        results[(2, 3)] = res2.pvalue[1]  # Rehab Pillar vs 2-4 Post

    return results


def is_valid_tray(pellet_values):
    for v in pellet_values:
        if v is not None and v != 'N/A' and isinstance(v, (int, float)):
            return True
    return False


def score_tray(pellet_values):
    valid = [v for v in pellet_values if v is not None and v != 'N/A' and isinstance(v, (int, float))]
    if not valid:
        return None, None
    n = len(valid)
    eaten = sum(1 for v in valid if v >= 2) / n * 100
    contacted = sum(1 for v in valid if v >= 1) / n * 100
    return eaten, contacted


def read_group_data(filepath):
    """Read raw pellet data and return per-animal per-test-type metrics + test metadata."""
    wb = openpyxl.load_workbook(filepath, data_only=True, read_only=True)
    ws = wb['1 - ENTER DATA HERE']

    headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    tray_offsets = TRAY_OFFSETS_4 if len(headers) > 80 else TRAY_OFFSETS_2

    data = defaultdict(dict)
    all_test_types = set()
    test_meta = {}  # tt -> {date, tray}

    for row in ws.iter_rows(min_row=2, values_only=True):
        vals = list(row)
        if vals[0] is None:
            continue

        test_type = str(vals[1])
        animal = vals[5]
        all_test_types.add(test_type)

        if test_type not in test_meta:
            test_meta[test_type] = {'date': vals[0], 'tray': str(vals[2])}

        tray_eaten, tray_contacted = [], []
        for start, end in tray_offsets:
            if start >= len(vals):
                continue
            pellets = vals[start:end]
            if not is_valid_tray(pellets):
                continue
            e, c = score_tray(pellets)
            if e is not None:
                tray_eaten.append(e)
                tray_contacted.append(c)

        if tray_eaten:
            data[animal][test_type] = {
                'eaten': np.mean(tray_eaten),
                'contacted': np.mean(tray_contacted),
            }

    wb.close()
    return data, sorted(all_test_types), test_meta


def get_time_windows(test_types_sorted, test_meta):
    """Determine time windows, using dates and tray info for correct Last 2."""
    windows = {k: [] for k in WINDOW_KEYS}

    first_post_idx = None
    for i, tt in enumerate(test_types_sorted):
        if 'post' in tt.lower() and 'injury' in tt.lower():
            first_post_idx = i
            break

    if first_post_idx is None:
        return windows

    pre = test_types_sorted[:first_post_idx]
    windows['final_3'] = pre[-3:] if len(pre) >= 3 else pre

    for tt in test_types_sorted[first_post_idx:]:
        tt_lower = tt.lower()
        if '1 week post' in tt_lower:
            windows['immediate_post'].append(tt)
        else:
            m = re.search(r'(\d+)\s*week\s*post', tt_lower)
            if m and 2 <= int(m.group(1)) <= 4:
                windows['2_4_post'].append(tt)

    # Last 2: find last 2 Pillar days in continuous rehab (no gap > 5 days)
    rehab_tts = [tt for tt in test_types_sorted if 'rehab' in tt.lower()]

    if not rehab_tts:
        windows['last_2'] = test_types_sorted[-2:]
        return windows

    # Split rehab into continuous blocks (gap > 5 days = new block)
    blocks = [[rehab_tts[0]]]
    for i in range(1, len(rehab_tts)):
        prev_d = test_meta.get(rehab_tts[i-1], {}).get('date')
        curr_d = test_meta.get(rehab_tts[i], {}).get('date')
        if prev_d and curr_d:
            try:
                gap = (curr_d - prev_d).days
            except (TypeError, AttributeError):
                gap = 1
            if 5 < gap < 100:  # Real breaks (5-100 days); ignore date-entry errors
                blocks.append([rehab_tts[i]])
                continue
        blocks[-1].append(rehab_tts[i])

    # Use the first (main) rehab block - find last 2 Pillar days in it
    main_block = blocks[0]
    pillar_in_main = [tt for tt in main_block
                      if test_meta.get(tt, {}).get('tray', '').lower().startswith('p')]

    if len(pillar_in_main) >= 2:
        windows['last_2'] = pillar_in_main[-2:]
    elif len(pillar_in_main) == 1:
        windows['last_2'] = pillar_in_main
    else:
        windows['last_2'] = main_block[-2:] if len(main_block) >= 2 else main_block

    return windows


def get_animal_window_values(data, windows):
    """Compute per-animal averages for each time window.
    Only includes animals that:
      1. Are present in ALL non-empty windows (for paired tests)
      2. Meet the learner criterion (Final 3 avg eaten% > LEARNER_EATEN_THRESHOLD)
    Returns (result_dict, n_excluded) where n_excluded is the learner filter count.
    """
    # First compute Final 3 eaten% per animal to apply learner criterion
    learners = set()
    non_learners = set()
    final3_tts = windows.get('final_3', [])
    for animal, tests in data.items():
        if final3_tts:
            e_vals = [tests[tt]['eaten'] for tt in final3_tts if tt in tests]
            if e_vals and np.mean(e_vals) > LEARNER_EATEN_THRESHOLD:
                learners.add(animal)
            else:
                non_learners.add(animal)
        else:
            learners.add(animal)  # No training data = can't filter

    animals_per_window = {}
    for wkey in WINDOW_KEYS:
        tt_list = windows[wkey]
        if not tt_list:
            animals_per_window[wkey] = set()
            continue
        animals_with_data = set()
        for animal in learners:
            if animal in data and any(tt in data[animal] for tt in tt_list):
                animals_with_data.add(animal)
        animals_per_window[wkey] = animals_with_data

    non_empty_windows = [k for k in WINDOW_KEYS if windows[k]]
    if non_empty_windows:
        paired_animals = set.intersection(*(animals_per_window[k] for k in non_empty_windows))
    else:
        paired_animals = set()

    result = {}
    for wkey in WINDOW_KEYS:
        tt_list = windows[wkey]
        if not tt_list:
            result[wkey] = {'eaten': np.array([]), 'contacted': np.array([]), 'animals': []}
            continue

        eaten_vals, contacted_vals, animal_ids = [], [], []
        for animal in sorted(paired_animals):
            e_vals, c_vals = [], []
            for tt in tt_list:
                if tt in data[animal]:
                    e_vals.append(data[animal][tt]['eaten'])
                    c_vals.append(data[animal][tt]['contacted'])
            if e_vals:
                eaten_vals.append(np.mean(e_vals))
                contacted_vals.append(np.mean(c_vals))
                animal_ids.append(animal)

        result[wkey] = {
            'eaten': np.array(eaten_vals),
            'contacted': np.array(contacted_vals),
            'animals': animal_ids,
        }

    return result, len(non_learners)


def plot_group(group_name, injury_type, window_data, windows, output_dir):
    """Create 2x2 bar panel figure with Dunnett's test."""
    n_animals = len(window_data['final_3']['animals']) if window_data['final_3']['animals'] else 0

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        f'Behavior Performance: Manual Scoring\n({group_name} - {injury_type}, N={n_animals} learners, eaten >{LEARNER_EATEN_THRESHOLD}% at training)',
        fontsize=16, fontweight='bold'
    )

    for col, (metric_key, metric_name, ylabel) in enumerate([
        ('eaten', 'Retrieved', '% Pellets Retrieved'),
        ('contacted', 'Contacted', '% Pellets Contacted'),
    ]):
        data_by_phase = [window_data[wk][metric_key] for wk in WINDOW_KEYS]

        # Run Dunnett's tests (post-injury windows as controls)
        dunnett_pvals = run_dunnett_tests(data_by_phase)

        for row, show_pval in enumerate([False, True]):
            ax = axes[row, col]

            means = [np.mean(d) if len(d) > 0 else 0 for d in data_by_phase]
            sems = [stats.sem(d) if len(d) > 1 else 0 for d in data_by_phase]

            x = np.arange(len(WINDOW_KEYS))

            bars = ax.bar(x, means, yerr=sems, color=COLORS['bar'],
                         capsize=5, alpha=0.8, edgecolor='black', linewidth=1)

            np.random.seed(42)
            for i, d in enumerate(data_by_phase):
                if len(d) > 0:
                    jitter = np.random.uniform(-0.15, 0.15, len(d))
                    ax.scatter(x[i] + jitter, d, color=COLORS['point'],
                              s=40, zorder=5, alpha=0.6, edgecolor='white', linewidth=0.5)

            # Statistical brackets with Dunnett's p-values
            all_vals = [v for d in data_by_phase for v in d]
            data_ceil = max(all_vals) if all_vals else 1
            data_ceil = max(data_ceil, 5)

            bracket_h = 1.5
            bracket_step = 4.0
            first_bracket_y = data_ceil + 3

            bracket_idx = 0
            for (i, j) in BRACKET_ORDER:
                if (i, j) in dunnett_pvals:
                    y_pos = first_bracket_y + bracket_idx * bracket_step
                    add_stat_bracket(ax, x[i], x[j], y_pos, bracket_h,
                                    dunnett_pvals[(i, j)], show_pval=show_pval)
                    bracket_idx += 1

            if bracket_idx > 0:
                top_bracket = first_bracket_y + bracket_idx * bracket_step
                ax.set_ylim(0, top_bracket + bracket_step)
            else:
                ax.set_ylim(0, data_ceil * 1.2)

            ax.set_xticks(x)
            ax.set_xticklabels(WINDOW_LABELS, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=12)

            title_suffix = "(Dunnett's p-values)" if show_pval else "(asterisks, Dunnett's test)"
            ax.set_title(f'{metric_name} {title_suffix}', fontsize=11, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'behavior_{group_name}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_recovery(group_name, injury_type, window_data, output_dir):
    """Per-animal paired recovery plot: Final 3 vs Rehab Pillar for each subject."""
    pre_eaten = window_data['final_3']['eaten']
    post_eaten = window_data['last_2']['eaten']
    pre_contacted = window_data['final_3']['contacted']
    post_contacted = window_data['last_2']['contacted']
    animals = window_data['final_3']['animals']

    if len(pre_eaten) < 2:
        return

    n = len(animals)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(
        f'Per-Animal Recovery: {group_name} - {injury_type} (N={n})\n'
        f'Paired t-test | Pre-Injury (Final 3) vs Rehab Pillar (Last 2)',
        fontsize=13, fontweight='bold'
    )

    for col, (pre, post, metric_name, ylabel) in enumerate([
        (pre_eaten, post_eaten, 'Retrieved', '% Pellets Retrieved'),
        (pre_contacted, post_contacted, 'Contacted', '% Pellets Contacted'),
    ]):
        ax = axes[col]

        # Paired lines for each animal
        for i in range(n):
            # Color by recovery direction
            if post[i] > pre[i] * 0.8:
                color = '#2ca02c'  # green = recovered
                alpha = 0.7
            elif post[i] > 0:
                color = '#ff7f0e'  # orange = partial
                alpha = 0.6
            else:
                color = '#d62728'  # red = no recovery
                alpha = 0.5
            ax.plot([0, 1], [pre[i], post[i]], 'o-', color=color, alpha=alpha,
                    markersize=6, linewidth=1.5, zorder=3)

        # Mean line
        ax.plot([0, 1], [np.mean(pre), np.mean(post)], 's-', color='black',
                markersize=10, linewidth=3, zorder=5, label='Group Mean')

        # Stats
        t_stat, p_val = stats.ttest_rel(pre, post)
        diff = post - pre
        mean_diff = np.mean(diff)
        sem_diff = stats.sem(diff)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Pre-Injury\n(Final 3)', 'Rehab Pillar\n(Last 2)'], fontsize=11)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlim(-0.3, 1.3)

        # Count recovery categories
        recovered = np.sum(post > pre * 0.8)
        partial = np.sum((post > 0) & (post <= pre * 0.8))
        none = np.sum(post == 0)

        stats_text = (f'p = {p_val:.4f}\n'
                     f'Mean diff = {mean_diff:.1f} +/- {sem_diff:.1f}\n'
                     f'Recovered: {recovered}/{n}\n'
                     f'Partial: {partial}/{n}\n'
                     f'None: {none}/{n}')
        ax.text(0.5, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#2ca02c', marker='o', label='Recovered (>80% of pre)'),
            Line2D([0], [0], color='#ff7f0e', marker='o', label='Partial (>0%)'),
            Line2D([0], [0], color='#d62728', marker='o', label='No recovery (0%)'),
            Line2D([0], [0], color='black', marker='s', linewidth=3, label='Group Mean'),
        ]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'recovery_{group_name}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_combined_overview(all_group_data, output_dir):
    """Create a combined overview: all groups side by side per window."""
    group_names = list(all_group_data.keys())
    n_groups = len(group_names)

    fig, axes = plt.subplots(2, 2, figsize=(max(14, n_groups * 2.5), 11))
    fig.suptitle('Behavior Performance: All Historical Groups\nManual Scoring Comparison',
                 fontsize=16, fontweight='bold')

    group_colors = plt.cm.Set2(np.linspace(0, 1, n_groups))

    for row, (metric_key, metric_name, ylabel) in enumerate([
        ('eaten', 'Retrieved', '% Pellets Retrieved'),
        ('contacted', 'Contacted', '% Pellets Contacted'),
    ]):
        for col, (wk, wlabel) in enumerate([
            ('final_3', 'Final 3 (Pre-Injury)'),
            ('last_2', 'Rehab Pillar (Last 2)'),
        ]):
            ax = axes[row, col]
            x = np.arange(n_groups)

            means, sems = [], []
            for gname in group_names:
                d = all_group_data[gname]['window_data'][wk][metric_key]
                means.append(np.mean(d) if len(d) > 0 else 0)
                sems.append(stats.sem(d) if len(d) > 1 else 0)

            bars = ax.bar(x, means, yerr=sems, color=[group_colors[i] for i in range(n_groups)],
                         capsize=4, alpha=0.85, edgecolor='black', linewidth=0.8)

            np.random.seed(42)
            for i, gname in enumerate(group_names):
                d = all_group_data[gname]['window_data'][wk][metric_key]
                if len(d) > 0:
                    jitter = np.random.uniform(-0.2, 0.2, len(d))
                    ax.scatter(x[i] + jitter, d, color='black', s=20, zorder=5, alpha=0.5,
                              edgecolor='white', linewidth=0.3)

            labels = [f"{gn}\n({all_group_data[gn]['injury_type']})" for gn in group_names]
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8, ha='center')
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f'{metric_name} - {wlabel}', fontsize=12, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'behavior_all_groups_overview.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved overview: {out_path}")


def main():
    print("=" * 70)
    print("GENERATING HISTORICAL BEHAVIOR PLOTS")
    print("  Style: Matching fig5_behavior_layout4.png (CNT reference)")
    print("  Stats: Dunnett's test (post-injury controls vs pre/rehab)")
    print("  Last 2: Last 2 Pillar days in continuous rehab block")
    print(f"  Learner criterion: Final 3 avg eaten% > {LEARNER_EATEN_THRESHOLD}%")
    print("  Data: Raw pellet scores from '1 - ENTER DATA HERE' tabs")
    print("=" * 70)

    all_group_data = {}

    for group_name, (filename, injury_type) in GROUP_FILES.items():
        filepath = os.path.join(SCRIPT_DIR, filename)
        if not os.path.exists(filepath):
            print(f"\nWARNING: {filename} not found, skipping {group_name}")
            continue

        print(f"\n--- {group_name} ({injury_type}) ---")
        data, test_types, test_meta = read_group_data(filepath)
        windows = get_time_windows(test_types, test_meta)
        window_data, n_excluded = get_animal_window_values(data, windows)

        n_included = len(window_data['final_3']['animals'])
        print(f"  Total animals: {len(data)}, Learners: {len(data) - n_excluded}, Non-learners excluded: {n_excluded}")
        print(f"  Animals (paired + learner): {n_included}")
        print(f"  Windows:")
        for wk in WINDOW_KEYS:
            tts = windows[wk]
            n = len(window_data[wk]['eaten'])
            print(f"    {wk:20s}: n={n:2d} | {', '.join(tts)}")

        plot_group(group_name, injury_type, window_data, windows, OUTPUT_DIR)
        plot_recovery(group_name, injury_type, window_data, OUTPUT_DIR)

        all_group_data[group_name] = {
            'injury_type': injury_type,
            'window_data': window_data,
            'windows': windows,
        }

    print(f"\n--- Combined Overview ---")
    plot_combined_overview(all_group_data, OUTPUT_DIR)

    print(f"\nAll figures saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
