"""
Generate behavior bar plots matching fig5_behavior_layout4.png style
for all historical (non-CNT, non-ENCR) injury groups.

Creates per-group figures:
  - 2x2 bar panels (asterisks / p-values x Retrieved / Contacted)
  - Per-animal recovery panel (paired lines: baseline vs end-of-rehab)

Data is read directly from each group's "1 - ENTER DATA HERE" sheet,
computing Eaten% and Contacted% from raw pellet scores (0/1/2).

Stats: GEE (Generalized Estimating Equations) with binomial family,
exchangeable correlation structure, and animal as clustering variable.
Post-hoc: pairwise contrasts with Holm correction.
Fallback: chi-square / Fisher's exact when GEE can't converge (complete separation).
Last 2: Uses last 2 Pillar tray days in continuous rehab (before any >5-day gap).
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openpyxl
import os
import re
import warnings
from collections import defaultdict
from scipy import stats
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable

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
    (0, 1),  # Final 3 vs Post Injury 1 (span=1)
    (2, 3),  # Post Injury 2-4 vs Rehab Pillar (span=1)
    (0, 2),  # Final 3 vs Post Injury 2-4 (span=2)
    (1, 3),  # Post Injury 1 vs Rehab Pillar (span=2)
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


def is_valid_tray(pellet_values):
    for v in pellet_values:
        if v is not None and v != 'N/A' and isinstance(v, (int, float)):
            return True
    return False


def read_group_data(filepath):
    """Read raw pellet data from Excel.

    Returns:
        data: {animal: {test_type: {eaten: float%, contacted: float%}}} - animal-level averages
        sorted test types list
        test_meta: {test_type: {date, tray}}
        pellet_records: list of (animal, test_type, eaten_01, contacted_01) - one per pellet
    """
    wb = openpyxl.load_workbook(filepath, data_only=True, read_only=True)
    ws = wb['1 - ENTER DATA HERE']

    headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    tray_offsets = TRAY_OFFSETS_4 if len(headers) > 80 else TRAY_OFFSETS_2

    data = defaultdict(dict)
    pellet_records = []
    all_test_types = set()
    test_meta = {}

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

            valid_in_tray = []
            for v in pellets:
                if v is not None and v != 'N/A' and isinstance(v, (int, float)):
                    valid_in_tray.append(v)
                    pellet_records.append((
                        animal, test_type,
                        1 if v >= 2 else 0,   # eaten binary
                        1 if v >= 1 else 0,   # contacted binary
                    ))

            if valid_in_tray:
                n_p = len(valid_in_tray)
                tray_eaten.append(sum(1 for v in valid_in_tray if v >= 2) / n_p * 100)
                tray_contacted.append(sum(1 for v in valid_in_tray if v >= 1) / n_p * 100)

        if tray_eaten:
            data[animal][test_type] = {
                'eaten': np.mean(tray_eaten),
                'contacted': np.mean(tray_contacted),
            }

    wb.close()
    return data, sorted(all_test_types), test_meta, pellet_records


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


def get_learners(data, final3_tts):
    """Return set of animals meeting learner criterion (Final 3 avg eaten% > threshold)."""
    learners = set()
    for animal, tests in data.items():
        if final3_tts:
            e_vals = [tests[tt]['eaten'] for tt in final3_tts if tt in tests]
            if e_vals and np.mean(e_vals) > LEARNER_EATEN_THRESHOLD:
                learners.add(animal)
        else:
            learners.add(animal)
    return learners


def get_animal_window_values(data, windows, learners):
    """Compute per-animal averages for each time window (no pairing requirement).

    Each window independently includes all learners with data in that window.
    Returns dict: {window_key: {eaten: array, contacted: array, animals: list}}
    """
    result = {}
    for wkey in WINDOW_KEYS:
        tt_list = windows[wkey]
        if not tt_list:
            result[wkey] = {'eaten': np.array([]), 'contacted': np.array([]), 'animals': []}
            continue

        eaten_vals, contacted_vals, animal_ids = [], [], []
        for animal in sorted(learners):
            if animal not in data:
                continue
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

    return result


def build_pellet_dataframe(pellet_records, windows, learners):
    """Build pandas DataFrame from pellet records for GEE analysis.

    Each row is one pellet observation with columns:
        animal, window (int 0-3), eaten (0/1), contacted (0/1), animal_code (int)
    Filtered to learner animals and test types within defined windows.
    """
    tt_to_window = {}
    for wk_idx, wk in enumerate(WINDOW_KEYS):
        for tt in windows[wk]:
            tt_to_window[tt] = wk_idx

    rows = []
    for animal, test_type, eaten, contacted in pellet_records:
        if animal not in learners:
            continue
        if test_type not in tt_to_window:
            continue
        rows.append({
            'animal': animal,
            'window': tt_to_window[test_type],
            'eaten': eaten,
            'contacted': contacted,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    animal_codes = {a: i for i, a in enumerate(sorted(df['animal'].unique()))}
    df['animal_code'] = df['animal'].map(animal_codes)
    return df


def _holm_correct(pvals):
    """Apply Holm-Bonferroni correction to a dict of p-values."""
    if not pvals:
        return pvals
    sorted_pairs = sorted(pvals.keys(), key=lambda k: pvals[k])
    m = len(sorted_pairs)
    corrected = {}
    for rank, pair in enumerate(sorted_pairs):
        corrected[pair] = min(1.0, pvals[pair] * (m - rank))
    # Enforce monotonicity
    prev_p = 0
    for pair in sorted_pairs:
        corrected[pair] = max(corrected[pair], prev_p)
        prev_p = corrected[pair]
    return corrected


def _chi_square_fallback(df, metric, comparisons, n_per_window):
    """Fallback to chi-square/Fisher's exact when GEE can't converge."""
    raw_pvals = {}
    for (i, j) in comparisons:
        di = df[df['window'] == i]
        dj = df[df['window'] == j]
        if len(di) == 0 or len(dj) == 0:
            continue
        a = int(di[metric].sum())
        b = len(di) - a
        c = int(dj[metric].sum())
        d_val = len(dj) - c
        table = np.array([[a, b], [c, d_val]])
        if table.min() < 5:
            _, p = stats.fisher_exact(table)
        else:
            _, p, _, _ = stats.chi2_contingency(table, correction=True)
        raw_pvals[(i, j)] = p

    raw_pvals = _holm_correct(raw_pvals)

    # Omnibus: chi-square on full contingency table across available windows
    available = [w for w in range(4) if n_per_window.get(w, 0) > 0]
    if len(available) >= 2:
        table_rows = []
        for w in available:
            subset = df[df['window'] == w]
            table_rows.append([int(subset[metric].sum()), len(subset) - int(subset[metric].sum())])
        table = np.array(table_rows)
        if table.sum() > 0:
            _, omnibus_p, _, _ = stats.chi2_contingency(table)
        else:
            omnibus_p = 1.0
    else:
        omnibus_p = 1.0

    return raw_pvals, omnibus_p, n_per_window, 'chi-sq/Fisher'


def run_gee_posthoc(df, metric):
    """Run GEE (binomial, exchangeable, animal cluster) + pairwise contrasts with Holm.

    Returns (pval_dict, omnibus_p, n_pellets_per_window, method_label)
    """
    comparisons = [(0, 1), (0, 2), (1, 3), (2, 3)]

    n_per_window = {}
    if not df.empty:
        for w in range(4):
            n_per_window[w] = int((df['window'] == w).sum())
    else:
        n_per_window = {w: 0 for w in range(4)}

    if df.empty or df['window'].nunique() < 2:
        return {}, 1.0, n_per_window, 'N/A'

    available_windows = sorted([w for w in range(4) if n_per_window.get(w, 0) > 0])
    if len(available_windows) < 2:
        return {}, 1.0, n_per_window, 'N/A'

    df_gee = df[df['window'].isin(available_windows)].copy()

    if df_gee['animal_code'].nunique() < 2:
        return _chi_square_fallback(df, metric, comparisons, n_per_window)

    # Check for complete separation in any window
    for w in available_windows:
        subset = df_gee[df_gee['window'] == w][metric]
        if len(subset) > 0 and (subset.sum() == 0 or subset.sum() == len(subset)):
            return _chi_square_fallback(df, metric, comparisons, n_per_window)

    try:
        ref_window = available_windows[0]  # 0 (final_3) if present
        df_gee = df_gee.sort_values('animal_code').reset_index(drop=True)

        formula = f'{metric} ~ C(window, Treatment({ref_window}))'
        model = GEE.from_formula(
            formula,
            groups='animal_code',
            data=df_gee,
            family=Binomial(),
            cov_struct=Exchangeable(),
        )

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = model.fit(maxiter=100)

        # Omnibus Wald test for all window effects
        param_names = list(result.params.index)
        window_params = [p for p in param_names if 'C(window' in p]

        if len(window_params) >= 1:
            idx = [param_names.index(p) for p in window_params]
            beta_w = result.params.values[idx]
            vcov_w = result.cov_params().values[np.ix_(idx, idx)]
            try:
                wald_stat = float(beta_w @ np.linalg.inv(vcov_w) @ beta_w)
                omnibus_p = float(1 - stats.chi2.cdf(wald_stat, len(window_params)))
            except np.linalg.LinAlgError:
                omnibus_p = 1.0
        else:
            omnibus_p = 1.0

        # Pairwise contrasts
        params = result.params.values
        vcov = result.cov_params().values

        raw_pvals = {}
        for (i, j) in comparisons:
            if i not in available_windows or j not in available_windows:
                continue

            L = np.zeros(len(params))

            if i == ref_window:
                j_names = [p for p in param_names if f'T.{j}]' in p]
                if not j_names:
                    continue
                L[param_names.index(j_names[0])] = 1.0
            elif j == ref_window:
                i_names = [p for p in param_names if f'T.{i}]' in p]
                if not i_names:
                    continue
                L[param_names.index(i_names[0])] = -1.0
            else:
                i_names = [p for p in param_names if f'T.{i}]' in p]
                j_names = [p for p in param_names if f'T.{j}]' in p]
                if not i_names or not j_names:
                    continue
                L[param_names.index(j_names[0])] = 1.0
                L[param_names.index(i_names[0])] = -1.0

            est = float(L @ params)
            var_est = float(L @ vcov @ L)
            if var_est <= 0:
                continue
            z = est / np.sqrt(var_est)
            p = float(2 * (1 - stats.norm.cdf(abs(z))))
            raw_pvals[(i, j)] = p

        raw_pvals = _holm_correct(raw_pvals)

        return raw_pvals, omnibus_p, n_per_window, 'GEE'

    except Exception as e:
        print(f"    GEE failed ({e}), falling back to chi-square")
        return _chi_square_fallback(df, metric, comparisons, n_per_window)


def plot_group(group_name, injury_type, window_data, pellet_df, n_learners, output_dir):
    """Create 2x2 bar panel figure with GEE post-hoc tests.

    Bars/scatter = animal-level means and SEM (visual).
    Stats = GEE on pellet-level binary outcomes with animal clustering.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    fig.suptitle(
        f'Behavior Performance: Manual Scoring\n'
        f'({group_name} - {injury_type}, N={n_learners} learners, '
        f'eaten >{LEARNER_EATEN_THRESHOLD}% at training)',
        fontsize=16, fontweight='bold'
    )

    for col, (metric_key, metric_name, ylabel) in enumerate([
        ('eaten', 'Retrieved', '% Pellets Retrieved'),
        ('contacted', 'Contacted', '% Pellets Contacted'),
    ]):
        data_by_phase = [window_data[wk][metric_key] for wk in WINDOW_KEYS]

        # GEE on pellet-level data
        posthoc_pvals, omnibus_p, n_pellets, method = run_gee_posthoc(pellet_df, metric_key)

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

            # Statistical brackets
            all_vals = [v for d in data_by_phase for v in d]
            data_ceil = max(all_vals) if all_vals else 1
            data_ceil = max(data_ceil, 5)

            bracket_h = 1.5
            bracket_step = 4.0
            first_bracket_y = data_ceil + 3

            bracket_idx = 0
            for (i, j) in BRACKET_ORDER:
                if (i, j) in posthoc_pvals:
                    y_pos = first_bracket_y + bracket_idx * bracket_step
                    add_stat_bracket(ax, x[i], x[j], y_pos, bracket_h,
                                    posthoc_pvals[(i, j)], show_pval=show_pval)
                    bracket_idx += 1

            # N labels on x-axis (pellet counts + animal counts)
            xlabels = []
            for i, wl in enumerate(WINDOW_LABELS):
                n_p = n_pellets.get(i, 0)
                n_a = len(data_by_phase[i])
                xlabels.append(f'{wl}\n({n_a} mice, {n_p} pel.)')

            if bracket_idx > 0:
                top_bracket = first_bracket_y + bracket_idx * bracket_step
                ax.set_ylim(0, top_bracket + bracket_step)
            else:
                ax.set_ylim(0, data_ceil * 1.2)

            ax.set_xticks(x)
            ax.set_xticklabels(xlabels, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=12)

            omnibus_str = f'p={omnibus_p:.4f}' if omnibus_p >= 0.0001 else 'p<0.0001'
            if show_pval:
                title_suffix = f"({method} p-values + Holm | omnibus {omnibus_str})"
            else:
                title_suffix = f"(asterisks, {method} + Holm | omnibus {omnibus_str})"
            ax.set_title(f'{metric_name} {title_suffix}', fontsize=10, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'behavior_{group_name}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_recovery(group_name, injury_type, window_data, output_dir):
    """Per-animal paired recovery plot: Final 3 vs Rehab Pillar for each subject."""
    # Find animals present in BOTH final_3 and last_2 (pairing for this plot only)
    f3_animals = window_data['final_3']['animals']
    l2_animals = window_data['last_2']['animals']
    f3_set = set(f3_animals)
    l2_set = set(l2_animals)
    paired_animals = sorted(f3_set & l2_set)

    if len(paired_animals) < 2:
        return

    # Build paired arrays
    f3_idx = {a: i for i, a in enumerate(f3_animals)}
    l2_idx = {a: i for i, a in enumerate(l2_animals)}

    pre_eaten = np.array([window_data['final_3']['eaten'][f3_idx[a]] for a in paired_animals])
    post_eaten = np.array([window_data['last_2']['eaten'][l2_idx[a]] for a in paired_animals])
    pre_contacted = np.array([window_data['final_3']['contacted'][f3_idx[a]] for a in paired_animals])
    post_contacted = np.array([window_data['last_2']['contacted'][l2_idx[a]] for a in paired_animals])

    n = len(paired_animals)

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

        for i in range(n):
            if post[i] > pre[i] * 0.8:
                color = '#2ca02c'   # green = recovered
                alpha = 0.7
            elif post[i] > 0:
                color = '#ff7f0e'   # orange = partial
                alpha = 0.6
            else:
                color = '#d62728'   # red = no recovery
                alpha = 0.5
            ax.plot([0, 1], [pre[i], post[i]], 'o-', color=color, alpha=alpha,
                    markersize=6, linewidth=1.5, zorder=3)

        ax.plot([0, 1], [np.mean(pre), np.mean(post)], 's-', color='black',
                markersize=10, linewidth=3, zorder=5, label='Group Mean')

        t_stat, p_val = stats.ttest_rel(pre, post)
        diff = post - pre
        mean_diff = np.mean(diff)
        sem_diff = stats.sem(diff)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Pre-Injury\n(Final 3)', 'Rehab Pillar\n(Last 2)'], fontsize=11)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlim(-0.3, 1.3)

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
    print("  Stats: GEE (binomial, exchangeable, animal cluster) + Holm")
    print("         Fallback: chi-sq/Fisher when GEE can't converge")
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
        data, test_types, test_meta, pellet_records = read_group_data(filepath)
        windows = get_time_windows(test_types, test_meta)
        learners = get_learners(data, windows['final_3'])
        n_excluded = len(data) - len(learners)

        window_data = get_animal_window_values(data, windows, learners)
        pellet_df = build_pellet_dataframe(pellet_records, windows, learners)

        print(f"  Total animals: {len(data)}, Learners: {len(learners)}, "
              f"Non-learners excluded: {n_excluded}")
        print(f"  Total pellet observations: {len(pellet_df)}")
        print(f"  Windows:")
        for wk_idx, wk in enumerate(WINDOW_KEYS):
            tts = windows[wk]
            n_animals = len(window_data[wk]['animals'])
            n_pellets = int((pellet_df['window'] == wk_idx).sum()) if not pellet_df.empty else 0
            print(f"    {wk:20s}: {n_animals:2d} animals, {n_pellets:5d} pellets | "
                  f"{', '.join(tts)}")

        plot_group(group_name, injury_type, window_data, pellet_df, len(learners), OUTPUT_DIR)
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
