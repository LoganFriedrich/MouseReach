"""
DLC Pellet Tracking Quality Analysis
Compares Group L (especially L02, L10, L12, L13) to Groups K and M
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict
import re

# Paths
GROUP_PATHS = {
    'K': Path(r"X:\! DLC Output\Analyzed\K\Post-Processing"),
    'L': Path(r"X:\! DLC Output\Analyzed\L\Post-Processing"),
    'M': Path(r"X:\! DLC Output\Analyzed\M\Post-Processing")
}

FLAGGED_ANIMALS = ['L02', 'L10', 'L12', 'L13']

def extract_animal_id(filename):
    """Extract animal ID from filename: {DATE}_{ANIMALID}_{POSITION}DLC_*.csv"""
    pattern = r'\d{8}_([A-Z]\d{2})_'
    match = re.search(pattern, filename)
    return match.group(1) if match else None

def load_dlc_csv(csv_path):
    """Load DLC CSV with 3-header format and return pellet/pillar data"""
    # Read the 3 header rows
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)

    try:
        # Get the scorer name (first level) - should be consistent across all columns
        scorer_level = df.columns.get_level_values(0)
        scorer = scorer_level[0] if len(scorer_level) > 0 else None

        if scorer is None:
            return None, None

        # Extract pellet and pillar likelihood columns
        # The MultiIndex is: (scorer, bodypart, coord)
        pellet_likelihood = df[(scorer, 'Pellet', 'likelihood')].values
        pillar_likelihood = df[(scorer, 'Pillar', 'likelihood')].values

        return pellet_likelihood, pillar_likelihood
    except (KeyError, IndexError) as e:
        print(f"Warning: Could not find columns in {csv_path.name}: {e}")
        return None, None

def calculate_tracking_metrics(likelihood_values):
    """Calculate tracking quality metrics from likelihood array"""
    if likelihood_values is None or len(likelihood_values) == 0:
        return None

    metrics = {
        'mean_likelihood': np.mean(likelihood_values),
        'pct_below_0.5': 100 * np.mean(likelihood_values < 0.5),
        'pct_below_0.3': 100 * np.mean(likelihood_values < 0.3),
        'pct_above_0.95': 100 * np.mean(likelihood_values > 0.95),
        'total_frames': len(likelihood_values)
    }

    # Count dropout events (consecutive frames with likelihood < 0.5)
    below_threshold = likelihood_values < 0.5
    dropout_events = []
    in_dropout = False
    dropout_start = 0

    for i, is_bad in enumerate(below_threshold):
        if is_bad and not in_dropout:
            in_dropout = True
            dropout_start = i
        elif not is_bad and in_dropout:
            in_dropout = False
            dropout_events.append(i - dropout_start)

    if in_dropout:  # Handle case where dropout extends to end
        dropout_events.append(len(below_threshold) - dropout_start)

    metrics['num_dropout_events'] = len(dropout_events)
    metrics['mean_dropout_length'] = np.mean(dropout_events) if dropout_events else 0
    metrics['max_dropout_length'] = np.max(dropout_events) if dropout_events else 0

    return metrics

def process_group(group_name, max_files=50):
    """Process all CSVs for a group"""
    group_path = GROUP_PATHS[group_name]
    csv_files = list(group_path.glob('*DLC*.csv'))

    print(f"\n{'='*80}")
    print(f"Processing Group {group_name}: Found {len(csv_files)} CSV files")
    print(f"{'='*80}")

    # For flagged animals in L, get ALL their sessions
    flagged_files = []
    other_files = []

    for csv_file in csv_files:
        animal_id = extract_animal_id(csv_file.name)
        if animal_id and animal_id in FLAGGED_ANIMALS:
            flagged_files.append(csv_file)
        else:
            other_files.append(csv_file)

    # Sample other files if too many
    if len(other_files) > max_files:
        print(f"Sampling {max_files} files from {len(other_files)} non-flagged sessions")
        np.random.seed(42)
        other_files = list(np.random.choice(other_files, max_files, replace=False))

    files_to_process = flagged_files + other_files
    print(f"Processing {len(flagged_files)} flagged + {len(other_files)} other = {len(files_to_process)} total files")

    results = []

    for csv_file in files_to_process:
        animal_id = extract_animal_id(csv_file.name)
        if not animal_id:
            continue

        pellet_lik, pillar_lik = load_dlc_csv(csv_file)

        if pellet_lik is None:
            continue

        pellet_metrics = calculate_tracking_metrics(pellet_lik)
        pillar_metrics = calculate_tracking_metrics(pillar_lik)

        if pellet_metrics is None:
            continue

        result = {
            'group': group_name,
            'animal_id': animal_id,
            'session_file': csv_file.name,
            'is_flagged': animal_id in FLAGGED_ANIMALS,
            # Pellet metrics
            'pellet_mean_lik': pellet_metrics['mean_likelihood'],
            'pellet_pct_below_0.5': pellet_metrics['pct_below_0.5'],
            'pellet_pct_below_0.3': pellet_metrics['pct_below_0.3'],
            'pellet_pct_above_0.95': pellet_metrics['pct_above_0.95'],
            'pellet_num_dropouts': pellet_metrics['num_dropout_events'],
            'pellet_mean_dropout_len': pellet_metrics['mean_dropout_length'],
            'pellet_max_dropout_len': pellet_metrics['max_dropout_length'],
            'total_frames': pellet_metrics['total_frames'],
            # Pillar metrics for control
            'pillar_mean_lik': pillar_metrics['mean_likelihood'] if pillar_metrics else None,
            'pillar_pct_below_0.5': pillar_metrics['pct_below_0.5'] if pillar_metrics else None,
        }

        results.append(result)

    return pd.DataFrame(results)

def print_summary_table(df, title):
    """Print summary statistics for a group"""
    print(f"\n{title}")
    print("="*80)

    summary = df[[
        'pellet_mean_lik', 'pellet_pct_below_0.5', 'pellet_pct_below_0.3',
        'pellet_pct_above_0.95', 'pellet_num_dropouts', 'pellet_mean_dropout_len',
        'pellet_max_dropout_len', 'pillar_mean_lik', 'pillar_pct_below_0.5'
    ]].describe()

    print(summary.to_string())
    print(f"\nTotal sessions: {len(df)}")
    print(f"Total animals: {df['animal_id'].nunique()}")

def compare_groups(df, group1, group2):
    """Statistical comparison between two groups"""
    g1 = df[df['group'] == group1]
    g2 = df[df['group'] == group2]

    print(f"\n{'='*80}")
    print(f"Statistical Comparison: Group {group1} vs Group {group2}")
    print(f"{'='*80}")

    metrics = [
        ('pellet_mean_lik', 'Mean Pellet Likelihood'),
        ('pellet_pct_below_0.5', '% Frames Below 0.5'),
        ('pellet_pct_below_0.3', '% Frames Below 0.3'),
        ('pellet_num_dropouts', 'Number of Dropouts'),
        ('pellet_mean_dropout_len', 'Mean Dropout Length'),
        ('pillar_mean_lik', 'Pillar Mean Likelihood')
    ]

    for metric, label in metrics:
        v1 = g1[metric].dropna()
        v2 = g2[metric].dropna()

        if len(v1) > 0 and len(v2) > 0:
            stat, p = stats.mannwhitneyu(v1, v2, alternative='two-sided')

            print(f"\n{label}:")
            print(f"  Group {group1}: {v1.mean():.3f} ± {v1.std():.3f} (n={len(v1)})")
            print(f"  Group {group2}: {v2.mean():.3f} ± {v2.std():.3f} (n={len(v2)})")
            print(f"  Mann-Whitney U: {stat:.1f}, p-value: {p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'}")

def analyze_flagged_vs_unflagged(df_l):
    """Compare flagged vs unflagged animals within Group L"""
    flagged = df_l[df_l['is_flagged']]
    unflagged = df_l[~df_l['is_flagged']]

    print(f"\n{'='*80}")
    print("Group L: Flagged vs Unflagged Animals")
    print(f"{'='*80}")

    print(f"\nFlagged animals (L02, L10, L12, L13): {len(flagged)} sessions")
    print(f"Unflagged animals: {len(unflagged)} sessions")

    metrics = [
        ('pellet_mean_lik', 'Mean Pellet Likelihood'),
        ('pellet_pct_below_0.5', '% Frames Below 0.5'),
        ('pellet_pct_below_0.3', '% Frames Below 0.3'),
        ('pellet_num_dropouts', 'Number of Dropouts'),
        ('pellet_mean_dropout_len', 'Mean Dropout Length'),
        ('pillar_mean_lik', 'Pillar Mean Likelihood')
    ]

    for metric, label in metrics:
        v_flag = flagged[metric].dropna()
        v_unflag = unflagged[metric].dropna()

        if len(v_flag) > 0 and len(v_unflag) > 0:
            stat, p = stats.mannwhitneyu(v_flag, v_unflag, alternative='two-sided')

            print(f"\n{label}:")
            print(f"  Flagged:   {v_flag.mean():.3f} ± {v_flag.std():.3f} (n={len(v_flag)})")
            print(f"  Unflagged: {v_unflag.mean():.3f} ± {v_unflag.std():.3f} (n={len(v_unflag)})")
            print(f"  Mann-Whitney U: {stat:.1f}, p-value: {p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'}")

def per_session_flagged_analysis(df_l):
    """Detailed per-session analysis for flagged animals"""
    flagged = df_l[df_l['is_flagged']].sort_values(['animal_id', 'session_file'])

    print(f"\n{'='*80}")
    print("Per-Session Analysis: Flagged Animals (L02, L10, L12, L13)")
    print(f"{'='*80}")

    for animal_id in FLAGGED_ANIMALS:
        animal_sessions = flagged[flagged['animal_id'] == animal_id]

        if len(animal_sessions) == 0:
            print(f"\n{animal_id}: NO SESSIONS FOUND")
            continue

        print(f"\n{animal_id}: {len(animal_sessions)} sessions")
        print("-"*80)

        # Create compact table
        for _, row in animal_sessions.iterrows():
            flag = "[POOR]" if row['pellet_pct_below_0.5'] > 30 else "[OK]  "
            print(f"{flag} {row['session_file'][:30]:30s} | "
                  f"Mean Lik: {row['pellet_mean_lik']:.3f} | "
                  f"<0.5: {row['pellet_pct_below_0.5']:5.1f}% | "
                  f"Dropouts: {row['pellet_num_dropouts']:3.0f} | "
                  f"Pillar: {row['pillar_mean_lik']:.3f}")

        # Summary for this animal
        print(f"\n  {animal_id} SUMMARY:")
        print(f"    Mean pellet likelihood: {animal_sessions['pellet_mean_lik'].mean():.3f}")
        print(f"    Mean % frames <0.5: {animal_sessions['pellet_pct_below_0.5'].mean():.1f}%")
        print(f"    Sessions with >30% poor frames: {(animal_sessions['pellet_pct_below_0.5'] > 30).sum()}/{len(animal_sessions)}")

def main():
    print("="*80)
    print("DLC PELLET TRACKING QUALITY ANALYSIS")
    print("="*80)

    # Process all groups
    all_data = []
    for group in ['K', 'L', 'M']:
        df = process_group(group, max_files=50)
        all_data.append(df)

    df_all = pd.concat(all_data, ignore_index=True)

    # Overall summaries
    for group in ['K', 'L', 'M']:
        df_group = df_all[df_all['group'] == group]
        print_summary_table(df_group, f"GROUP {group} SUMMARY")

    # Between-group comparisons
    compare_groups(df_all, 'L', 'K')
    compare_groups(df_all, 'L', 'M')
    compare_groups(df_all, 'K', 'M')

    # Within Group L: flagged vs unflagged
    df_l = df_all[df_all['group'] == 'L']
    analyze_flagged_vs_unflagged(df_l)

    # Per-session analysis for flagged animals
    per_session_flagged_analysis(df_l)

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total sessions analyzed: {len(df_all)}")
    print(f"  Group K: {len(df_all[df_all['group'] == 'K'])} sessions")
    print(f"  Group L: {len(df_all[df_all['group'] == 'L'])} sessions")
    print(f"  Group M: {len(df_all[df_all['group'] == 'M'])} sessions")
    print(f"\nFlagged animals (L02, L10, L12, L13): {len(df_l[df_l['is_flagged']])} sessions")

    # Key findings
    print(f"\nKEY FINDINGS:")
    flagged_pct_bad = df_l[df_l['is_flagged']]['pellet_pct_below_0.5'].mean()
    unflagged_pct_bad = df_l[~df_l['is_flagged']]['pellet_pct_below_0.5'].mean()

    print(f"  Flagged animals: {flagged_pct_bad:.1f}% of frames have pellet likelihood < 0.5")
    print(f"  Unflagged Group L: {unflagged_pct_bad:.1f}% of frames have pellet likelihood < 0.5")

    if flagged_pct_bad > unflagged_pct_bad * 1.5:
        print(f"  WARNING: Flagged animals have {flagged_pct_bad/unflagged_pct_bad:.1f}x worse pellet tracking")
    else:
        print(f"  No major difference in pellet tracking quality")

if __name__ == '__main__':
    main()
