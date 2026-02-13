"""
Deep dive analysis of Group L (Contusion 50kD) from UCSF collaboration data.
"""

import pandas as pd
import numpy as np

# Load data
data_path = r"C:\Users\friedrichl\OneDrive - Marquette University\Blackmore Lab Notes - Sharepoint\3 Lab Projects\Automated single pellet apparatus\!UCSF_Collab\May2025_Uploads\ODC Uploads\Session_Data.csv"

print("="*80)
print("GROUP L (CONTUSION 50kD) DEEP DIVE ANALYSIS")
print("="*80)

df = pd.read_csv(data_path)

# Filter to Pillar trays only
df = df[df['Tray_Type'] == 'Pillar'].copy()

print(f"\nTotal Pillar trays in dataset: {len(df)}")
print(f"Groups in dataset: {sorted(df['Group'].unique())}")

# Focus on Group L
group_L = df[df['Group'] == 'L'].copy()
print(f"\nGroup L Pillar trays: {len(group_L)}")

if len(group_L) == 0:
    print("\n*** ERROR: No Group L data found! ***")
    print("\nAll groups in data:")
    print(df['Group'].value_counts())
    exit()

print("\n" + "="*80)
print("1. GROUP L OVERVIEW")
print("="*80)

subjects_L = sorted(group_L['SubjectID'].unique())
print(f"\nSubjects in Group L: {len(subjects_L)}")
for subj in subjects_L:
    subj_data = group_L[group_L['SubjectID'] == subj]
    n_trays = len(subj_data)
    print(f"  {subj}: {n_trays} trays")

print("\n" + "-"*80)
print("Test_Type distribution in Group L:")
print(group_L['Test_Type'].value_counts().sort_index())

print("\n" + "-"*80)
print("Test_Type_Grouped_1 distribution in Group L:")
print(group_L['Test_Type_Grouped_1'].value_counts().sort_index())

print("\n" + "="*80)
print("2. PER-ANIMAL DETAILED BREAKDOWN")
print("="*80)

for subj in subjects_L:
    subj_data = group_L[group_L['SubjectID'] == subj].copy()
    subj_data = subj_data.sort_values('Test_Type')

    print(f"\n{'='*80}")
    print(f"SUBJECT: {subj}")
    print(f"{'='*80}")
    print(f"Total trays: {len(subj_data)}")

    # Test days
    test_days = subj_data['Test_Type'].unique()
    print(f"Test days present: {sorted(test_days)}")

    # Per test day breakdown
    print("\nPer Test Day Scores:")
    print("-"*80)

    for test_day in sorted(test_days):
        day_data = subj_data[subj_data['Test_Type'] == test_day]
        n_trays = len(day_data)

        print(f"\n  Test Day: {test_day} ({n_trays} trays)")

        # Manual scores - convert to numeric and handle NaN
        manual_contacted = pd.to_numeric(day_data['Manual_Contacted'], errors='coerce').fillna(0).sum()
        manual_retrieved = pd.to_numeric(day_data['Manual_Retrieved'], errors='coerce').fillna(0).sum()
        manual_displaced = pd.to_numeric(day_data['Manual_Displaced'], errors='coerce').fillna(0).sum()

        # Video scores - convert to numeric and handle NaN
        video_contacted = pd.to_numeric(day_data['Video_Contacted'], errors='coerce').fillna(0).sum()
        video_retrieved = pd.to_numeric(day_data['Video_Retrieved'], errors='coerce').fillna(0).sum()
        video_displaced = pd.to_numeric(day_data['Video_Displaced'], errors='coerce').fillna(0).sum()

        print(f"    Manual:  Contacted={manual_contacted:.0f}, Retrieved={manual_retrieved:.0f}, Displaced={manual_displaced:.0f}")
        print(f"    Video:   Contacted={video_contacted:.0f}, Retrieved={video_retrieved:.0f}, Displaced={video_displaced:.0f}")

        # Differences
        diff_contacted = video_contacted - manual_contacted
        diff_retrieved = video_retrieved - manual_retrieved
        diff_displaced = video_displaced - manual_displaced

        print(f"    Diff:    Contacted={diff_contacted:+.0f}, Retrieved={diff_retrieved:+.0f}, Displaced={diff_displaced:+.0f}")

        # Flag large discrepancies
        if abs(diff_contacted) > 3 or abs(diff_retrieved) > 3 or abs(diff_displaced) > 3:
            print(f"    *** LARGE DISCREPANCY detected!")

    # Within-animal variance analysis
    print("\n  Within-Animal Variance:")
    print("-"*80)

    # Group by test type grouped
    for test_group in sorted(subj_data['Test_Type_Grouped_1'].unique()):
        group_data = subj_data[subj_data['Test_Type_Grouped_1'] == test_group]

        if len(group_data) > 1:
            manual_ret_vals = []
            video_ret_vals = []

            for test_day in sorted(group_data['Test_Type'].unique()):
                day_data = group_data[group_data['Test_Type'] == test_day]
                manual_ret_vals.append(pd.to_numeric(day_data['Manual_Retrieved'], errors='coerce').fillna(0).sum())
                video_ret_vals.append(pd.to_numeric(day_data['Video_Retrieved'], errors='coerce').fillna(0).sum())

            if len(manual_ret_vals) > 1:
                manual_std = np.std(manual_ret_vals)
                video_std = np.std(video_ret_vals)
                print(f"    {test_group}: Manual Retrieved std={manual_std:.2f}, Video Retrieved std={video_std:.2f}")

                if manual_std > 5 or video_std > 5:
                    print(f"      *** HIGH VARIANCE within {test_group}! Values: Manual={manual_ret_vals}, Video={video_ret_vals}")

print("\n" + "="*80)
print("3. GROUP L ANOMALY DETECTION")
print("="*80)

# Check for animals with unusual patterns
print("\nChecking for unusual patterns...")

for subj in subjects_L:
    subj_data = group_L[group_L['SubjectID'] == subj].copy()

    anomalies = []

    # Get pre-injury baseline
    # Use Test_Type_Grouped_1 to identify pre-injury (starts with "2_Pre-injury")
    pre_data = subj_data[subj_data['Test_Type_Grouped_1'].str.startswith('2_Pre-injury', na=False)]
    if len(pre_data) > 0:
        pre_manual = pre_data.groupby('Test_Type').apply(lambda x: pd.to_numeric(x['Manual_Retrieved'], errors='coerce').fillna(0).sum())
        pre_video = pre_data.groupby('Test_Type').apply(lambda x: pd.to_numeric(x['Video_Retrieved'], errors='coerce').fillna(0).sum())

        if len(pre_manual) > 0:
            pre_manual_mean = pre_manual.mean()
            pre_video_mean = pre_video.mean()
        else:
            pre_manual_mean = 0
            pre_video_mean = 0
    else:
        anomalies.append("NO PRE-INJURY DATA")
        pre_manual_mean = 0
        pre_video_mean = 0

    # Get post-injury (starts with "3_")
    post_data = subj_data[subj_data['Test_Type_Grouped_1'].str.startswith('3_', na=False)]
    if len(post_data) > 0:
        post_manual = post_data.groupby('Test_Type').apply(lambda x: pd.to_numeric(x['Manual_Retrieved'], errors='coerce').fillna(0).sum())
        post_video = post_data.groupby('Test_Type').apply(lambda x: pd.to_numeric(x['Video_Retrieved'], errors='coerce').fillna(0).sum())

        if len(post_manual) > 0:
            post_manual_mean = post_manual.mean()
            post_video_mean = post_video.mean()
        else:
            post_manual_mean = 0
            post_video_mean = 0
    else:
        anomalies.append("NO POST-INJURY DATA")
        post_manual_mean = 0
        post_video_mean = 0

    # Check if post-injury HIGHER than pre-injury
    if post_manual_mean > pre_manual_mean and pre_manual_mean > 0:
        anomalies.append(f"Manual Retrieved HIGHER post-injury ({post_manual_mean:.1f}) than pre-injury ({pre_manual_mean:.1f})")

    if post_video_mean > pre_video_mean and pre_video_mean > 0:
        anomalies.append(f"Video Retrieved HIGHER post-injury ({post_video_mean:.1f}) than pre-injury ({pre_video_mean:.1f})")

    # Check for large manual-video discrepancy across all data
    total_manual_retrieved = pd.to_numeric(subj_data['Manual_Retrieved'], errors='coerce').fillna(0).sum()
    total_video_retrieved = pd.to_numeric(subj_data['Video_Retrieved'], errors='coerce').fillna(0).sum()

    if total_manual_retrieved > 0:
        pct_diff = 100 * (total_video_retrieved - total_manual_retrieved) / total_manual_retrieved
        if abs(pct_diff) > 20:
            anomalies.append(f"Large Manual-Video discrepancy: {pct_diff:+.1f}%")

    # Check for missing test phases
    test_phases = set(subj_data['Test_Type_Grouped_1'].unique())
    expected_phases = {'2_Pre-injury_1', '2_Pre-injury_2', '2_Pre-injury_3',
                       '3_1wk_Post-injury', '3_2wk_Post-injury', '3_3wk_Post-injury'}
    missing_phases = expected_phases - test_phases

    if missing_phases:
        anomalies.append(f"Missing test phases: {missing_phases}")

    if anomalies:
        print(f"\n{subj}:")
        for anom in anomalies:
            print(f"  ! {anom}")

print("\n" + "="*80)
print("4. COMPARISON: GROUP L vs K (70kD) vs M (60kD)")
print("="*80)

# Get other contusion groups
group_K = df[df['Group'] == 'K'].copy()
group_M = df[df['Group'] == 'M'].copy()

print(f"\nGroup K (70kD) trays: {len(group_K)}")
print(f"Group L (50kD) trays: {len(group_L)}")
print(f"Group M (60kD) trays: {len(group_M)}")

def analyze_group(group_df, group_name):
    """Calculate summary stats for a group"""

    results = {
        'group': group_name,
        'n_subjects': len(group_df['SubjectID'].unique()),
        'n_trays': len(group_df)
    }

    # Pre-injury
    pre = group_df[group_df['Test_Type_Grouped_1'].str.startswith('2_Pre-injury', na=False)]
    if len(pre) > 0:
        # Per test day averages - convert to numeric first
        pre_numeric = pre.copy()
        pre_numeric['Manual_Retrieved'] = pd.to_numeric(pre_numeric['Manual_Retrieved'], errors='coerce').fillna(0)
        pre_numeric['Video_Retrieved'] = pd.to_numeric(pre_numeric['Video_Retrieved'], errors='coerce').fillna(0)

        pre_by_day = pre_numeric.groupby('Test_Type').agg({
            'Manual_Retrieved': 'sum',
            'Video_Retrieved': 'sum',
            'SubjectID': 'nunique'
        })

        # Average per animal per day
        results['pre_manual_mean'] = pre_by_day['Manual_Retrieved'].mean()
        results['pre_video_mean'] = pre_by_day['Video_Retrieved'].mean()
        results['pre_manual_std'] = pre_by_day['Manual_Retrieved'].std()
        results['pre_video_std'] = pre_by_day['Video_Retrieved'].std()
    else:
        results['pre_manual_mean'] = np.nan
        results['pre_video_mean'] = np.nan
        results['pre_manual_std'] = np.nan
        results['pre_video_std'] = np.nan

    # Post-injury
    post = group_df[group_df['Test_Type_Grouped_1'].str.startswith('3_', na=False)]
    if len(post) > 0:
        post_numeric = post.copy()
        post_numeric['Manual_Retrieved'] = pd.to_numeric(post_numeric['Manual_Retrieved'], errors='coerce').fillna(0)
        post_numeric['Video_Retrieved'] = pd.to_numeric(post_numeric['Video_Retrieved'], errors='coerce').fillna(0)

        post_by_day = post_numeric.groupby('Test_Type').agg({
            'Manual_Retrieved': 'sum',
            'Video_Retrieved': 'sum',
            'SubjectID': 'nunique'
        })

        results['post_manual_mean'] = post_by_day['Manual_Retrieved'].mean()
        results['post_video_mean'] = post_by_day['Video_Retrieved'].mean()
        results['post_manual_std'] = post_by_day['Manual_Retrieved'].std()
        results['post_video_std'] = post_by_day['Video_Retrieved'].std()
    else:
        results['post_manual_mean'] = np.nan
        results['post_video_mean'] = np.nan
        results['post_manual_std'] = np.nan
        results['post_video_std'] = np.nan

    # Manual-Video agreement
    total_manual = pd.to_numeric(group_df['Manual_Retrieved'], errors='coerce').fillna(0).sum()
    total_video = pd.to_numeric(group_df['Video_Retrieved'], errors='coerce').fillna(0).sum()

    if total_manual > 0:
        results['manual_video_pct_diff'] = 100 * (total_video - total_manual) / total_manual
    else:
        results['manual_video_pct_diff'] = np.nan

    # Calculate recovery (post/pre ratio)
    if not np.isnan(results['pre_manual_mean']) and results['pre_manual_mean'] > 0:
        results['recovery_manual'] = results['post_manual_mean'] / results['pre_manual_mean']
    else:
        results['recovery_manual'] = np.nan

    if not np.isnan(results['pre_video_mean']) and results['pre_video_mean'] > 0:
        results['recovery_video'] = results['post_video_mean'] / results['pre_video_mean']
    else:
        results['recovery_video'] = np.nan

    return results

# Analyze all three groups
stats_K = analyze_group(group_K, "K (70kD)")
stats_L = analyze_group(group_L, "L (50kD)")
stats_M = analyze_group(group_M, "M (60kD)")

print("\n" + "-"*80)
print("PRE-INJURY BASELINE PERFORMANCE")
print("-"*80)
print(f"{'Group':<12} {'Manual Mean':<15} {'Video Mean':<15} {'Manual Std':<15} {'Video Std':<15}")
print("-"*80)

for stats in [stats_K, stats_L, stats_M]:
    print(f"{stats['group']:<12} {stats['pre_manual_mean']:>8.2f}       {stats['pre_video_mean']:>8.2f}       {stats['pre_manual_std']:>8.2f}       {stats['pre_video_std']:>8.2f}")

print("\n" + "-"*80)
print("POST-INJURY PERFORMANCE")
print("-"*80)
print(f"{'Group':<12} {'Manual Mean':<15} {'Video Mean':<15} {'Manual Std':<15} {'Video Std':<15}")
print("-"*80)

for stats in [stats_K, stats_L, stats_M]:
    print(f"{stats['group']:<12} {stats['post_manual_mean']:>8.2f}       {stats['post_video_mean']:>8.2f}       {stats['post_manual_std']:>8.2f}       {stats['post_video_std']:>8.2f}")

print("\n" + "-"*80)
print("RECOVERY (Post/Pre Ratio)")
print("-"*80)
print(f"{'Group':<12} {'Manual Recovery':<18} {'Video Recovery':<18}")
print("-"*80)

for stats in [stats_K, stats_L, stats_M]:
    print(f"{stats['group']:<12} {stats['recovery_manual']:>12.3f}       {stats['recovery_video']:>12.3f}")

print("\n" + "-"*80)
print("MANUAL-VIDEO AGREEMENT")
print("-"*80)
print(f"{'Group':<12} {'% Difference':<18}")
print("-"*80)

for stats in [stats_K, stats_L, stats_M]:
    print(f"{stats['group']:<12} {stats['manual_video_pct_diff']:>12.1f}%")

print("\n" + "="*80)
print("5. RAW TRAY DATA FOR SUSPICIOUS ANIMALS")
print("="*80)

# Define "suspicious" as animals with large manual-video discrepancies or unusual patterns
print("\nShowing raw tray data for animals flagged above...\n")

for subj in subjects_L:
    subj_data = group_L[group_L['SubjectID'] == subj].copy()

    # Calculate discrepancy
    total_manual = pd.to_numeric(subj_data['Manual_Retrieved'], errors='coerce').fillna(0).sum()
    total_video = pd.to_numeric(subj_data['Video_Retrieved'], errors='coerce').fillna(0).sum()

    if total_manual > 0:
        pct_diff = 100 * (total_video - total_manual) / total_manual
    else:
        pct_diff = 0

    # Check for post > pre
    pre_data = subj_data[subj_data['Test_Type'] < 0]
    post_data = subj_data[subj_data['Test_Type'] > 0]

    pre_manual_mean = pre_data.groupby('Test_Type').apply(lambda x: pd.to_numeric(x['Manual_Retrieved'], errors='coerce').fillna(0).sum()).mean() if len(pre_data) > 0 else 0
    post_manual_mean = post_data.groupby('Test_Type').apply(lambda x: pd.to_numeric(x['Manual_Retrieved'], errors='coerce').fillna(0).sum()).mean() if len(post_data) > 0 else 0

    is_suspicious = (abs(pct_diff) > 20) or (post_manual_mean > pre_manual_mean and pre_manual_mean > 0)

    if is_suspicious:
        print(f"{'='*80}")
        print(f"RAW TRAY DATA: {subj}")
        print(f"{'='*80}")

        subj_sorted = subj_data.sort_values(['Test_Type', 'Tray_ID'])

        print(f"\n{'Tray_ID':<15} {'Test_Type':<30} {'Manual C/R/D':<20} {'Video C/R/D':<20}")
        print("-"*80)

        for _, row in subj_sorted.iterrows():
            manual_c = pd.to_numeric(row['Manual_Contacted'], errors='coerce')
            manual_r = pd.to_numeric(row['Manual_Retrieved'], errors='coerce')
            manual_d = pd.to_numeric(row['Manual_Displaced'], errors='coerce')
            video_c = pd.to_numeric(row['Video_Contacted'], errors='coerce')
            video_r = pd.to_numeric(row['Video_Retrieved'], errors='coerce')
            video_d = pd.to_numeric(row['Video_Displaced'], errors='coerce')

            manual_str = f"{manual_c:.0f}/{manual_r:.0f}/{manual_d:.0f}" if not pd.isna(manual_c) else "NaN/NaN/NaN"
            video_str = f"{video_c:.0f}/{video_r:.0f}/{video_d:.0f}" if not pd.isna(video_c) else "NaN/NaN/NaN"

            print(f"{row['Tray_ID']:<15} {row['Test_Type']:<30} {manual_str:<20} {video_str:<20}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
