"""
ASPA Version Mismatch Analysis — Three Branches

Branch 1: Quality metrics — can we identify which M swipes are "extra"?
Branch 2: Is the 2x swipes/animal ratio consistent across phases?
Branch 3: Subsample M to match K/L density — do kinematics change?
"""

import pandas as pd
import numpy as np
import os

base = os.path.join(
    r'C:\Users\friedrichl\OneDrive - Marquette University\Blackmore Lab Notes - Sharepoint',
    r'3 Lab Projects\Automated single pellet apparatus\!UCSF_Collab',
    r'May2025_Uploads\ODC Uploads'
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_connectome_root = SCRIPT_DIR
while os.path.basename(_connectome_root) != '2_Connectome' and os.path.dirname(_connectome_root) != _connectome_root:
    _connectome_root = os.path.dirname(_connectome_root)
OUTPUT_DIR = os.path.join(_connectome_root, 'Databases', 'figures', 'bodypart_tracking_quality')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
df1 = pd.read_csv(os.path.join(base, 'Swipe_Contusion_Data.csv'))
df2 = pd.read_csv(os.path.join(base, 'Swipe_Contusion_Data_2.csv'))
df_all = pd.concat([df1, df2], ignore_index=True)
df_all['group'] = df_all['SubjectID'].str[0]

# Parse kinematic columns
for col in ['Path_over_Frames', 'Swipe_speed', 'Swipe_length', 'Swipe_area',
            'Swipe_breadth', 'Swipe_Duration_Frames', 'Path_length', 'Swipe_Duration']:
    df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

# Phase mapping
PHASE_MAP = {
    'Pre-Injury': ['2_Pre-injury_1', '2_Pre-injury_2', '2_Pre-injury_3'],
    'Immediate Post': ['3_1wk_Post-injury', '3_1wk_Post-injury_First_Test'],
    '2-4wk Post': ['3_2wk_Post-injury', '3_3wk_Post-injury', '3_4wk_Post-injury'],
    'Post-Rehab': ['5_Post-rehab_Test_1', '5_Post-rehab_Test_2'],
}
test_to_phase = {}
for phase, tests in PHASE_MAP.items():
    for t in tests:
        test_to_phase[t] = phase
df_all['phase'] = df_all['Test_Type_Grouped_1'].map(test_to_phase)
df_phased = df_all[df_all['phase'].notna()].copy()

print("=" * 80)
print("BRANCH 1: QUALITY METRICS -- Can we identify extra swipes?")
print("=" * 80)

print("\nSwipe duration stats by group (phased data only):")
for grp in ['K', 'L', 'M']:
    g = df_phased[df_phased['group'] == grp]
    dur = g['Swipe_Duration_Frames'].dropna()
    pof = g['Path_over_Frames'].dropna()
    length = g['Swipe_length'].dropna()
    area = g['Swipe_area'].dropna()
    print(f"  {grp}: duration mean={dur.mean():.1f} med={dur.median():.0f} "
          f"min={dur.min():.0f} max={dur.max():.0f}")
    print(f"     path/frames mean={pof.mean():.2f} med={pof.median():.2f}")
    print(f"     length mean={length.mean():.2f} med={length.median():.2f}")
    print(f"     area mean={area.mean():.2f} med={area.median():.2f}")

# Short swipes
for threshold in [3, 5, 10]:
    print(f"\n% swipes with duration <= {threshold} frames:")
    for grp in ['K', 'L', 'M']:
        g = df_phased[df_phased['group'] == grp]
        dur = g['Swipe_Duration_Frames'].dropna()
        short = (dur <= threshold).sum()
        print(f"  {grp}: {short}/{len(dur)} ({100*short/len(dur):.1f}%)")

# Very small area swipes
for threshold in [50, 100, 500]:
    print(f"\n% swipes with area < {threshold}:")
    for grp in ['K', 'L', 'M']:
        g = df_phased[df_phased['group'] == grp]
        a = g['Swipe_area'].dropna()
        small = (a < threshold).sum()
        print(f"  {grp}: {small}/{len(a)} ({100*small/len(a):.1f}%)")

# Distribution of reach outcomes
print("\nReach outcome distribution (phased data):")
for grp in ['K', 'L', 'M']:
    g = df_phased[df_phased['group'] == grp]
    print(f"\n  {grp} ({len(g)} swipes):")
    for val, cnt in g['Reach_outcome'].value_counts().items():
        print(f"    {val}: {cnt} ({100*cnt/len(g):.1f}%)")


print("\n" + "=" * 80)
print("BRANCH 2: SWIPES/ANIMAL RATIO BY PHASE")
print("=" * 80)

print("\nPer-animal swipes by group and phase:")
for phase in ['Pre-Injury', 'Immediate Post', '2-4wk Post', 'Post-Rehab']:
    print(f"\n  {phase}:")
    for grp in ['K', 'L', 'M']:
        g = df_phased[(df_phased['group'] == grp) & (df_phased['phase'] == phase)]
        if len(g) == 0:
            print(f"    {grp}: no data")
            continue
        per_animal = g.groupby('SubjectID').size()
        print(f"    {grp}: {len(g)} total, {g['SubjectID'].nunique()} animals, "
              f"{per_animal.mean():.0f} swipes/animal (range {per_animal.min()}-{per_animal.max()})")

print("\nM-to-K+L swipes/animal ratio by phase:")
for phase in ['Pre-Injury', 'Immediate Post', '2-4wk Post', 'Post-Rehab']:
    m_g = df_phased[(df_phased['group'] == 'M') & (df_phased['phase'] == phase)]
    kl_g = df_phased[(df_phased['group'].isin(['K', 'L'])) & (df_phased['phase'] == phase)]
    if len(m_g) == 0 or len(kl_g) == 0:
        continue
    m_per = len(m_g) / m_g['SubjectID'].nunique()
    kl_per = len(kl_g) / kl_g['SubjectID'].nunique()
    print(f"  {phase:18s}: M={m_per:.0f}/animal, K+L={kl_per:.0f}/animal, ratio={m_per/kl_per:.2f}x")


print("\n" + "=" * 80)
print("BRANCH 3: SUBSAMPLE M TO MATCH K/L DENSITY")
print("=" * 80)

kin_features = ['Path_over_Frames', 'Swipe_length', 'Swipe_area', 'Swipe_breadth']
np.random.seed(42)

# Get K+L target density per phase
kl_targets = {}
for phase in ['Pre-Injury', 'Immediate Post', '2-4wk Post', 'Post-Rehab']:
    kl_g = df_phased[(df_phased['group'].isin(['K', 'L'])) & (df_phased['phase'] == phase)]
    if len(kl_g) > 0:
        kl_targets[phase] = int(len(kl_g) / kl_g['SubjectID'].nunique())

print(f"K+L target swipes/animal by phase: {kl_targets}")

# Subsample M
m_subsampled = []
for phase, target in kl_targets.items():
    m_phase = df_phased[(df_phased['group'] == 'M') & (df_phased['phase'] == phase)]
    for aid in m_phase['SubjectID'].unique():
        animal_data = m_phase[m_phase['SubjectID'] == aid]
        if len(animal_data) <= target:
            m_subsampled.append(animal_data)
        else:
            m_subsampled.append(animal_data.sample(n=target, random_state=42))

m_sub = pd.concat(m_subsampled, ignore_index=True) if m_subsampled else pd.DataFrame()
print(f"M subsampled: {len(m_sub)} swipes (from {len(df_phased[df_phased['group']=='M'])} original)")

# Compare kinematics
print("\n--- Per-animal mean kinematics by phase ---")
for feature in kin_features:
    print(f"\n  {feature}:")
    for phase in ['Pre-Injury', 'Immediate Post', '2-4wk Post', 'Post-Rehab']:
        kl = df_phased[(df_phased['group'].isin(['K', 'L'])) & (df_phased['phase'] == phase)]
        kl_means = kl.groupby('SubjectID')[feature].mean().dropna()

        m_full = df_phased[(df_phased['group'] == 'M') & (df_phased['phase'] == phase)]
        m_full_means = m_full.groupby('SubjectID')[feature].mean().dropna()

        m_s = m_sub[m_sub['phase'] == phase]
        m_sub_means = m_s.groupby('SubjectID')[feature].mean().dropna()

        if len(kl_means) < 2 or len(m_full_means) < 2:
            continue

        print(f"    {phase:18s}: K+L={kl_means.mean():.2f}  "
              f"M_full={m_full_means.mean():.2f}  M_subsamp={m_sub_means.mean():.2f}")

# Pre-to-post shifts
print("\n--- Pre-to-Post Shift (% change from pre-injury mean) ---")
for feature in kin_features:
    results = {}
    for label, data in [('K+L', df_phased[df_phased['group'].isin(['K', 'L'])]),
                        ('M_full', df_phased[df_phased['group'] == 'M']),
                        ('M_sub', m_sub)]:
        pre = data[data['phase'] == 'Pre-Injury'].groupby('SubjectID')[feature].mean()
        post = data[data['phase'].isin(['Immediate Post', '2-4wk Post'])].groupby('SubjectID')[feature].mean()
        common = pre.index.intersection(post.index)
        if len(common) < 2:
            results[label] = 'n/a'
            continue
        pct = ((post.loc[common] - pre.loc[common]) / pre.loc[common].abs().replace(0, np.nan) * 100).dropna()
        results[label] = f"{pct.mean():+.1f}% (n={len(pct)})"

    print(f"  {feature:20s}: K+L={results['K+L']}  M_full={results['M_full']}  M_sub={results['M_sub']}")

# Also: filter M to only contact swipes (matching what you said matters)
print("\n\n" + "=" * 80)
print("BONUS: CONTACTS ONLY (displaced + successful)")
print("=" * 80)

contact_outcomes = ['pellet displaced', 'swipe successful']
df_contacts = df_phased[df_phased['Reach_outcome'].isin(contact_outcomes)].copy()

print(f"\nContact swipes only:")
for grp in ['K', 'L', 'M']:
    g = df_contacts[df_contacts['group'] == grp]
    n_animals = g['SubjectID'].nunique()
    per_animal = len(g) / n_animals if n_animals > 0 else 0
    print(f"  {grp}: {len(g)} contacts, {n_animals} animals, {per_animal:.0f} contacts/animal")

print("\nContacts per animal by phase:")
for phase in ['Pre-Injury', 'Immediate Post', '2-4wk Post', 'Post-Rehab']:
    print(f"\n  {phase}:")
    for grp in ['K', 'L', 'M']:
        g = df_contacts[(df_contacts['group'] == grp) & (df_contacts['phase'] == phase)]
        if len(g) == 0:
            print(f"    {grp}: no contact data")
            continue
        per_animal = g.groupby('SubjectID').size()
        print(f"    {grp}: {len(g)} contacts, {g['SubjectID'].nunique()} animals, "
              f"{per_animal.mean():.0f}/animal (range {per_animal.min()}-{per_animal.max()})")

print("\n--- Contact-Only Pre-to-Post Shift ---")
for feature in kin_features:
    results = {}
    for label, data in [('K+L', df_contacts[df_contacts['group'].isin(['K', 'L'])]),
                        ('M', df_contacts[df_contacts['group'] == 'M'])]:
        pre = data[data['phase'] == 'Pre-Injury'].groupby('SubjectID')[feature].mean()
        post = data[data['phase'].isin(['Immediate Post', '2-4wk Post'])].groupby('SubjectID')[feature].mean()
        common = pre.index.intersection(post.index)
        if len(common) < 2:
            results[label] = 'n/a'
            continue
        pct = ((post.loc[common] - pre.loc[common]) / pre.loc[common].abs().replace(0, np.nan) * 100).dropna()
        results[label] = f"{pct.mean():+.1f}% (n={len(pct)})"

    print(f"  {feature:20s}: K+L={results['K+L']}  M={results['M']}")


# Also filter to missed swipes only
print("\n\n" + "=" * 80)
print("BONUS: MISSED SWIPES ONLY")
print("=" * 80)

miss_outcomes = ['swipe missed', 'swipe missed (on pillar)']
df_misses = df_phased[df_phased['Reach_outcome'].isin(miss_outcomes)].copy()

print("\n--- Missed-Only Pre-to-Post Shift ---")
for feature in kin_features:
    results = {}
    for label, data in [('K+L', df_misses[df_misses['group'].isin(['K', 'L'])]),
                        ('M', df_misses[df_misses['group'] == 'M'])]:
        pre = data[data['phase'] == 'Pre-Injury'].groupby('SubjectID')[feature].mean()
        post = data[data['phase'].isin(['Immediate Post', '2-4wk Post'])].groupby('SubjectID')[feature].mean()
        common = pre.index.intersection(post.index)
        if len(common) < 2:
            results[label] = 'n/a'
            continue
        pct = ((post.loc[common] - pre.loc[common]) / pre.loc[common].abs().replace(0, np.nan) * 100).dropna()
        results[label] = f"{pct.mean():+.1f}% (n={len(pct)})"

    print(f"  {feature:20s}: K+L={results['K+L']}  M={results['M']}")

print("\n\nDone.")
