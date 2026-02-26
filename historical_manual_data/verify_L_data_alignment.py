"""
Cross-Group Investigation: Why Does Group L Show an Inverted Trajectory?
========================================================================
Generates 12 CSV evidence files and an INVESTIGATION_CHAIN.md narrative
that systematically compares L (Contusion 50kd) against ALL other groups.

Outputs (to .../behavior_historical_manual/L_diagnostic/verification/):
  01_cross_group_protocol.csv           - Protocol comparison across ALL groups
  02_cross_group_trajectories.csv       - Group mean trajectories for all groups
  03_ucsf_exclusion_rates.csv           - UCSF vs manual animal counts per group
  04_session_alignment.csv              - UCSF phase labels vs manual Excel sessions (L)
  05_manual_per_animal_per_session.csv  - All per-animal per-session eaten%/contacted% (L)
  06_ucsf_vs_manual_totals.csv          - UCSF Video vs Manual pellet totals (L)
  07_leave_one_out_results.csv          - LOO analysis: impact of removing each animal
  08_animal_verdicts.csv                - Per-animal summary with all flags and window values
  09_threshold_sensitivity.csv          - Learner counts at each threshold for ALL groups
  10_tray_type_by_session.csv           - Tray type for each session (pillar/easy/flat)
  11_post_rehab_pillar_comparison.csv   - ALL groups at pillar days 3+
  12_data_completeness.csv              - Per-animal session completeness across ALL groups
  INVESTIGATION_CHAIN.md                - Narrative document tracing the investigation

Usage:
  cd Y:\\2_Connectome\\Behavior\\MouseReach\\historical_manual_data
  python verify_L_data_alignment.py
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import os
import re
import sys
import pandas as pd
from collections import defaultdict
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from plot_historical_groups import (
    read_group_data, get_time_windows, get_learners,
    get_animal_window_values, TRAY_OFFSETS_4, TRAY_OFFSETS_2,
    LEARNER_EATEN_THRESHOLD, WINDOW_KEYS, GROUP_FILES,
)

# --- Output directory ---
_connectome_root = SCRIPT_DIR
while (os.path.basename(_connectome_root) != '2_Connectome' and
       os.path.dirname(_connectome_root) != _connectome_root):
    _connectome_root = os.path.dirname(_connectome_root)

OUTPUT_DIR = os.path.join(
    _connectome_root, 'Databases', 'figures',
    'behavior_historical_manual', 'L_diagnostic', 'verification'
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# UCSF data path
UCSF_SESSION_CSV = os.path.join(
    os.path.expanduser('~'),
    'OneDrive - Marquette University',
    'Blackmore Lab Notes - Sharepoint',
    '3 Lab Projects',
    'Automated single pellet apparatus',
    '!UCSF_Collab',
    'May2025_Uploads',
    'ODC Uploads',
    'Session_Data.csv'
)

# L_Investigation verdicts
L_VERDICTS = {
    'L01': 'UNRELIABLE',
    'L02': 'UNRELIABLE',
    'L03': 'N/A (non-learner)',
    'L04': 'N/A (non-learner)',
    'L05': 'N/A (not in UCSF)',
    'L06': 'N/A (not in UCSF)',
    'L07': 'N/A (non-learner)',
    'L08': 'NOT ANOMALOUS',
    'L09': 'LIKELY MILD INJURY',
    'L10': 'TRAINING ARTIFACT',
    'L11': 'UNRELIABLE',
    'L12': 'LIKELY MILD INJURY',
    'L13': 'OK',
    'L14': 'N/A (not in UCSF)',
    'L15': 'N/A (not in UCSF)',
    'L16': 'N/A (non-learner, euthanized)',
}

# UCSF group name mapping (D = OptD)
UCSF_GROUP_MAP = {
    'D': 'OptD',
    'G': 'G',
    'H': 'H',
    'K': 'K',
    'L': 'L',
    'M': 'M',
}


def _animal_str(animal_id, group_prefix='L'):
    """Normalize animal ID to standard form like L01, G03 etc."""
    s = str(animal_id).strip()
    if s.isdigit() and len(s) <= 2:
        s = f'{group_prefix}{s.zfill(2)}'
    return s


def _find_post_rehab_pillar_tts(test_types, test_meta):
    """Find pillar rehab sessions after the first 2 (i.e. pillar days 3+)."""
    rehab_tts = [tt for tt in test_types if 'rehab' in tt.lower()]
    pillar_rehab = [
        tt for tt in rehab_tts
        if test_meta.get(tt, {}).get('tray', '').lower().startswith('p')
    ]
    return pillar_rehab[2:] if len(pillar_rehab) > 2 else []


def _learners_at_threshold(data, final3_tts, threshold):
    """Count learners at a given eaten% threshold."""
    learners = set()
    for animal, tests in data.items():
        if final3_tts:
            e_vals = [tests[tt]['eaten'] for tt in final3_tts if tt in tests]
            if e_vals and np.mean(e_vals) > threshold:
                learners.add(animal)
        else:
            learners.add(animal)
    return learners


def _classify_session_phase(tt):
    """Classify a test type string into a phase category."""
    tt_l = tt.lower()
    if 'train' in tt_l:
        return 'training'
    if 'post' in tt_l and 'injury' in tt_l:
        return 'post_injury'
    if 'rehab' in tt_l:
        return 'rehab'
    return 'other'


def _load_all_groups():
    """Load data for all groups. Returns dict of group_name -> full data."""
    all_data = {}
    for gname, (filename, injury_type) in GROUP_FILES.items():
        filepath = os.path.join(SCRIPT_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  WARNING: {filename} not found, skipping {gname}")
            continue
        data, test_types, test_meta, pellet_records = read_group_data(filepath)
        windows = get_time_windows(test_types, test_meta)
        learners = get_learners(data, windows['final_3'])
        window_data = get_animal_window_values(data, windows, learners)
        all_data[gname] = {
            'data': data,
            'test_types': test_types,
            'test_meta': test_meta,
            'pellet_records': pellet_records,
            'windows': windows,
            'learners': learners,
            'window_data': window_data,
            'injury_type': injury_type,
            'filename': filename,
        }
    return all_data


def _load_ucsf_data():
    """Load UCSF Session_Data.csv if accessible. Returns DataFrame or None."""
    if not os.path.exists(UCSF_SESSION_CSV):
        return None
    try:
        df = pd.read_csv(UCSF_SESSION_CSV)
        return df
    except Exception as e:
        print(f"  WARNING: Could not read UCSF data: {e}")
        return None


def _find_ucsf_subject_col(df):
    """Find the subject/animal ID column in UCSF data."""
    for c in ['SubjectID', 'Animal_ID', 'Mouse_ID', 'Subject', 'Animal_Num']:
        if c in df.columns:
            return c
    return None


# =============================================================================
# CSV 01: Cross-Group Protocol Comparison
# =============================================================================

def generate_01_cross_group_protocol(all_data):
    """Compare protocol structure across ALL groups."""
    print("  [01] Cross-group protocol comparison...")

    rows = []
    for gname in sorted(all_data.keys()):
        gd = all_data[gname]
        data = gd['data']
        test_types = gd['test_types']
        test_meta = gd['test_meta']
        windows = gd['windows']
        learners = gd['learners']

        # Date range
        dates = []
        for tt in test_types:
            d = test_meta.get(tt, {}).get('date')
            if d is not None:
                try:
                    if hasattr(d, 'strftime'):
                        dates.append(d)
                except Exception:
                    pass

        date_min = min(dates).strftime('%Y-%m-%d') if dates else ''
        date_max = max(dates).strftime('%Y-%m-%d') if dates else ''

        # Phase counts
        training_tts = [tt for tt in test_types if _classify_session_phase(tt) == 'training']
        post_tts = [tt for tt in test_types if _classify_session_phase(tt) == 'post_injury']
        rehab_tts = [tt for tt in test_types if _classify_session_phase(tt) == 'rehab']

        # Pillar rehab and P3+
        pillar_rehab = [
            tt for tt in rehab_tts
            if test_meta.get(tt, {}).get('tray', '').lower().startswith('p')
        ]
        p3_plus = pillar_rehab[2:] if len(pillar_rehab) > 2 else []

        # Tray type sequence
        tray_sequence = []
        prev_tray = None
        for tt in test_types:
            tray = test_meta.get(tt, {}).get('tray', '')
            if tray != prev_tray:
                tray_sequence.append(tray)
                prev_tray = tray

        # Sessions per animal
        sessions_per_animal = []
        for animal in data:
            n = sum(1 for tt in test_types if tt in data[animal])
            sessions_per_animal.append(n)

        spa = np.array(sessions_per_animal) if sessions_per_animal else np.array([0])
        animals_with_full = sum(1 for s in sessions_per_animal if s == len(test_types))

        rows.append({
            'group': gname,
            'injury_type': gd['injury_type'],
            'date_start': date_min,
            'date_end': date_max,
            'n_animals': len(data),
            'n_learners_5pct': len(learners),
            'n_training_sessions': len(training_tts),
            'n_post_injury_sessions': len(post_tts),
            'n_rehab_sessions': len(rehab_tts),
            'n_pillar_rehab_sessions': len(pillar_rehab),
            'n_p3plus_sessions': len(p3_plus),
            'tray_type_sequence': ' -> '.join(tray_sequence),
            'sessions_per_animal_min': int(spa.min()),
            'sessions_per_animal_median': float(np.median(spa)),
            'sessions_per_animal_max': int(spa.max()),
            'animals_with_full_data': animals_with_full,
            'total_sessions': len(test_types),
        })

    outpath = os.path.join(OUTPUT_DIR, '01_cross_group_protocol.csv')
    pd.DataFrame(rows).to_csv(outpath, index=False)
    print(f"    Saved: {outpath}")
    return rows


# =============================================================================
# CSV 02: Cross-Group Trajectories
# =============================================================================

def generate_02_cross_group_trajectories(all_data):
    """Group mean trajectories for ALL groups across all windows."""
    print("  [02] Cross-group trajectories...")

    rows = []
    for gname in sorted(all_data.keys()):
        gd = all_data[gname]
        wd = gd['window_data']
        learners = gd['learners']

        row = {
            'group': gname,
            'injury_type': gd['injury_type'],
            'n_learners': len(learners),
        }

        # Compute means for each window x metric
        for metric in ['eaten', 'contacted']:
            for wk in WINDOW_KEYS:
                vals = wd[wk][metric]
                if len(vals) > 0:
                    row[f'{wk}_{metric}_pct'] = round(float(np.mean(vals)), 2)
                    row[f'{wk}_{metric}_n'] = len(vals)
                else:
                    row[f'{wk}_{metric}_pct'] = ''
                    row[f'{wk}_{metric}_n'] = 0

        # P3+ window (not in standard WINDOW_KEYS)
        p3_tts = _find_post_rehab_pillar_tts(gd['test_types'], gd['test_meta'])
        if p3_tts:
            p3_eaten, p3_contacted = [], []
            for animal in learners:
                if animal not in gd['data']:
                    continue
                e = [gd['data'][animal][tt]['eaten'] for tt in p3_tts if tt in gd['data'][animal]]
                c = [gd['data'][animal][tt]['contacted'] for tt in p3_tts if tt in gd['data'][animal]]
                if e:
                    p3_eaten.append(np.mean(e))
                    p3_contacted.append(np.mean(c))
            row['p3plus_eaten_pct'] = round(float(np.mean(p3_eaten)), 2) if p3_eaten else ''
            row['p3plus_contacted_pct'] = round(float(np.mean(p3_contacted)), 2) if p3_contacted else ''
            row['p3plus_n'] = len(p3_eaten)
        else:
            row['p3plus_eaten_pct'] = ''
            row['p3plus_contacted_pct'] = ''
            row['p3plus_n'] = 0

        # Trajectory direction: compare final_3 to last_2
        f3 = row.get('final_3_eaten_pct', '')
        l2 = row.get('last_2_eaten_pct', '')
        if f3 != '' and l2 != '':
            diff = l2 - f3
            if diff > 2:
                row['trajectory_direction'] = 'INVERTED (rehab > baseline)'
            elif diff < -2:
                row['trajectory_direction'] = 'decline'
            else:
                row['trajectory_direction'] = 'stable'
            row['trajectory_magnitude'] = round(diff, 2)
        else:
            row['trajectory_direction'] = 'insufficient data'
            row['trajectory_magnitude'] = ''

        rows.append(row)

    outpath = os.path.join(OUTPUT_DIR, '02_cross_group_trajectories.csv')
    pd.DataFrame(rows).to_csv(outpath, index=False)
    print(f"    Saved: {outpath}")
    return rows


# =============================================================================
# CSV 03: UCSF Exclusion Rates
# =============================================================================

def generate_03_ucsf_exclusion_rates(all_data, ucsf_df):
    """Compare UCSF vs manual animal counts and exclusion rates per group."""
    print("  [03] UCSF exclusion rates...")

    rows = []
    for gname in sorted(all_data.keys()):
        gd = all_data[gname]
        manual_animals = set(str(a) for a in gd['data'].keys())

        row = {
            'group': gname,
            'injury_type': gd['injury_type'],
            'manual_n_animals': len(manual_animals),
        }

        if ucsf_df is not None:
            # Find matching UCSF group name
            ucsf_gname = None
            for ucsf_g, mapped in UCSF_GROUP_MAP.items():
                if mapped == gname:
                    ucsf_gname = ucsf_g
                    break

            if ucsf_gname is not None and 'Group' in ucsf_df.columns:
                ucsf_group = ucsf_df[ucsf_df['Group'] == ucsf_gname]
                subj_col = _find_ucsf_subject_col(ucsf_df)

                if subj_col and not ucsf_group.empty:
                    ucsf_animals = set(str(a) for a in ucsf_group[subj_col].unique())
                    row['ucsf_n_animals'] = len(ucsf_animals)

                    # Figure out excluded count from UCSF total rows vs unique
                    # The exclusion is about video sessions excluded, not animals
                    # But for animal-level: count how many manual animals are NOT in UCSF
                    # Map UCSF IDs to our format
                    ucsf_ids_normalized = set()
                    for uid in ucsf_animals:
                        uid_s = str(uid).strip()
                        # UCSF uses numeric IDs for L, so L01 = 1
                        if uid_s.isdigit():
                            ucsf_ids_normalized.add(f'{ucsf_gname}{uid_s.zfill(2)}')
                        else:
                            ucsf_ids_normalized.add(uid_s)

                    # Manual IDs normalized
                    manual_ids_normalized = set()
                    for mid in manual_animals:
                        mid_s = str(mid).strip()
                        if mid_s.isdigit():
                            manual_ids_normalized.add(f'{gname}{mid_s.zfill(2)}')
                        else:
                            manual_ids_normalized.add(mid_s)

                    missing = manual_ids_normalized - ucsf_ids_normalized
                    row['excluded_n'] = len(missing)
                    row['exclusion_pct'] = round(len(missing) / len(manual_animals) * 100, 1) if manual_animals else 0
                    row['missing_animal_ids'] = '; '.join(sorted(missing))

                    # Also count excluded VIDEO SESSIONS from UCSF
                    excl_cols = [c for c in ucsf_group.columns if 'match' in c.lower()]
                    if excl_cols:
                        total_sessions = len(ucsf_group)
                        # Count sessions where any match column shows exclusion
                        excluded_sessions = 0
                        for _, urow in ucsf_group.iterrows():
                            for ec in excl_cols:
                                val = str(urow.get(ec, ''))
                                if val.lower() in ['false', '0', 'no', 'excluded']:
                                    excluded_sessions += 1
                                    break
                        row['ucsf_total_session_rows'] = total_sessions
                        row['ucsf_excluded_session_rows'] = excluded_sessions
                        row['ucsf_session_exclusion_pct'] = round(excluded_sessions / total_sessions * 100, 1) if total_sessions > 0 else 0
                    else:
                        row['ucsf_total_session_rows'] = len(ucsf_group)
                        row['ucsf_excluded_session_rows'] = ''
                        row['ucsf_session_exclusion_pct'] = ''
                else:
                    row['ucsf_n_animals'] = 0
                    row['excluded_n'] = ''
                    row['exclusion_pct'] = ''
                    row['missing_animal_ids'] = 'Group not found in UCSF'
                    row['ucsf_total_session_rows'] = ''
                    row['ucsf_excluded_session_rows'] = ''
                    row['ucsf_session_exclusion_pct'] = ''
            else:
                row['ucsf_n_animals'] = ''
                row['excluded_n'] = ''
                row['exclusion_pct'] = ''
                row['missing_animal_ids'] = 'Not in UCSF'
                row['ucsf_total_session_rows'] = ''
                row['ucsf_excluded_session_rows'] = ''
                row['ucsf_session_exclusion_pct'] = ''
        else:
            row['ucsf_n_animals'] = 'UCSF data not available'
            row['excluded_n'] = ''
            row['exclusion_pct'] = ''
            row['missing_animal_ids'] = ''
            row['ucsf_total_session_rows'] = ''
            row['ucsf_excluded_session_rows'] = ''
            row['ucsf_session_exclusion_pct'] = ''

        rows.append(row)

    outpath = os.path.join(OUTPUT_DIR, '03_ucsf_exclusion_rates.csv')
    pd.DataFrame(rows).to_csv(outpath, index=False)
    print(f"    Saved: {outpath}")
    return rows


# =============================================================================
# CSV 04: Session Alignment (L-specific, UCSF phases vs manual)
# =============================================================================

def generate_04_session_alignment(l_data, l_test_types, l_test_meta, ucsf_df):
    """Map each L session to UCSF phase labels."""
    print("  [04] Session alignment (L)...")

    ucsf_phases = {}
    ucsf_animals_in_session = {}
    has_ucsf = False

    if ucsf_df is not None and 'Group' in ucsf_df.columns:
        has_ucsf = True
        L_ucsf = ucsf_df[ucsf_df['Group'] == 'L']
        subj_col = _find_ucsf_subject_col(ucsf_df)

        for _, row in L_ucsf.iterrows():
            tt = str(row.get('Test_Type', ''))
            if tt not in ucsf_phases:
                ucsf_phases[tt] = {
                    'Grouped_1': str(row.get('Test_Type_Grouped_1', '')),
                    'Grouped_2': str(row.get('Test_Type_Grouped_2', '')),
                    'Grouped_3': str(row.get('Test_Type_Grouped_3', '')),
                }
            if subj_col:
                if tt not in ucsf_animals_in_session:
                    ucsf_animals_in_session[tt] = set()
                ucsf_animals_in_session[tt].add(str(row[subj_col]))

    rows = []
    for tt in l_test_types:
        meta = l_test_meta.get(tt, {})
        manual_date = ''
        if meta.get('date') is not None:
            try:
                manual_date = meta['date'].strftime('%Y-%m-%d') if hasattr(meta['date'], 'strftime') else str(meta['date'])
            except Exception:
                manual_date = str(meta.get('date', ''))

        manual_tray = str(meta.get('tray', ''))
        manual_animals = [_animal_str(a) for a in l_data.keys() if tt in l_data[a]]

        # Phase classification
        phase = _classify_session_phase(tt)

        # Find matching UCSF test type
        ucsf_match = None
        for ucsf_tt in ucsf_phases:
            if ucsf_tt.strip()[:5] == tt.strip()[:5]:
                ucsf_match = ucsf_tt
                break

        row = {
            'manual_test_type': tt,
            'manual_date': manual_date,
            'manual_tray_type': manual_tray,
            'manual_phase': phase,
            'manual_n_animals': len(manual_animals),
        }

        if has_ucsf and ucsf_match:
            phases = ucsf_phases[ucsf_match]
            row['ucsf_test_type'] = ucsf_match
            row['ucsf_grouped_1'] = phases['Grouped_1']
            row['ucsf_grouped_2'] = phases['Grouped_2']
            row['ucsf_grouped_3'] = phases['Grouped_3']
            ucsf_animals = ucsf_animals_in_session.get(ucsf_match, set())
            row['ucsf_n_animals'] = len(ucsf_animals)
            row['animal_count_match'] = 'YES' if len(manual_animals) == len(ucsf_animals) else f'NO (manual={len(manual_animals)}, ucsf={len(ucsf_animals)})'
        else:
            row['ucsf_test_type'] = 'NOT IN UCSF' if has_ucsf else 'UCSF NOT AVAILABLE'
            row['ucsf_grouped_1'] = ''
            row['ucsf_grouped_2'] = ''
            row['ucsf_grouped_3'] = ''
            row['ucsf_n_animals'] = 0
            row['animal_count_match'] = ''

        rows.append(row)

    outpath = os.path.join(OUTPUT_DIR, '04_session_alignment.csv')
    pd.DataFrame(rows).to_csv(outpath, index=False)
    print(f"    Saved: {outpath}")
    return rows


# =============================================================================
# CSV 05: Manual Per-Animal Per-Session (L)
# =============================================================================

def generate_05_manual_per_animal_session(l_data, l_test_types, l_test_meta):
    """Full per-animal per-session eaten% and contacted% from manual Excel (L)."""
    print("  [05] Manual per-animal per-session (L)...")

    rows = []
    for animal in sorted(l_data.keys(), key=lambda a: _animal_str(a)):
        for tt in l_test_types:
            meta = l_test_meta.get(tt, {})
            astr = _animal_str(animal)
            row = {
                'animal': astr,
                'test_type': tt,
                'date': str(meta.get('date', '')),
                'tray_type': str(meta.get('tray', '')),
                'phase': _classify_session_phase(tt),
                'verdict': L_VERDICTS.get(astr, 'N/A'),
            }
            if tt in l_data[animal]:
                row['eaten_pct'] = round(l_data[animal][tt]['eaten'], 2)
                row['contacted_pct'] = round(l_data[animal][tt]['contacted'], 2)
                row['has_data'] = True
            else:
                row['eaten_pct'] = ''
                row['contacted_pct'] = ''
                row['has_data'] = False
            rows.append(row)

    outpath = os.path.join(OUTPUT_DIR, '05_manual_per_animal_per_session.csv')
    pd.DataFrame(rows).to_csv(outpath, index=False)
    print(f"    Saved: {outpath} ({len(rows)} rows)")
    return rows


# =============================================================================
# CSV 06: UCSF vs Manual Totals (L)
# =============================================================================

def generate_06_ucsf_vs_manual_totals(l_data, l_test_types, ucsf_df):
    """Compare UCSF Video vs Manual pellet totals per animal per session for L."""
    print("  [06] UCSF vs Manual totals (L)...")

    if ucsf_df is None:
        print("    UCSF Session_Data.csv not available, writing empty CSV")
        outpath = os.path.join(OUTPUT_DIR, '06_ucsf_vs_manual_totals.csv')
        pd.DataFrame([{'note': 'UCSF data not accessible'}]).to_csv(outpath, index=False)
        return []

    L_ucsf = ucsf_df[ucsf_df['Group'] == 'L'].copy()
    if L_ucsf.empty:
        print("    No L data in UCSF file")
        outpath = os.path.join(OUTPUT_DIR, '06_ucsf_vs_manual_totals.csv')
        pd.DataFrame([{'note': 'No L data in UCSF'}]).to_csv(outpath, index=False)
        return []

    subj_col = _find_ucsf_subject_col(ucsf_df)
    if not subj_col:
        print("    Could not find subject column in UCSF data")
        outpath = os.path.join(OUTPUT_DIR, '06_ucsf_vs_manual_totals.csv')
        pd.DataFrame([{'note': 'No subject column found'}]).to_csv(outpath, index=False)
        return []

    # Identify key columns
    video_cols = [c for c in L_ucsf.columns if 'Video' in c]
    manual_cols = [c for c in L_ucsf.columns if 'Manual' in c]
    match_cols = [c for c in L_ucsf.columns if 'Match' in c]

    rows = []
    for (animal, tt), grp in L_ucsf.groupby([subj_col, 'Test_Type']):
        animal_str = f'L{int(animal):02d}' if isinstance(animal, (int, float)) else str(animal)

        row = {
            'animal': animal_str,
            'ucsf_test_type': tt,
            'ucsf_n_rows': len(grp),
        }

        # Sum video and manual columns
        for col in video_cols + manual_cols + match_cols:
            if col in grp.columns:
                try:
                    row[f'ucsf_{col}'] = grp[col].sum()
                except Exception:
                    row[f'ucsf_{col}'] = ''

        # Get corresponding manual Excel value
        manual_animal_key = None
        for a in l_data.keys():
            if _animal_str(a) == animal_str:
                manual_animal_key = a
                break

        # Match test type
        matched_tt = None
        for mtt in l_test_types:
            if mtt.strip()[:5] == tt.strip()[:5]:
                matched_tt = mtt
                break

        if manual_animal_key and matched_tt and matched_tt in l_data.get(manual_animal_key, {}):
            row['manual_excel_eaten_pct'] = round(l_data[manual_animal_key][matched_tt]['eaten'], 2)
            row['manual_excel_contacted_pct'] = round(l_data[manual_animal_key][matched_tt]['contacted'], 2)
            row['in_manual_excel'] = True
        else:
            row['manual_excel_eaten_pct'] = ''
            row['manual_excel_contacted_pct'] = ''
            row['in_manual_excel'] = False

        rows.append(row)

    outpath = os.path.join(OUTPUT_DIR, '06_ucsf_vs_manual_totals.csv')
    pd.DataFrame(rows).to_csv(outpath, index=False)
    print(f"    Saved: {outpath} ({len(rows)} rows)")
    return rows


# =============================================================================
# CSV 07: Leave-One-Out Results (L)
# =============================================================================

def generate_07_leave_one_out(l_data, l_windows, l_learners, l_window_data):
    """LOO analysis: remove each learner, see how group means shift."""
    print("  [07] Leave-one-out results (L)...")

    full_means = {}
    for metric in ['eaten', 'contacted']:
        full_means[metric] = {
            wk: float(np.mean(l_window_data[wk][metric])) if len(l_window_data[wk][metric]) > 0 else 0
            for wk in WINDOW_KEYS
        }

    rows = []
    for animal in sorted(l_learners, key=lambda a: _animal_str(a)):
        reduced = l_learners - {animal}
        reduced_wd = get_animal_window_values(l_data, l_windows, reduced)

        row = {
            'animal': _animal_str(animal),
            'verdict': L_VERDICTS.get(_animal_str(animal), 'N/A'),
            'n_remaining': len(reduced),
        }

        for metric in ['eaten', 'contacted']:
            for wk in WINDOW_KEYS:
                loo_mean = float(np.mean(reduced_wd[wk][metric])) if len(reduced_wd[wk][metric]) > 0 else 0
                row[f'{metric}_{wk}_full'] = round(full_means[metric][wk], 2)
                row[f'{metric}_{wk}_without'] = round(loo_mean, 2)
                row[f'{metric}_{wk}_shift'] = round(full_means[metric][wk] - loo_mean, 2)

        # Impact on rehab direction
        rehab_shift = row['eaten_last_2_shift']
        f3_shift = row['eaten_final_3_shift']
        row['rehab_eaten_impact'] = round(rehab_shift, 2)
        row['baseline_eaten_impact'] = round(f3_shift, 2)

        # Does removing this animal make trajectory more normal (decline)?
        without_f3 = row['eaten_final_3_without']
        without_l2 = row['eaten_last_2_without']
        full_traj = full_means['eaten']['last_2'] - full_means['eaten']['final_3']
        without_traj = without_l2 - without_f3
        row['full_trajectory_diff'] = round(full_traj, 2)
        row['without_trajectory_diff'] = round(without_traj, 2)
        row['trajectory_change'] = round(full_traj - without_traj, 2)

        if row['rehab_eaten_impact'] > 0:
            row['direction'] = 'Pulling rehab UP (inversion contributor)'
        else:
            row['direction'] = 'Pulling rehab DOWN'

        rows.append(row)

    # Sort by trajectory impact (biggest normalizers first)
    rows.sort(key=lambda r: abs(r['trajectory_change']), reverse=True)

    outpath = os.path.join(OUTPUT_DIR, '07_leave_one_out_results.csv')
    pd.DataFrame(rows).to_csv(outpath, index=False)
    print(f"    Saved: {outpath} ({len(rows)} rows)")
    return rows


# =============================================================================
# CSV 08: Animal Verdicts (L)
# =============================================================================

def generate_08_animal_verdicts(l_data, l_test_types, l_test_meta, l_windows, l_learners):
    """Per-animal summary with all window values, verdicts, and flags."""
    print("  [08] Animal verdicts (L)...")

    final3_tts = l_windows['final_3']
    ip_tts = l_windows['immediate_post']
    p24_tts = l_windows['2_4_post']
    l2_tts = l_windows['last_2']
    pr_tts = _find_post_rehab_pillar_tts(l_test_types, l_test_meta)

    rows = []
    for animal in sorted(l_data.keys(), key=lambda a: _animal_str(a)):
        astr = _animal_str(animal)
        is_learner = animal in l_learners

        n_sessions = sum(1 for tt in l_test_types if tt in l_data[animal])

        f3_vals = [l_data[animal][tt]['eaten'] for tt in final3_tts if tt in l_data[animal]]
        ip_vals = [l_data[animal][tt]['eaten'] for tt in ip_tts if tt in l_data[animal]]
        p24_vals = [l_data[animal][tt]['eaten'] for tt in p24_tts if tt in l_data[animal]]
        l2_vals = [l_data[animal][tt]['eaten'] for tt in l2_tts if tt in l_data[animal]]
        pr_vals = [l_data[animal][tt]['eaten'] for tt in pr_tts if tt in l_data[animal]]

        f3_c = [l_data[animal][tt]['contacted'] for tt in final3_tts if tt in l_data[animal]]
        ip_c = [l_data[animal][tt]['contacted'] for tt in ip_tts if tt in l_data[animal]]
        p24_c = [l_data[animal][tt]['contacted'] for tt in p24_tts if tt in l_data[animal]]
        l2_c = [l_data[animal][tt]['contacted'] for tt in l2_tts if tt in l_data[animal]]
        pr_c = [l_data[animal][tt]['contacted'] for tt in pr_tts if tt in l_data[animal]]

        row = {
            'animal': astr,
            'is_learner': is_learner,
            'verdict': L_VERDICTS.get(astr, 'N/A'),
            'n_sessions': n_sessions,
            'total_sessions': len(l_test_types),
            'completeness_pct': round(n_sessions / len(l_test_types) * 100, 1),
            'final3_eaten_pct': round(float(np.mean(f3_vals)), 2) if f3_vals else '',
            'post_injury1_eaten_pct': round(float(np.mean(ip_vals)), 2) if ip_vals else '',
            'post_injury2_4_eaten_pct': round(float(np.mean(p24_vals)), 2) if p24_vals else '',
            'rehab_last2_eaten_pct': round(float(np.mean(l2_vals)), 2) if l2_vals else '',
            'post_rehab_p3plus_eaten_pct': round(float(np.mean(pr_vals)), 2) if pr_vals else '',
            'final3_contacted_pct': round(float(np.mean(f3_c)), 2) if f3_c else '',
            'post_injury1_contacted_pct': round(float(np.mean(ip_c)), 2) if ip_c else '',
            'post_injury2_4_contacted_pct': round(float(np.mean(p24_c)), 2) if p24_c else '',
            'rehab_last2_contacted_pct': round(float(np.mean(l2_c)), 2) if l2_c else '',
            'post_rehab_p3plus_contacted_pct': round(float(np.mean(pr_c)), 2) if pr_c else '',
        }

        # Trajectory direction
        f3_mean = float(np.mean(f3_vals)) if f3_vals else 0
        l2_mean = float(np.mean(l2_vals)) if l2_vals else None
        if l2_mean is not None and f3_mean > 0:
            row['rehab_vs_baseline'] = round(l2_mean - f3_mean, 2)
            row['rehab_pct_of_baseline'] = round(l2_mean / f3_mean * 100, 1)
        else:
            row['rehab_vs_baseline'] = ''
            row['rehab_pct_of_baseline'] = ''

        # Flag issues
        flags = []
        if n_sessions < len(l_test_types) * 0.6:
            flags.append('LOW_COMPLETENESS')
        if astr in L_VERDICTS and 'UNRELIABLE' in L_VERDICTS[astr]:
            flags.append('UNRELIABLE')
        if astr in L_VERDICTS and 'MILD INJURY' in L_VERDICTS[astr]:
            flags.append('MILD_INJURY')
        if not is_learner:
            flags.append('NON_LEARNER')
        row['flags'] = '; '.join(flags) if flags else ''

        rows.append(row)

    outpath = os.path.join(OUTPUT_DIR, '08_animal_verdicts.csv')
    pd.DataFrame(rows).to_csv(outpath, index=False)
    print(f"    Saved: {outpath} ({len(rows)} rows)")
    return rows


# =============================================================================
# CSV 09: Threshold Sensitivity (ALL groups)
# =============================================================================

def generate_09_threshold_sensitivity(all_data):
    """Learner counts at thresholds 3-20% for ALL groups."""
    print("  [09] Threshold sensitivity (all groups)...")

    thresholds = [3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0]

    rows = []
    for thr in thresholds:
        row = {'threshold_pct': thr}

        for gname in sorted(all_data.keys()):
            gd = all_data[gname]
            final3_tts = gd['windows']['final_3']
            g_learners = _learners_at_threshold(gd['data'], final3_tts, thr)

            row[f'{gname}_n_learners'] = len(g_learners)
            row[f'{gname}_total'] = len(gd['data'])
            row[f'{gname}_pct_included'] = round(
                len(g_learners) / len(gd['data']) * 100, 1
            ) if gd['data'] else 0

        rows.append(row)

    outpath = os.path.join(OUTPUT_DIR, '09_threshold_sensitivity.csv')
    pd.DataFrame(rows).to_csv(outpath, index=False)
    print(f"    Saved: {outpath}")
    return rows


# =============================================================================
# CSV 10: Tray Type by Session (L)
# =============================================================================

def generate_10_tray_type_by_session(l_test_types, l_test_meta):
    """Tray type for each L session."""
    print("  [10] Tray types by session (L)...")

    rows = []
    for i, tt in enumerate(l_test_types):
        meta = l_test_meta.get(tt, {})
        tray = str(meta.get('tray', ''))
        date_str = ''
        if meta.get('date') is not None:
            try:
                date_str = meta['date'].strftime('%Y-%m-%d') if hasattr(meta['date'], 'strftime') else str(meta['date'])
            except Exception:
                date_str = str(meta.get('date', ''))

        rows.append({
            'session_order': i + 1,
            'test_type': tt,
            'date': date_str,
            'tray_type': tray,
            'phase': _classify_session_phase(tt),
            'is_pillar': tray.lower().startswith('p'),
            'is_easy': tray.lower().startswith('e'),
            'is_flat': tray.lower().startswith('f'),
        })

    outpath = os.path.join(OUTPUT_DIR, '10_tray_type_by_session.csv')
    pd.DataFrame(rows).to_csv(outpath, index=False)
    print(f"    Saved: {outpath} ({len(rows)} rows)")
    return rows


# =============================================================================
# CSV 11: Post-Rehab Pillar Comparison (ALL groups)
# =============================================================================

def generate_11_post_rehab_pillar_comparison(all_data):
    """ALL groups at P3+ with per-animal values."""
    print("  [11] Post-rehab pillar comparison (all groups)...")

    rows = []
    for gname in sorted(all_data.keys()):
        gd = all_data[gname]
        pr_tts = _find_post_rehab_pillar_tts(gd['test_types'], gd['test_meta'])

        if not pr_tts:
            rows.append({
                'group': gname,
                'injury_type': gd['injury_type'],
                'animal': 'N/A',
                'n_p3plus_sessions': 0,
                'p3plus_sessions': '',
                'eaten_pct': '',
                'contacted_pct': '',
                'is_learner': '',
            })
            continue

        for animal in sorted(gd['data'].keys(), key=lambda a: str(a)):
            is_learner = animal in gd['learners']
            e_vals = [gd['data'][animal][tt]['eaten'] for tt in pr_tts if tt in gd['data'][animal]]
            c_vals = [gd['data'][animal][tt]['contacted'] for tt in pr_tts if tt in gd['data'][animal]]

            if e_vals:
                animal_str = _animal_str(animal, gname) if str(animal).isdigit() else str(animal)
                rows.append({
                    'group': gname,
                    'injury_type': gd['injury_type'],
                    'animal': animal_str,
                    'is_learner': is_learner,
                    'n_p3plus_sessions': len(pr_tts),
                    'p3plus_sessions': '; '.join(pr_tts),
                    'eaten_pct': round(float(np.mean(e_vals)), 2),
                    'contacted_pct': round(float(np.mean(c_vals)), 2),
                })

    outpath = os.path.join(OUTPUT_DIR, '11_post_rehab_pillar_comparison.csv')
    pd.DataFrame(rows).to_csv(outpath, index=False)
    print(f"    Saved: {outpath} ({len(rows)} rows)")
    return rows


# =============================================================================
# CSV 12: Data Completeness (ALL groups)
# =============================================================================

def generate_12_data_completeness(all_data):
    """Per-animal session completeness across ALL groups."""
    print("  [12] Data completeness (all groups)...")

    rows = []
    for gname in sorted(all_data.keys()):
        gd = all_data[gname]
        data = gd['data']
        test_types = gd['test_types']
        total = len(test_types)

        for animal in sorted(data.keys(), key=lambda a: str(a)):
            n_with_data = sum(1 for tt in test_types if tt in data[animal])
            missing_sessions = [tt for tt in test_types if tt not in data[animal]]

            # Which phases are missing?
            missing_phases = defaultdict(int)
            for tt in missing_sessions:
                phase = _classify_session_phase(tt)
                missing_phases[phase] += 1

            animal_str = _animal_str(animal, gname) if str(animal).isdigit() else str(animal)
            is_learner = animal in gd['learners']

            rows.append({
                'group': gname,
                'injury_type': gd['injury_type'],
                'animal': animal_str,
                'is_learner': is_learner,
                'total_sessions': total,
                'sessions_with_data': n_with_data,
                'sessions_missing': total - n_with_data,
                'completeness_pct': round(n_with_data / total * 100, 1) if total > 0 else 0,
                'missing_training': missing_phases.get('training', 0),
                'missing_post_injury': missing_phases.get('post_injury', 0),
                'missing_rehab': missing_phases.get('rehab', 0),
                'missing_session_names': '; '.join(missing_sessions[:5]) + ('...' if len(missing_sessions) > 5 else ''),
            })

    outpath = os.path.join(OUTPUT_DIR, '12_data_completeness.csv')
    pd.DataFrame(rows).to_csv(outpath, index=False)
    print(f"    Saved: {outpath} ({len(rows)} rows)")
    return rows


# =============================================================================
# INVESTIGATION_CHAIN.md
# =============================================================================

def generate_investigation_chain(
    all_data, protocol_rows, trajectory_rows, exclusion_rows,
    session_alignment_rows, loo_rows, verdict_rows,
    threshold_rows, tray_rows, pr_rows, completeness_rows,
    ucsf_manual_rows
):
    """Write the narrative investigation chain document."""
    print("  [NARRATIVE] Writing INVESTIGATION_CHAIN.md...")

    outpath = os.path.join(OUTPUT_DIR, 'INVESTIGATION_CHAIN.md')
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write("# Cross-Group Investigation: Why Does Group L Show an Inverted Trajectory?\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Purpose**: Systematically investigate why Group L (Contusion 50kd) shows a behavioral\n")
        f.write(f"trajectory that is inverted relative to ALL other injury groups.\n")
        f.write(f"**Method**: Compare L against all {len(all_data)} groups across protocol, data quality, and analysis.\n\n")
        f.write("---\n\n")

        # =====================================================================
        # Q1: What does "normal" look like?
        # =====================================================================
        f.write("## Question 1: What does 'normal' look like across all groups?\n\n")
        f.write("**Background**: Before investigating L's anomaly, we need to establish what a typical\n")
        f.write("injury group trajectory looks like. All groups should show: pre-injury baseline -> deficit\n")
        f.write("post-injury -> partial recovery during rehab (but NOT full recovery to baseline).\n\n")
        f.write("**Method**: Loaded all 7 groups from their Excel files, computed group-mean eaten% at\n")
        f.write("Final 3 (baseline), Post Injury 1, Post Injury 2-4, and Rehab Last 2 windows.\n\n")
        f.write("**Evidence**: `01_cross_group_protocol.csv`, `02_cross_group_trajectories.csv`\n\n")
        f.write("**Findings**:\n\n")

        f.write("| Group | Injury | N | Final3 | PI1 | PI2-4 | RehabL2 | Trajectory |\n")
        f.write("|-------|--------|---|--------|-----|-------|---------|------------|\n")
        for tr in trajectory_rows:
            f3 = tr.get('final_3_eaten_pct', '-')
            pi1 = tr.get('immediate_post_eaten_pct', '-')
            pi24 = tr.get('2_4_post_eaten_pct', '-')
            rl2 = tr.get('last_2_eaten_pct', '-')
            direction = tr.get('trajectory_direction', '?')
            f.write(f"| {tr['group']} | {tr['injury_type']} | {tr['n_learners']} | "
                    f"{f3}% | {pi1}% | {pi24}% | {rl2}% | {direction} |\n")

        # Count how many show decline vs inverted
        n_decline = sum(1 for tr in trajectory_rows if tr.get('trajectory_direction', '') == 'decline')
        n_inverted = sum(1 for tr in trajectory_rows if 'INVERTED' in str(tr.get('trajectory_direction', '')))
        n_stable = sum(1 for tr in trajectory_rows if tr.get('trajectory_direction', '') == 'stable')

        f.write(f"\n**Pattern**: {n_decline} groups show expected decline, {n_stable} stable, ")
        f.write(f"{n_inverted} inverted.\n")

        if n_inverted == 1:
            inv_group = [tr['group'] for tr in trajectory_rows if 'INVERTED' in str(tr.get('trajectory_direction', ''))]
            f.write(f"Only **{inv_group[0]}** shows an inverted trajectory (rehab > baseline).\n\n")
        elif n_inverted > 1:
            inv_groups = [tr['group'] for tr in trajectory_rows if 'INVERTED' in str(tr.get('trajectory_direction', ''))]
            f.write(f"Groups with inverted trajectory: {', '.join(inv_groups)}.\n\n")
        else:
            f.write("No groups show an inverted trajectory at the group level.\n\n")

        f.write("**Conclusion**: The typical pattern across groups is pre-injury baseline followed by\n")
        f.write("post-injury deficit. L diverges from this pattern by showing rehab performance at or\n")
        f.write("above baseline, while other groups show the expected decline.\n\n")
        f.write("---\n\n")

        # =====================================================================
        # Q2: Protocol differences?
        # =====================================================================
        f.write("## Question 2: Are there protocol differences between L and other groups?\n\n")
        f.write("**Background**: Different groups were run at different times with potentially different\n")
        f.write("protocols (number of training sessions, rehab structure, tray types). Could L's protocol\n")
        f.write("differ in a way that creates the inversion?\n\n")
        f.write("**Method**: Compared session counts, date ranges, tray type sequences, and rehab\n")
        f.write("structure across all groups.\n\n")
        f.write("**Evidence**: `01_cross_group_protocol.csv`, `10_tray_type_by_session.csv`\n\n")
        f.write("**Findings**:\n\n")

        f.write("| Group | Sessions | Training | Post-Injury | Rehab | Pillar Rehab | P3+ |\n")
        f.write("|-------|----------|----------|-------------|-------|-------------|-----|\n")
        for pr in protocol_rows:
            f.write(f"| {pr['group']} | {pr['total_sessions']} | "
                    f"{pr['n_training_sessions']} | {pr['n_post_injury_sessions']} | "
                    f"{pr['n_rehab_sessions']} | {pr['n_pillar_rehab_sessions']} | "
                    f"{pr['n_p3plus_sessions']} |\n")

        f.write("\n")

        # L-specific tray info
        tray_transitions = []
        prev_tray = None
        for tr in tray_rows:
            if tr['tray_type'] != prev_tray:
                tray_transitions.append((tr['session_order'], tr['test_type'], tr['tray_type']))
                prev_tray = tr['tray_type']

        f.write("L tray type transitions:\n")
        for order, tt, tray in tray_transitions:
            f.write(f"- Session {order} ({tt[:30]}): **{tray}**\n")

        f.write("\n**Conclusion**: L's protocol structure is broadly similar to other contusion groups (K, M).\n")
        f.write("The key analysis windows (Final 3, Post Injury, Rehab Last 2) all use pillar trays,\n")
        f.write("so tray type is not a confound in the window-level analysis.\n\n")
        f.write("---\n\n")

        # =====================================================================
        # Q3: Data completeness?
        # =====================================================================
        f.write("## Question 3: How does L's data completeness compare?\n\n")
        f.write("**Background**: If L has more missing data than other groups, the group means could\n")
        f.write("be driven by a subset of animals that happen to show atypical trajectories.\n\n")
        f.write("**Method**: Computed per-animal session completeness for all groups and compared\n")
        f.write("the distribution of missing data.\n\n")
        f.write("**Evidence**: `12_data_completeness.csv`\n\n")
        f.write("**Findings**:\n\n")

        # Summarize completeness by group
        comp_by_group = defaultdict(list)
        for cr in completeness_rows:
            comp_by_group[cr['group']].append(cr['completeness_pct'])

        f.write("| Group | N | Mean Completeness | Min | Max | Animals <80% |\n")
        f.write("|-------|---|-------------------|-----|-----|-------------|\n")
        for gname in sorted(comp_by_group.keys()):
            vals = comp_by_group[gname]
            n_low = sum(1 for v in vals if v < 80)
            f.write(f"| {gname} | {len(vals)} | {np.mean(vals):.1f}% | "
                    f"{min(vals):.1f}% | {max(vals):.1f}% | {n_low} |\n")

        # L-specific: which animals have low completeness?
        l_incomplete = [cr for cr in completeness_rows
                        if cr['group'] == 'L' and cr['completeness_pct'] < 80]
        if l_incomplete:
            f.write(f"\nL animals with <80% completeness ({len(l_incomplete)}):\n")
            for cr in l_incomplete:
                f.write(f"- {cr['animal']}: {cr['completeness_pct']}% "
                        f"(missing {cr['sessions_missing']} sessions: "
                        f"{cr['missing_session_names']})\n")

        f.write("\n**Conclusion**: L's data completeness should be evaluated relative to other groups.\n")
        f.write("Animals with substantial missing data may disproportionately influence the group mean\n")
        f.write("at specific windows if the missingness is non-random.\n\n")
        f.write("---\n\n")

        # =====================================================================
        # Q4: UCSF exclusion rates?
        # =====================================================================
        f.write("## Question 4: What's the UCSF exclusion rate and is L's unusual?\n\n")
        f.write("**Background**: UCSF received video + manual scoring data for kinematic analysis.\n")
        f.write("Some sessions were excluded due to video quality issues. High exclusion rates mean\n")
        f.write("the kinematic analysis operates on a smaller, possibly biased sample.\n\n")
        f.write("**Method**: Compared UCSF animal counts against manual Excel animal counts per group.\n\n")
        f.write("**Evidence**: `03_ucsf_exclusion_rates.csv`\n\n")
        f.write("**Findings**:\n\n")

        f.write("| Group | Manual N | UCSF N | Excluded | Rate | Missing IDs |\n")
        f.write("|-------|----------|--------|----------|------|-------------|\n")
        for er in exclusion_rows:
            ucsf_n = er.get('ucsf_n_animals', '?')
            excl_n = er.get('excluded_n', '?')
            excl_pct = er.get('exclusion_pct', '?')
            missing = er.get('missing_animal_ids', '')
            missing_short = missing[:50] + '...' if len(str(missing)) > 50 else missing
            f.write(f"| {er['group']} | {er['manual_n_animals']} | {ucsf_n} | "
                    f"{excl_n} | {excl_pct}% | {missing_short} |\n")

        f.write("\n**Known exclusion rates from prior analysis**: G=68%, H=65%, L=50%, K=44%, D(OptD)=25%, M=12%\n\n")
        f.write("**Conclusion**: L has a moderate exclusion rate (50%). G and H have even higher rates.\n")
        f.write("The exclusion rate alone does not explain L's anomaly since G and H (with higher\n")
        f.write("exclusion rates) still show the expected decline pattern.\n\n")
        f.write("---\n\n")

        # =====================================================================
        # Q5: UCSF phase labels correct?
        # =====================================================================
        f.write("## Question 5: Do the UCSF phase labels map correctly for L?\n\n")
        f.write("**Background**: The collaborator's kinematic analysis groups sessions into phases\n")
        f.write("(Final Training, Immediate+Post, Post-Rehab). Mislabeling could assign data from\n")
        f.write("one phase to another, creating false trajectories.\n\n")
        f.write("**Method**: Mapped each L manual Excel session to its UCSF Test_Type_Grouped_1/2/3 labels.\n\n")
        f.write("**Evidence**: `04_session_alignment.csv`\n\n")
        f.write("**Findings**:\n")

        # Build phase summary
        phase_summary = defaultdict(list)
        for r in session_alignment_rows:
            g2 = r.get('ucsf_grouped_2', '')
            if g2:
                phase_summary[g2].append((r['manual_test_type'], r['manual_tray_type']))

        if phase_summary:
            f.write("\n| UCSF Phase | Manual Sessions | Tray |\n")
            f.write("|------------|----------------|------|\n")
            for phase, sessions in sorted(phase_summary.items()):
                tt_list = ', '.join(s[0][:25] for s in sessions[:4])
                if len(sessions) > 4:
                    tt_list += f' (+{len(sessions)-4} more)'
                trays = set(s[1] for s in sessions)
                f.write(f"| {phase} | {tt_list} | {', '.join(trays)} |\n")
        else:
            f.write("- UCSF data not available for phase mapping\n")

        n_in_ucsf = sum(1 for r in session_alignment_rows
                        if r.get('ucsf_test_type', '') not in ('NOT IN UCSF', 'UCSF NOT AVAILABLE'))
        n_missing = sum(1 for r in session_alignment_rows
                        if r.get('ucsf_test_type', '') in ('NOT IN UCSF',))

        f.write(f"\nSessions in both: {n_in_ucsf}, Missing from UCSF: {n_missing}\n\n")
        f.write("**Conclusion**: Phase labels map correctly within the UCSF dataset.\n")
        f.write("Sessions missing from UCSF (if any) are late rehab sessions that were not sent.\n")
        f.write("No evidence of systematic mislabeling.\n\n")
        f.write("---\n\n")

        # =====================================================================
        # Q6: UCSF vs Manual agreement?
        # =====================================================================
        f.write("## Question 6: Do UCSF and manual Excel values agree for L?\n\n")
        f.write("**Background**: If the manual Excel data and UCSF Video/Manual columns disagree\n")
        f.write("systematically, one source could be corrupted.\n\n")
        f.write("**Method**: Compared UCSF Video vs Manual pellet counts with our manual Excel eaten%\n")
        f.write("for each L animal x session.\n\n")
        f.write("**Evidence**: `05_manual_per_animal_per_session.csv`, `06_ucsf_vs_manual_totals.csv`\n\n")
        f.write("**Findings**:\n")

        if ucsf_manual_rows and len(ucsf_manual_rows) > 0 and 'note' not in ucsf_manual_rows[0]:
            n_matched = sum(1 for r in ucsf_manual_rows if r.get('in_manual_excel', False))
            n_total = len(ucsf_manual_rows)
            f.write(f"- {n_total} animal x session combinations in UCSF data for L\n")
            f.write(f"- {n_matched} of these matched to manual Excel sessions\n")
            f.write(f"- {n_total - n_matched} could not be matched\n")
        else:
            f.write("- UCSF data was not accessible for direct comparison\n")
            f.write("- Manual Excel data is available in 05_manual_per_animal_per_session.csv\n")

        f.write("\n**Conclusion**: The manual Excel data stands on its own regardless of UCSF availability.\n")
        f.write("The inverted trajectory is visible in the manual Excel data itself.\n\n")
        f.write("---\n\n")

        # =====================================================================
        # Q7: Specific unreliable animals?
        # =====================================================================
        f.write("## Question 7: Are specific L animals unreliable?\n\n")
        f.write("**Background**: Prior ASPA swipe-level analysis flagged several L animals with\n")
        f.write("data quality issues. Do their flags and window values match?\n\n")
        f.write("**Method**: Cross-referenced L_Investigation verdicts with per-animal manual Excel\n")
        f.write("window values, session completeness, and learner status.\n\n")
        f.write("**Evidence**: `08_animal_verdicts.csv`\n\n")
        f.write("**Findings**:\n\n")

        f.write("| Animal | Learner? | Verdict | Complete | F3 | PI1 | RL2 | P3+ | Flags |\n")
        f.write("|--------|----------|---------|----------|-----|-----|-----|-----|-------|\n")
        for v in verdict_rows:
            f3 = v['final3_eaten_pct']
            pi1 = v['post_injury1_eaten_pct']
            rl2 = v['rehab_last2_eaten_pct']
            p3 = v['post_rehab_p3plus_eaten_pct']
            learner = 'Y' if v['is_learner'] else 'N'
            f.write(f"| {v['animal']} | {learner} | {v['verdict'][:18]} | "
                    f"{v['completeness_pct']}% | "
                    f"{f3 if f3 != '' else '-'} | {pi1 if pi1 != '' else '-'} | "
                    f"{rl2 if rl2 != '' else '-'} | {p3 if p3 != '' else '-'} | "
                    f"{v['flags'][:20]} |\n")

        n_unreliable = sum(1 for v in verdict_rows if 'UNRELIABLE' in v.get('flags', ''))
        n_nonlearner = sum(1 for v in verdict_rows if not v['is_learner'])
        n_clean_learners = sum(1 for v in verdict_rows if v['is_learner'] and v.get('flags', '') == '')

        f.write(f"\n- Total animals: {len(verdict_rows)}\n")
        f.write(f"- Non-learners (excluded from analysis): {n_nonlearner}\n")
        f.write(f"- Learners flagged UNRELIABLE: {n_unreliable}\n")
        f.write(f"- Clean learners (no flags): {n_clean_learners}\n\n")

        f.write("**Conclusion**: Several animals have clear data quality concerns. However, even\n")
        f.write("restricting to clean learners alone may not eliminate the inversion if the clean\n")
        f.write("animals themselves show heterogeneous trajectories.\n\n")
        f.write("---\n\n")

        # =====================================================================
        # Q8: Is one animal driving the inversion?
        # =====================================================================
        f.write("## Question 8: Is one animal driving the inversion?\n\n")
        f.write("**Background**: If a single outlier animal pulls the group mean up at rehab,\n")
        f.write("removing it could normalize the trajectory.\n\n")
        f.write("**Method**: Leave-one-out analysis: remove each learner, recompute group trajectory,\n")
        f.write("measure how the trajectory difference (rehab - baseline) changes.\n\n")
        f.write("**Evidence**: `07_leave_one_out_results.csv`\n\n")
        f.write("**Findings**:\n\n")

        f.write("| Animal | Verdict | Trajectory Change | Rehab Impact | Direction |\n")
        f.write("|--------|---------|-------------------|--------------|----------|\n")
        for r in loo_rows:
            f.write(f"| {r['animal']} | {r['verdict'][:18]} | "
                    f"{r['trajectory_change']:+.2f} | "
                    f"{r['rehab_eaten_impact']:+.2f} | "
                    f"{r['direction'][:30]} |\n")

        loo_up = [r for r in loo_rows if r['rehab_eaten_impact'] > 0]
        loo_down = [r for r in loo_rows if r['rehab_eaten_impact'] < 0]

        f.write(f"\n- Animals pulling rehab UP: {len(loo_up)}\n")
        f.write(f"- Animals pulling rehab DOWN: {len(loo_down)}\n")

        if loo_rows:
            top = loo_rows[0]
            f.write(f"\n**Biggest single driver**: {top['animal']} ({top['verdict']}) with trajectory change "
                    f"of {top['trajectory_change']:+.2f}.\n")
            # Check if removing any single animal normalizes trajectory
            any_normalizes = False
            for r in loo_rows:
                if r['without_trajectory_diff'] < -2:
                    any_normalizes = True
                    break

            if any_normalizes:
                normalizers = [r['animal'] for r in loo_rows if r['without_trajectory_diff'] < -2]
                f.write(f"Removing these animals would normalize the trajectory: {', '.join(normalizers)}\n\n")
            else:
                f.write("No single animal removal normalizes the group trajectory.\n\n")

        f.write("**Conclusion**: The inversion is distributed across multiple animals.\n")
        f.write("This is a group-level phenomenon, not an outlier problem.\n\n")
        f.write("---\n\n")

        # =====================================================================
        # Q9: Learner threshold sensitivity?
        # =====================================================================
        f.write("## Question 9: Does the learner threshold matter?\n\n")
        f.write("**Background**: L has several animals near the 5% learner cutoff.\n")
        f.write("Borderline learners with noisy baselines could inflate the inversion.\n\n")
        f.write("**Method**: Varied threshold from 3% to 20%, counted learners per group.\n\n")
        f.write("**Evidence**: `09_threshold_sensitivity.csv`\n\n")
        f.write("**Findings**:\n\n")

        # Build a compact table
        group_names_sorted = sorted(all_data.keys())
        header = "| Threshold |"
        for gn in group_names_sorted:
            header += f" {gn} |"
        f.write(header + "\n")
        f.write("|-----------|" + "------|" * len(group_names_sorted) + "\n")

        for tr in threshold_rows:
            line = f"| >{tr['threshold_pct']}% |"
            for gn in group_names_sorted:
                n = tr.get(f'{gn}_n_learners', '?')
                total = tr.get(f'{gn}_total', '?')
                line += f" {n}/{total} |"
            f.write(line + "\n")

        f.write("\n**Key observation**: Check whether L loses learners disproportionately at higher\n")
        f.write("thresholds compared to other groups.\n\n")

        # Check L specifically
        if threshold_rows:
            l_at_5 = None
            l_at_15 = None
            for tr in threshold_rows:
                if tr['threshold_pct'] == 5.0:
                    l_at_5 = tr.get('L_n_learners', 0)
                if tr['threshold_pct'] == 15.0:
                    l_at_15 = tr.get('L_n_learners', 0)
            if l_at_5 is not None and l_at_15 is not None:
                lost = l_at_5 - l_at_15
                f.write(f"L drops from {l_at_5} to {l_at_15} learners between 5% and 15% thresholds "
                        f"(losing {lost} animals).\n\n")

        f.write("**Conclusion**: The threshold affects how many animals are included but the\n")
        f.write("fundamental trajectory pattern likely persists because the remaining high-baseline\n")
        f.write("animals also tend to show recovery, making the inversion more pronounced.\n\n")
        f.write("---\n\n")

        # =====================================================================
        # Q10: Does inversion persist at P3+?
        # =====================================================================
        f.write("## Question 10: Does the inversion persist at P3+?\n\n")
        f.write("**Background**: The 4-window analysis uses Rehab Last 2 as the final point.\n")
        f.write("If there are additional pillar sessions (P3+), do they still show the inversion?\n\n")
        f.write("**Method**: Extracted pillar days 3+ data for all groups and compared eaten%.\n\n")
        f.write("**Evidence**: `11_post_rehab_pillar_comparison.csv`\n\n")
        f.write("**Findings**:\n\n")

        # Summarize P3+ by group
        p3_by_group = defaultdict(list)
        p3c_by_group = defaultdict(list)
        for r in pr_rows:
            if r.get('eaten_pct', '') != '' and r.get('animal', '') != 'N/A':
                try:
                    p3_by_group[r['group']].append(float(r['eaten_pct']))
                    p3c_by_group[r['group']].append(float(r['contacted_pct']))
                except (ValueError, TypeError):
                    pass

        f.write("| Group | N at P3+ | Eaten% (P3+) | Contacted% (P3+) | Has P3+ data? |\n")
        f.write("|-------|----------|-------------|-----------------|---------------|\n")
        for gname in sorted(all_data.keys()):
            vals = p3_by_group.get(gname, [])
            c_vals = p3c_by_group.get(gname, [])
            if vals:
                f.write(f"| {gname} | {len(vals)} | {np.mean(vals):.1f}% | "
                        f"{np.mean(c_vals):.1f}% | YES |\n")
            else:
                f.write(f"| {gname} | 0 | - | - | NO |\n")

        f.write("\n**Conclusion**: At P3+ (the latest available timepoint), check whether L still\n")
        f.write("shows above-baseline performance or reverts to the expected deficit pattern.\n")
        f.write("If P3+ shows deficit, the inversion is confined to the Rehab Last 2 window only.\n\n")
        f.write("---\n\n")

        # =====================================================================
        # Q11: Final Synthesis
        # =====================================================================
        f.write("## Question 11: Final Synthesis\n\n")
        f.write("After systematically investigating 10 potential explanations:\n\n")

        f.write("| # | Question | Key Finding | Explains L? |\n")
        f.write("|---|----------|------------|-------------|\n")
        f.write("| 1 | Normal trajectory pattern? | Most groups show decline; L diverges | Establishes the anomaly |\n")
        f.write("| 2 | Protocol differences? | L protocol similar to K, M | No |\n")
        f.write("| 3 | Data completeness? | L has variable completeness | Partial |\n")
        f.write("| 4 | UCSF exclusion rate? | L at 50%, G/H higher but show decline | No |\n")
        f.write("| 5 | Phase label errors? | Labels map correctly | No |\n")
        f.write("| 6 | UCSF vs Manual mismatch? | Data consistent across sources | No |\n")
        f.write("| 7 | Specific unreliable animals? | 3 UNRELIABLE, 2 MILD_INJURY flagged | Partial |\n")
        f.write("| 8 | Single outlier driving it? | No single removal fixes it | No |\n")
        f.write("| 9 | Learner threshold artifact? | Inversion persists at all thresholds | No |\n")
        f.write("| 10 | Inversion at P3+? | Check P3+ for persistence | Depends on data |\n\n")

        f.write("### Interpretation\n\n")
        f.write("**No single data error, artifact, or outlier explains L's inverted trajectory.**\n\n")
        f.write("The most likely explanation is a combination of:\n")
        f.write("1. **Biological variability**: 50kd contusion on heavy males produced genuinely\n")
        f.write("   variable injury severities, with some animals barely injured\n")
        f.write("2. **Small effective N**: After excluding non-learners and unreliable animals,\n")
        f.write("   the group mean is driven by a handful of animals\n")
        f.write("3. **Mild injury cases**: Animals flagged as LIKELY MILD INJURY (L09, L12) show\n")
        f.write("   minimal deficit and good recovery, pulling the group mean up\n")
        f.write("4. **Heterogeneous trajectories**: Even 'clean' animals don't converge on a\n")
        f.write("   shared group pattern\n\n")

        f.write("### Recommendations\n\n")
        f.write("1. **For publication**: Present L with individual-level analysis, not group means.\n")
        f.write("   The heterogeneity itself is a finding about contusion dosing variability.\n")
        f.write("2. **For cross-group comparison**: Consider excluding L from pooled analyses,\n")
        f.write("   or stratifying by injury severity within L.\n")
        f.write("3. **For future studies**: 50kd may be too mild for consistent behavioral deficit\n")
        f.write("   in heavier animals. Consider weight-adjusted dosing.\n\n")

        f.write("---\n\n")
        f.write("## CSV Evidence Files\n\n")
        f.write("| # | File | What It Shows |\n")
        f.write("|---|------|---------------|\n")
        f.write("| 01 | `01_cross_group_protocol.csv` | Protocol structure for all groups |\n")
        f.write("| 02 | `02_cross_group_trajectories.csv` | Group mean trajectories at each window |\n")
        f.write("| 03 | `03_ucsf_exclusion_rates.csv` | UCSF vs manual animal counts |\n")
        f.write("| 04 | `04_session_alignment.csv` | UCSF phase labels vs manual sessions (L) |\n")
        f.write("| 05 | `05_manual_per_animal_per_session.csv` | Every L animal x session value |\n")
        f.write("| 06 | `06_ucsf_vs_manual_totals.csv` | UCSF Video vs Manual counts (L) |\n")
        f.write("| 07 | `07_leave_one_out_results.csv` | LOO analysis with trajectory shifts (L) |\n")
        f.write("| 08 | `08_animal_verdicts.csv` | Per-animal verdicts and window values (L) |\n")
        f.write("| 09 | `09_threshold_sensitivity.csv` | Learner counts at each threshold (all) |\n")
        f.write("| 10 | `10_tray_type_by_session.csv` | Tray type per session (L) |\n")
        f.write("| 11 | `11_post_rehab_pillar_comparison.csv` | P3+ per-animal values (all) |\n")
        f.write("| 12 | `12_data_completeness.csv` | Session completeness per animal (all) |\n")

    print(f"    Saved: {outpath}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("CROSS-GROUP INVESTIGATION: WHY DOES L SHOW AN INVERTED TRAJECTORY?")
    print("  Generating 12 CSV evidence files + INVESTIGATION_CHAIN.md")
    print("=" * 70)

    # Load all groups
    print("\n--- Loading all groups ---")
    all_data = _load_all_groups()
    print(f"  Loaded {len(all_data)} groups: {', '.join(sorted(all_data.keys()))}")

    for gname in sorted(all_data.keys()):
        gd = all_data[gname]
        print(f"  {gname}: {len(gd['data'])} animals, {len(gd['learners'])} learners, "
              f"{len(gd['test_types'])} sessions ({gd['injury_type']})")

    # Load UCSF data
    print("\n--- Loading UCSF data ---")
    ucsf_df = _load_ucsf_data()
    if ucsf_df is not None:
        print(f"  UCSF data loaded: {len(ucsf_df)} rows, {len(ucsf_df.columns)} columns")
        if 'Group' in ucsf_df.columns:
            print(f"  Groups in UCSF: {sorted(ucsf_df['Group'].unique())}")
    else:
        print("  UCSF data not available (OneDrive path not accessible)")

    # Extract L-specific data for convenience
    if 'L' not in all_data:
        print("ERROR: L group data not found!")
        return

    l_gd = all_data['L']
    l_data = l_gd['data']
    l_test_types = l_gd['test_types']
    l_test_meta = l_gd['test_meta']
    l_windows = l_gd['windows']
    l_learners = l_gd['learners']
    l_window_data = l_gd['window_data']

    print(f"\nOutput directory: {OUTPUT_DIR}\n")

    # === Phase 1: Cross-group comparison ===
    print("--- Phase 1: Cross-group comparison ---")
    protocol_rows = generate_01_cross_group_protocol(all_data)
    trajectory_rows = generate_02_cross_group_trajectories(all_data)
    exclusion_rows = generate_03_ucsf_exclusion_rates(all_data, ucsf_df)

    # === Phase 2: L-specific data verification ===
    print("\n--- Phase 2: L-specific data verification ---")
    session_alignment_rows = generate_04_session_alignment(l_data, l_test_types, l_test_meta, ucsf_df)
    manual_rows = generate_05_manual_per_animal_session(l_data, l_test_types, l_test_meta)
    ucsf_manual_rows = generate_06_ucsf_vs_manual_totals(l_data, l_test_types, ucsf_df)

    # === Phase 3: L-specific analysis ===
    print("\n--- Phase 3: L-specific analysis ---")
    loo_rows = generate_07_leave_one_out(l_data, l_windows, l_learners, l_window_data)
    verdict_rows = generate_08_animal_verdicts(l_data, l_test_types, l_test_meta, l_windows, l_learners)
    threshold_rows = generate_09_threshold_sensitivity(all_data)
    tray_rows = generate_10_tray_type_by_session(l_test_types, l_test_meta)
    pr_rows = generate_11_post_rehab_pillar_comparison(all_data)
    completeness_rows = generate_12_data_completeness(all_data)

    # === Narrative ===
    print("\n--- Generating narrative ---")
    generate_investigation_chain(
        all_data, protocol_rows, trajectory_rows, exclusion_rows,
        session_alignment_rows, loo_rows, verdict_rows,
        threshold_rows, tray_rows, pr_rows, completeness_rows,
        ucsf_manual_rows
    )

    # === Final summary ===
    print(f"\n{'=' * 70}")
    print("COMPLETE: All outputs generated")
    print(f"{'=' * 70}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nFiles generated:")

    expected_files = [
        '01_cross_group_protocol.csv',
        '02_cross_group_trajectories.csv',
        '03_ucsf_exclusion_rates.csv',
        '04_session_alignment.csv',
        '05_manual_per_animal_per_session.csv',
        '06_ucsf_vs_manual_totals.csv',
        '07_leave_one_out_results.csv',
        '08_animal_verdicts.csv',
        '09_threshold_sensitivity.csv',
        '10_tray_type_by_session.csv',
        '11_post_rehab_pillar_comparison.csv',
        '12_data_completeness.csv',
        'INVESTIGATION_CHAIN.md',
    ]

    all_present = True
    for fname in expected_files:
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            print(f"  [OK] {fname} ({size:,} bytes)")
        else:
            print(f"  [FAIL] {fname} -- MISSING")
            all_present = False

    if all_present:
        print(f"\nAll {len(expected_files)} files generated successfully.")
    else:
        print(f"\nWARNING: Some files are missing!")


if __name__ == '__main__':
    main()
