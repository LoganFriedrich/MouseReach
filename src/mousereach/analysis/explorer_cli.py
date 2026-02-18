"""
CLI for Reach Data Explorer.

Commands:
    mousereach-build-explorer   Build the pre-computed statistics database
    mousereach-explore          Query the database interactively
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


def main_build():
    """Build the explorer database from pipeline data."""
    parser = argparse.ArgumentParser(
        description="Build reach explorer database with pre-computed statistics"
    )
    parser.add_argument(
        '-i', '--input',
        help="Input Excel or CSV file with reach data, or directory with reach JSON files"
    )
    parser.add_argument(
        '-o', '--output',
        help="Output database path (default: reach_explorer.db in input directory)"
    )
    parser.add_argument(
        '--features-dir',
        help="Directory with *_features.json files for kinematic features"
    )

    args = parser.parse_args()

    # Import here to avoid slow startup
    from .explorer import build_explorer_database, ReachExplorer

    # Find input data
    if args.input:
        input_path = Path(args.input)
    else:
        # Default to pipeline processing directory
        from mousereach.config import Paths
        input_path = Paths.PROCESSING

    print("=" * 60)
    print("REACH EXPLORER - BUILD DATABASE")
    print("=" * 60)

    # Load reach data
    if input_path.is_file():
        print(f"\nLoading from file: {input_path}")
        if input_path.suffix == '.xlsx':
            df = pd.read_excel(input_path)
        else:
            df = pd.read_csv(input_path)
        output_dir = input_path.parent
    else:
        print(f"\nLoading from directory: {input_path}")
        df = load_reaches_from_jsons(input_path)
        output_dir = input_path.parent

    print(f"  Loaded {len(df):,} reaches from {df['animal'].nunique()} mice")

    # Load features if available
    features_df = None
    features_dir = Path(args.features_dir) if args.features_dir else output_dir / 'Step5_Features'
    if features_dir.exists():
        print(f"\nLoading kinematic features from: {features_dir}")
        features_df = load_features_from_jsons(features_dir)
        if len(features_df) > 0:
            print(f"  Loaded features for {len(features_df):,} reaches")

    # Build database
    output_path = Path(args.output) if args.output else output_dir / 'reach_explorer.db'

    explorer = build_explorer_database(df, output_path, features_df)

    # Print summary
    print("\n" + "=" * 60)
    print("DATABASE BUILT SUCCESSFULLY")
    print("=" * 60)
    print(f"Location: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024:.1f} KB")

    # Quick stats
    pop = explorer.get_population_stats()
    if pop:
        profile = pop['profile']
        print(f"\nPopulation Summary:")
        print(f"  Total reaches: {profile['n_reaches']:,}")
        print(f"  Success rate: {profile['success_rate']*100:.1f}%")
        print(f"  Mean extent: {profile['extent_mean']:.2f} mm")

    print(f"\nMice in database: {', '.join(explorer.list_mice())}")

    explorer.close()


def main_explore():
    """Interactive query interface for the explorer database."""
    parser = argparse.ArgumentParser(
        description="Query the reach explorer database"
    )
    parser.add_argument(
        '-d', '--database',
        help="Path to explorer database (default: auto-detect)"
    )
    parser.add_argument(
        '--mouse',
        help="Show stats for specific mouse"
    )
    parser.add_argument(
        '--session',
        help="Show stats for specific session"
    )
    parser.add_argument(
        '--compare',
        nargs='+',
        help="Compare multiple mice (e.g., --compare CNT0110 CNT0111 CNT0112)"
    )
    parser.add_argument(
        '--population',
        action='store_true',
        help="Show population-level statistics"
    )
    parser.add_argument(
        '--list-mice',
        action='store_true',
        help="List all mice in database"
    )
    parser.add_argument(
        '--list-sessions',
        help="List sessions for a mouse"
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help="Output as JSON"
    )

    args = parser.parse_args()

    from .explorer import ReachExplorer

    # Find database
    if args.database:
        db_path = Path(args.database)
    else:
        # Auto-detect
        from mousereach.config import Paths, PROCESSING_ROOT
        db_path = PROCESSING_ROOT / 'reach_explorer.db'
        if not db_path.exists():
            # Try Processing folder
            db_path = Paths.PROCESSING / 'reach_explorer.db'

    if not db_path.exists():
        print(f"Database not found: {db_path}")
        print("Run 'mousereach-build-explorer' first to create the database.")
        sys.exit(1)

    explorer = ReachExplorer(db_path)
    explorer.connect()

    try:
        if args.list_mice:
            mice = explorer.list_mice()
            if args.json:
                print(json.dumps(mice))
            else:
                print("Mice in database:")
                for m in mice:
                    stats = explorer.get_mouse_stats(m)
                    if stats:
                        print(f"  {m}: {stats['n_sessions']} sessions, {stats['profile']['n_reaches']} reaches, {stats['profile']['success_rate']*100:.1f}% success")

        elif args.list_sessions:
            sessions = explorer.list_sessions(args.list_sessions)
            if args.json:
                print(json.dumps(sessions))
            else:
                print(f"Sessions for {args.list_sessions}:")
                for s in sessions:
                    stats = explorer.get_session_stats(s)
                    if stats:
                        print(f"  {s}: {stats['profile']['n_reaches']} reaches, {stats['profile']['success_rate']*100:.1f}% success")

        elif args.population:
            pop = explorer.get_population_stats()
            if args.json:
                print(json.dumps(pop, indent=2))
            else:
                print_population_stats(pop)

        elif args.mouse:
            profile = explorer.get_mouse_success_profile(args.mouse)
            if args.json:
                print(json.dumps(profile, indent=2, default=str))
            else:
                print_mouse_profile(profile)

        elif args.session:
            stats = explorer.get_session_stats(args.session)
            if args.json:
                print(json.dumps(stats, indent=2))
            else:
                print_session_stats(stats)

        elif args.compare:
            comparison = explorer.compare_mice(args.compare)
            if args.json:
                print(comparison.to_json(orient='records', indent=2))
            else:
                print("\nMouse Comparison:")
                print(comparison.to_string(index=False))

        else:
            # Default: show summary
            print("=" * 60)
            print("REACH EXPLORER DATABASE")
            print("=" * 60)
            print(f"Database: {db_path}")
            print(f"Built: {explorer.get_metadata('build_date')}")
            print(f"Total reaches: {explorer.get_metadata('n_reaches')}")
            print(f"Total animals: {explorer.get_metadata('n_animals')}")
            print("\nUse --help to see query options")

    finally:
        explorer.close()


def print_population_stats(pop: dict):
    """Pretty-print population statistics."""
    profile = pop.get('profile', {})
    comparisons = pop.get('comparisons', [])
    temporal = pop.get('temporal', [])

    print("=" * 60)
    print("POPULATION STATISTICS")
    print("=" * 60)

    print(f"\nTotal Reaches: {profile.get('n_reaches', 0):,}")
    print(f"Success Rate: {profile.get('success_rate', 0)*100:.1f}%")
    print(f"  Retrieved: {profile.get('n_success', 0):,}")
    print(f"  Failed: {profile.get('n_fail', 0):,}")

    print(f"\nKinematics:")
    print(f"  Extent: {profile.get('extent_mean', 0):.2f} +/- {profile.get('extent_std', 0):.2f} mm")
    print(f"  Duration: {profile.get('duration_mean', 0):.3f} +/- {profile.get('duration_std', 0):.3f} sec")
    print(f"  Velocity: {profile.get('velocity_mean', 0):.2f} +/- {profile.get('velocity_std', 0):.2f} px/frame")

    if comparisons:
        print(f"\nSuccess vs Fail Comparisons:")
        for comp in comparisons:
            sig = "*" if comp.get('significant') else ""
            print(f"  {comp['feature']}: p={comp['p_value']:.4f}{sig}, d={comp['cohens_d']:.2f}")

    if temporal:
        print(f"\nTemporal Patterns (Fatigue):")
        for t in temporal:
            print(f"  {t['phase'].capitalize()}: {t['n_reaches']} reaches, {t['success_rate']*100:.1f}% success")


def print_mouse_profile(profile: dict):
    """Pretty-print mouse success profile."""
    if not profile:
        print("Mouse not found.")
        return

    print("=" * 60)
    print(f"MOUSE: {profile.get('animal', 'Unknown')}")
    print("=" * 60)

    overall = profile.get('overall', {})
    print(f"\nOverall: {overall.get('n_reaches', 0)} reaches, {overall.get('success_rate', 0)*100:.1f}% success")

    success = profile.get('success_profile', {})
    fail = profile.get('fail_profile', {})

    print(f"\nSUCCESS PROFILE (n={success.get('n', 0)}):")
    print(f"  Extent: {success.get('extent_mean', 0):.2f} +/- {success.get('extent_std', 0):.2f} mm")
    print(f"  Duration: {success.get('duration_mean', 0):.3f} sec")
    print(f"  Velocity: {success.get('velocity_mean', 0):.2f} px/frame")

    print(f"\nFAIL PROFILE (n={fail.get('n', 0)}):")
    print(f"  Extent: {fail.get('extent_mean', 0):.2f} +/- {fail.get('extent_std', 0):.2f} mm")
    print(f"  Duration: {fail.get('duration_mean', 0):.3f} sec")
    print(f"  Velocity: {fail.get('velocity_mean', 0):.2f} px/frame")

    comparisons = profile.get('comparisons', [])
    if comparisons:
        print(f"\nSuccess vs Fail (within this mouse):")
        for comp in comparisons:
            sig = "*" if comp.get('significant') else ""
            print(f"  {comp['feature']}: p={comp['p_value']:.4f}{sig}")

    learning = profile.get('learning_curve', [])
    if learning:
        print(f"\nLearning Curve:")
        for point in learning:
            print(f"  {point['date']}: {point['n_reaches']} reaches, {point['success_rate']*100:.1f}% success")


def print_session_stats(stats: dict):
    """Pretty-print session statistics."""
    if not stats:
        print("Session not found.")
        return

    print("=" * 60)
    print(f"SESSION: {stats.get('video_name', 'Unknown')}")
    print("=" * 60)

    print(f"Animal: {stats.get('animal')}")
    print(f"Date: {stats.get('date')}")
    print(f"Tray: {stats.get('tray_type')} (run {stats.get('run_num')})")

    profile = stats.get('profile', {})
    print(f"\nReaches: {profile.get('n_reaches', 0)}")
    print(f"Success Rate: {profile.get('success_rate', 0)*100:.1f}%")
    print(f"Extent: {profile.get('extent_mean', 0):.2f} mm")

    fatigue = stats.get('fatigue', [])
    if fatigue:
        print(f"\nFatigue Pattern:")
        for f in fatigue:
            print(f"  {f['phase'].capitalize()}: {f['n_reaches']} reaches, {f['success_rate']*100:.1f}% success")


# =============================================================================
# HELPERS
# =============================================================================

def load_reaches_from_jsons(directory: Path) -> pd.DataFrame:
    """Load reach data from *_reaches.json files."""
    import re

    all_reaches = []
    reach_files = list(directory.glob('*_reaches.json'))

    for rf in reach_files:
        try:
            with open(rf) as f:
                data = json.load(f)

            video_name = data.get('video_name', rf.stem.replace('_reaches', ''))

            # Parse filename
            match = re.match(r'(\d{8})_([A-Z]+\d+)_([PF])(\d)?', video_name)
            if match:
                date_str, animal, tray_code, run_num = match.groups()
                date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                cohort = f"CNT_{animal[3:5]}"
                tray_type = 'Pillar' if tray_code == 'P' else 'Flat'
                run_num = int(run_num) if run_num else 1
            else:
                date, animal, tray_type, run_num, cohort = None, video_name, None, None, None

            for seg in data.get('segments', []):
                for reach in seg.get('reaches', []):
                    all_reaches.append({
                        'video_name': video_name,
                        'date': date,
                        'animal': animal,
                        'tray_type': tray_type,
                        'run_num': run_num,
                        'cohort': cohort,
                        'segment_num': seg.get('segment_num'),
                        'reach_id': reach.get('reach_id'),
                        'reach_num_in_segment': reach.get('reach_num'),
                        'start_frame': reach.get('start_frame'),
                        'apex_frame': reach.get('apex_frame'),
                        'end_frame': reach.get('end_frame'),
                        'duration_frames': reach.get('duration_frames'),
                        'max_extent_pixels': reach.get('max_extent_pixels'),
                        'max_extent_mm': reach.get('max_extent_ruler', 0) * 10 if reach.get('max_extent_ruler') else None,
                        'duration_sec': reach.get('duration_frames', 0) / 30.0 if reach.get('duration_frames') else None,
                    })
        except Exception as e:
            print(f"  Warning: Error loading {rf.name}: {e}")

    # Load outcomes
    outcome_files = list(directory.glob('*_outcome_ground_truth.json')) + list(directory.glob('*_pellet_outcomes.json'))
    outcomes = {}

    for of in outcome_files:
        try:
            with open(of) as f:
                data = json.load(f)
            video_name = data.get('video_name', of.stem.split('_outcome')[0].split('_pellet')[0])
            outcomes[video_name] = {seg['segment_num']: seg for seg in data.get('segments', [])}
        except (OSError, json.JSONDecodeError, KeyError) as e:
            pass  # Skip files that can't be loaded

    # Merge outcomes
    for reach in all_reaches:
        video = reach['video_name']
        seg = reach['segment_num']
        if video in outcomes and seg in outcomes[video]:
            out = outcomes[video][seg]
            reach['outcome'] = out.get('outcome')

    return pd.DataFrame(all_reaches)


def load_features_from_jsons(directory: Path) -> pd.DataFrame:
    """Load kinematic features from *_features.json files."""
    all_features = []

    for ff in directory.glob('*_features.json'):
        try:
            with open(ff) as f:
                data = json.load(f)

            video_name = data.get('video_name')

            for seg in data.get('segments', []):
                for reach in seg.get('reaches', []):
                    all_features.append({
                        'video_name': video_name,
                        'segment_num': seg.get('segment_num'),
                        'reach_num': reach.get('reach_num'),
                        'peak_velocity_px_per_frame': reach.get('peak_velocity_px_per_frame'),
                        'mean_velocity_px_per_frame': reach.get('mean_velocity_px_per_frame'),
                        'trajectory_straightness': reach.get('trajectory_straightness'),
                    })
        except (OSError, KeyError, ValueError) as e:
            pass  # Skip videos with missing kinematic data

    return pd.DataFrame(all_features)


if __name__ == '__main__':
    main_explore()
