"""
Query the reach explorer database.

Usage:
    python explore.py                    # Show overview
    python explore.py --mouse CNT0110    # Show mouse profile
    python explore.py --compare CNT0110 CNT0216  # Compare mice
    python explore.py --population       # Population stats
"""
import argparse
import json
import sqlite3
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Query reach explorer database")
    parser.add_argument('--mouse', help="Show stats for specific mouse")
    parser.add_argument('--session', help="Show stats for specific session")
    parser.add_argument('--compare', nargs='+', help="Compare multiple mice")
    parser.add_argument('--population', action='store_true', help="Show population stats")
    parser.add_argument('--list-mice', action='store_true', help="List all mice")

    args = parser.parse_args()

    db_path = Path(__file__).parent / 'reach_explorer.db'
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    if args.list_mice:
        rows = conn.execute("SELECT animal, cohort, n_sessions, profile_json FROM mouse_stats ORDER BY animal").fetchall()
        print("\n" + "="*70)
        print("MICE IN DATABASE")
        print("="*70)
        print(f"{'Animal':<12} {'Cohort':<10} {'Sessions':<10} {'Reaches':<10} {'Success':<10}")
        print("-"*70)
        for r in rows:
            profile = json.loads(r['profile_json'])
            print(f"{r['animal']:<12} {r['cohort'] or 'N/A':<10} {r['n_sessions']:<10} {profile['n_reaches']:<10} {profile['success_rate']*100:.1f}%")

    elif args.population:
        row = conn.execute("SELECT * FROM population_stats WHERE id=1").fetchone()
        profile = json.loads(row['profile_json'])
        print("\n" + "="*70)
        print("POPULATION STATISTICS")
        print("="*70)
        print(f"Total Reaches: {profile['n_reaches']:,}")
        print(f"Success Rate: {profile['success_rate']*100:.1f}%")
        print(f"  Retrieved: {profile['n_success']:,}")
        print(f"  Failed: {profile['n_fail']:,}")
        print(f"\nKinematics:")
        print(f"  Extent: {profile['extent_mean']:.2f} ± {profile['extent_std']:.2f} mm (median: {profile['extent_median']:.2f})")
        print(f"  Duration: {profile['duration_mean']:.3f} ± {profile['duration_std']:.3f} sec")
        print(f"  Velocity: {profile['velocity_mean']:.2f} ± {profile['velocity_std']:.2f} px/frame")

    elif args.mouse:
        row = conn.execute("SELECT * FROM mouse_stats WHERE animal=?", (args.mouse,)).fetchone()
        if not row:
            print(f"Mouse {args.mouse} not found.")
            return

        profile = json.loads(row['profile_json'])
        learning = json.loads(row['learning_curve_json'])

        print("\n" + "="*70)
        print(f"MOUSE: {args.mouse}")
        print("="*70)
        print(f"Cohort: {row['cohort']}")
        print(f"Sessions: {row['n_sessions']}")
        print(f"\nOverall: {profile['n_reaches']} reaches, {profile['success_rate']*100:.1f}% success")
        print(f"  Extent: {profile['extent_mean']:.2f} ± {profile['extent_std']:.2f} mm")
        print(f"  Duration: {profile['duration_mean']:.3f} sec")
        print(f"  Velocity: {profile['velocity_mean']:.2f} px/frame")

        # Success vs fail for this mouse
        success = pd.read_sql_query(
            "SELECT * FROM reaches WHERE animal=? AND outcome='retrieved'",
            conn, params=(args.mouse,))
        fail = pd.read_sql_query(
            "SELECT * FROM reaches WHERE animal=? AND outcome!='retrieved'",
            conn, params=(args.mouse,))

        if len(success) > 0 and len(fail) > 0:
            print(f"\nSUCCESS PROFILE (n={len(success)}):")
            print(f"  Extent: {success['extent_mm'].mean():.2f} ± {success['extent_mm'].std():.2f} mm")
            print(f"  Duration: {success['duration_sec'].mean():.3f} sec")
            print(f"\nFAIL PROFILE (n={len(fail)}):")
            print(f"  Extent: {fail['extent_mm'].mean():.2f} ± {fail['extent_mm'].std():.2f} mm")
            print(f"  Duration: {fail['duration_sec'].mean():.3f} sec")

        if learning:
            print(f"\nLEARNING CURVE:")
            for pt in learning:
                print(f"  {pt['date']}: {pt['n_reaches']} reaches, {pt['success_rate']*100:.1f}% success")

    elif args.compare:
        print("\n" + "="*70)
        print("MOUSE COMPARISON")
        print("="*70)
        print(f"{'Animal':<12} {'Reaches':<10} {'Success':<10} {'Extent':<12} {'Velocity':<10}")
        print("-"*70)
        for animal in args.compare:
            row = conn.execute("SELECT * FROM mouse_stats WHERE animal=?", (animal,)).fetchone()
            if row:
                profile = json.loads(row['profile_json'])
                print(f"{animal:<12} {profile['n_reaches']:<10} {profile['success_rate']*100:.1f}%{' '*5} {profile['extent_mean']:.2f} mm{' '*4} {profile['velocity_mean']:.2f}")

    else:
        # Default: overview
        meta_date = conn.execute("SELECT value FROM metadata WHERE key='build_date'").fetchone()
        meta_n = conn.execute("SELECT value FROM metadata WHERE key='n_reaches'").fetchone()
        meta_animals = conn.execute("SELECT value FROM metadata WHERE key='n_animals'").fetchone()

        print("\n" + "="*70)
        print("REACH EXPLORER DATABASE")
        print("="*70)
        print(f"Database: {db_path}")
        print(f"Built: {meta_date[0] if meta_date else 'Unknown'}")
        print(f"Total reaches: {meta_n[0] if meta_n else 'Unknown'}")
        print(f"Total animals: {meta_animals[0] if meta_animals else 'Unknown'}")
        print("\nQuery options:")
        print("  --list-mice       List all mice with summary stats")
        print("  --population      Show population-level statistics")
        print("  --mouse CNT0110   Show detailed profile for a mouse")
        print("  --compare A B C   Compare multiple mice side-by-side")

    conn.close()


if __name__ == '__main__':
    main()
