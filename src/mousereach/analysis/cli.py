"""
CLI entry points for MouseReach analysis.

Commands:
    mousereach-analyze                    # Launch Streamlit dashboard
    mousereach-analyze --export data.csv  # Export all data to CSV
    mousereach-analyze --tracking-dir DIR # Include experimental metadata

    mousereach-build-database             # Build unified reach database (wide format)
"""

import argparse
import sys
from pathlib import Path


# =============================================================================
# mousereach-build-database: Build unified wide-format database
# =============================================================================
def main_build_database():
    """
    Build the unified reach-centric database with ALL hierarchical data.

    This creates a single wide-format file where each row is a reach with:
    - Reach identification (video, segment, reach IDs)
    - Temporal bounds (start/apex/end frames)
    - Kinematic features (extent, velocity, trajectory, etc.)
    - Pellet outcome + reach-level outcome derivation
    - Segment context (reach position, first/last flags)
    - Session context (position in day's testing)
    - Experimental metadata (Test_Phase, timepoint, Weight)
    - Surgery data (injury details, days_post_injury)

    Examples:
        mousereach-build-database
        mousereach-build-database -o reaches.parquet
        mousereach-build-database --tracking-dir /path/to/Animal_Tracking
    """
    parser = argparse.ArgumentParser(
        description="Build unified reach database (one row per reach, all metadata attached)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output Formats:
    .csv     - CSV file (readable in Excel, larger size)
    .parquet - Parquet file (faster loading, smaller size, preserves dtypes)
    .xlsx    - Excel file with multiple sheets (Reaches, Sessions, Mice)

Examples:
    mousereach-build-database
        Build database using default locations, output to unified_reaches.parquet

    mousereach-build-database -o all_reaches.csv
        Export as CSV instead of Parquet

    mousereach-build-database --tracking-dir /path/to/Animal_Tracking
        Include experimental metadata (Test_Phase, Weight, surgery data)

    mousereach-build-database --derive-outcomes
        Add reach-level outcome derivation (miss_on_pillar vs causal reach)
        """
    )

    parser.add_argument(
        '-i', '--input', '-d', '--data-dir',
        dest='data_dir',
        type=Path,
        help="Directory with pipeline output files (*_features.json). Default: Processing folder"
    )

    parser.add_argument(
        '-o', '--output',
        type=Path,
        help="Output file path. Supports .csv, .parquet, .xlsx. Default: ./unified_reaches.parquet"
    )

    parser.add_argument(
        '-b', '--brainglobe-path',
        type=Path,
        help="Path to BrainGlobe region_counts.csv (default: auto-detect)"
    )

    parser.add_argument(
        '--no-brainglobe',
        action='store_true',
        help="Skip loading BrainGlobe connectomics data"
    )

    parser.add_argument(
        '-t', '--tracking-dir',
        type=Path,
        help="Directory with Connectome_XX_Animal_Tracking.xlsx files (adds Test_Phase, Weight, surgery)"
    )

    parser.add_argument(
        '--no-surgery',
        action='store_true',
        help="Skip loading surgery/mouse-level metadata"
    )

    parser.add_argument(
        '--derive-outcomes',
        action='store_true',
        help="Derive per-reach outcomes (miss_on_pillar, miss_off_pillar, causal reach)"
    )

    parser.add_argument(
        '--include-flagged',
        action='store_true',
        help="Include flagged/excluded reaches (normally excluded)"
    )

    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help="Force rebuild even if database is current"
    )

    parser.add_argument(
        '--check',
        action='store_true',
        help="Only check if database needs rebuilding (don't actually build)"
    )

    args = parser.parse_args()

    # Import here to avoid slow startup
    from mousereach.analysis.data import (
        build_unified_reach_data,
        derive_reach_outcomes,
        check_database_current,
        get_source_fingerprint,
        save_database_metadata
    )
    from mousereach.config import Paths

    # Resolve paths
    data_dir = args.data_dir or Paths.PROCESSING
    tracking_dir = args.tracking_dir

    # Tracking dir is optional - external data
    # User must provide --tracking-dir explicitly if they want experimental metadata

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Default to current directory
        output_path = Path("./unified_reaches.parquet")

    # Check if database is current
    if not args.force:
        is_current, reason = check_database_current(output_path, data_dir)
        if args.check:
            # Just report status
            if is_current:
                print(f"✓ Database is current: {reason}")
            else:
                print(f"✗ Database needs rebuild: {reason}")
            return

        if is_current:
            print(f"Database already current: {reason}")
            print("Use --force to rebuild anyway.")
            return

        print(f"Rebuilding: {reason}")

    # Build the unified dataset
    reach_data = build_unified_reach_data(
        data_dir=data_dir,
        tracking_dir=tracking_dir,
        brainglobe_path=args.brainglobe_path,
        use_features=True,
        exclude_flagged=not args.include_flagged,
        include_surgery=not args.no_surgery,
        include_brainglobe=not args.no_brainglobe
    )

    if len(reach_data) == 0:
        print("No data found. Check your data directory.")
        return

    df = reach_data.df

    # Derive per-reach outcomes if requested
    if args.derive_outcomes:
        print("\nDeriving per-reach outcomes...")
        df = derive_reach_outcomes(df)

        # Print summary
        if 'reach_outcome' in df.columns:
            outcome_counts = df['reach_outcome'].value_counts()
            print("  Reach outcomes:")
            for outcome, count in outcome_counts.items():
                print(f"    {outcome}: {count}")

    # Export based on suffix
    suffix = output_path.suffix.lower()

    print(f"\nExporting to {output_path}...")

    if suffix == '.parquet':
        df.to_parquet(output_path, index=False)
    elif suffix == '.xlsx':
        import pandas as pd
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Reaches', index=False)

            # Add session-level aggregation
            session_cols = ['session_id', 'mouse_id', 'date', 'timepoint']
            session_cols = [c for c in session_cols if c in df.columns]
            if session_cols:
                session_df = df.groupby(session_cols).agg({
                    'reach_id': 'count',
                    'outcome': lambda x: (x == 'retrieved').sum() / len(x) if len(x) > 0 else 0
                }).rename(columns={'reach_id': 'n_reaches', 'outcome': 'retrieval_rate'}).reset_index()
                session_df.to_excel(writer, sheet_name='Sessions', index=False)

            # Add mouse-level aggregation
            if 'mouse_id' in df.columns:
                mouse_df = df.groupby('mouse_id').agg({
                    'reach_id': 'count',
                    'session_id': 'nunique'
                }).rename(columns={'reach_id': 'n_reaches', 'session_id': 'n_sessions'}).reset_index()
                mouse_df.to_excel(writer, sheet_name='Mice', index=False)
    else:
        # Default to CSV
        df.to_csv(output_path.with_suffix('.csv'), index=False)
        output_path = output_path.with_suffix('.csv')

    # Save metadata for versioning
    fingerprint = get_source_fingerprint(data_dir)
    meta_path = save_database_metadata(output_path, fingerprint, len(df), df.columns.tolist())

    # Report
    file_size = output_path.stat().st_size / (1024 * 1024)  # MB
    print(f"\n{'='*60}")
    print("DATABASE BUILT SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Output: {output_path}")
    print(f"Metadata: {meta_path}")
    print(f"Size: {file_size:.1f} MB")
    print(f"Rows: {len(df):,} reaches")
    print(f"Columns: {len(df.columns)}")

    # Column summary
    print(f"\nColumn categories:")
    id_cols = [c for c in df.columns if any(x in c.lower() for x in ['id', 'name', 'num'])]
    temporal_cols = [c for c in df.columns if 'frame' in c.lower()]
    kinematic_cols = [c for c in df.columns if any(x in c.lower() for x in ['extent', 'velocity', 'trajectory', 'angle'])]
    meta_cols = [c for c in df.columns if any(x in c.lower() for x in ['phase', 'timepoint', 'weight', 'surgery', 'injury'])]

    print(f"  Identifiers: {len(id_cols)}")
    print(f"  Temporal: {len(temporal_cols)}")
    print(f"  Kinematic: {len(kinematic_cols)}")
    print(f"  Metadata: {len(meta_cols)}")

    if 'timepoint' in df.columns:
        print(f"\nTimepoints present: {sorted(df['timepoint'].dropna().unique())}")
    if 'days_post_injury' in df.columns:
        valid_dpi = df['days_post_injury'].notna().sum()
        print(f"Days post injury: {valid_dpi}/{len(df)} reaches have this computed")


def main():
    """Main entry point for mousereach-analyze command."""
    parser = argparse.ArgumentParser(
        description="MouseReach Analysis Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    mousereach-analyze
        Launch interactive dashboard

    mousereach-analyze --export results.csv
        Export all data to CSV

    mousereach-analyze --tracking-dir /path/to/Animal_Tracking
        Include experimental metadata (Test_Phase, Weight, etc.) from tracking spreadsheets
        """
    )

    parser.add_argument(
        '--data-dir', '-d',
        type=Path,
        help="Directory containing pipeline output files (default: Processing)"
    )

    parser.add_argument(
        '--tracking-dir', '-t',
        type=Path,
        help="Directory with Connectome_XX_Animal_Tracking.xlsx files for experimental metadata"
    )

    parser.add_argument(
        '--export',
        type=Path,
        metavar='OUTPUT',
        help="Export all data to CSV/Excel file instead of launching dashboard"
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help="Port for Streamlit server (default: 8501)"
    )

    args = parser.parse_args()

    if args.export:
        # Export mode - no GUI
        from mousereach.analysis.data import load_all_data, load_data_with_metadata
        from mousereach.config import Paths

        data_dir = args.data_dir or Paths.PROCESSING

        print(f"Loading data from: {data_dir}")

        if args.tracking_dir:
            data = load_data_with_metadata(
                data_dir,
                tracking_dir=args.tracking_dir,
                use_features=True,
                exclude_flagged=True
            )
        else:
            data = load_all_data(data_dir, use_features=True, exclude_flagged=True)

        output_path = args.export
        if output_path.suffix == '.xlsx':
            data.to_excel(output_path)
        else:
            data.to_csv(output_path.with_suffix('.csv'))

        print(f"Exported {len(data)} reaches to {output_path}")

    else:
        # Dashboard mode - launch Streamlit
        import subprocess

        dashboard_path = Path(__file__).parent / "dashboard.py"

        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.port", str(args.port),
            "--browser.gatherUsageStats", "false"
        ]

        print("Launching MouseReach Analysis Dashboard...")
        print(f"Open http://localhost:{args.port} in your browser")
        print("Press Ctrl+C to stop\n")

        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\nDashboard stopped.")


if __name__ == "__main__":
    main()
