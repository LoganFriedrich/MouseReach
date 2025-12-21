"""
ASPA2 Pipeline Runner
=====================

Run all pipeline steps in sequence.

Usage:
    python 0_run_pipeline.py              # Interactive
    python 0_run_pipeline.py /path/to/dir  # Process directory
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_pipeline(data_dir: Path):
    """Run full pipeline on a directory."""
    print("ASPA2 Pipeline")
    print("=" * 50)
    
    # Step 1: Segment
    print("\n[Step 1] Segmenting videos...")
    from scripts import segment_videos  # TODO
    
    # Step 2: Calibrate
    print("\n[Step 2] Calibrating...")
    # TODO
    
    # Step 3: Detect reaches
    print("\n[Step 3] Detecting reaches...")
    # TODO
    
    # Step 4: Score
    print("\n[Step 4] Scoring...")
    # TODO
    
    # Step 5: Compile
    print("\n[Step 5] Compiling results...")
    # TODO
    
    print("\n" + "=" * 50)
    print("Pipeline complete!")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        run_pipeline(Path(sys.argv[1]))
    else:
        print("Usage: python 0_run_pipeline.py /path/to/data")
        print("\nTODO: Implement interactive mode")
