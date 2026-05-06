"""Run segmentation evaluator end-to-end."""
import sys
from pathlib import Path
from .analyze import analyze
from .graph import graph

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m mousereach.improvement.segmentation <snapshot_dir>")
        sys.exit(1)
    sd = Path(sys.argv[1])
    res = analyze(sd)
    out = graph(sd)
    print(f"Matched: {res['n_matched']}  FP: {res['n_fp']}  FN: {res['n_fn']}  triage: {res['triage_count']}")
    print(f"Figure: {out}")
