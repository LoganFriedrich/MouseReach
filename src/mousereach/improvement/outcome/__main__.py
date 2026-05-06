"""Run outcome (algo 3) evaluator end-to-end -- per-pellet Sankey."""
import sys
from pathlib import Path
from .analyze import analyze
from .graph import graph

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m mousereach.improvement.outcome <snapshot_dir>")
        sys.exit(1)
    sd = Path(sys.argv[1])
    res = analyze(sd)
    out = graph(sd)
    print(f"n_segments={res['n_segments']}  n_correct={res['n_correct']}  triage={res['triage_count']}")
    print(f"Sankey: {out}")
