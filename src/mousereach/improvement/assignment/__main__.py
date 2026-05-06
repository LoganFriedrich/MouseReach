"""Run assignment (algo 4) evaluator end-to-end -- per-reach Sankey + holistic view."""
import sys
from pathlib import Path
from .analyze import analyze
from .graph import graph

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m mousereach.improvement.assignment <snapshot_dir>")
        sys.exit(1)
    sd = Path(sys.argv[1])
    res = analyze(sd)
    out = graph(sd)
    print(f"n_reaches_universe={res['n_reaches_universe']}  triage={res['triage_count']}")
    print(f"Sankey: {out}")
