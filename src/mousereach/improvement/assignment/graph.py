"""
Assignment grapher (algo 4) -- per-reach Sankey + directional summary.
Reads <snapshot>/metrics/scalars.json. Writes Sankey via run_sankey
(canonical for both algo 4 and pipeline-wide holistic view).
"""
from __future__ import annotations

from pathlib import Path

from mousereach.improvement.lib.inputs import load_snapshot_paths
from mousereach.improvement.outcome._run_notebooks import run_sankey


def graph(snapshot_dir: Path) -> Path:
    paths = load_snapshot_paths(snapshot_dir)
    run_sankey(paths.snapshot_dir)
    return paths.figures_dir / "sankey.png"


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m mousereach.improvement.assignment.graph <snapshot_dir>")
        sys.exit(1)
    out = graph(Path(sys.argv[1]))
    print(f"Wrote: {out}")
