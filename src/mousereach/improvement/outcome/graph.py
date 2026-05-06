"""
Outcome grapher (algo 3) -- per-PELLET (per-segment) Sankey.

Reads <snapshot>/metrics/scalars.json (written by analyze.py).
Calls the existing run_sankey from _run_notebooks.py.

Writes:
  <snapshot>/figures/sankey.png
  <snapshot>/figures/sankey_legend.md
"""
from __future__ import annotations

from pathlib import Path

from mousereach.improvement.lib.inputs import load_snapshot_paths
from ._run_notebooks import run_sankey


def graph(snapshot_dir: Path) -> Path:
    paths = load_snapshot_paths(snapshot_dir)
    run_sankey(paths.snapshot_dir)
    return paths.figures_dir / "sankey.png"


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m mousereach.improvement.outcome.graph <snapshot_dir>")
        sys.exit(1)
    out = graph(Path(sys.argv[1]))
    print(f"Wrote: {out}")
