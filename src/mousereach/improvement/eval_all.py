"""Unified pipeline-eval runner.

Hard rule (from project instructions): when evaluating "how well do all of
our algos generalize?" you have to look at every algo element. This module
runs every analyzer + figure renderer in the canonical sequence:

    1. segmentation  (boundary-delta violin + summary table)
    2. reach_detection  (TP/FP/FN delta violins + summary table)
    3. outcome  (per-segment Sankey [pre/post] + summary table + IFR violin)
    4. assignment  (per-reach Sankey)

The runner also calls GT auto-resolve on ``algo_outputs_current/`` before
analyzers fire, so the outcome two-level metrics populate correctly.

Default behavior is "run everything". Use ``--only`` to restrict to a
specific algo when iterating on one piece.

CLI:

    mousereach-eval-all <snapshot_dir> [--only seg|reach|outcome|assign]
                                       [--no-gt-resolve]
                                       [--gt-dir <dir>]

The snapshot dir must contain (or fall back to) ``algo_outputs_current/``
and ``gt/`` per the quarantine convention. The eval framework's
``load_snapshot_paths`` resolves defaults automatically.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import List, Optional

# Each step is (label, analyze_fn, [renderer_fns])
# Analyzer and renderers all take a single snapshot_dir Path argument.


def _safe_call(label: str, fn, *args, **kwargs):
    """Run ``fn``, print a short status line, swallow exceptions so a
    single step's failure doesn't block the rest."""
    try:
        fn(*args, **kwargs)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {label}: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=2)
        return False


def _archive_step_figures(step: str, snapshot_dir: Path) -> None:
    """Copy whatever PNGs / legend MDs are now in ``figures/`` into
    ``figures/<step>/`` so the next step's render doesn't overwrite them
    (e.g. outcome's sankey.png vs assignment's sankey.png)."""
    import shutil
    fig_dir = snapshot_dir / "figures"
    if not fig_dir.is_dir():
        return
    step_dir = fig_dir / step
    step_dir.mkdir(parents=True, exist_ok=True)
    # Only top-level files; don't recurse into the step subdirs themselves.
    for f in fig_dir.iterdir():
        if f.is_file() and (f.suffix.lower() in (".png", ".md")):
            shutil.copy2(f, step_dir / f.name)


def _open_figures_in_vscode(snapshot_dir: Path, steps: List[str]) -> None:
    """Open the rendered canonical PNGs for each step in VS Code, so a
    scorecard is never produced without its figures being shown. PNGs only
    (no legend .md, no manifest). Never raises -- a viewer failure must not
    fail the eval; the paths are printed as a fallback."""
    import shutil as _sh
    import subprocess
    pngs: List[str] = []
    for step in steps:
        step_dir = snapshot_dir / "figures" / step
        if step_dir.is_dir():
            pngs.extend(str(p) for p in sorted(step_dir.glob("*.png")))
    if not pngs:
        print("  [open] no figures found to open")
        return
    code = _sh.which("code") or _sh.which("code.cmd")
    if not code:
        print("  [open] 'code' CLI not on PATH -- open these manually:")
        for p in pngs:
            print(f"    {p}")
        return
    try:
        subprocess.run([code, *pngs], check=False)
        print(f"  [open] opened {len(pngs)} canonical figures in VS Code")
    except Exception as exc:  # noqa: BLE001
        print(f"  [open] could not open in VS Code ({exc}); paths:")
        for p in pngs:
            print(f"    {p}")


def run_one_step(step: str, snapshot_dir: Path, *,
                 archive: bool = True) -> None:
    """Run analyzer + every available renderer for one algo step.

    When ``archive`` is True (default), each step's outputs are mirrored
    into ``figures/<step>/`` so cross-step filename collisions (e.g.
    outcome's sankey.png vs assignment's sankey.png) don't clobber
    earlier renders.
    """
    print(f"\n========== {step.upper()} :: {snapshot_dir.name} ==========")
    # Clear top-level files in figures/ so this step starts clean and the
    # archive captures only this step's output. Step subdirs are preserved.
    fig_dir = snapshot_dir / "figures"
    if fig_dir.is_dir():
        for f in fig_dir.iterdir():
            if f.is_file() and (f.suffix.lower() in (".png", ".md")):
                try:
                    f.unlink()
                except OSError:
                    pass

    if step == "segmentation":
        from mousereach.improvement.segmentation import analyze as seg_analyze
        from mousereach.improvement.segmentation import _run_notebooks as seg_rn
        _safe_call("analyze", seg_analyze.analyze, snapshot_dir)
        _safe_call("violin", seg_rn.run_violin, snapshot_dir)
        _safe_call("summary_table", seg_rn.run_summary_table, snapshot_dir)
    elif step == "reach_detection":
        from mousereach.improvement.reach_detection import analyze as r_analyze
        from mousereach.improvement.reach_detection import _run_notebooks as r_rn
        _safe_call("analyze", r_analyze.analyze, snapshot_dir)
        _safe_call("violin", r_rn.run_violin, snapshot_dir)
        _safe_call("summary_table", r_rn.run_summary_table, snapshot_dir)
    elif step == "outcome":
        from mousereach.improvement.outcome import analyze as o_analyze
        from mousereach.improvement.outcome import _run_notebooks as o_rn
        _safe_call("analyze", o_analyze.analyze, snapshot_dir)
        _safe_call("sankey", o_rn.run_sankey, snapshot_dir)
        _safe_call("interaction_violin", o_rn.run_interaction_violin, snapshot_dir)
        _safe_call("summary_table", o_rn.run_summary_table, snapshot_dir)
    elif step == "assignment":
        from mousereach.improvement.assignment import analyze as a_analyze
        from mousereach.improvement.assignment import graph as a_graph
        _safe_call("analyze", a_analyze.analyze, snapshot_dir)
        _safe_call("graph", a_graph.graph, snapshot_dir)
    else:
        raise ValueError(f"unknown step: {step}")

    if archive:
        _archive_step_figures(step, snapshot_dir)


STEP_ALIASES = {
    "seg": "segmentation",
    "segmentation": "segmentation",
    "reach": "reach_detection",
    "reach_detection": "reach_detection",
    "outcome": "outcome",
    "assign": "assignment",
    "assignment": "assignment",
}

ALL_STEPS = ("segmentation", "reach_detection", "outcome", "assignment")


def _gt_resolve_if_present(snapshot_dir: Path, gt_dir: Optional[Path]) -> None:
    """Run GT auto-resolve on ``algo_outputs_current/`` so outcome's
    two-level metrics populate. No-op if nothing to resolve."""
    try:
        from mousereach.triage.gt_resolve import resolve_dir
    except ImportError:
        print("  [skip] mousereach.triage.gt_resolve unavailable")
        return
    cand_algo = None
    for name in ("algo_outputs_current", "algo_outputs"):
        if (snapshot_dir / name).is_dir():
            cand_algo = snapshot_dir / name
            break
    if cand_algo is None:
        print("  [skip] no algo_outputs dir under snapshot")
        return
    cand_gt = gt_dir or (snapshot_dir / "gt" if (snapshot_dir / "gt").is_dir() else None)
    if cand_gt is None or not Path(cand_gt).is_dir():
        print("  [skip] no GT dir found for gt_resolve")
        return
    print(f"\n========== GT auto-resolve :: {snapshot_dir.name} ==========")
    resolve_dir(cand_algo, gt_dir=Path(cand_gt), verbose=False)


def eval_all(
    snapshot_dir: Path,
    *,
    only: Optional[str] = None,
    skip_gt_resolve: bool = False,
    gt_dir: Optional[Path] = None,
    open_figures: bool = False,
) -> None:
    snapshot_dir = Path(snapshot_dir).resolve()
    print(f"Snapshot: {snapshot_dir}")
    if not snapshot_dir.is_dir():
        raise FileNotFoundError(snapshot_dir)

    if only:
        steps: List[str] = [STEP_ALIASES[only.lower()]]
    else:
        steps = list(ALL_STEPS)

    # Outcome's two-level metrics need GT auto-resolve to have stamped
    # cleared markers. Run it once up front unless suppressed.
    if not skip_gt_resolve and ("outcome" in steps or "assignment" in steps):
        _gt_resolve_if_present(snapshot_dir, gt_dir)

    for step in steps:
        run_one_step(step, snapshot_dir)

    print(f"\n========== DONE :: {snapshot_dir.name} ==========")
    fig_dir = snapshot_dir / "figures"
    if fig_dir.is_dir():
        pngs = sorted(p.name for p in fig_dir.glob("*.png"))
        print(f"Figures emitted in {fig_dir} ({len(pngs)}):")
        for p in pngs:
            print(f"  - {p}")
    if open_figures:
        _open_figures_in_vscode(snapshot_dir, steps)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run every algo evaluator on a snapshot (segmentation, "
            "reach detection, outcome detection, reach assignment). "
            "Default is all four; --only restricts to one. GT auto-resolve "
            "is invoked up front so outcome two-level metrics populate."
        )
    )
    parser.add_argument("snapshot_dir", type=Path,
                        help="Snapshot / quarantine directory.")
    parser.add_argument("--only", choices=sorted(STEP_ALIASES.keys()),
                        help="Run a single step only.")
    parser.add_argument("--no-gt-resolve", action="store_true",
                        help="Skip GT auto-resolve pre-step.")
    parser.add_argument("--gt-dir", type=Path, default=None,
                        help="GT directory (defaults to <snapshot>/gt/).")
    parser.add_argument("--no-open", action="store_true",
                        help="Do not open the rendered figures in VS Code "
                             "(default: open them -- the figures ARE the report).")
    args = parser.parse_args()

    try:
        eval_all(
            args.snapshot_dir,
            only=args.only,
            skip_gt_resolve=args.no_gt_resolve,
            gt_dir=args.gt_dir,
            open_figures=not args.no_open,
        )
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
