"""Copy-free per-video staging for the Model-4.0 causal-review loop.

This module prepares "review bundles" for the causal review tool. For each
video it runs the shipped-best 4.0 algorithms directly, in this order:

    segmentation (v2.2.2)  ->  reach detection (v8.1.0, w=0.7 model4.0)
    ->  outcome cascade (v6.1.0)  ->  reach assignment (v2.0.0 agreement gate)

Design goal (explicit user constraint): **do not duplicate the bulky video /
pose files, and never write anything to the C: system drive.** The algos are
called on the video's *canonical* Y: locations in place:

    mp4  :  <Connectome>/CNT##/{stem}.mp4
    pose :  <Connectome>/DLC Model 4/CNT##/{stem}DLC_resnet101_..shuffle3_100000.h5

The bundle written into ``<pending_dir>/{stem}/`` therefore contains only:

    {stem}_segments.json
    {stem}_reaches.json
    {stem}_pellet_outcomes.json
    {stem}_reach_assignments.json
    manifest.json          <- pointers back to the canonical mp4 + pose, plus provenance

The review tool reads the four JSONs from the bundle and loads the mp4/pose
from the manifest's canonical paths (see ``CausalReviewWidget.load_from_manifest``).

Nothing here writes to the production ``Processing`` dir or to C:. The only new
bytes are a few hundred KB of JSON per video, on Y:.

NB: this file is run against Y: master code (shipped-best). Because production's
C: editable install is intentionally frozen at the pre-4.0 versions until the
coordinated activation flip, invoke with ``PYTHONPATH`` pinned to the Y: master
``src`` so ``import mousereach`` resolves to the shipped-best tree:

    $env:PYTHONPATH = "Y:\\2_Connectome\\Behavior\\MouseReach\\src"
    <mousereach-env-python> -m mousereach.review.staging --cnt CNT01 --limit 3
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Canonical roots (Y: NAS). Everything read/written lives under here; never C:.
DEFAULT_CONNECTOME_ROOT = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Analyzed\Connectome"
)
DEFAULT_PENDING_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Model40_Review\Pending"
)
DLC4_SUBDIR = "DLC Model 4"
# 4.0 pose scorer signature (ResNet101, shuffle 3, snapshot 100000).
DLC4_SCORER = "DLC_resnet101_MPSAOct27shuffle3_100000"

MANIFEST_TYPE = "mousereach_causal_review_manifest"
MANIFEST_SCHEMA_VERSION = "1.0"

_CNT_RE = re.compile(r"CNT(\d{2})\d{2}", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Canonical path resolution
# ---------------------------------------------------------------------------
def cohort_dir_for_stem(stem: str) -> str:
    """Return the CNT## cohort folder name for a video stem.

    e.g. ``20250624_CNT0101_P1`` -> ``CNT01`` (cohort 01, subject 01).
    The bulk 4.0 tree groups videos by cohort, matching this convention.
    """
    m = _CNT_RE.search(stem)
    if not m:
        raise ValueError(f"Cannot derive CNT cohort from stem: {stem!r}")
    return f"CNT{m.group(1)}"


def resolve_canonical_paths(
    stem: str, connectome_root: Path,
    pose_dirs: Optional[List[Path]] = None,
    mp4_dirs: Optional[List[Path]] = None,
) -> Dict[str, Path]:
    """Resolve the canonical mp4 + 4.0 pose h5 for a stem. Raises if missing.

    ``pose_dirs`` are extra directories searched for the 4.0 pose h5 when it is
    not in the standard ``DLC Model 4/<cohort>`` tree. ``mp4_dirs`` are extra
    directories searched for the mp4 when it is not archived in the canonical
    Connectome tree (e.g. still sitting in a DLC_Queue / DLC_Complete staging
    dir). Both are used in place -- nothing is copied into the archive.
    """
    cohort = cohort_dir_for_stem(stem)
    mp4 = connectome_root / cohort / f"{stem}.mp4"
    dlc_dir = connectome_root / DLC4_SUBDIR / cohort
    # Prefer the exact scorer name; fall back to a shuffle3 glob for robustness.
    h5 = dlc_dir / f"{stem}{DLC4_SCORER}.h5"
    if not h5.exists():
        cands = sorted(dlc_dir.glob(f"{stem}DLC_resnet101*shuffle3*.h5"))
        h5 = cands[0] if cands else h5
    if not h5.exists() and pose_dirs:
        for pdir in pose_dirs:
            cands = sorted(Path(pdir).glob(f"{stem}DLC_resnet101*shuffle3*.h5"))
            if cands:
                h5 = cands[0]
                break
    if not mp4.exists() and mp4_dirs:
        for mdir in mp4_dirs:
            cand = Path(mdir) / f"{stem}.mp4"
            if cand.exists():
                mp4 = cand
                break
    if not mp4.exists():
        raise FileNotFoundError(f"mp4 not found for {stem}: {mp4}")
    if not h5.exists():
        raise FileNotFoundError(f"canonical 4.0 pose h5 not found for {stem} in {dlc_dir}")
    return {"mp4": mp4, "h5": h5, "mp4_dir": mp4.parent, "cohort": cohort}


# ---------------------------------------------------------------------------
# JSON parsing helpers (tolerate the known schema shapes)
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_boundaries(seg_doc: Dict[str, Any]) -> List[int]:
    """Extract the ordered boundary frame list from a _segments.json doc."""
    if "boundaries" in seg_doc:
        out = []
        for b in seg_doc["boundaries"]:
            out.append(int(b["frame"]) if isinstance(b, dict) else int(b))
        return out
    if "segmentation" in seg_doc and "boundaries" in seg_doc["segmentation"]:
        return [int(b["frame"]) if isinstance(b, dict) else int(b)
                for b in seg_doc["segmentation"]["boundaries"]]
    raise ValueError("could not parse boundaries from segments doc")


def _segments_as_tuples(seg_doc: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Return per-segment (start_frame, end_frame) inclusive tuples."""
    b = _parse_boundaries(seg_doc)
    return [(b[i], b[i + 1] - 1) for i in range(len(b) - 1)]


def _reaches_as_tuples(reach_doc: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Return (start_frame, end_frame) tuples for the outcome cascade input."""
    from mousereach.assignment.cli import _reaches_list
    out: List[Tuple[int, int]] = []
    for r in _reaches_list(reach_doc):
        s = r.get("start_frame", r.get("start"))
        e = r.get("end_frame", r.get("end"))
        if s is not None and e is not None:
            out.append((int(s), int(e)))
    return out


# ---------------------------------------------------------------------------
# Side-effect neutralization
# ---------------------------------------------------------------------------
_SIDE_EFFECTS_DISABLED = False


def _neutralize_pipeline_side_effects() -> None:
    """Silence the production pipeline's write-side effects for this process.

    ``save_segmentation`` and ``ReachDetector.save_results`` each best-effort
    update the pipeline index (``C:\\...\\pipeline_index.json``) and, for reach,
    sync to a central DB. During staging those are (a) a violation of the
    "never write to C:" rule and (b) a correctness hazard under parallel
    staging -- every worker rewrites the same index file, which yields
    ``WinError 32`` + "Corrupt index file, rebuilding". The review bundles are
    self-contained JSON on Y:, so the index/DB updates are simply not wanted.

    Idempotent; monkeypatches only this process's imported objects (workers get
    it too because ProcessPoolExecutor re-imports this module on spawn).
    """
    global _SIDE_EFFECTS_DISABLED
    if _SIDE_EFFECTS_DISABLED:
        return
    try:
        from mousereach.index import PipelineIndex

        def _noop_load(self):
            self._loaded = True
            if getattr(self, "_data", None) is None:
                self._data = {"videos": {}, "folder_mtimes": {}}
            return self._data

        PipelineIndex.load = _noop_load
        PipelineIndex.save = lambda self: None
        PipelineIndex.record_file_created = lambda self, *a, **k: None
    except Exception:
        pass
    try:
        import mousereach.sync.database as _db  # local import target of save_results
        _db.sync_file_to_database = lambda *a, **k: None
    except Exception:
        pass
    _SIDE_EFFECTS_DISABLED = True


# ---------------------------------------------------------------------------
# The stager
# ---------------------------------------------------------------------------
def stage_video(
    stem: str,
    *,
    connectome_root: Path = DEFAULT_CONNECTOME_ROOT,
    pending_dir: Path = DEFAULT_PENDING_DIR,
    overwrite: bool = False,
    verbose: bool = True,
    pose_dirs: Optional[List[Path]] = None,
    mp4_dirs: Optional[List[Path]] = None,
    boundaries_override: Optional[List[int]] = None,
    preserve_clears: bool = True,
) -> Path:
    """Run the 4.0 algos on a video's canonical files and write a review bundle.

    Returns the bundle directory. Reads only canonical Y: paths; writes only
    the four small JSONs + manifest into ``pending_dir/{stem}/``. No copies.
    """
    _neutralize_pipeline_side_effects()  # no C: index / DB writes; parallel-safe

    # Imports are local so the module loads even if a dependency is unavailable.
    from mousereach.segmentation.core.segmenter_multi import segment_video_multi
    from mousereach.segmentation.core.segmenter_robust import save_segmentation
    from mousereach.reach.core.span_to_reaches import detect_video_reaches
    from mousereach.reach.core.reach_detector import ReachDetector
    from mousereach.outcomes.v6_cascade import detect_outcomes_v6_cascade
    from mousereach.reach.v8.features import load_dlc_h5
    from mousereach.assignment.cli import _segments_with_outcomes, _reaches_list
    from mousereach.assignment.v2 import assign_reaches_v2

    def log(msg: str) -> None:
        if verbose:
            print(f"[stage {stem}] {msg}")

    paths = resolve_canonical_paths(stem, connectome_root, pose_dirs=pose_dirs, mp4_dirs=mp4_dirs)
    mp4, h5, mp4_dir = paths["mp4"], paths["h5"], paths["mp4_dir"]

    bundle = pending_dir / stem
    seg_out = bundle / f"{stem}_segments.json"
    reach_out = bundle / f"{stem}_reaches.json"
    outcome_out = bundle / f"{stem}_pellet_outcomes.json"
    assign_out = bundle / f"{stem}_reach_assignments.json"
    manifest_out = bundle / "manifest.json"

    if bundle.exists() and manifest_out.exists() and not overwrite:
        log(f"already staged (manifest present); skipping. Use overwrite=True to redo.")
        return bundle
    bundle.mkdir(parents=True, exist_ok=True)

    # Clear-guard: capture any human triage-clears BEFORE the fresh run
    # overwrites the JSONs, so they can be re-applied afterwards. Skipped when
    # boundaries change (manual re-seg renumbers segments -> can't map clears).
    _old_outcome = _load_json(outcome_out) if (preserve_clears and outcome_out.exists()) else None
    _old_reach = _load_json(reach_out) if (preserve_clears and reach_out.exists()) else None

    # 1. SEGMENTATION (v2.2.2) -- OR a manual re-segmentation: when the reviewer
    # supplies boundaries, skip auto-segmentation and write a minimal manual
    # segments.json (downstream reach/outcome/assignment read only the boundary
    # list + frame count). Re-running the algos on the reviewer's cuts means
    # UNCHANGED segments keep the algo's scoring and every segment still shows an
    # algo verdict -- fixing one boundary no longer forces a full re-score.
    if boundaries_override is not None:
        log("manual re-segmentation (given boundaries; skipping auto-segment)...")
        _b = sorted({int(x) for x in boundaries_override})
        _total = None
        if seg_out.exists():
            try:
                _total = int(_load_json(seg_out).get("total_frames"))
            except Exception:
                _total = None
        if _total is None:
            _total = int(len(load_dlc_h5(h5)))
        seg_out.write_text(json.dumps({
            "segmenter_version": "manual_resegmentation",
            "segmenter_algorithm": "manual",
            "video_name": stem,
            "total_frames": _total,
            "fps": 60.0,
            "boundaries": _b,
            "reference_quality": "manual",
            "overall_confidence": 1.0,
            "anomalies": [],
            "anomaly_summary": {"critical": 0, "warning": 0, "info": 0},
            "manual_resegmentation": True,
        }, indent=2), encoding="utf-8")
    else:
        log("segment...")
        boundaries, diagnostics = segment_video_multi(h5)
        save_segmentation(boundaries, diagnostics, seg_out)

    # 2. REACH DETECTION (v8.1.0 w=0.7 model4.0) -- reads pose + segments, no gate
    log("detect reaches...")
    reach_results = detect_video_reaches(h5, seg_out)
    ReachDetector.save_results(reach_results, reach_out, validation_status="needs_review")

    # 3. OUTCOME CASCADE (v6.1.0) -- reads pose + segments + reaches + mp4 (CV gate)
    log("detect outcomes (CV gate via canonical mp4 dir)...")
    dlc_df = load_dlc_h5(h5)
    seg_doc = _load_json(seg_out)
    reach_doc = _load_json(reach_out)
    segments = _segments_as_tuples(seg_doc)
    reaches_tuples = _reaches_as_tuples(reach_doc)
    outcome_result = detect_outcomes_v6_cascade(
        dlc_df=dlc_df,
        segments=segments,
        reaches=reaches_tuples,
        video_id=stem,
        video_dir=mp4_dir,  # points at canonical CNT##/ containing {stem}.mp4
    )
    outcome_out.write_text(json.dumps(outcome_result, indent=2), encoding="utf-8")

    # 4. ASSIGNMENT v2 (2.0.0 agreement gate) -- CLI still v1, call v2 directly
    log("assign reaches (v2 agreement gate)...")
    merged_segs = _segments_with_outcomes(seg_doc, outcome_result)
    reaches_list = _reaches_list(reach_doc)
    assign_result = assign_reaches_v2(
        reaches=reaches_list,
        segments_with_outcomes=merged_segs,
        dlc_df=dlc_df,
        video_id=stem,
    )
    assign_out.write_text(json.dumps(assign_result, indent=2) + "\n", encoding="utf-8")

    # 4b. CLEAR-GUARD: re-apply human triage-clears the fresh run just
    # overwrote. Human calls win over the recomputed algo call. Skipped for
    # manual re-segmentation (boundaries changed -> segment_num can't be mapped
    # safely; the reviewer is intentionally re-scoring).
    if preserve_clears and boundaries_override is None and (_old_outcome or _old_reach):
        from .clear_guard import (
            merge_preserving_clears, is_outcome_locked, is_reach_locked)
        if _old_outcome:
            _new_o = _load_json(outcome_out)
            _new_o, kept, skipped = merge_preserving_clears(_new_o, _old_outcome, is_outcome_locked)
            if kept:
                outcome_out.write_text(json.dumps(_new_o, indent=2), encoding="utf-8")
                log(f"clear-guard: preserved {len(kept)} human-cleared outcome segment(s) {kept}")
            if skipped:
                log(f"clear-guard: WARNING {len(skipped)} cleared outcome segment(s) "
                    f"not re-applied (segment_num absent after re-run): {skipped}")
        if _old_reach:
            _new_r = _load_json(reach_out)
            _new_r, kept, skipped = merge_preserving_clears(_new_r, _old_reach, is_reach_locked)
            if kept:
                reach_out.write_text(json.dumps(_new_r, indent=2), encoding="utf-8")
                log(f"clear-guard: preserved {len(kept)} human-cleared reach segment(s) {kept}")
            if skipped:
                log(f"clear-guard: WARNING {len(skipped)} cleared reach segment(s) "
                    f"not re-applied (segment_num absent after re-run): {skipped}")

    # 5. MANIFEST -- pointers to canonical files + provenance (versions, sources)
    manifest = _build_manifest(
        stem=stem, mp4=mp4, h5=h5,
        seg_doc=seg_doc, reach_doc=reach_doc,
        outcome_result=outcome_result, assign_result=assign_result,
    )
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log(f"bundle ready: {bundle}")
    return bundle


def _mtime_iso(p: Path) -> Optional[str]:
    try:
        return datetime.fromtimestamp(Path(p).stat().st_mtime).isoformat()
    except OSError:
        return None


def _build_manifest(*, stem: str, mp4: Path, h5: Path,
                    seg_doc: Dict[str, Any], reach_doc: Dict[str, Any],
                    outcome_result: Dict[str, Any], assign_result: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble the bundle manifest: canonical pointers + algo provenance.

    Version fields record the AUTHORITATIVE running constants, not the JSON
    fields: ``save_segmentation`` mislabels seg with segmenter_robust's 2.1.3
    even though ``segment_video_multi`` runs 2.2.2, and the cascade stamps its
    version under ``detector_version``. The mislabeled seg field is kept
    transparently under ``segmenter_version_json_field``.
    """
    try:
        from mousereach.segmentation.core.segmenter_multi import SEGMENTER_VERSION as _seg_v
    except Exception:
        _seg_v = seg_doc.get("segmenter_version")
    try:
        from mousereach.outcomes.v6_cascade import VERSION as _out_v
    except Exception:
        _out_v = outcome_result.get("detector_version")
    provenance = {
        "staged_at": datetime.now().isoformat(),
        "dlc_model": DLC4_SCORER,
        "segmenter_version": _seg_v,
        "segmenter_version_json_field": seg_doc.get("segmenter_version"),
        "reach_detector_version": reach_doc.get("version") or reach_doc.get("detector_version"),
        "outcome_detector_version": _out_v,
        "assignment_version": assign_result.get("version"),
        "assignment_detector": assign_result.get("detector"),
        "source_mp4": {"path": str(mp4), "mtime": _mtime_iso(mp4)},
        "source_pose_h5": {"path": str(h5), "mtime": _mtime_iso(h5)},
    }
    return {
        "type": MANIFEST_TYPE,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "video_stem": stem,
        "canonical_video_path": str(mp4),
        "canonical_dlc_h5_path": str(h5),
        "provenance": provenance,
    }


# ---------------------------------------------------------------------------
# Batch discovery + CLI
# ---------------------------------------------------------------------------
def list_stems_for_cohort(cohort: str, connectome_root: Path = DEFAULT_CONNECTOME_ROOT) -> List[str]:
    """List _P# video stems in a cohort that have a 4.0 pose h5 available."""
    dlc_dir = connectome_root / DLC4_SUBDIR / cohort
    stems = []
    for h5 in sorted(dlc_dir.glob("*shuffle3*.h5")):
        stem = h5.name.split("DLC")[0]
        stems.append(stem)
    return stems


def list_all_cohorts(connectome_root: Path = DEFAULT_CONNECTOME_ROOT) -> List[str]:
    """Every CNT## cohort folder that has a 4.0 pose tree (skips ``_logs`` etc.)."""
    dlc_root = connectome_root / DLC4_SUBDIR
    if not dlc_root.exists():
        return []
    return sorted(d.name for d in dlc_root.iterdir()
                  if d.is_dir() and d.name.upper().startswith("CNT"))


def list_all_stems(connectome_root: Path = DEFAULT_CONNECTOME_ROOT) -> List[str]:
    """All DLC-4.0-processed video stems across every cohort."""
    stems: List[str] = []
    for c in list_all_cohorts(connectome_root):
        stems.extend(list_stems_for_cohort(c, connectome_root))
    return stems


def _stage_one_result(stem: str, connectome_root: Path, pending_dir: Path,
                      overwrite: bool) -> Tuple[str, bool, Optional[str]]:
    """Top-level worker for parallel batches: stage one video, never raise.

    Must be module-level (picklable) so ProcessPoolExecutor can dispatch it.
    """
    try:
        stage_video(stem, connectome_root=connectome_root,
                    pending_dir=pending_dir, overwrite=overwrite, verbose=False)
        return (stem, True, None)
    except Exception as e:  # noqa: BLE001 -- keep the batch alive on a bad video
        return (stem, False, repr(e))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 4.0 review bundles (seg->reach->outcome->assignment v2) "
                    "into Model40_Review/Pending. Copy-free: reads canonical Y: paths, "
                    "writes only small JSONs + manifest. Never touches C:.",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--stems", nargs="+", help="Explicit video stems, e.g. 20250624_CNT0101_P1")
    src.add_argument("--cnt", help="Cohort folder (e.g. CNT01); stage its videos")
    src.add_argument("--all", action="store_true",
                     help="Stage EVERY DLC-4.0 video across all cohorts (the full review corpus)")
    parser.add_argument("--limit", type=int, default=None, help="With --cnt/--all, cap number of videos")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel staging processes (default 1 = serial)")
    parser.add_argument("--include-gt", action="store_true",
                        help="Also stage already-ground-truthed videos (default: skip; GT is the answer)")
    parser.add_argument("--connectome-root", type=Path, default=DEFAULT_CONNECTOME_ROOT)
    parser.add_argument("--pending-dir", type=Path, default=DEFAULT_PENDING_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Restage even if a bundle exists")
    args = parser.parse_args(argv)

    if args.stems:
        stems = list(args.stems)
    elif args.all:
        stems = list_all_stems(args.connectome_root)
    else:
        stems = list_stems_for_cohort(args.cnt, args.connectome_root)

    # For cohort/all batches, skip work that would never be reviewed anyway:
    # already-ground-truthed videos (GT stands in for a review) and, unless
    # restaging, bundles already present. Explicit --stems are always honored.
    if not args.stems:
        n0 = len(stems)
        if not args.include_gt:
            from mousereach.review.causal_review_io import has_gt
            stems = [s for s in stems if not has_gt(s)]
        if not args.overwrite:
            already = ({d.name for d in args.pending_dir.iterdir() if d.is_dir()}
                       if args.pending_dir.exists() else set())
            stems = [s for s in stems if s not in already]
        skipped = n0 - len(stems)
        if skipped:
            print(f"Skipped {skipped} video(s) already GT'd or staged.")
    if args.limit:
        stems = stems[: args.limit]

    print(f"Staging {len(stems)} video(s) into {args.pending_dir} "
          f"(workers={args.workers})", flush=True)
    ok, failed = [], []
    total = len(stems)

    if args.workers > 1 and total > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_stage_one_result, s, args.connectome_root,
                              args.pending_dir, args.overwrite): s for s in stems}
            for i, fut in enumerate(as_completed(futs), 1):
                stem, good, err = fut.result()
                if good:
                    ok.append(stem)
                else:
                    failed.append((stem, err))
                    print(f"[{i}/{total}] FAILED {stem}: {err}", flush=True)
                if good and (i % 25 == 0 or i == total):
                    print(f"[{i}/{total}] staged (ok={len(ok)} fail={len(failed)})", flush=True)
    else:
        for i, stem in enumerate(stems, 1):
            try:
                stage_video(stem, connectome_root=args.connectome_root,
                            pending_dir=args.pending_dir, overwrite=args.overwrite,
                            verbose=False)
                ok.append(stem)
            except Exception as e:  # keep the batch alive on a single bad video
                failed.append((stem, repr(e)))
                print(f"[{i}/{total}] FAILED {stem}: {e}", flush=True)
            if i % 25 == 0 or i == total:
                print(f"[{i}/{total}] staged (ok={len(ok)} fail={len(failed)})", flush=True)

    print(f"\nDone. staged={len(ok)}  failed={len(failed)}", flush=True)
    for stem, err in failed:
        print(f"  FAILED {stem}: {err}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
