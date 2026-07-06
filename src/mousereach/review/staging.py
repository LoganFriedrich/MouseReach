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


def resolve_canonical_paths(stem: str, connectome_root: Path) -> Dict[str, Path]:
    """Resolve the canonical mp4 + 4.0 pose h5 for a stem. Raises if missing."""
    cohort = cohort_dir_for_stem(stem)
    mp4 = connectome_root / cohort / f"{stem}.mp4"
    dlc_dir = connectome_root / DLC4_SUBDIR / cohort
    # Prefer the exact scorer name; fall back to a shuffle3 glob for robustness.
    h5 = dlc_dir / f"{stem}{DLC4_SCORER}.h5"
    if not h5.exists():
        cands = sorted(dlc_dir.glob(f"{stem}DLC_resnet101*shuffle3*.h5"))
        h5 = cands[0] if cands else h5
    if not mp4.exists():
        raise FileNotFoundError(f"canonical mp4 not found: {mp4}")
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
# The stager
# ---------------------------------------------------------------------------
def stage_video(
    stem: str,
    *,
    connectome_root: Path = DEFAULT_CONNECTOME_ROOT,
    pending_dir: Path = DEFAULT_PENDING_DIR,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Run the 4.0 algos on a video's canonical files and write a review bundle.

    Returns the bundle directory. Reads only canonical Y: paths; writes only
    the four small JSONs + manifest into ``pending_dir/{stem}/``. No copies.
    """
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

    paths = resolve_canonical_paths(stem, connectome_root)
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

    # 1. SEGMENTATION (v2.2.2) -- reads pose only
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 4.0 review bundles (seg->reach->outcome->assignment v2) "
                    "into Model40_Review/Pending. Copy-free: reads canonical Y: paths, "
                    "writes only small JSONs + manifest. Never touches C:.",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--stems", nargs="+", help="Explicit video stems, e.g. 20250624_CNT0101_P1")
    src.add_argument("--cnt", help="Cohort folder (e.g. CNT01); stage its videos")
    parser.add_argument("--limit", type=int, default=None, help="With --cnt, cap number of videos")
    parser.add_argument("--connectome-root", type=Path, default=DEFAULT_CONNECTOME_ROOT)
    parser.add_argument("--pending-dir", type=Path, default=DEFAULT_PENDING_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Restage even if a bundle exists")
    args = parser.parse_args(argv)

    if args.stems:
        stems = list(args.stems)
    else:
        stems = list_stems_for_cohort(args.cnt, args.connectome_root)
        if args.limit:
            stems = stems[: args.limit]

    print(f"Staging {len(stems)} video(s) into {args.pending_dir}")
    ok, failed = [], []
    for stem in stems:
        try:
            stage_video(stem, connectome_root=args.connectome_root,
                        pending_dir=args.pending_dir, overwrite=args.overwrite)
            ok.append(stem)
        except Exception as e:  # keep the batch alive on a single bad video
            failed.append((stem, repr(e)))
            print(f"[stage {stem}] FAILED: {e}")
            traceback.print_exc()

    print(f"\nDone. staged={len(ok)}  failed={len(failed)}")
    for stem, err in failed:
        print(f"  FAILED {stem}: {err}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
