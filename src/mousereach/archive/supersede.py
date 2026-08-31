"""
mousereach.archive.supersede -- non-destructive, versioned archiving of superseded
outputs.

When a video is reprocessed to a newer version its OLD outputs are NOT overwritten
-- they are moved aside, checksum-verified, into a version-stamped Archive tree so
the history is preserved and the CURRENT outputs (next to the video) stay clean.

Layout (the DLC pose is stored ONCE per DLC generation; algo outputs live under
their algo-version subfolder, so N algo variants that share one pose never
duplicate the multi-MB h5/csv)::

    Archive/
      DLC Model 3.1/
        <pose: {stem}DLC_resnet50_..._100000.h5/.csv/_meta.pickle>   one copy/gen
        seg2.1.0_reach5.3.0_out2.4.4_asnNA/
          {stem}_reaches.json  {stem}_features.json  ...             algo outputs
          figures/   summary/                                        stats + figures
      README.txt                                                     the scheme

Rules honored (from the agreed Y:/X: storage model):
 - GT (``{stem}_unified_ground_truth.json``) and human reviews
   (``{stem}_causal_review.json``) are NEVER archived -- human truth stays with
   the video.
 - The video itself (``{stem}.mp4`` / ``.mkv``) is NEVER archived -- one live copy
   stays with the video; the collage can regenerate it if ever needed.
 - Moves are CHECKSUM-verified (sha256); the source is deleted ONLY after the copy
   is confirmed byte-identical at the destination.
 - The archive is never clobbered: an identical file already there -> the source is
   simply dropped; a same-named but DIFFERENT file -> the incoming one is versioned
   (``.1``, ``.2`` ...), never overwritten.

ASCII-only console output (Windows cp1252 consoles cannot print Unicode).
"""
from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from ..watcher.transfer import verify_file_hash

logger = logging.getLogger(__name__)

# --- file classification ----------------------------------------------------
_POSE_EXTS = (".h5", ".csv", ".pickle")
_ALGO_SUFFIXES = (
    "_reaches.json", "_features.json", "_pellet_outcomes.json",
    "_segments.json", "_processing_manifest.json", "_triage.json",
    "_reach_assignments.json",
    # The finalized per-video results table. Every entry above is .json, so this
    # one fell through: it is not a pose file either (no "DLC" in the name), so
    # _classify returned None and it was silently overwritten on every reprocess.
    # 1,250 of these exist on disk.
    "_results.csv",
)
# Human truth + the video: never archived by a reprocess.
_NEVER_SUFFIXES = (
    "_unified_ground_truth.json", "_causal_review.json", ".mp4", ".mkv",
)

# Known DLC generations -> friendly folder name (fallback derives from the scorer).
_DLC_GENERATION = {
    "DLC_resnet50_MPSAOct27shuffle1_100000": "DLC Model 3.1",
    "DLC_resnet101_MPSAOct27shuffle3_100000": "DLC Model 4.0",
}


def default_archive_root() -> Optional[Path]:
    """``MouseReach_Pipeline/Archive`` -- inside the pipe root, mirrored to X: by
    the backup watcher (per the storage model). None if the NAS isn't configured."""
    from ..config import Paths
    return (Path(Paths.NAS_ROOT) / "Archive") if Paths.NAS_ROOT else None


def dlc_generation_label(scorer: Optional[str]) -> str:
    """Friendly ``DLC Model X.Y`` folder name for a DLC scorer string."""
    if scorer and scorer in _DLC_GENERATION:
        return _DLC_GENERATION[scorer]
    m = re.match(r"DLC_(\w+?)_.+?shuffle(\d+)_(\d+)", scorer or "")
    if m:
        return f"DLC {m.group(1)} sh{m.group(2)} snap{m.group(3)}"
    return "DLC unknown"


def algo_stack_label(versions: Optional[Dict]) -> str:
    """Compact algo-version subfolder name from a manifest's pipeline_versions."""
    v = versions or {}

    def s(key: str) -> str:
        val = v.get(key)
        return str(val) if val else "NA"

    return (f"seg{s('segmenter')}_reach{s('reach_detector')}"
            f"_out{s('outcome_detector')}_asn{s('assignment')}")


def _versioned(dst: Path) -> Path:
    """A non-clobbering variant of ``dst`` (append .1/.2/... before the suffix)."""
    i = 1
    while True:
        cand = dst.with_name(f"{dst.stem}.{i}{dst.suffix}")
        if not cand.exists():
            return cand
        i += 1


def _checksummed_move(src: Path, dst: Path) -> Optional[Path]:
    """Copy ``src`` to ``dst``, verify sha256, then delete ``src``. Never clobbers:
    an identical file already at ``dst`` -> drop src, return the existing dst; a
    different same-named file -> version the incoming one. Returns the final dest,
    or None on failure (source left intact)."""
    if dst.exists():
        if verify_file_hash(src, dst, "sha256"):
            src.unlink()               # already archived, byte-identical
            return dst
        dst = _versioned(dst)          # different content -> keep both
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(str(src), str(dst))
    except OSError as e:
        logger.error(f"supersede copy failed {src} -> {dst}: {e}")
        if dst.exists():
            dst.unlink(missing_ok=True)
        return None
    if not verify_file_hash(src, dst, "sha256"):
        logger.error(f"supersede checksum mismatch, source kept: {src.name}")
        dst.unlink(missing_ok=True)
        return None
    src.unlink()
    return dst


def _classify(path: Path, video_id: str, old_scorer: Optional[str]) -> Optional[str]:
    """'pose' | 'algo' | 'never' | None (not this video's superseded output)."""
    name = path.name
    low = name.lower()
    for suf in _NEVER_SUFFIXES:
        if low.endswith(suf):
            return "never"
    if path.suffix.lower() in _POSE_EXTS and "DLC" in name:
        # Only the OLD generation's pose is superseded; a current-gen pose that
        # happens to sit here is left alone.
        if old_scorer and old_scorer in name:
            return "pose"
        return None
    for suf in _ALGO_SUFFIXES:
        if name.endswith(suf):
            return "algo"
    return None


def _load_manifest(source_dir: Path, video_id: str) -> Optional[dict]:
    import json
    p = source_dir / f"{video_id}_processing_manifest.json"
    try:
        if p.is_file() and p.stat().st_size > 0:
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def supersede_video_outputs(
    video_id: str,
    source_dir,
    archive_root=None,
    *,
    manifest: Optional[dict] = None,
    extra_files: Optional[List[Path]] = None,
    dry_run: bool = False,
    only: Optional[set] = None,
) -> Dict:
    """Move ``video_id``'s current (about-to-be-superseded) outputs from
    ``source_dir`` into the versioned Archive, checksum-verified, leaving the
    video + any GT/review in place.

    The old version is read from the video's processing manifest in ``source_dir``
    (or the ``manifest`` override): its DLC scorer picks the generation folder and
    its pipeline_versions picks the algo-stack subfolder. ``extra_files`` archives
    additional artifacts (e.g. figures/summary) into the algo-stack subfolder.

    Returns a summary dict; never raises on an individual file (logs + continues)."""
    source_dir = Path(source_dir)
    archive_root = Path(archive_root) if archive_root else default_archive_root()
    if archive_root is None:
        return {"error": "no archive root (NAS not configured)", "pose": [], "algo": []}
    if manifest is None:
        manifest = _load_manifest(source_dir, video_id)
    scorer = ((manifest or {}).get("dlc_model") or {}).get("dlc_scorer")
    versions = (manifest or {}).get("pipeline_versions") or {}
    gen = dlc_generation_label(scorer)
    stack = algo_stack_label(versions)
    pose_dir = archive_root / gen
    algo_dir = archive_root / gen / stack

    summary = {"video_id": video_id, "generation": gen, "stack": stack,
               "archive_root": str(archive_root), "pose": [], "algo": [],
               "kept": [], "failed": [], "dry_run": dry_run}

    candidates = sorted(source_dir.glob(f"{video_id}*"))
    for f in candidates:
        if not f.is_file():
            continue
        # ``only``: sweep nothing the caller is not about to replace. WHY: a
        # RETRY of a partially-failed archive arrives with few local files, and
        # an unscoped sweep then treats the just-archived outputs as a previous
        # generation and guts the bundle -- observed 2026-08-30, two videos
        # archived with their fresh outputs swept away (recovered from here,
        # because this move never clobbers). The manual bring-current tool
        # passes no ``only`` and keeps the sweep-everything behaviour.
        if only is not None and f.name not in only:
            continue
        cat = _classify(f, video_id, scorer)
        if cat is None:
            continue
        if cat == "never":
            summary["kept"].append(f.name)
            continue
        dst = (pose_dir if cat == "pose" else algo_dir) / f.name
        if dry_run:
            summary[cat].append(f.name)
            continue
        res = _checksummed_move(f, dst)
        if res is None:
            summary["failed"].append(f.name)
        else:
            summary[cat].append(f.name)

    for f in (extra_files or []):
        f = Path(f)
        if not f.is_file():
            continue
        dst = algo_dir / f.name
        if dry_run:
            summary["algo"].append(f.name)
            continue
        res = _checksummed_move(f, dst)
        (summary["algo"] if res else summary["failed"]).append(f.name)

    if not dry_run and (summary["pose"] or summary["algo"]):
        write_archive_readme(archive_root)
    return summary


def write_archive_readme(archive_root: Path) -> None:
    """Write/refresh the Archive README documenting the layout. Best-effort."""
    archive_root = Path(archive_root)
    readme = archive_root / "README.txt"
    if readme.exists():
        return
    try:
        archive_root.mkdir(parents=True, exist_ok=True)
        readme.write_text(
            "MouseReach Archive -- superseded (old-version) outputs\n"
            "=====================================================\n\n"
            "This holds outputs from OLD processing generations, moved aside when a\n"
            "video was reprocessed to a newer version so the current outputs (next\n"
            "to the video in Analyzed/) stay clean. Nothing here was deleted from a\n"
            "video -- it was moved, checksum-verified, out of the live tree.\n\n"
            "Layout:\n"
            "  DLC Model <gen>/                      one DLC generation\n"
            "    <stem>DLC_<net>_...h5/.csv/.pickle  the pose, stored ONCE per gen\n"
            "    seg<v>_reach<v>_out<v>_asn<v>/      one algo stack that ran on it\n"
            "        <stem>_reaches.json ...         that stack's algo outputs\n"
            "        figures/  summary/              its figures + summary stats\n\n"
            "The pose is shared across every algo stack in the same DLC generation,\n"
            "so many algo variants never duplicate the multi-MB pose files.\n\n"
            "NEVER archived (these stay with the video): the video (.mp4/.mkv),\n"
            "ground truth (_unified_ground_truth.json), human review\n"
            "(_causal_review.json). Those are human truth or the irreplaceable-ish\n"
            "source and belong with the current data.\n",
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"could not write Archive README: {e}")
