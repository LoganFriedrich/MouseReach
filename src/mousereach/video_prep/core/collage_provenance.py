"""Collage provenance -- what was done to a collage's offspring, attached to the
collage.

A multi-animal collage (``{date}_{id1,...,id8}_{last}.mkv``) is cropped into up
to 8 single-animal videos named ``{date}_{animal_id}_{last}.mp4`` (cohort-00
positions are blank and skipped). The offspring names are therefore FULLY
DETERMINED by the collage filename -- which is what lets us tell, at any time,
whether a collage has been cropped and how far each child got, even though the
raw crop step historically wrote nothing back.

This module makes that provenance explicit and durable:

* ``write_crop_manifest`` persists ``{collage_stem}_crop_manifest.json`` NEXT TO
  the collage at crop time (origin-source-title-first, per the project naming
  rule): every position -> animal_id / status / offspring name, plus timestamp
  and tool version. "What was done to my offspring" now lives with the collage.
* ``expected_offspring`` / ``derive_offspring_status`` reconstruct that mapping
  from the filename alone, so a collage with no manifest yet (all the historical
  ones) can still be resolved by checking whether its offspring exist downstream.
* ``backfill_manifests`` writes manifests for already-cropped collages so history
  is captured, not just future crops.

ASCII-only console output (Windows cp1252 consoles cannot print Unicode).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from .cropper import parse_collage_filename, is_blank_animal

logger = logging.getLogger(__name__)

CROP_MANIFEST_SUFFIX = "_crop_manifest.json"

# Crop status buckets for a collage, from its offspring:
CROP_UNCROPPED = "uncropped"   # no offspring exist yet
CROP_PARTIAL = "partial"       # some but not all offspring exist
CROP_CROPPED = "cropped"       # every expected offspring exists

# An offspring has "made it through the entire pipeline" when it reaches the final
# Analyzed output. A collage is retirement-eligible only when EVERY offspring is
# there -- a child still processing, or held in a review queue, keeps the collage
# in the active intake folder.
COMPLETE_STAGES = {"analyzed"}


def normalize_video_stem(p) -> str:
    """Single-animal session stem from any pipeline file name (video, .h5,
    DLC output, '_full' copy): strips the DLC suffix and the '_full' marker.
    THE one place this rule lives -- the downstream index and the census both
    normalize through here, so a DLC artifact can never be counted as a
    different video than its session."""
    return Path(p).stem.split("DLC")[0].replace("_full", "").strip("_")


def _tool_version() -> Optional[str]:
    try:
        from mousereach import __version__  # type: ignore
        return __version__
    except Exception:
        return None


def expected_offspring(collage: str) -> List[Dict]:
    """From a collage filename OR bare stem, the deterministic offspring list:
    ``[{position, animal_id, offspring_stem, blank}]`` (blanks included but marked
    so callers can skip them). Returns [] if the name cannot be parsed."""
    name = collage if collage.lower().endswith((".mkv", ".mp4")) else f"{collage}.mkv"
    try:
        info = parse_collage_filename(name)
    except Exception:
        return []
    out: List[Dict] = []
    for i, aid in enumerate(info["animal_ids"]):
        blank = is_blank_animal(aid)
        out.append({
            "position": i + 1,
            "animal_id": aid,
            "offspring_stem": None if blank else f"{info['date']}_{aid}_{info['last_part']}",
            "blank": blank,
        })
    return out


_PROC_MANIFEST_SUFFIX = "_processing_manifest.json"


_STAGE_PRIORITY = {"cropped": 30, "dlc_complete": 40, "processing": 50,
                   "analyzed": 100, "triage": 120, "deep_review": 120}


def build_downstream_index() -> Dict[str, str]:
    """``{single-animal stem -> furthest pipeline stage}`` across the pipeline
    folders (the same furthest-wins logic the dashboard uses). Used by the
    watcher's auto-retire and by ``retire_completed_collages`` callers to resolve
    each collage's offspring. Never raises on a missing folder."""
    from mousereach.config import Paths

    idx: Dict[str, str] = {}

    def _norm(p: Path) -> str:
        return normalize_video_stem(p)

    def _add(folder, state):
        if not folder or not Path(folder).exists():
            return
        for p in Path(folder).rglob("*"):
            if p.suffix.lower() in (".mp4", ".mkv", ".h5"):
                s = _norm(p)
                if s and (s not in idx or
                          _STAGE_PRIORITY.get(state, 0) > _STAGE_PRIORITY.get(idx[s], 0)):
                    idx[s] = state

    _add(Paths.SINGLE_ANIMAL_OUTPUT, "cropped")
    _add(Paths.DLC_STAGING, "dlc_complete")
    _add(Paths.PROCESSING, "processing")
    _add(Paths.ANALYZED_OUTPUT, "analyzed")
    for root, state in ((Paths.TRIAGE_REVIEW, "triage"), (Paths.DEEP_REVIEW, "deep_review")):
        if root and Path(root).exists():
            for d in Path(root).iterdir():
                if d.is_dir():
                    idx[d.name] = state
    return idx


def _review_pending(stem: str, manifest_dir: Path) -> bool:
    """True if a saved human review for this offspring is NEWER than its archived
    features (its triage resolution has not been applied to the shipped product),
    or a review exists with no features yet. Never raises."""
    try:
        from mousereach.review.causal_review_io import resolve_review_path
        review = resolve_review_path(stem)
        if review is None:
            return False
        feats = Path(manifest_dir) / f"{stem}_features.json"
        if not feats.exists():
            return True
        return review.stat().st_mtime > feats.stat().st_mtime
    except Exception:
        return False


def build_complete_stems(analyzed_root=None, nas_root=None) -> set:
    """Set of single-animal stems that are TRULY complete -- i.e. safe to treat a
    collage as retirement-eligible on.

    An offspring counts ONLY when it is (a) in the final Analyzed output, (b)
    processed with the CURRENTLY shipped versions (its processing manifest matches
    pipeline_versions.json -- not an outdated DLC model or algo version), and (c)
    has no human review still pending application. An offspring sitting in Analyzed
    from an old version, or with an unresolved review, is deliberately EXCLUDED --
    it is not done, it is stale. Never raises; missing manifest -> not complete."""
    from mousereach.config import Paths
    from mousereach.pipeline.versions import (
        get_current_versions, compare_manifest_to_current)

    analyzed_root = Path(analyzed_root or Paths.ANALYZED_OUTPUT)
    nas_root = nas_root or Paths.NAS_ROOT
    complete: set = set()
    if not analyzed_root.exists():
        return complete
    current = get_current_versions(nas_root)
    for mp in analyzed_root.rglob(f"*{_PROC_MANIFEST_SUFFIX}"):
        stem = mp.name[: -len(_PROC_MANIFEST_SUFFIX)]
        try:
            manifest = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            continue
        cmp = compare_manifest_to_current(manifest, current)
        if not cmp.get("is_current"):
            continue  # outdated version -> not done
        if _review_pending(stem, mp.parent):
            continue  # a human review still to apply -> not done
        complete.add(stem)
    return complete


def crop_manifest_path(collage_path, manifest_dir=None) -> Path:
    """Where a collage's crop manifest lives -- next to the collage by default."""
    collage_path = Path(collage_path)
    base = Path(manifest_dir) if manifest_dir else collage_path.parent
    return base / f"{collage_path.stem}{CROP_MANIFEST_SUFFIX}"


def read_crop_manifest(collage_path, manifest_dir=None) -> Optional[dict]:
    p = crop_manifest_path(collage_path, manifest_dir)
    try:
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def build_crop_manifest(collage_path, results: List[dict], output_dir) -> dict:
    """Assemble the crop-manifest dict from a ``crop_collage`` results list."""
    collage_path = Path(collage_path)
    info = parse_collage_filename(collage_path.name)
    offspring = []
    for r in results:
        aid = r.get("animal_id")
        out_name = None
        if r.get("output_path"):
            out_name = Path(r["output_path"]).name
        elif r.get("status") == "success" and aid:
            out_name = f"{info['date']}_{aid}_{info['last_part']}.mp4"
        offspring.append({
            "position": r.get("position"),
            "animal_id": aid,
            "status": r.get("status"),          # success | skipped | failed | error
            "reason": r.get("reason"),
            "output_name": out_name,
        })
    n_expected = sum(1 for a in info["animal_ids"] if not is_blank_animal(a))
    n_written = sum(1 for r in results if r.get("status") == "success")
    return {
        "type": "collage_crop_manifest",
        "schema_version": "1.0",
        "collage": collage_path.name,
        "date": info["date"],
        "last_part": info["last_part"],
        "animal_ids": info["animal_ids"],
        "output_dir": str(output_dir),
        "tool_version": _tool_version(),
        "cropped_at": datetime.now().isoformat(),
        "n_offspring_expected": n_expected,
        "n_offspring_written": n_written,
        "offspring": offspring,
    }


def write_crop_manifest(collage_path, results: List[dict], output_dir,
                        manifest_dir=None) -> Optional[Path]:
    """Persist the crop manifest next to the collage. Never raises (best-effort
    provenance must not break the crop)."""
    try:
        manifest = build_crop_manifest(collage_path, results, output_dir)
        p = crop_manifest_path(collage_path, manifest_dir)
        p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return p
    except Exception as e:
        logger.warning("could not write crop manifest for %s: %s", collage_path, e)
        return None


def derive_offspring_status(collage: str, downstream: Mapping[str, str],
                            complete_stems: Optional[set] = None) -> Dict:
    """Reconstruct a collage's crop status from the filename + a downstream index.

    ``downstream`` maps a single-animal stem -> its furthest pipeline stage (e.g.
    the dashboard's stem->state map). Returns::

        {crop_state, n_expected, n_present, offspring: [{stem, animal_id, stage}]}

    where ``crop_state`` is uncropped / partial / cropped. A collage with a saved
    manifest is trusted for the offspring set, but stages are always read live
    from ``downstream`` so the rollup reflects current reality.

    ``complete_stems`` (from ``build_complete_stems``) is the authoritative set of
    offspring that are truly done -- in Analyzed AND version-current AND with no
    pending review. When given, ``all_complete`` requires every offspring to be in
    it (an outdated or review-pending child does NOT count). When omitted, the
    weaker "in the Analyzed folder" test is used (display-only fallback)."""
    children = [c for c in expected_offspring(collage) if not c["blank"]]
    rows = []
    present = 0
    complete = 0
    for c in children:
        stem = c["offspring_stem"]
        stage = downstream.get(stem)
        if stage:
            present += 1
        if complete_stems is not None:
            is_complete = stem in complete_stems
        else:
            is_complete = stage in COMPLETE_STAGES
        if is_complete:
            complete += 1
        rows.append({"stem": stem, "animal_id": c["animal_id"],
                     "stage": stage, "complete": is_complete})
    n_expected = len(children)
    if present == 0:
        state = CROP_UNCROPPED
    elif present == n_expected:
        state = CROP_CROPPED
    else:
        state = CROP_PARTIAL
    return {
        "crop_state": state,
        "n_expected": n_expected,
        "n_present": present,
        "n_complete": complete,
        # retirement-eligible: every offspring has reached the final Analyzed output
        "all_complete": n_expected > 0 and complete == n_expected,
        "offspring": rows,
    }


def backfill_manifests(source_dir, downstream: Mapping[str, str],
                       write: bool = True) -> Dict:
    """Derive + (optionally) write crop manifests for every collage in
    ``source_dir`` that is already cropped but has no manifest. Returns a summary
    with counts. Existing manifests are left untouched. ``downstream`` is a
    stem->stage index of every single-animal video that exists downstream."""
    source_dir = Path(source_dir)
    summary = {"scanned": 0, "cropped": 0, "partial": 0, "uncropped": 0,
               "written": 0, "already_had_manifest": 0, "errors": 0}
    if not source_dir.exists():
        return summary
    collages = sorted(list(source_dir.glob("*.mkv")) + list(source_dir.glob("*.mp4")))
    for c in collages:
        summary["scanned"] += 1
        try:
            st = derive_offspring_status(c.name, downstream)
            summary[st["crop_state"]] = summary.get(st["crop_state"], 0) + 1
            if st["crop_state"] == CROP_UNCROPPED:
                continue
            if read_crop_manifest(c) is not None:
                summary["already_had_manifest"] += 1
                continue
            if not write:
                continue
            # Synthesize crop results from the derived offspring so the manifest
            # schema matches a live crop; mark it backfilled provenance.
            results = []
            for c_off in expected_offspring(c.name):
                if c_off["blank"]:
                    results.append({"position": c_off["position"],
                                    "animal_id": c_off["animal_id"],
                                    "status": "skipped", "reason": "blank_cohort_00"})
                else:
                    exists = downstream.get(c_off["offspring_stem"]) is not None
                    results.append({
                        "position": c_off["position"],
                        "animal_id": c_off["animal_id"],
                        "status": "success" if exists else "unknown",
                        "output_path": f"{c_off['offspring_stem']}.mp4" if exists else None,
                    })
            manifest = build_crop_manifest(c, results, output_dir="(derived)")
            manifest["backfilled"] = True
            manifest["backfilled_at"] = datetime.now().isoformat()
            manifest["backfill_note"] = ("Derived from offspring existence; the "
                                         "original crop wrote no manifest.")
            crop_manifest_path(c).write_text(json.dumps(manifest, indent=2),
                                             encoding="utf-8")
            summary["written"] += 1
        except Exception as e:
            summary["errors"] += 1
            logger.warning("backfill error on %s: %s", c.name, e)
    return summary


def _completion_results(collage_name: str, downstream: Mapping[str, str]) -> List[dict]:
    """Synthesize a crop-results list (for the manifest) from a collage's derived
    offspring + their current stages."""
    results = []
    for c in expected_offspring(collage_name):
        if c["blank"]:
            results.append({"position": c["position"], "animal_id": c["animal_id"],
                            "status": "skipped", "reason": "blank_cohort_00"})
        else:
            stage = downstream.get(c["offspring_stem"])
            results.append({
                "position": c["position"],
                "animal_id": c["animal_id"],
                "status": "success" if stage else "unknown",
                "output_path": f"{c['offspring_stem']}.mp4" if stage else None,
                "final_stage": stage,
            })
    return results


def retire_completed_collages(source_dir, downstream: Mapping[str, str],
                              complete_stems: Optional[set] = None,
                              dest_dir=None, dry_run: bool = True) -> Dict:
    """Move every collage whose offspring have ALL reached the final Analyzed
    output to ultimate storage.

    A collage is retired only when ``derive_offspring_status(...).all_complete`` --
    i.e. every single-mouse child has made it through the entire pipeline. Children
    still processing, or held in a review queue, keep the collage in the active
    intake folder. Retired collages (and a completion-stamped crop manifest) move
    to ``dest_dir`` (default ``Analyzed/Multi-Animal``), which the backup watcher
    already mirrors to the backup NAS -- so "ultimate storage + copied to the backup"
    happens without a special copy. NEVER deletes; a move is fully reversible.

    ``downstream`` must be a fresh stem->furthest-stage index. ``complete_stems``
    (from ``build_complete_stems``) is the authoritative version-current +
    review-clean completion set -- a collage retires only when EVERY offspring is
    in it. Returns a summary with the list of collages moved (or that would move,
    when ``dry_run``)."""
    import shutil
    from mousereach.config import Paths

    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir) if dest_dir else (Paths.ANALYZED_OUTPUT / "Multi-Animal")
    summary = {"scanned": 0, "retired": 0, "not_complete": 0, "uncropped": 0,
               "errors": 0, "dest": str(dest_dir), "moved": []}
    if not source_dir.exists():
        return summary
    collages = sorted(list(source_dir.glob("*.mkv")) + list(source_dir.glob("*.mp4")))
    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)
    for c in collages:
        summary["scanned"] += 1
        try:
            st = derive_offspring_status(c.name, downstream, complete_stems)
            if st["n_expected"] == 0 or st["n_present"] == 0:
                summary["uncropped"] += 1
                continue
            if not st["all_complete"]:
                summary["not_complete"] += 1
                continue
            summary["moved"].append(c.name)
            if dry_run:
                summary["retired"] += 1
                continue
            # Stamp completion into the manifest, then move collage + manifest.
            manifest = build_crop_manifest(c, _completion_results(c.name, downstream),
                                           output_dir="(retired)")
            manifest["retired"] = True
            manifest["retired_at"] = datetime.now().isoformat()
            manifest["offspring_all_complete"] = True
            mpath = crop_manifest_path(c)
            mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            shutil.move(str(c), str(dest_dir / c.name))
            shutil.move(str(mpath), str(dest_dir / mpath.name))
            summary["retired"] += 1
        except Exception as e:
            summary["errors"] += 1
            logger.warning("retire error on %s: %s", c.name, e)
    return summary
