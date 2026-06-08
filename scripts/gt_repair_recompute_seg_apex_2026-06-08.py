"""
GT repair: recompute segment_num + remove exact-duplicate reach records.
Both corpora. DRY-RUN by default -- writes NOTHING unless --apply is passed.

SCOPE (decided with Logan 2026-06-08)
    Repairs the reload-bug field-crossing (other Claude found it in
    unified_gt.py:1007-1019 and fixed it going forward). On reload, determined
    reaches kept their CORRECT human start/end but inherited segment_num +
    apex_frame from a mismatched algo reach (unstable reach_id key).

    THIS REPAIR TOUCHES EXACTLY TWO THINGS:
      1. segment_num  -> segment whose [boundary_i, boundary_{i+1}) contains the
         reach's start_frame (deterministic, from the file's own boundaries).
      2. exact-duplicate reach records -> collapse (start,end)-identical records
         to one, keeping a determined copy (then lowest reach_id).

    EXPLICITLY NOT TOUCHED:
      - start_frame / end_frame (intact human truth)
      - apex_frame  (LEFT AS-IS: neither _find_apex (47%) nor norm_pos-argmax
        (12%) reproduces the existing apex convention -- likely a different DLC
        vintage -- so recompute would invent values, not restore them. Deferred.)
      - segmentation block, outcomes block
      - near-duplicate (1-2 frame off) records -> REPORTED, not removed.

GUARDRAILS (memory: feedback_forbidden_to_edit_gt, no_agentic_behavior)
    - Modifies no src/ module.  - DLC not needed (seg+dedup are DLC-free).
    - --apply backs up every file first, then HARD-ASSERTS per-reach
      start/end/apex unchanged and segmentation/outcomes blocks byte-identical.
    - Default dry-run. GT writes only under Logan's explicit per-action go.
"""
from __future__ import annotations

import argparse
import json
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime
from pathlib import Path

CORPORA = [
    ("47-corpus",
     Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs\DLC_2026_03_27\gt")),
    ("generalization",
     Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations\generalization_test_2026-05-11\gt")),
]


def seg_of(frames, f):
    if f is None or not frames or f < frames[0] or f >= frames[-1]:
        return None
    return bisect_right(frames, f)


def plan_file(gt_path):
    """Return (json_obj, seg_changes, removals, near_dups) without writing."""
    j = json.loads(gt_path.read_text(encoding="utf-8"))
    frames = [(b.get("frame") if isinstance(b, dict) else b)
              for b in j.get("segmentation", {}).get("boundaries", [])]
    reaches = j.get("reaches", {}).get("reaches", []) or []

    # segment_num changes
    seg_changes = []
    for r in reaches:
        ns = seg_of(frames, r.get("start_frame"))
        if ns is not None and ns != r.get("segment_num"):
            seg_changes.append((r.get("reach_id"), r.get("segment_num"), ns))

    # exact-duplicate removals: keep determined, then lowest reach_id
    groups = defaultdict(list)
    for idx, r in enumerate(reaches):
        s, e = r.get("start_frame"), r.get("end_frame")
        if s is not None and e is not None:
            groups[(s, e)].append(idx)
    remove_idx = set()
    for (s, e), idxs in groups.items():
        if len(idxs) <= 1:
            continue
        def keyf(i):
            r = reaches[i]
            determined = bool(r.get("start_determined") or r.get("end_determined"))
            return (0 if determined else 1, r.get("reach_id", 1 << 30))
        keep = min(idxs, key=keyf)
        for i in idxs:
            if i != keep:
                remove_idx.add(i)
    removals = [(reaches[i].get("reach_id"), reaches[i].get("start_frame"),
                 reaches[i].get("end_frame"),
                 bool(reaches[i].get("start_determined") or reaches[i].get("end_determined")))
                for i in sorted(remove_idx)]

    # near-duplicates (report only): |dstart|<=2 and |dend|<=2, not exact
    near = []
    rl = [(r.get("reach_id"), r.get("start_frame"), r.get("end_frame")) for r in reaches]
    for a in range(len(rl)):
        for b in range(a + 1, len(rl)):
            ra, rb = rl[a], rl[b]
            if None in (ra[1], ra[2], rb[1], rb[2]):
                continue
            if (ra[1], ra[2]) == (rb[1], rb[2]):
                continue
            if abs(ra[1] - rb[1]) <= 2 and abs(ra[2] - rb[2]) <= 2:
                near.append((ra[0], ra[1], ra[2], rb[0], rb[1], rb[2]))
    return j, frames, seg_changes, remove_idx, removals, near


def apply_file(gt_path):
    j, frames, seg_changes, remove_idx, removals, near = plan_file(gt_path)
    reaches = j["reaches"]["reaches"]
    orig_seg = json.dumps(j.get("segmentation"), sort_keys=True)
    orig_out = json.dumps(j.get("outcomes"), sort_keys=True)
    orig_by_id = {r.get("reach_id"): (r.get("start_frame"), r.get("end_frame"), r.get("apex_frame"))
                  for r in reaches}

    new_reaches = []
    for idx, r in enumerate(reaches):
        if idx in remove_idx:
            continue
        ns = seg_of(frames, r.get("start_frame"))
        if ns is not None:
            r["segment_num"] = ns
        new_reaches.append(r)
    j["reaches"]["reaches"] = new_reaches
    j["reaches"]["total_reaches"] = len(new_reaches)

    # HARD ASSERTS: start/end/apex untouched on every kept reach; seg/outcomes blocks intact
    for r in new_reaches:
        o = orig_by_id[r.get("reach_id")]
        assert (r.get("start_frame"), r.get("end_frame"), r.get("apex_frame")) == o, \
            f"start/end/apex changed for reach {r.get('reach_id')} in {gt_path.name}"
    assert json.dumps(j.get("segmentation"), sort_keys=True) == orig_seg, f"segmentation changed in {gt_path.name}"
    assert json.dumps(j.get("outcomes"), sort_keys=True) == orig_out, f"outcomes changed in {gt_path.name}"

    gt_path.write_text(json.dumps(j, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(seg_changes), len(removals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="WRITE changes (backs up first).")
    args = ap.parse_args()

    if args.apply:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        changelog = {"applied_at": stamp, "files": {}}

    grand_seg = grand_dup = grand_near = 0
    for label, gt_dir in CORPORA:
        print("=" * 92)
        print(f"{label}   {gt_dir}")
        print("=" * 92)
        if args.apply:
            backup = gt_dir.parent / f"gt_backup_pre_seg_dedup_{stamp}" if label == "47-corpus" \
                else gt_dir.parent / f"gt_backup_pre_seg_dedup_{stamp}"
            backup.mkdir(parents=True, exist_ok=True)
        files = sorted(gt_dir.glob("*_unified_ground_truth.json"))
        tseg = tdup = tnear = 0
        print(f"{'video':32} {'segChg':>6} {'dupRemoved':>10} {'nearDup(report)':>16}")
        for f in files:
            j, frames, seg_changes, remove_idx, removals, near = plan_file(f)
            tseg += len(seg_changes); tdup += len(removals); tnear += len(near)
            print(f"{f.name.replace('_unified_ground_truth.json',''):32} "
                  f"{len(seg_changes):>6} {len(removals):>10} {len(near):>16}")
            if args.apply:
                import shutil
                shutil.copy2(f, backup / f.name)
                nseg, ndup = apply_file(f)
                changelog["files"][f.name] = {
                    "seg_changes": seg_changes, "removed": removals}
        print("-" * 92)
        print(f"TOTALS  files={len(files)}  segChanges={tseg}  dupRemoved={tdup}  nearDup(reported,not removed)={tnear}")
        if args.apply:
            print(f"BACKUP: {backup}")
        print()
        grand_seg += tseg; grand_dup += tdup; grand_near += tnear

    print("#" * 92)
    if args.apply:
        log_path = Path(CORPORA[0][1]).parent / f"gt_repair_seg_dedup_changelog_{stamp}.json"
        log_path.write_text(json.dumps(changelog, indent=2), encoding="utf-8")
        print(f"APPLIED. segChanges={grand_seg} dupRemoved={grand_dup}. Changelog: {log_path}")
    else:
        print(f"DRY-RUN. Would change segment_num on {grand_seg} reaches, remove {grand_dup} "
              f"exact-duplicate records. {grand_near} near-dup pairs reported (NOT removed).")
        print("No GT modified. start/end/apex/segmentation/outcomes untouched by design.")


if __name__ == "__main__":
    main()
