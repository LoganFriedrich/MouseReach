"""Per-action authorized GT edit (Logan 2026-06-03): set
20251029_CNT0408_P1 segments 9,11,13,17 outcome -> retrieved, with revert
provenance. Fidelity-gated: only writes if json round-trip reproduces the file
exactly (so only the 4 targeted segments change). Run with --write to apply."""
import json, sys, copy
from pathlib import Path

GT = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs\DLC_2026_03_27\iterations\2026-04-28_outcome_v4.0.0_dev_walkthrough\gt\20251029_CNT0408_P1_unified_ground_truth.json")
TARGETS = {9, 11, 13, 17}
NEW = "retrieved"
orig_text = GT.read_text(encoding="utf-8")
data = json.loads(orig_text)

# pick serialization that reproduces the file exactly
def ser(d, ascii_):
    s = json.dumps(d, indent=2, ensure_ascii=ascii_)
    return s, s + "\n"
match_ascii = None
for a in (True, False):
    s, sn = ser(data, a)
    if orig_text == s or orig_text == sn:
        match_ascii = a; trailing = "\n" if orig_text == sn else ""
        break
print(f"round-trip fidelity: {'CLEAN (ensure_ascii=%s)'%match_ascii if match_ascii is not None else 'MISMATCH -- will NOT write'}")
if match_ascii is None:
    # show how far off, then stop
    s,_=ser(data,True)
    print(f"  orig len={len(orig_text)} dump len={len(s)}")
    sys.exit(1)

new = copy.deepcopy(data)
segs = new.get("outcomes", {}).get("segments", [])
changed = []
for seg in segs:
    if int(seg["segment_num"]) in TARGETS:
        before = seg.get("outcome")
        seg["outcome"] = NEW
        seg["reverted_outcome_to"] = NEW
        seg["reverted_by"] = "claude (per Logan explicit per-action permission, 2026-06-03)"
        seg["reverted_at"] = "2026-06-03"
        seg["revert_reason"] = ("Logan review of the CNT0408_P1 tray-attached-artifact cluster: "
                                "set to retrieved. The v4.0.0 abnormal_exception relabel is reverted "
                                "for this segment. original_outcome preserved above.")
        changed.append((int(seg["segment_num"]), before, seg.get("original_outcome")))

print("changed segments (segnum, was, original_outcome):")
for c in changed: print(f"  s{c[0]}: {c[1]} -> {NEW}   (original_outcome={c[2]})")
print(f"n changed: {len(changed)} (expect {len(TARGETS)})")

# verify ONLY the 4 segments differ
diffs = sum(1 for a, b in zip(data["outcomes"]["segments"], new["outcomes"]["segments"]) if a != b)
print(f"segments differing between old and new: {diffs}")

if "--write" in sys.argv and len(changed) == len(TARGETS) and diffs == len(TARGETS):
    GT.write_text(json.dumps(new, indent=2, ensure_ascii=match_ascii) + trailing, encoding="utf-8")
    print("WROTE file.")
else:
    print("DRY RUN (pass --write to apply; write also requires all checks pass).")
