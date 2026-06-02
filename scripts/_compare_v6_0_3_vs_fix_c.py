"""Compare v6.0.3 vs Fix C per-segment outcomes. Identify any case where
Stage 21 deferred under Fix C and a different stage committed."""
import json
from pathlib import Path

V603 = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots\outcome"
    r"\v6.0.3_fix_b_retrieved_rescue_2026-06-02\algo_outputs"
)
FIXC = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots\outcome"
    r"\v6.0.4_fix_c_stage_21_corroboration_2026-06-02\algo_outputs"
)

n_total_diff = 0
n_stage_diff = 0
n_outcome_diff = 0
fixc_fired_cases = []

for f in sorted(FIXC.glob("*_pellet_outcomes.json")):
    vid = f.stem.replace("_pellet_outcomes", "")
    a = json.loads(f.read_text())
    b = json.loads((V603 / f"{vid}_pellet_outcomes.json").read_text())
    a_by = {s["segment_num"]: s for s in a["segments"]}
    b_by = {s["segment_num"]: s for s in b["segments"]}
    for sn, a_s in a_by.items():
        b_s = b_by.get(sn)
        if not b_s:
            continue
        if a_s["stage"] != b_s["stage"] or a_s["outcome"] != b_s["outcome"]:
            n_total_diff += 1
            if a_s["stage"] != b_s["stage"]:
                n_stage_diff += 1
            if a_s["outcome"] != b_s["outcome"]:
                n_outcome_diff += 1
            # Was this a Stage 21 case in v6.0.3?
            was_21 = "stage_21" in (b_s.get("stage") or "")
            now_21 = "stage_21" in (a_s.get("stage") or "")
            if was_21 and not now_21:
                fixc_fired_cases.append({
                    "video": vid,
                    "seg": sn,
                    "v6.0.3_stage": b_s["stage"],
                    "v6.0.3_outcome": b_s["outcome"],
                    "fixc_stage": a_s["stage"],
                    "fixc_outcome": a_s["outcome"],
                })

print(f"Segments with any diff: {n_total_diff}")
print(f"Segments with stage diff: {n_stage_diff}")
print(f"Segments with outcome diff: {n_outcome_diff}")
print()
print(f"Cases where Fix C Edit 2 caused Stage 21 to defer: {len(fixc_fired_cases)}")
for c in fixc_fired_cases:
    print(f"  {c['video']:30s} seg={c['seg']:>3}: "
          f"v6.0.3 [{c['v6.0.3_stage']}, {c['v6.0.3_outcome']}] -> "
          f"FixC [{c['fixc_stage']}, {c['fixc_outcome']}]")
