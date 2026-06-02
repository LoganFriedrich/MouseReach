"""Probe: on the 20-video 2026-05-11 generalization corpus + current v6.0.3
cascade, how many errors are committed by Stage 21? And what are they?

The A-side proposal targets Stage 21 errors on the cal/train_pool corpus
(v6_cascade_2026-05-04). Need to verify those Stage 21 errors also exist
on the holdout corpus we shipped v6.0.3 against, before pursuing the
proposal.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

OUTCOME_SNAPSHOT = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\Improvement_Snapshots\outcome\v6.0.3_fix_b_retrieved_rescue_2026-06-02"
)
ALGO_DIR = OUTCOME_SNAPSHOT / "algo_outputs"
GT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\iterations\generalization_test_2026-05-11\gt"
)


def load_gt(video_id: str):
    p = GT_DIR / f"{video_id}_unified_ground_truth.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def main():
    outcomes_files = sorted(ALGO_DIR.glob("*_pellet_outcomes.json"))
    print(f"Outcome files in v6.0.3 snapshot: {len(outcomes_files)}")
    print()

    stage_counts: Counter = Counter()
    errors_by_stage: dict = defaultdict(list)
    per_video_summary = []

    for f in outcomes_files:
        video = f.stem.replace("_pellet_outcomes", "")
        with open(f) as fh:
            outcomes = json.load(fh)
        gt = load_gt(video)
        if gt is None:
            print(f"WARN: no GT for {video}")
            continue
        # GT structure: gt["outcomes"]["segments"][...]
        gt_segments = gt.get("outcomes", {}).get("segments", [])
        gt_by_seg_idx = {}
        for seg in gt_segments:
            seg_idx = seg.get("segment_num")
            gt_outcome = seg.get("outcome")
            gt_by_seg_idx[seg_idx] = gt_outcome

        n_correct = 0
        n_triage = 0
        n_errors = 0
        for o in outcomes.get("segments", []):
            seg_idx = o.get("segment_num")
            algo_outcome = o.get("outcome")
            stage = o.get("stage") or "(missing)"
            reason = o.get("flag_reason", "") or ""
            gt_outcome = gt_by_seg_idx.get(seg_idx)
            stage_counts[stage] += 1
            if algo_outcome == "triaged":
                n_triage += 1
            elif algo_outcome == gt_outcome:
                n_correct += 1
            elif gt_outcome == "displaced_outside":
                # cascade collapses displaced_outside -> displaced_sa
                if algo_outcome == "displaced_sa":
                    n_correct += 1
                else:
                    n_errors += 1
                    errors_by_stage[stage].append({
                        "video": video,
                        "segment": seg_idx,
                        "gt": gt_outcome,
                        "algo": algo_outcome,
                        "stage": stage,
                        "reason": reason,
                    })
            else:
                n_errors += 1
                errors_by_stage[stage].append({
                    "video": video,
                    "segment": seg_idx,
                    "gt": gt_outcome,
                    "algo": algo_outcome,
                    "stage": stage,
                    "reason": reason,
                })
        per_video_summary.append({
            "video": video,
            "correct": n_correct,
            "triaged": n_triage,
            "errors": n_errors,
        })

    print("Per-stage commit counts (all segments, not just errors):")
    for stage, n in sorted(stage_counts.items(), key=lambda x: -x[1])[:30]:
        print(f"  stage {stage}: {n}")
    print()

    print("Errors by stage on the 20-video 2026-05-11 corpus, v6.0.3:")
    for stage, errs in sorted(errors_by_stage.items(), key=lambda x: -len(x[1])):
        print(f"  Stage {stage}: {len(errs)} errors")
        for e in errs[:15]:
            reason_short = (e["reason"] or "")[:80]
            gt_str = str(e["gt"]) if e["gt"] is not None else "None"
            algo_str = str(e["algo"]) if e["algo"] is not None else "None"
            print(
                f"    {e['video']:30s} seg={e['segment']!s:>3} gt={gt_str:>20s} "
                f"algo={algo_str:>15s} reason='{reason_short}'"
            )
    print()

    total_errors = sum(len(v) for v in errors_by_stage.values())
    total_correct = sum(v["correct"] for v in per_video_summary)
    total_triage = sum(v["triaged"] for v in per_video_summary)
    print(f"Total: correct={total_correct} triaged={total_triage} errors={total_errors}")


if __name__ == "__main__":
    main()
