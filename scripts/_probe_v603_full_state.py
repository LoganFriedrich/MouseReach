"""Full state of v6.0.3 on the 20-video 2026-05-11 corpus: errors + triages
+ confusion matrix grouped by GT class and by committing stage."""
import json
from collections import Counter, defaultdict
from pathlib import Path

V603 = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots\outcome"
    r"\v6.0.3_fix_b_retrieved_rescue_2026-06-02\algo_outputs"
)
GT = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\iterations\generalization_test_2026-05-11\gt"
)


def main():
    correct = 0
    errors_by_pattern: Counter = Counter()
    errors_detail = []
    triage_detail = []
    stage_correct: Counter = Counter()
    stage_error: Counter = Counter()
    stage_triage: Counter = Counter()

    for f in sorted(V603.glob("*_pellet_outcomes.json")):
        vid = f.stem.replace("_pellet_outcomes", "")
        outs = json.loads(f.read_text())
        gt = json.loads((GT / f"{vid}_unified_ground_truth.json").read_text())
        gt_by = {s["segment_num"]: s.get("outcome") for s in gt["outcomes"]["segments"]}
        for o in outs["segments"]:
            sn = o["segment_num"]
            algo = o["outcome"]
            stage = o.get("stage")
            reason = o.get("flag_reason", "")
            gtout = gt_by.get(sn)
            # Cascade collapses displaced_outside -> displaced_sa for scoring
            algo_eff = algo
            gt_eff = "displaced_sa" if gtout == "displaced_outside" else gtout

            if algo == "triaged":
                triage_detail.append({
                    "video": vid,
                    "seg": sn,
                    "gt": gtout,
                    "stage": stage,
                    "reason": reason,
                })
                stage_triage[stage] += 1
            elif gt_eff == algo_eff:
                correct += 1
                stage_correct[stage] += 1
            else:
                errors_by_pattern[f"{gtout} -> {algo}"] += 1
                errors_detail.append({
                    "video": vid,
                    "seg": sn,
                    "gt": gtout,
                    "algo": algo,
                    "stage": stage,
                    "reason": reason,
                })
                stage_error[stage] += 1

    print(f"Total correct: {correct} / 400 ({correct/400*100:.2f}%)")
    print(f"Triaged: {len(triage_detail)}")
    print(f"Errors: {sum(errors_by_pattern.values())}")
    print()
    print("Error patterns:")
    for pat, n in errors_by_pattern.most_common():
        print(f"  {pat}: {n}")
    print()
    print("Errors by committing stage:")
    for stage, n in sorted(stage_error.items(), key=lambda x: -x[1]):
        print(f"  {stage}: {n}")
    print()
    print("=== ALL 9 ERRORS (detail) ===")
    for e in sorted(errors_detail, key=lambda x: (x["stage"], x["video"], x["seg"])):
        print(f"  {e['stage']:55s} {e['video']:30s} s{e['seg']:>3}  "
              f"gt={str(e['gt']):>20s}  algo={e['algo']}")
    print()
    print("=== ALL 9 TRIAGES (detail) ===")
    for t in sorted(triage_detail, key=lambda x: (x["stage"], x["video"], x["seg"])):
        reason_short = (t["reason"] or "")[:90]
        print(f"  {t['stage']:55s} {t['video']:30s} s{t['seg']:>3}  "
              f"gt={str(t['gt']):>20s}")
        print(f"    reason: {reason_short}")
    print()
    print("=== Stages by total commits ===")
    all_stages = set(stage_correct) | set(stage_error) | set(stage_triage)
    rows = []
    for s in all_stages:
        c, e, t = stage_correct[s], stage_error[s], stage_triage[s]
        rows.append((s, c, e, t, c + e + t))
    rows.sort(key=lambda r: -r[4])
    print(f"{'stage':55s} {'correct':>8s} {'errors':>7s} {'triages':>8s} {'total':>6s} {'acc':>6s}")
    for s, c, e, t, tot in rows:
        if tot == 0:
            continue
        acc = c / max(c + e, 1) * 100 if (c + e) > 0 else 0
        print(f"  {s:55s} {c:>8d} {e:>7d} {t:>8d} {tot:>6d} {acc:>5.1f}%")


if __name__ == "__main__":
    main()
