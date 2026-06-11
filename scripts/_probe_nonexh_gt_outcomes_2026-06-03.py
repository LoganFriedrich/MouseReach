"""Read-only: per-video GT segment outcomes for the 27 non-exhaustive corpus
videos (for Logan to double-check GT against the videos). Shows GT per segment;
flags '*' where the v6.0.4 cascade disagrees (normalized). Prints only."""
import csv, json, importlib.util
from pathlib import Path

S = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_lva", S / "outcome_leverA_net_displaced_sa_2026-06-03.py")
lva = importlib.util.module_from_spec(spec); spec.loader.exec_module(lva)

GT_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs\DLC_2026_03_27\iterations\2026-04-28_outcome_v4.0.0_dev_walkthrough\gt")
SNAP = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots\outcome\v6.0.4_eval_full47_corpus_2026-06-03")
cv = json.loads(Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots\_corpus\2026-04-30_restart_inventory\cv_folds.json").read_text())
train = set(cv["train_pool"]["video_ids"]); hold = set(cv["test_holdout"]["video_ids"])
exh = set(lva.M31["ids"])
NON_TRAIN = sorted(train - exh); NON_HOLD = sorted(hold - exh)

ABBR = {"untouched":"untouched","displaced_sa":"displaced_sa","displaced_outside":"displaced_outside",
        "retrieved":"retrieved","abnormal_exception":"abnormal","uncertain":"uncertain"}
def norm(x): return "displaced_sa" if x in ("displaced_outside","displaced_sa") else x

# algo outcomes from full47 snapshot
algo = {}
for r in csv.DictReader(open(SNAP / "metrics" / "outcome_per_segment.csv")):
    algo[(r["video_id"], int(r["segment_num"]))] = norm(r["algo_outcome"])

def dump(vid):
    gt = json.loads((GT_DIR / f"{vid}_unified_ground_truth.json").read_text())
    segs = sorted(gt.get("outcomes", {}).get("segments", []), key=lambda s: int(s["segment_num"]))
    print(f"\n{vid}  ({len(segs)} segments)")
    mism = []
    for s in segs:
        sn = int(s["segment_num"]); o = s.get("outcome", "?")
        a = algo.get((vid, sn))
        flag = ""
        if a is not None and norm(o) != a:
            flag = f"   <-- cascade: {a}"; mism.append(sn)
        print(f"   s{sn:>2}  {ABBR.get(o, o)}{flag}")
    if mism: print(f"   (cascade disagrees on segs: {mism})")

print("="*70); print("NON-EXHAUSTIVE TRAIN_POOL (21)"); print("="*70)
for v in NON_TRAIN: dump(v)
print("\n"+"="*70); print("NON-EXHAUSTIVE TEST_HOLDOUT (6)"); print("="*70)
for v in NON_HOLD: dump(v)
