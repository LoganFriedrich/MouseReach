"""
Generate v8.0.4 FP/FN review manifests for MODEL 3.1 (production detector),
refreshed against the CURRENT live GT, covering ALL videos:

  - corpus_47:           all 47 validation-corpus videos (DLC_2026_03_27)
  - holdout_2026_05_11:  all 20 generalization videos

This is the "updated" successor to:
  - v8.0.4/newly_exhaustive (was only the 33 corpus videos lacking a manifest)
  - v8.0.3/holdout_2026_05_11 (was 19 videos, GT pre-2026-06-19 edits)

WHY: GT was edited (generalization on 2026-06-19; corpus exhaustive pass done).
Logan asked for a fresh full set in a NEW folder so nothing existing is touched.

SOURCES (guideline-compliant, identical method to the newly_exhaustive generator)
  - Algo:  production detect_reaches_v8 (v8.0.4, bundled production model
           v8.0.0_bsw_w0.8, all postprocess applied internally) -- INFERENCE --
           on the MODEL 3.1 DLC. This is the PRODUCTION detector + PRODUCTION
           model (NOT the recalibration sandbox).
  - GT:    LIVE canonical (validation_runs/DLC_2026_03_27/gt for corpus_47;
           iterations/generalization_test_2026-05-11/gt for holdout) -- never a
           snapshot.
  - Topology + manifest schema + matcher: reused VERBATIM from
           generate_fpfn_review_manifests_v8_0_3_new_topology.py (imported).

OUTPUT
  fpfn_review_manifests/v8.04 updated/corpus_47/<video_id>.json
  fpfn_review_manifests/v8.04 updated/holdout_2026_05_11/<video_id>.json

GUARDRAILS
  - Does NOT call the imported generator's main()/archive -- only
    build_manifests + write_manifests, pointed at the NEW dir. No existing
    manifest dir is moved or overwritten.
  - DLC + GT read from Y: canonical only.
  - Per-step breakdown fields (v802/v804 counts) are placeholders: the production
    detect_reaches_v8 applies postprocess internally and does not expose the
    intermediate counts (same as the newly_exhaustive set; events/topology/counts
    are correct). The detector_version stamp records the full config.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

SRC = r"Y:\2_Connectome\Behavior\MouseReach\src"
sys.path.insert(0, SRC)

_GEN_PATH = Path(
    r"Y:\2_Connectome\Behavior\MouseReach\scripts"
    r"\generate_fpfn_review_manifests_v8_0_3_new_topology.py"
)
_spec = importlib.util.spec_from_file_location("_gen_v803_topo", _GEN_PATH)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

from mousereach.reach.v8 import detect_reaches_v8
from mousereach.reach.v8.features import load_dlc_h5

# --- corpora (Y: canonical; both use the model-3.1 shuffle1_100000 DLC) ---
CORPORA = {
    "corpus_47": {
        "gt": Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs\DLC_2026_03_27\gt"),
        "dlc": Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs\DLC_2026_03_27\Processing\updated dlc model 3.1"),
    },
    "holdout_2026_05_11": {
        "gt": Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations\generalization_test_2026-05-11\gt"),
        "dlc": Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations\generalization_test_2026-05-11\dlc"),
    },
}
SNAPSHOT_NAME = "v8.0.4_production_inference_on_model31_dlc_current_gt"

# --- override imported-module globals: NEW output dir + detector stamp ---
gen.OUTPUT_ROOT = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\fpfn_review_manifests\v8.04 updated")
gen.DETECTOR_VERSION = ("v8.0.4_production_bsw_w0.8_mg0_trim_n3_t0.6_"
                        "apex_p0.12_d0.5_pk2_0.85_trailtrim_n3_t0.6")
for label, cfg in CORPORA.items():
    gen.GT_ROOTS[label] = cfg["gt"]   # used by _load_live_gt_reaches + _load_gt_boundaries


def build_label(label, gt_dir, dlc_dir, limit=0):
    vids = sorted(p.name.replace("_unified_ground_truth.json", "")
                  for p in gt_dir.glob("*_unified_ground_truth.json"))
    if limit:
        vids = vids[:limit]
    print(f"\n[{label}] GT={gt_dir}\n[{label}] DLC={dlc_dir}\n[{label}] {len(vids)} videos")
    print("-" * 70)
    per_video = {}
    for vid in vids:
        h5 = sorted(dlc_dir.glob(f"{vid}DLC_*.h5"))
        if not h5:
            print(f"  [skip: no model-3.1 DLC] {vid}")
            continue
        algos = detect_reaches_v8(load_dlc_h5(h5[0]))
        gts, last_mod = gen._load_live_gt_reaches(label, vid)
        per_video[vid] = {
            "algos": sorted(set((int(s), int(e)) for s, e in algos)),
            "gts": gts,
            "gt_last_modified_at": last_mod,
        }
        print(f"  {vid:32} algo={len(per_video[vid]['algos']):>4}  gt={len(gts):>4}  gt_last_mod={last_mod}")
    return per_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="first N videos per corpus (validation)")
    args = ap.parse_args()

    print("=" * 70)
    print("v8.0.4 UPDATED manifests -- model 3.1 production detector, current GT")
    print(f"Output: {gen.OUTPUT_ROOT}")
    print("=" * 70)

    for label, cfg in CORPORA.items():
        per_video = build_label(label, cfg["gt"], cfg["dlc"], limit=args.limit)
        print(f"\nBuilding manifests for [{label}] ...")
        manifests = gen.build_manifests(per_video, label, SNAPSHOT_NAME)
        gen._print_summary(manifests, label)
        n = gen.write_manifests(manifests, label)
        print(f"Wrote {n} manifests to {gen.OUTPUT_ROOT / label}/")


if __name__ == "__main__":
    main()
