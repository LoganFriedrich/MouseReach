"""Read-only probe to enumerate manifest event kinds + categories.

Used to ground Phase A diagnostic against the canonical v8.0.4 manifest.
Not a permanent script -- delete after probing.
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter

MANIFEST_ROOTS = {
    "holdout": (
        r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
        r"\fpfn_review_manifests\v8.0.3\holdout_2026_05_11"
    ),
    "calibration": (
        r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
        r"\fpfn_review_manifests\v8.0.3\calibration_loocv"
    ),
}


def run_one(label: str, root: str) -> None:
    print("=" * 76)
    print(f"CORPUS: {label}    dir: {root}")
    print("=" * 76)
    files = sorted(glob.glob(os.path.join(root, "*.json")))
    print(f"manifest files: {len(files)}")
    print()
    kinds: Counter = Counter()
    topos: Counter = Counter()
    # Component-level: gather all events per component_id per video.
    # For TOL_pair: a component should have exactly 1 FP + 1 FN with category
    # starting "tolerance_miss". Reconstruct algo+GT spans from those pairs.
    end_off_cases = []
    start_off_cases = []
    span_short_cases = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        vid = d["video_id"]
        comps: dict = {}
        for e in d.get("events", []):
            kinds[(e.get("kind"), e.get("category", ""))] += 1
            topos[(e.get("topology"), e.get("topology_sub"))] += 1
            cid = e.get("component_id")
            if cid is None:
                continue
            comps.setdefault(cid, []).append(e)
        # Find TOL_pair components: those that are exactly 1 FP + 1 FN both
        # with topology == "TOLERANCE_ERROR"
        for cid, evs in comps.items():
            tol_fp = [e for e in evs if e.get("kind") == "FP" and e.get("topology") == "TOLERANCE_ERROR"]
            tol_fn = [e for e in evs if e.get("kind") == "FN" and e.get("topology") == "TOLERANCE_ERROR"]
            if len(tol_fp) == 1 and len(tol_fn) == 1 and len(evs) == 2:
                fp = tol_fp[0]
                fn = tol_fn[0]
                algo = fp.get("detector") or {}
                gt = fn.get("gt") or {}
                ase, aend = algo.get("start"), algo.get("end")
                gst, gend = gt.get("start"), gt.get("end")
                if ase is not None and gst is not None and aend is not None and gend is not None:
                    start_delta = ase - gst
                    end_delta = aend - gend
                    rec = {
                        "video": vid,
                        "cid": cid,
                        "algo": [ase, aend],
                        "gt": [gst, gend],
                        "start_delta": start_delta,
                        "end_delta": end_delta,
                        "fn_cat": fn.get("category"),
                        "fp_cat": fp.get("category"),
                    }
                    if end_delta > 5:
                        end_off_cases.append(rec)
                    if start_delta < -2:
                        start_off_cases.append(rec)
                    if end_delta < -5:
                        span_short_cases.append(rec)

    print("event kind + category counts:")
    for (kind, cat), c in sorted(kinds.items()):
        print(f"  {kind!r:6s} cat={cat!r:30s} count={c}")
    print()
    print("topology + sub counts:")
    for (t, sub), c in sorted(topos.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
        print(f"  topology={t!r:18s} sub={sub!r:24s} count={c}")
    print()
    print(f"==== END-EXTENDED TOL pairs (end_delta > +5): {len(end_off_cases)} ====")
    by_video: Counter = Counter()
    for c in end_off_cases:
        by_video[c["video"]] += 1
    for vid, n in by_video.most_common():
        print(f"  {vid}: {n}")
    print()
    print("Per-case detail (end-extended TOL):")
    for c in sorted(end_off_cases, key=lambda r: (r["video"], r["gt"][0])):
        print(
            f"  {c['video']:30s} algo=[{c['algo'][0]:>6},{c['algo'][1]:>6}] "
            f"gt=[{c['gt'][0]:>6},{c['gt'][1]:>6}] sd={c['start_delta']:>+4} ed={c['end_delta']:>+4} "
            f"fn_cat={c['fn_cat']}"
        )
    print()
    print(f"==== START-EARLY TOL pairs (start_delta < -2): {len(start_off_cases)} ====")
    by_video = Counter()
    for c in start_off_cases:
        by_video[c["video"]] += 1
    for vid, n in by_video.most_common():
        print(f"  {vid}: {n}")
    print()
    print(f"==== SPAN-SHORT TOL pairs (end_delta < -5, algo shorter than GT): {len(span_short_cases)} ====")
    for c in sorted(span_short_cases, key=lambda r: (r["video"], r["gt"][0])):
        print(
            f"  {c['video']:30s} algo=[{c['algo'][0]:>6},{c['algo'][1]:>6}] "
            f"gt=[{c['gt'][0]:>6},{c['gt'][1]:>6}] sd={c['start_delta']:>+4} ed={c['end_delta']:>+4} "
            f"fn_cat={c['fn_cat']}"
        )


def main() -> None:
    for label, root in MANIFEST_ROOTS.items():
        run_one(label, root)
        print()


if __name__ == "__main__":
    main()
