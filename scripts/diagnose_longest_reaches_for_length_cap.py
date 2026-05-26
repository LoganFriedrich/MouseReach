"""Diagnostic: enumerate the longest GT reaches and longest algo (v8.0.4) reaches
across the calibration LOOCV + holdout corpora.

Purpose
-------
Inform the length-cap-filter decision (Postprocess-side lever #3 on the
2026-05-26 inventory). We need to see:

  - Where does the TP-span upper tail end? (i.e., what is the longest
    real reach? This is the MIN safe length cap.)
  - Where does the FP/MERGED-span upper tail start? (i.e., how long
    are the longest algo emissions, and what fraction of them are
    actually wrong?)

If there's a clean gap between the longest TP and the longest unwanted
algo emission, a length cap is viable. If not, capping into matched
MERGED would convert TP -> FN (wrong direction per FN-over-FP rule).

Data source
-----------
Fresh v8.0.4 manifests at
`Improvement_Snapshots/.../fpfn_review_manifests/v8.0.3/{calibration_loocv,holdout_2026_05_11}/`
regenerated 2026-05-26 as part of the v8.0.4 trailing-trim ship (commit d8b4da1).
Detector_version confirmed v8.0.4 in each manifest.

Outputs (stdout only)
---------------------
- Top-30 GT spans (with kind/topology/video/start-end)
- Top-30 algo spans (with kind/topology/video/start-end)
- Distribution percentiles for GT and algo spans, both corpora
- Per-kind breakdown of algo spans in the top tail
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict


ROOT = Path("y:/2_Connectome/Behavior/MouseReach_Improvement/fpfn_review_manifests/v8.0.3")
CORPORA = ["calibration_loocv", "holdout_2026_05_11"]


def span_of(seg):
    if seg is None:
        return None
    return seg["end"] - seg["start"] + 1


def collect_events():
    """Returns list of dicts with span info per event side (gt and detector)."""
    gt_rows = []   # (span, kind, video_id, corpus, gt_start, gt_end, det_start, det_end)
    algo_rows = [] # (span, kind, video_id, corpus, gt_start, gt_end, det_start, det_end)
    for corpus in CORPORA:
        for f in sorted((ROOT / corpus).glob("*.json")):
            d = json.load(open(f))
            vid = d["video_id"]
            for ev in d.get("events", []):
                kind = ev.get("kind")
                topo = ev.get("topology_kind", kind)
                gt = ev.get("gt")
                det = ev.get("detector")
                gt_span = span_of(gt)
                det_span = span_of(det)
                if gt_span is not None:
                    gt_rows.append({
                        "span": gt_span, "kind": kind, "topology": topo,
                        "video": vid, "corpus": corpus,
                        "gt_start": gt["start"], "gt_end": gt["end"],
                        "det_start": det["start"] if det else None,
                        "det_end": det["end"] if det else None,
                    })
                if det_span is not None:
                    algo_rows.append({
                        "span": det_span, "kind": kind, "topology": topo,
                        "video": vid, "corpus": corpus,
                        "gt_start": gt["start"] if gt else None,
                        "gt_end": gt["end"] if gt else None,
                        "det_start": det["start"], "det_end": det["end"],
                    })
    return gt_rows, algo_rows


def percentiles(spans, ps=(50, 75, 90, 95, 99, 99.5, 100)):
    spans = sorted(spans)
    out = []
    for p in ps:
        if not spans:
            out.append((p, None))
            continue
        idx = min(len(spans) - 1, int(round((p / 100.0) * (len(spans) - 1))))
        out.append((p, spans[idx]))
    return out


def print_top(rows, n, label, with_det=True):
    rows = sorted(rows, key=lambda r: -r["span"])[:n]
    print(f"\n=== Top {n} {label} ===")
    print(f"{'rank':>4} {'span':>5} {'corpus':<22} {'video':<22} {'topo':<22} "
          f"{'gt_start':>9} {'gt_end':>7} {'det_start':>10} {'det_end':>8}")
    for i, r in enumerate(rows, 1):
        print(f"{i:>4} {r['span']:>5} {r['corpus']:<22} {r['video']:<22} "
              f"{str(r['topology']):<22} "
              f"{str(r['gt_start']):>9} {str(r['gt_end']):>7} "
              f"{str(r['det_start']):>10} {str(r['det_end']):>8}")


def main():
    gt_rows, algo_rows = collect_events()

    # Split by corpus
    by_corpus_gt = defaultdict(list)
    by_corpus_algo = defaultdict(list)
    for r in gt_rows:
        by_corpus_gt[r["corpus"]].append(r["span"])
    for r in algo_rows:
        by_corpus_algo[r["corpus"]].append(r["span"])

    print("=== Counts ===")
    print(f"GT events:   {len(gt_rows)}  (cal={len(by_corpus_gt['calibration_loocv'])}, "
          f"hol={len(by_corpus_gt['holdout_2026_05_11'])})")
    print(f"Algo events: {len(algo_rows)}  (cal={len(by_corpus_algo['calibration_loocv'])}, "
          f"hol={len(by_corpus_algo['holdout_2026_05_11'])})")

    print("\n=== GT span percentiles ===")
    for corpus in CORPORA:
        print(f"\n  {corpus}:")
        for p, v in percentiles(by_corpus_gt[corpus]):
            print(f"    p{p:>5}: {v}")

    print("\n=== Algo span percentiles ===")
    for corpus in CORPORA:
        print(f"\n  {corpus}:")
        for p, v in percentiles(by_corpus_algo[corpus]):
            print(f"    p{p:>5}: {v}")

    print_top(gt_rows, 30, "longest GT reaches")
    print_top(algo_rows, 30, "longest algo reaches")

    # Top-tail breakdown by kind/topology for algo
    print("\n=== Algo top-tail topology breakdown (>= 30 frames) ===")
    tail = [r for r in algo_rows if r["span"] >= 30]
    by_topo = defaultdict(int)
    by_topo_corpus = defaultdict(lambda: defaultdict(int))
    for r in tail:
        by_topo[r["topology"]] += 1
        by_topo_corpus[r["corpus"]][r["topology"]] += 1
    for topo, n in sorted(by_topo.items(), key=lambda kv: -kv[1]):
        print(f"  {str(topo):<25} total={n}  "
              f"(cal={by_topo_corpus['calibration_loocv'][topo]}, "
              f"hol={by_topo_corpus['holdout_2026_05_11'][topo]})")

    # Same for >= 50 frames
    print("\n=== Algo top-tail topology breakdown (>= 50 frames) ===")
    tail = [r for r in algo_rows if r["span"] >= 50]
    by_topo = defaultdict(int)
    by_topo_corpus = defaultdict(lambda: defaultdict(int))
    for r in tail:
        by_topo[r["topology"]] += 1
        by_topo_corpus[r["corpus"]][r["topology"]] += 1
    for topo, n in sorted(by_topo.items(), key=lambda kv: -kv[1]):
        print(f"  {str(topo):<25} total={n}  "
              f"(cal={by_topo_corpus['calibration_loocv'][topo]}, "
              f"hol={by_topo_corpus['holdout_2026_05_11'][topo]})")


if __name__ == "__main__":
    main()
