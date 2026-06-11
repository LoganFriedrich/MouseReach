"""Where is the last-confident pellet before it vanishes? Compare:
- 3 displaced_outside error targets (CNT0102_P4 s10, CNT0209_P2 s17, CNT0303_P2 s6)
- Sample of retrieved TPs on those same videos
- Sample of displaced_sa TPs on those same videos

Hypothesis: in displaced_outside cases, last-confident pellet position is
LATERAL away from the slit center and OUTSIDE the SA polygon. In retrieved
cases, last-confident pellet position is at the apex / SA area (occluded by
paw before going through slit). In displaced_sa cases, pellet remains
visible or vanishes briefly INSIDE the SA polygon.

Computes per segment:
- Last frame with Pellet_lk >= 0.7
- Last-confident pellet (x, y)
- Slit center (BoxL+BoxR midpoint)
- SA polygon corners (SABL/SABR/SATL/SATR)
- Distance from slit center (total + lateral)
- Inside-SA boolean (point-in-polygon)
- Pre-vanish velocity (last 5 confident frames)
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from mousereach.reach.v8.features import load_dlc_h5

V603 = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots\outcome"
    r"\v6.0.3_fix_b_retrieved_rescue_2026-06-02\algo_outputs"
)
GT = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\iterations\generalization_test_2026-05-11\gt"
)
DLC_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\iterations\generalization_test_2026-05-11\algo_outputs_current"
)

# Target cases: 3 displaced_outside errors
TARGETS = [
    ("20250625_CNT0102_P4", 10),
    ("20250715_CNT0209_P2", 17),
    ("20251008_CNT0303_P2", 6),
]


def _load_gt_segments_with_outcomes(vid):
    gt = json.loads((GT / f"{vid}_unified_ground_truth.json").read_text())
    bs = [int(b["frame"]) for b in gt.get("segmentation", {}).get("boundaries", [])]
    outcomes = {s["segment_num"]: s.get("outcome") for s in gt["outcomes"]["segments"]}
    segs = []
    for i in range(len(bs) - 1):
        sn = i + 1
        segs.append((sn, bs[i], bs[i + 1] - 1, outcomes.get(sn)))
    return segs


def _find_dlc(vid):
    matches = sorted(DLC_DIR.glob(f"{vid}DLC_*.h5"))
    if not matches:
        raise FileNotFoundError(vid)
    return matches[0]


def _point_in_polygon(x, y, poly):
    """Standard ray-casting test. poly = list of (x, y) tuples."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / max(yj - yi, 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def compute_features(dlc, seg_start, seg_end):
    """For a segment [seg_start, seg_end], compute:
    - last_conf_frame: last frame where Pellet_lk >= 0.7
    - last_conf_pos: (x, y) at that frame
    - slit_center: avg(BoxL, BoxR) at that frame
    - sa_poly: (SABL, SABR, SATR, SATL) at that frame (CCW order)
    - dist_from_slit_total, dist_from_slit_lateral (x), dist_from_slit_axial (y)
    - inside_sa: bool
    - pre_vanish_velocity: mean speed over last 5 confident frames
    - n_confident_frames_in_seg
    """
    sub = dlc.iloc[seg_start:seg_end + 1]
    plk = sub["Pellet_likelihood"].to_numpy()
    px = sub["Pellet_x"].to_numpy()
    py = sub["Pellet_y"].to_numpy()
    conf_mask = plk >= 0.7
    n_conf = int(conf_mask.sum())
    if n_conf == 0:
        return None
    conf_indices = np.where(conf_mask)[0]
    last_conf_idx = conf_indices[-1]
    last_conf_frame = seg_start + last_conf_idx

    # Slit center + SA polygon at last_conf_frame
    bxl = (float(dlc["BOXL_x"].iloc[last_conf_frame]),
           float(dlc["BOXL_y"].iloc[last_conf_frame]))
    bxr = (float(dlc["BOXR_x"].iloc[last_conf_frame]),
           float(dlc["BOXR_y"].iloc[last_conf_frame]))
    sabl = (float(dlc["SABL_x"].iloc[last_conf_frame]),
            float(dlc["SABL_y"].iloc[last_conf_frame]))
    sabr = (float(dlc["SABR_x"].iloc[last_conf_frame]),
            float(dlc["SABR_y"].iloc[last_conf_frame]))
    satl = (float(dlc["SATL_x"].iloc[last_conf_frame]),
            float(dlc["SATL_y"].iloc[last_conf_frame]))
    satr = (float(dlc["SATR_x"].iloc[last_conf_frame]),
            float(dlc["SATR_y"].iloc[last_conf_frame]))
    slit_cx = (bxl[0] + bxr[0]) / 2.0
    slit_cy = (bxl[1] + bxr[1]) / 2.0

    last_px = float(px[last_conf_idx])
    last_py = float(py[last_conf_idx])
    dx = last_px - slit_cx
    dy = last_py - slit_cy
    dist_total = float(np.sqrt(dx * dx + dy * dy))

    # SA polygon order: SATL -> SATR -> SABR -> SABL (top-left, top-right, bot-right, bot-left)
    sa_poly = [satl, satr, sabr, sabl]
    inside_sa = _point_in_polygon(last_px, last_py, sa_poly)

    # Pre-vanish velocity: mean speed over last 5 confident frames
    last_n = min(5, n_conf)
    if last_n >= 2:
        idxs = conf_indices[-last_n:]
        sx = px[idxs]
        sy = py[idxs]
        vmags = np.sqrt(np.diff(sx) ** 2 + np.diff(sy) ** 2)
        pre_vanish_speed_mean = float(vmags.mean())
        pre_vanish_speed_max = float(vmags.max())
        # Net displacement vector over the last_n frames
        net_dx = float(sx[-1] - sx[0])
        net_dy = float(sy[-1] - sy[0])
        net_dist = float(np.sqrt(net_dx * net_dx + net_dy * net_dy))
    else:
        pre_vanish_speed_mean = float("nan")
        pre_vanish_speed_max = float("nan")
        net_dx = net_dy = net_dist = float("nan")

    return {
        "last_conf_frame": int(last_conf_frame),
        "last_conf_x": last_px,
        "last_conf_y": last_py,
        "slit_cx": slit_cx,
        "slit_cy": slit_cy,
        "dx_to_slit": dx,
        "dy_to_slit": dy,
        "dist_from_slit": dist_total,
        "abs_lateral_dist": abs(dx),
        "inside_sa": bool(inside_sa),
        "n_confident_frames": n_conf,
        "pre_vanish_speed_mean": pre_vanish_speed_mean,
        "pre_vanish_speed_max": pre_vanish_speed_max,
        "pre_vanish_net_dx": net_dx,
        "pre_vanish_net_dy": net_dy,
        "pre_vanish_net_dist": net_dist,
    }


def main():
    # Group: per video, find target + retrieved samples + displaced_sa samples
    by_video = {}
    for vid, sn in TARGETS:
        by_video.setdefault(vid, set()).add(sn)

    # For each affected video, pull all retrieved + displaced_sa segments + the target
    print("=" * 110)
    for vid in sorted(by_video):
        print()
        print(f"VIDEO: {vid}")
        print("-" * 110)
        dlc = load_dlc_h5(_find_dlc(vid))
        gt_segs = _load_gt_segments_with_outcomes(vid)

        # Build (sn, kind) tuples
        targets_this_vid = by_video[vid]
        records = []
        for sn, s, e, gtout in gt_segs:
            kind = None
            if sn in targets_this_vid:
                kind = "TARGET (displaced_outside)"
            elif gtout == "retrieved":
                kind = "retrieved"
            elif gtout == "displaced_sa":
                kind = "displaced_sa"
            if kind is None:
                continue
            feat = compute_features(dlc, s, e)
            if feat is None:
                feat = {"n_confident_frames": 0}
            feat["kind"] = kind
            feat["seg"] = sn
            feat["gt"] = gtout
            feat["seg_start"] = s
            feat["seg_end"] = e
            records.append(feat)

        # Sort: target first, then retrieved (limited), then displaced_sa (limited)
        targets = [r for r in records if r["kind"].startswith("TARGET")]
        retrievs = [r for r in records if r["kind"] == "retrieved"][:6]
        disps = [r for r in records if r["kind"] == "displaced_sa"][:6]

        header = (f"  {'kind':30s} {'seg':>4s} {'n_conf':>7s} "
                  f"{'dx_slit':>9s} {'dy_slit':>9s} {'dist':>7s} "
                  f"{'|lat|':>6s} {'in_sa':>6s} "
                  f"{'pre_vmean':>10s} {'pre_vmax':>9s} {'net_dx':>7s} {'net_dy':>7s}")
        print(header)
        for r in targets + retrievs + disps:
            if r.get("n_confident_frames", 0) == 0:
                print(f"  {r['kind']:30s} {r['seg']:>4d}   <no confident pellet>")
                continue
            print(f"  {r['kind']:30s} {r['seg']:>4d} {r['n_confident_frames']:>7d} "
                  f"{r['dx_to_slit']:>+9.1f} {r['dy_to_slit']:>+9.1f} "
                  f"{r['dist_from_slit']:>7.1f} {r['abs_lateral_dist']:>6.1f} "
                  f"{str(r['inside_sa']):>6s} "
                  f"{r['pre_vanish_speed_mean']:>10.2f} {r['pre_vanish_speed_max']:>9.2f} "
                  f"{r['pre_vanish_net_dx']:>+7.1f} {r['pre_vanish_net_dy']:>+7.1f}")

    print()
    print("=" * 110)
    print("Sign convention: x = lateral (signed); y = axial (positive = away from box, toward apex).")
    print("inside_sa: True = last-conf pellet position inside the SA quadrilateral polygon.")
    print("pre_vanish_*: computed over last 5 confident frames before pellet vanishes.")
    print("net_dx / net_dy: net displacement of pellet across those last 5 frames (direction of last motion).")


if __name__ == "__main__":
    main()
