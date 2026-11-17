"""Only ground-truth reaches a person determined reach the results.

The ground-truth tool seeds its reach list from the algorithm and saves
those seeds too. A reach whose start and end nobody determined is the
algorithm's guess, not ground truth: it used to be applied anyway, stamped
reach_source "ground_truth", and an exhaustive set made of such seeds
replaced the reaches actually detected in the run being measured.
"""
from mousereach.review.truth_resolver import _apply_gt_reaches


def _data(reaches):
    return {"segments": [{"segment_num": 1, "start_frame": 0, "end_frame": 999,
                          "reaches": [dict(r) for r in reaches]}]}


DETECTED = [{"reach_id": 1, "start_frame": 100, "end_frame": 120},
            {"reach_id": 2, "start_frame": 500, "end_frame": 530}]


def _gt(exhaustive, reaches):
    return {"reaches": {"exhaustive": exhaustive, "reaches": reaches}}


def _rows(out):
    return [(r["start_frame"], r["end_frame"], r.get("reach_source", "algo"))
            for r in out["segments"][0]["reaches"]]


def _done(s, e):
    return {"segment_num": 1, "start_frame": s, "end_frame": e,
            "start_determined": True, "end_determined": True}


def _seed(s, e):
    return {"segment_num": 1, "start_frame": s, "end_frame": e,
            "start_determined": False, "end_determined": False}


def test_an_undetermined_seed_is_not_applied():
    out = _apply_gt_reaches(_data(DETECTED), _gt(False, [_seed(98, 125)]))
    assert _rows(out) == [(100, 120, "algo"), (500, 530, "algo")]


def test_a_half_determined_reach_is_not_applied():
    half = dict(_done(98, 125), end_determined=False)
    out = _apply_gt_reaches(_data(DETECTED), _gt(False, [half]))
    assert _rows(out) == [(100, 120, "algo"), (500, 530, "algo")]


def test_a_determined_reach_replaces_the_detected_one_it_overlaps():
    out = _apply_gt_reaches(_data(DETECTED), _gt(False, [_done(98, 125)]))
    assert _rows(out) == [(98, 125, "ground_truth"), (500, 530, "algo")]


def test_an_exhaustive_set_with_an_undetermined_reach_does_not_drop_detected_reaches():
    # The person determined one reach and left a seed undetermined: the
    # segment is not finished, so the detected reach at 500 stays.
    gt = _gt(True, [_done(98, 125), _seed(900, 910)])
    out = _apply_gt_reaches(_data(DETECTED), gt)
    assert _rows(out) == [(98, 125, "ground_truth"), (500, 530, "algo")]


def test_a_fully_determined_exhaustive_set_is_the_whole_reach_list():
    out = _apply_gt_reaches(_data(DETECTED), _gt(True, [_done(98, 125)]))
    assert _rows(out) == [(98, 125, "ground_truth")]


def test_older_files_spell_it_verified():
    old = {"segment_num": 1, "start_frame": 98, "end_frame": 125,
           "start_verified": True, "end_verified": True}

    out = _apply_gt_reaches(_data(DETECTED), _gt(False, [old]))
    assert _rows(out) == [(98, 125, "ground_truth"), (500, 530, "algo")]
