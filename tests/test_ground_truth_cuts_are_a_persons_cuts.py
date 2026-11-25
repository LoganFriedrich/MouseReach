"""Cuts a person determined in the ground-truth tool are the video's cuts.

WHY: what a person said is the correct result for every
step after it. A ``<video>_unified_ground_truth.json`` whose cuts are ALL
determined is a person's segmentation; it used to be read only to evaluate the
segmenter, so reaches, outcomes, reach assignment and kinematics were computed
on the algorithm's cuts the person had corrected. It is now written exactly as
the segmentation fixer writes a person's cuts (review/hand_cuts.py).

Video ids are invented.
"""
import json

import pytest

V = "20990101_CNT9901_P1"
ALGO = [100, 200, 300]
GT = [102, 200, 305]


def _write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _doc(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _gt(frames, determined=True, at="2026-05-04T11:00:00", by="annotator"):
    return {"video_name": V, "segmentation": {"boundaries": [
        {"index": i, "frame": f, "determined": determined, "determined_by": by,
         "determined_at": at} for i, f in enumerate(frames)]},
        "reaches": {"reaches": []}, "outcomes": {"segments": []}}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    from mousereach.config import Paths
    import mousereach.review.causal_review_io as cio
    monkeypatch.setattr(cio, "gt_index", lambda refresh=False: {})
    monkeypatch.setattr(Paths, "REVIEW_ROOT", None, raising=False)
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", tmp_path / "Analyzed", raising=False)


def test_only_a_file_whose_every_cut_is_determined_counts(tmp_path):
    from mousereach.review.hand_cuts import ground_truth_cuts
    _write(tmp_path / f"{V}_unified_ground_truth.json", _gt(GT))
    got = ground_truth_cuts(V, (tmp_path,))
    assert got["boundaries"] == GT and got["determined_by"] == "annotator"

    doc = _gt(GT)
    doc["segmentation"]["boundaries"][1]["determined"] = False   # the tool's seed
    _write(tmp_path / f"{V}_unified_ground_truth.json", doc)
    assert ground_truth_cuts(V, (tmp_path,)) is None

    # Schema 1 files say "verified" for the same thing.
    old = {"segmentation": {"boundaries": [{"index": 0, "frame": 5, "verified": True}]}}
    _write(tmp_path / f"{V}_unified_ground_truth.json", old)
    assert ground_truth_cuts(V, (tmp_path,))["boundaries"] == [5]


def test_segmentation_writes_ground_truth_cuts_over_the_segmenters(tmp_path, monkeypatch):
    import mousereach.segmentation.core.batch as batch
    seg = _write(tmp_path / f"{V}_segments.json",
                 {"boundaries": ALGO, "segmenter_version": "9.9.9", "video_name": V})
    _write(tmp_path / f"{V}_reaches.json", {"segments": []})
    _write(tmp_path / f"{V}_unified_ground_truth.json", _gt(GT))

    # The segmenter runs as usual, writing the algorithm's cuts ...
    monkeypatch.setattr(batch, "segment_video_multi",
                        lambda p: (ALGO, type("D", (), {"interval_cv": 0.0, "anomalies": []})()))
    monkeypatch.setattr(batch, "save_segmentation",
                        lambda b, d, out, previous_from=None: _write(
                            out, {"boundaries": list(b), "segmenter_version": "9.9.9",
                                  "video_name": V}))
    dlc = tmp_path / f"{V}DLC_model.h5"
    dlc.write_bytes(b"")
    res = batch.process_single(dlc)

    # ... and the person's cuts replace them, marked as a person's.
    d = _doc(seg)
    assert d["boundaries"] == GT
    assert d["boundary_source"] == "human" and d["cuts_source"] == "ground_truth"
    assert d["algo_boundaries"] == ALGO and d["corrected_by"] == "annotator"
    assert d["ground_truth_file"] == f"{V}_unified_ground_truth.json"
    assert res["applied_ground_truth_cuts"] and res["success"]
    # Everything computed from the old cuts is gone (and archived).
    assert not (tmp_path / f"{V}_reaches.json").exists()
    assert list(tmp_path.glob(f"_archived/*/{V}_reaches.json"))

    # The next run keeps them: nothing is rewritten.
    before = seg.read_text(encoding="utf-8")
    res2 = batch.process_single(dlc)
    assert res2.get("preserved_human_boundaries") and not res2.get("applied_ground_truth_cuts")
    assert seg.read_text(encoding="utf-8") == before


def test_identical_cuts_are_left_alone(tmp_path):
    from mousereach.review.hand_cuts import apply_ground_truth_cuts
    seg = _write(tmp_path / f"{V}_segments.json", {"boundaries": GT, "video_name": V})
    _write(tmp_path / f"{V}_reaches.json", {"segments": []})
    _write(tmp_path / f"{V}_unified_ground_truth.json", _gt(GT))
    assert apply_ground_truth_cuts(seg, V) is None
    assert (tmp_path / f"{V}_reaches.json").exists()


def test_the_newer_of_a_fixer_save_and_the_ground_truth_stands(tmp_path):
    from mousereach.review.hand_cuts import apply_ground_truth_cuts
    fixer = {"boundaries": ALGO, "boundary_source": "human", "video_name": V,
             "corrected_at": "2026-06-01T09:00:00", "corrected_by": "fixer_person"}
    seg = _write(tmp_path / f"{V}_segments.json", fixer)
    _write(tmp_path / f"{V}_unified_ground_truth.json", _gt(GT, at="2026-05-04T11:00:00"))
    assert apply_ground_truth_cuts(seg, V) is None, "the fixer's later save stands"
    assert _doc(seg)["boundaries"] == ALGO

    _write(tmp_path / f"{V}_unified_ground_truth.json", _gt(GT, at="2026-07-01T11:00:00"))
    assert apply_ground_truth_cuts(seg, V)
    assert _doc(seg)["boundaries"] == GT


def test_never_in_the_filed_tree(tmp_path):
    from mousereach.review.hand_cuts import apply_ground_truth_cuts
    filed = tmp_path / "Analyzed" / "PRJ"
    seg = _write(filed / f"{V}_segments.json", {"boundaries": ALGO, "video_name": V})
    _write(filed / f"{V}_unified_ground_truth.json", _gt(GT))
    assert apply_ground_truth_cuts(seg, V) is None
    assert _doc(seg)["boundaries"] == ALGO


def test_a_review_return_starts_at_segmentation_when_ground_truth_cuts_are_due(tmp_path):
    from mousereach.watcher.resume_point import current_start
    _write(tmp_path / f"{V}_processing_manifest.json", {"dlc_model": {"dlc_scorer": "S"}})
    _write(tmp_path / f"{V}_segments.json", {"boundaries": ALGO, "video_name": V})
    declared = {"versions": {"dlc_scorer": "S"}}
    assert current_start(tmp_path, V, "checks", declared, "S")[0] != "segmentation"
    _write(tmp_path / f"{V}_unified_ground_truth.json", _gt(GT))
    start, why = current_start(tmp_path, V, "checks", declared, "S")
    assert start == "segmentation" and "ground truth" in why
