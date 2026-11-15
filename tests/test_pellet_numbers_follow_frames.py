"""Hand scores and ground truth are paired with segments by FRAMES.

The hand-score check compared segment N with hand pellet N, and the review
window's check-your-answer box did the same, although a person can record
that a segment is really another pellet (``true_segment_num``). Ground truth
was applied by segment number, although a re-cut renumbers segments: its
"segment 7" can be other footage than today's segment 7. Now:

- ground truth's outcomes are matched to the current segment their frames
  cover, in the ground truth's own segmentation;
- a pellet's hand score is looked up by the pellet the record is about;
- when a person said the cuts number a pellet wrongly and the current cuts
  still disagree, nothing is compared: the video is held for a person rather
  than compared with another pellet's score.
"""
import json

import pytest

import mousereach.review.causal_review_io as cio
import mousereach.watcher.hand_scores as hs
import mousereach.watcher.route_cli as rc
from mousereach.review.triage_status import gt_resolved_segments
from mousereach.review.truth_resolver import _seg_overrides_from_gt

VID = "20990101_ABC9901_P1"

# Ground truth made on cuts 100 / 200 / 300 / 400 (segments 1-3) ...
GT_BOUNDS = [100, 200, 300, 400]
# ... and the video since re-cut with one more cut in front: what the ground
# truth calls segment 2 (frames 200-299) is today's segment 3.
RECUT = [50, 100, 200, 300, 400]


def _gt(determined_segment=2, outcome="retrieved"):
    return {
        "segmentation": {"boundaries": [{"index": i, "frame": f, "determined": True}
                                        for i, f in enumerate(GT_BOUNDS)]},
        "outcomes": {"segments": [{"segment_num": determined_segment, "outcome": outcome,
                                   "determined": True}]},
    }


def _current(bounds):
    return cio.segments_from_segmentation({"boundaries": bounds})


def test_ground_truth_outcome_lands_on_the_segment_its_frames_cover():
    overrides, _ = _seg_overrides_from_gt(_gt(), _current(RECUT))
    assert set(overrides) == {3}
    assert overrides[3]["outcome"] == "retrieved"


def test_ground_truth_on_unchanged_cuts_keeps_its_number():
    overrides, _ = _seg_overrides_from_gt(_gt(), _current(GT_BOUNDS))
    assert set(overrides) == {2}


def test_ground_truth_about_frames_no_segment_covers_is_not_applied():
    overrides, _ = _seg_overrides_from_gt(_gt(), _current([5000, 5100, 5200]))
    assert overrides == {}


def test_ground_truth_resolves_the_current_segment_for_the_gate(tmp_path):
    (tmp_path / f"{VID}_unified_ground_truth.json").write_text(json.dumps(_gt()),
                                                              encoding="utf-8")
    (tmp_path / f"{VID}_segments.json").write_text(json.dumps({"boundaries": RECUT}),
                                                  encoding="utf-8")
    full, segs = gt_resolved_segments(VID, extra_dirs=(tmp_path,))
    assert (full, segs) == (False, {3})


# ---------------------------------------------------------------- hand scores

def _site(tmp_path, monkeypatch, review_records, bounds=(100, 200, 300, 400)):
    f = tmp_path / "hand.csv"
    f.write_text("video_key,pellet,score\n" + "".join(
        "%s,%d,%d\n" % (VID, p, s) for p, s in ((1, 0), (2, 0), (3, 2))), encoding="utf-8")
    f.with_suffix(".json").write_text(json.dumps({"last_successful_import": {}}),
                                      encoding="utf-8")
    monkeypatch.setattr(hs, "settings", lambda: {"path": str(f), "required_projects": []})
    hs._CACHE.update(key=None)
    monkeypatch.setattr(cio, "durable_review_dir", lambda: None)
    monkeypatch.setattr(rc, "answered_segments_of", lambda v, d: set())
    folder = tmp_path / "work"
    folder.mkdir()
    (folder / f"{VID}_segments.json").write_text(json.dumps({"boundaries": list(bounds)}),
                                                 encoding="utf-8")
    (folder / f"{VID}_pellet_outcomes.json").write_text(json.dumps({"segments": [
        {"segment_num": 1, "outcome": "untouched"},
        {"segment_num": 2, "outcome": "untouched"},
        {"segment_num": 3, "outcome": "untouched"}]}), encoding="utf-8")
    (folder / f"{VID}_causal_review.json").write_text(
        json.dumps({"segments": review_records}), encoding="utf-8")
    return folder


def _mislabel(seg_num, true_num, start):
    return {"segment_num": seg_num, "true_segment_num": true_num, "segmentation_wrong": True,
            "segment_span": {"start": start, "end": start + 99},
            "human": {"outcome": None}, "answers": {"reviewed": False}}


def test_hand_check_holds_when_a_person_said_the_cuts_number_a_pellet_wrongly(
        tmp_path, monkeypatch):
    # A person said segment 2 (frames 200-299) is really pellet 3, and the
    # cuts still call it 2: comparing segment 1 with hand pellet 1 is a guess.
    folder = _site(tmp_path, monkeypatch, [_mislabel(2, 3, 200)])
    with pytest.raises(hs.PelletNumbersUnknown):
        hs.check(VID, folder)


def test_hand_check_compares_again_once_the_cuts_agree_with_the_person(
        tmp_path, monkeypatch):
    # After the re-cut, frames 200-299 are segment 3, the number the person
    # gave: resolved, so the check runs (and segment 3 disagrees with its score).
    folder = _site(tmp_path, monkeypatch, [_mislabel(2, 3, 200)],
                   bounds=(0, 100, 200, 300, 400))
    v = hs.check(VID, folder)
    assert v.status == hs.DISAGREES and set(v.disagreements) == {3}


def test_answer_box_uses_the_pellet_the_record_is_about():
    rec = {"segment_num": 2, "true_segment_num": 3, "segmentation_wrong": True}
    assert hs.pellet_of_record(rec, {2: rec}) == (3, "")
    plain = {"segment_num": 1}
    pellet, why = hs.pellet_of_record(plain, {1: plain, 2: rec})
    assert pellet is None and "segment 2 shows pellet 3" in why
    assert hs.pellet_of_record(plain, {1: plain}) == (1, "")
