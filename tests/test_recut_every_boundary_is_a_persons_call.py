"""Every boundary must be explicitly accepted or moved before re-cut cuts count
(owner decision MRR-08d, 2026-11-19).

"Force a human to at least look at every single boundary ... can't save and
can't send the video out of resegmentation until they have explicitly accepted
that a boundary is already correctly placed or have corrected it."

Before: a fixer save stamped EVERY cut as a person's whether or not it was
walked, and the walk record ("guided_walk") was keyed by the boundary's
position, which shifts when a cut moves past another or is removed -- a later
answer overwrote an earlier one. Now:
- the fixer keeps each call under the cut's FRAME and refuses Save while any
  cut has no call;
- the saved file carries ``boundary_calls``; a video leaves re-segmentation
  (resegmented_since_entry, release_after_hand_cuts) only when every cut in it
  is a call;
- an older save's surviving calls are read (called_frames) and, on the same
  visit or when sent back for exactly this reason, kept so only the remaining
  boundaries are walked (carried_calls).

Built under tmp_path with invented ids.
"""
import json
from types import SimpleNamespace

import pytest

import mousereach.review.fix_segmentation_widget as fsw
from mousereach.review.fix_segmentation_widget import FixSegmentationWidget as W
from mousereach.review.hand_cuts import (RECUT_REMAINING_REASON, called_frames,
                                         release_after_hand_cuts, uncalled_boundaries)
from mousereach.review.resegmented import carried_calls, resegmented_since_entry

STEM = "20990101_ABC9901_P1"
ROUTED = "2099-01-01T09:00:00"
AFTER = "2099-01-02T09:00:00"
BEFORE = "2098-12-31T09:00:00"


def _calls(frames, action="accepted"):
    return [{"frame": f, "action": action} for f in frames]


def _bundle(tmp_path, seg_extra, reason="segmentation: cuts wrong"):
    b = tmp_path / "Review" / "deep_review" / STEM
    b.mkdir(parents=True)
    (b / f"{STEM}_routing.json").write_text(
        json.dumps({"routed_at": ROUTED, "routed_reason": reason}), encoding="utf-8")
    seg = {"boundaries": [100, 200, 300, 400], "boundary_source": "human",
           "corrected_at": AFTER, "corrected_by": "person-a"}
    seg.update(seg_extra)
    (b / f"{STEM}_segments.json").write_text(json.dumps(seg), encoding="utf-8")
    return b


# ---- the record --------------------------------------------------------------

def test_calls_are_read_by_frame():
    seg = {"boundaries": [100, 200, 300],
           "boundary_calls": _calls([100, 300]) + [{"frame": 250, "action": "removed"}]}
    assert called_frames(seg) == {100, 300}
    assert uncalled_boundaries(seg) == [200]


def test_an_older_walk_record_is_read_for_what_survived_of_it():
    # Position-keyed: answer 2 (frame 200) was overwritten by the answer for
    # the boundary that moved into position 2 after a removal.
    seg = {"boundaries": [100, 205, 300],
           "guided_walk": [{"boundary_num": 1, "algo_frame": 100, "confirmed": True},
                           {"boundary_num": 2, "algo_frame": 150, "confirmed": False,
                            "corrected_frame": 205}]}
    assert uncalled_boundaries(seg) == [300]
    assert uncalled_boundaries({"boundaries": [1, 2]}) == [1, 2]
    # A cut the algorithm never proposed was placed by the person (the old
    # fixer recorded no walk answer for a cut added by hand).
    seg["algo_boundaries"] = [100, 150, 250]
    seg["boundaries"] = [100, 205, 300, 333]
    assert uncalled_boundaries(seg) == []


# ---- leaving the re-segmentation queue ----------------------------------------

def test_a_save_with_unwalked_boundaries_does_not_let_the_video_leave(tmp_path):
    b = _bundle(tmp_path, {"boundary_calls": _calls([100, 200])})
    ok, why = resegmented_since_entry(b, STEM)
    assert not ok and "2 of 4" in why and "300, 400" in why
    assert "NOT released" in release_after_hand_cuts(b, STEM)
    assert not (b / f"{STEM}_deep_review_cleared.json").exists()


def test_a_save_with_every_boundary_walked_lets_it_leave(tmp_path):
    b = _bundle(tmp_path, {"boundary_calls": _calls([100, 200, 400]) + _calls([300], "moved")})
    assert resegmented_since_entry(b, STEM)[0]
    assert "Released" in release_after_hand_cuts(b, STEM)


def test_calls_are_kept_on_the_same_visit_or_when_sent_back_for_them(tmp_path):
    same_visit = _bundle(tmp_path / "a", {"boundary_calls": _calls([100, 200])})
    assert set(carried_calls(same_visit, STEM)) == {100, 200}
    earlier = _bundle(tmp_path / "b", {"corrected_at": BEFORE,
                                       "boundary_calls": _calls([100, 200])})
    assert carried_calls(earlier, STEM) == {}
    sent_back = _bundle(tmp_path / "c", {"corrected_at": BEFORE,
                                         "guided_walk": [{"boundary_num": 1, "algo_frame": 100,
                                                          "confirmed": True}]},
                        reason=RECUT_REMAINING_REASON)
    assert set(carried_calls(sent_back, STEM)) == {100}


# ---- the fixer ------------------------------------------------------------------

class _Text:
    def __init__(self):
        self.t = ""

    def setText(self, t):
        self.t = t

    def text(self):
        return self.t


class _Check:
    def __init__(self, on=False):
        self.on = on

    def isChecked(self):
        return self.on

    def setChecked(self, v):
        self.on = bool(v)


class _Spin:
    def __init__(self):
        self.v = 0

    def value(self):
        return self.v

    def setValue(self, v):
        self.v = int(v)


def _fixer(boundaries):
    stub = SimpleNamespace(
        boundaries=list(boundaries), original_boundaries=list(boundaries),
        _calls={}, _removed_calls=[], _gw_idx=0, candidates=[],
        gw_header=_Text(), gw_q=_Text(), gw_status=_Text(), status=_Text(),
        gw_yes=_Check(True), gw_no=_Check(False), gw_spin=_Spin(), gw_cands=None,
        _refresh_candidates=lambda: None, _refresh_segments=lambda: None,
        _goto=lambda f: None, _gw_fill_candidate_row=lambda *a, **k: None)
    for name in ("_call", "_uncall", "_uncalled", "_first_uncalled_idx", "_gw_load",
                 "_gw_confirm", "_gw_remove_boundary"):
        setattr(stub, name, (lambda f: lambda *a, **k: f(stub, *a, **k))(getattr(W, name)))
    return stub


def test_walk_records_follow_the_cut_not_its_position():
    w = _fixer([100, 200, 300, 400])
    w._gw_load()
    w._gw_confirm()                      # 100 accepted -> walk goes to 200
    assert w._gw_idx == 1
    w.gw_no.setChecked(True)
    w.gw_spin.setValue(350)              # 200 moved past 300
    w._gw_confirm()
    assert w.boundaries == [100, 300, 350, 400]
    assert w._calls[350]["action"] == "moved" and w._calls[350]["shown_frame"] == 200
    assert 200 not in w._calls
    # The walk goes to the next cut nobody called (400 after 350, then 300).
    assert sorted(w.boundaries)[w._gw_idx] == 400
    w.gw_no.setChecked(False)
    w._gw_remove_boundary()              # 400 removed
    assert w.boundaries == [100, 300, 350]
    assert sorted(w.boundaries)[w._gw_idx] == 300
    assert w._uncalled() == [300]
    w._gw_confirm()
    assert w._uncalled() == []
    assert {c["frame"]: c["action"] for c in w._calls.values()} == {
        100: "accepted", 350: "moved", 300: "accepted"}
    assert w._removed_calls[0]["frame"] == 400


class _Box:
    calls = []

    @classmethod
    def warning(cls, *a, **k):
        cls.calls.append(a[1:])

    critical = warning


def test_save_is_refused_while_a_boundary_is_unwalked(tmp_path, monkeypatch):
    monkeypatch.setattr(fsw, "QMessageBox", _Box)
    _Box.calls = []
    b = _bundle(tmp_path, {"boundary_source": None, "corrected_at": None})
    before = (b / f"{STEM}_segments.json").read_text(encoding="utf-8")
    w = _fixer([100, 200, 300, 400])
    w.seg_path = b / f"{STEM}_segments.json"
    w.seg = json.loads(before)
    w.video_stem = STEM
    w._call(100, "accepted")
    W._save(w)
    assert _Box.calls and "3 of 4" in _Box.calls[0][1]
    assert (b / f"{STEM}_segments.json").read_text(encoding="utf-8") == before
