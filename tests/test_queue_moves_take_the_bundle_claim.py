"""A move out of a review queue waits for the bundle claim the return scan takes.

The watcher's return scan takes a per-video claim before it touches a bundle.
Escalate, Flag Session and a route of a queue bundle moved it with no claim,
so the scan (on another machine, or the watcher on this one) could be moving
the same folder at that moment. Now such a move is refused, with nothing
moved, while another holder has the claim; a route started by the holder
itself still goes on (the claim is re-entrant per thread).

The "other program" here is another thread of this process holding the claim,
which the claim refuses exactly as it refuses another process. Built under
tmp_path; no real queue, claim folder or database.
"""
import threading
from types import SimpleNamespace

import pytest

import mousereach.review.causal_review_widget as crw
import mousereach.watcher.review_gate as rg
from mousereach.config import Paths
from mousereach.watcher import review_return

VID = "20990101_ABC9901_P1"


@pytest.fixture
def queues(tmp_path, monkeypatch):
    triage = tmp_path / "review" / "triage"
    deep = tmp_path / "review" / "deep_review"
    (triage / VID).mkdir(parents=True)
    for name in (f"{VID}.mp4", f"{VID}_segments.json"):
        (triage / VID / name).write_text("x")
    monkeypatch.setattr(Paths, "REVIEW_ROOT", tmp_path / "review", raising=False)
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(Paths, "DEEP_REVIEW", deep)
    return triage, deep


@pytest.fixture
def held_elsewhere():
    """Another holder (a second thread) keeps VID's claim until the test ends."""
    taken, done = threading.Event(), threading.Event()
    result = {}

    def hold():
        result["ok"] = review_return._claim_return(VID)
        taken.set()
        done.wait(10)
        if result["ok"]:
            review_return._release_return_claim(VID)

    t = threading.Thread(target=hold)
    t.start()
    taken.wait(10)
    assert result["ok"]
    yield
    done.set()
    t.join(10)


def _refused(fn):
    try:
        fn()
    except Exception as e:      # the refusal; nothing may have moved
        return e
    return None


def test_escalate_is_refused_while_another_program_holds_the_claim(queues, held_elsewhere):
    triage, deep = queues
    err = _refused(lambda: rg.escalate_to_deep_review(VID, triage / VID, "bad cuts"))
    assert (triage / VID / f"{VID}.mp4").is_file()            # nothing moved
    assert not (deep / VID).exists()
    assert isinstance(err, rg.AlreadyInQueueError)           # handled as "try later"


def test_route_out_of_a_queue_is_refused_while_held(queues, held_elsewhere):
    triage, deep = queues
    err = _refused(lambda: rg.route_to_queue(VID, triage / VID, deep, "divert",
                                             db_state="deep_review"))
    assert (triage / VID / f"{VID}.mp4").is_file()
    assert not (deep / VID).exists()
    assert isinstance(err, rg.AlreadyInQueueError)


def test_the_holder_itself_still_routes(queues):
    triage, deep = queues
    assert review_return._claim_return(VID)
    try:
        rg.route_to_queue(VID, triage / VID, deep, "divert", db_state="deep_review")
    finally:
        review_return._release_return_claim(VID)
    assert (deep / VID / f"{VID}.mp4").is_file()
    assert VID not in review_return._claims_held             # claim given back


def test_flag_session_touches_nothing_while_held(queues, held_elsewhere, monkeypatch):
    triage, deep = queues
    touched, errors, advanced = [], [], []
    monkeypatch.setattr(crw, "flag_session", lambda stem, root, reason="", **kw: "key-1")
    monkeypatch.setattr(crw, "show_info", lambda *a, **k: None)
    monkeypatch.setattr(crw, "show_error", lambda msg, *a, **k: errors.append(msg))
    monkeypatch.setattr(rg, "supersede_deep_review_clear", lambda *a, **k: touched.append(a))
    w = SimpleNamespace(
        _video_stem=VID, _bundle_dir=triage / VID,
        _review_root=lambda: triage.parent,
        _save_review=lambda: None,
        _notes_edit=SimpleNamespace(toPlainText=lambda: ""),
        _status_label=SimpleNamespace(setText=lambda text: None),
        _load_next_video=lambda: advanced.append(True),
        _watcher_db_for_move=lambda: (None, ""),
    )
    crw.CausalReviewWidget._flag_session(w)
    assert touched == [] and advanced == []
    assert (triage / VID / f"{VID}.mp4").is_file()
    assert errors and "NOT moved" in errors[0]
