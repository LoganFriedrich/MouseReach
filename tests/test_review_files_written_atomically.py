"""Files holding a person's answers (and the stage outputs the quality
verdict is stamped into) are replaced in one step, never rewritten in place.

WHY: opening a file for writing empties it first. A crash, a closed window or
a lost share connection part-way through a write left a truncated file, and
the readers take an unreadable review copy for "no answers" and an unreadable
ground-truth file for "no ground truth" -- a person's work lost silently.

The tests simulate a disk that fails part-way through a write: the first
write to any file opened for writing puts half its text on disk and then
raises. Whatever the writer does, the file that was there before must still
be complete afterwards, and no stray file named after the video may be left
(the router and the archiver carry every file whose name starts with the
video's name). Everything here uses invented video names.
"""
import builtins
import io
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

VID = "PRJ0101_20990101_P1"


# ---------------------------------------------------------------------------
# A disk that fails part-way through a write
# ---------------------------------------------------------------------------

class _HalfWriter:
    """Wraps a real file: the first write puts half its data on disk, then
    the disk 'fills up'."""

    def __init__(self, f):
        self._f = f

    def write(self, data):
        self._f.write(data[: max(1, len(data) // 2)])
        self._f.flush()
        raise OSError(28, "No space left on device (simulated)")

    def writelines(self, lines):
        for ln in lines:
            self.write(ln)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._f.close()
        return False

    def __getattr__(self, name):
        return getattr(self._f, name)


@pytest.fixture
def failing_disk(monkeypatch):
    """While active, every file opened for writing fails half-way through
    its first write. Covers open(), Path.write_text, os.fdopen and
    shutil.copy* (all of which go through io.open)."""
    real_open = io.open
    # "spare": substrings of paths whose writes still succeed (a backup the
    # writer makes first, so the failure reaches the write under test).
    state = {"on": False, "spare": ()}

    def fake_open(file, mode="r", *a, **kw):
        f = real_open(file, mode, *a, **kw)
        if state["on"] and any(c in mode for c in "wax+"):
            name = "" if isinstance(file, int) else os.fspath(file)
            if not (name and any(s in str(name) for s in state["spare"])):
                return _HalfWriter(f)
        return f

    monkeypatch.setattr(io, "open", fake_open)
    monkeypatch.setattr(builtins, "open", fake_open)
    # Python 3.10's pathlib bound io.open at import time (Path.open goes
    # through an accessor), so Path.write_text needs its own patch.
    import pathlib
    accessor = getattr(pathlib, "_NormalAccessor", None)
    if accessor is not None and hasattr(accessor, "open"):
        monkeypatch.setattr(accessor, "open", staticmethod(fake_open))
    # retries of a refused rename must not really sleep
    import mousereach.pipeline.fsutil as fsutil
    monkeypatch.setattr(fsutil.time, "sleep", lambda s: None, raising=False)
    return state


def _stray(folder: Path, keep):
    """Files in ``folder`` other than ``keep`` (names)."""
    return sorted(p.name for p in Path(folder).iterdir() if p.name not in keep)


def _write(path: Path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# MRR-12i: ground truth
# ---------------------------------------------------------------------------

def _gt(video_name):
    from mousereach.review.unified_gt import UnifiedGroundTruth
    return UnifiedGroundTruth(video_name=video_name)


def test_ground_truth_survives_a_failed_save(tmp_path, failing_disk):
    from mousereach.review import unified_gt as ug
    video = tmp_path / f"{VID}.mp4"
    video.write_bytes(b"")
    gt_path = ug.save_unified_gt(_gt(VID), video)
    before = gt_path.read_text(encoding="utf-8")
    assert json.loads(before)["video_name"] == VID

    failing_disk["on"] = True
    with pytest.raises(OSError):
        ug.save_unified_gt(_gt(VID), video)
    failing_disk["on"] = False

    # the earlier ground truth is complete and readable, and nothing else
    # named after the video was left beside it
    assert gt_path.read_text(encoding="utf-8") == before
    assert ug.load_unified_gt(video) is not None
    assert _stray(tmp_path, {video.name, gt_path.name}) == []


def test_ground_truth_unserialisable_content_leaves_old_file(tmp_path, monkeypatch):
    """Content that cannot be written as JSON fails BEFORE the file is
    touched (it used to be streamed into the emptied file)."""
    from mousereach.review import unified_gt as ug
    video = tmp_path / f"{VID}.mp4"
    video.write_bytes(b"")
    gt_path = ug.save_unified_gt(_gt(VID), video)
    before = gt_path.read_text(encoding="utf-8")

    monkeypatch.setattr(ug, "_unified_gt_to_dict",
                        lambda gt: {"video_name": VID, "pad": "x" * 200000,
                                    "bad": object()})
    with pytest.raises(TypeError):
        ug.save_unified_gt(_gt(VID), video)
    assert gt_path.read_text(encoding="utf-8") == before


# ---------------------------------------------------------------------------
# The shared writer itself
# ---------------------------------------------------------------------------

def test_atomic_writer_retries_a_refused_rename_then_succeeds(tmp_path, monkeypatch):
    from mousereach.pipeline import fsutil
    monkeypatch.setattr(fsutil.time, "sleep", lambda s: None)
    target = tmp_path / f"{VID}_x.json"
    _write(target, {"v": 1})
    real_replace = os.replace
    calls = {"n": 0}

    def refuse_twice(src, dst):
        calls["n"] += 1
        if calls["n"] <= 2:
            e = PermissionError(13, "file in use")
            e.winerror = 32
            raise e
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", refuse_twice)
    fsutil.write_json_atomic(target, {"v": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"v": 2}
    assert calls["n"] == 3
    assert _stray(tmp_path, {target.name}) == []


def test_atomic_writer_gives_up_loudly_and_cleans_up(tmp_path, monkeypatch):
    from mousereach.pipeline import fsutil
    monkeypatch.setattr(fsutil.time, "sleep", lambda s: None)
    target = tmp_path / f"{VID}_x.json"
    _write(target, {"v": 1})

    def always_refuse(src, dst):
        e = PermissionError(13, "file in use")
        e.winerror = 32
        raise e

    monkeypatch.setattr(os, "replace", always_refuse)
    with pytest.raises(PermissionError):
        fsutil.write_json_atomic(target, {"v": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"v": 1}
    assert _stray(tmp_path, {target.name}) == []


def test_temporary_name_never_starts_with_the_video_name(tmp_path, monkeypatch):
    """A temporary file a crash leaves behind must not be swept up with the
    video's files, and remove_leftover_temps finds it again."""
    from mousereach.pipeline import fsutil
    target = tmp_path / f"{VID}_segments.json"
    seen = []
    real_replace = os.replace

    def spy(src, dst):
        seen.append(Path(src).name)
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    fsutil.write_json_atomic(target, {"v": 1})
    assert seen and not seen[0].startswith(VID) and seen[0].startswith(".")

    left = tmp_path / f".{target.name}.abc123.tmp"
    left.write_text("{", encoding="utf-8")
    other = tmp_path / f".{VID}_reaches.json.abc123.tmp"
    other.write_text("{", encoding="utf-8")
    assert fsutil.remove_leftover_temps(target) == 1
    assert not left.exists() and other.exists()


# ---------------------------------------------------------------------------
# MRR-02b: review copies, clear markers, flagged sessions (causal_review_io)
# ---------------------------------------------------------------------------

def _seg_record(n, outcome="retrieved"):
    return {"segment_num": n, "segment_span": [n * 100, n * 100 + 99],
            "human": {"outcome": outcome, "agreed": True},
            "answers": {"reviewed": True, "outcome": outcome}}


@pytest.fixture
def review_roots(tmp_path, monkeypatch):
    from mousereach.review import causal_review_io as cio
    records = tmp_path / "review_records"
    monkeypatch.setattr(cio, "_default_index_dir", lambda: records)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    return SimpleNamespace(records=records, bundle=bundle)


def test_review_copies_survive_a_failed_save(review_roots, failing_disk):
    from mousereach.review import causal_review_io as cio
    out = cio.save_causal_review(VID, review_roots.bundle, [_seg_record(1)], {})
    durable = cio.durable_review_path(VID)
    working_before = out.read_text(encoding="utf-8")
    durable_before = durable.read_text(encoding="utf-8")

    failing_disk["on"] = True
    with pytest.raises(OSError):
        cio.save_causal_review(VID, review_roots.bundle,
                               [_seg_record(1), _seg_record(2, "displaced")], {})
    failing_disk["on"] = False

    # both copies still hold the earlier answers, complete
    assert out.read_text(encoding="utf-8") == working_before
    assert durable.read_text(encoding="utf-8") == durable_before
    docs = cio.review_copies(VID, [review_roots.bundle])
    assert len(docs) == 2
    assert all(d["segments"][0]["human"]["outcome"] == "retrieved" for d in docs)
    assert _stray(review_roots.bundle, {out.name}) == []
    assert _stray(durable.parent, {durable.name}) == []


def test_clear_marker_and_session_flag_never_torn(review_roots, failing_disk):
    from mousereach.review import causal_review_io as cio
    marker = review_roots.bundle / f"{VID}_deep_review_cleared.json"
    cio._write_json(marker, {"cleared_by": "someone"})
    before = marker.read_text(encoding="utf-8")
    cio.flag_session(VID, review_roots.records, reason="first")
    flags = review_roots.records / "flagged_sessions.json"
    flags_before = flags.read_text(encoding="utf-8")

    failing_disk["on"] = True
    with pytest.raises(OSError):
        cio._write_json(marker, {"cleared_by": "someone else"})
    with pytest.raises(OSError):
        cio.flag_session("PRJ0102_20990101_P1", review_roots.records, reason="x")
    failing_disk["on"] = False

    assert marker.read_text(encoding="utf-8") == before
    assert flags.read_text(encoding="utf-8") == flags_before
    assert _stray(review_roots.bundle, {marker.name}) == []


def test_durable_copy_at_return_survives_a_failed_copy(review_roots, failing_disk):
    """review_return copies the bundle's review over the durable copy before
    the bundle is emptied; a copy that fails part-way must leave the earlier
    durable copy complete."""
    from mousereach.review import causal_review_io as cio
    from mousereach.watcher import review_return as rr
    durable = cio.durable_review_path(VID)
    _write(durable, {"type": "causal_review", "segments": [_seg_record(1)]})
    before = durable.read_text(encoding="utf-8")
    src = review_roots.bundle / f"{VID}_causal_review.json"
    _write(src, {"type": "causal_review",
                 "segments": [_seg_record(1), _seg_record(2, "displaced")]})
    old = time.time() - 3600
    os.utime(durable, (old, old))

    failing_disk["on"] = True
    rr._ensure_durable_review(review_roots.bundle, VID)   # never raises
    failing_disk["on"] = False

    assert durable.read_text(encoding="utf-8") == before
    assert _stray(durable.parent, {durable.name}) == []

    # and when the disk is healthy the copy is made, with the source's time
    rr._ensure_durable_review(review_roots.bundle, VID)
    assert json.loads(durable.read_text(encoding="utf-8"))["segments"][1]["segment_num"] == 2
    assert abs(durable.stat().st_mtime - src.stat().st_mtime) < 2


# ---------------------------------------------------------------------------
# MRR-09e: routing records (route_cli.flag_segments / reopen_answered)
# ---------------------------------------------------------------------------

def test_flag_segments_never_tears_the_outcomes_file(tmp_path, failing_disk):
    from mousereach.watcher import route_cli
    p = tmp_path / f"{VID}_pellet_outcomes.json"
    _write(p, {"segments": [{"segment_num": n, "outcome": "retrieved"}
                            for n in range(1, 21)]})
    before = p.read_text(encoding="utf-8")

    failing_disk["on"] = True
    with pytest.raises(OSError):
        route_cli.flag_segments(p, [3], "checked again")
    failing_disk["on"] = False

    assert p.read_text(encoding="utf-8") == before
    assert _stray(tmp_path, {p.name}) == []
    # healthy disk: the flag lands
    assert route_cli.flag_segments(p, [3], "checked again") == [3]
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert doc["segments"][2]["flagged_for_review"] is True


def test_reopen_answered_never_tears_the_review(tmp_path, monkeypatch, failing_disk):
    from mousereach.config import Paths
    from mousereach.review import causal_review_io as cio
    from mousereach.watcher import route_cli
    monkeypatch.setattr(cio, "_default_index_dir", lambda: None)
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", tmp_path / "analyzed", raising=False)
    folder = tmp_path / "work"
    rp = folder / f"{VID}_causal_review.json"
    _write(rp, {"type": "causal_review", "video_stem": VID, "reviewer": "r1",
                "reviewed_at": "2099-01-01T00:00:00",
                "segments": [_seg_record(1), _seg_record(2)]})
    before = rp.read_text(encoding="utf-8")

    failing_disk["on"] = True
    failing_disk["spare"] = ("manual_corrections",)   # the backup is made
    with pytest.raises(OSError):
        route_cli.reopen_answered(VID, folder, [1], {1: "are you sure?"})
    failing_disk["on"] = False

    assert rp.read_text(encoding="utf-8") == before
    assert _stray(folder, {rp.name}) == []
    backups = list((tmp_path / "analyzed" / "Archive" / "manual_corrections").iterdir())
    assert len(backups) == 1 and backups[0].read_text(encoding="utf-8") == before

    # healthy disk: the answer is reopened, and kept under recheck.previous
    assert route_cli.reopen_answered(VID, folder, [1], {1: "are you sure?"}) == [1]
    rec = json.loads(rp.read_text(encoding="utf-8"))["segments"][0]
    assert rec["answers"]["reviewed"] is False
    assert rec["answers"]["recheck"]["previous"]["human"]["outcome"] == "retrieved"


def test_hand_score_record_leaves_no_file_named_after_the_video(tmp_path, failing_disk):
    """The hand-score record used its own "<video>_pellet_outcomes.json.tmp";
    one left by a failed write starts with the video's name, so the router
    and the archiver carried it along as one of the video's files."""
    from mousereach.watcher import hand_scores as hs
    p = tmp_path / f"{VID}_pellet_outcomes.json"
    _write(p, {"segments": [{"segment_num": 1}]})
    before = p.read_text(encoding="utf-8")
    v = hs.Verdict(status=hs.AGREES, reason="", hand={1: 2})

    failing_disk["on"] = True
    with pytest.raises(OSError):
        hs.record(tmp_path, VID, v, 1.0)
    failing_disk["on"] = False

    assert p.read_text(encoding="utf-8") == before
    assert [n for n in os.listdir(tmp_path) if n.startswith(VID) and n != p.name] == []
    assert _stray(tmp_path, {p.name}) == []

    assert hs.record(tmp_path, VID, v, 1.0) is True
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert doc["hand_scores"]["pellets"] == {"1": 2} and doc["segments"]


# ---------------------------------------------------------------------------
# MRR-14c: the quality verdict stamped into the three stage files
# ---------------------------------------------------------------------------

STAGES = ("_segments.json", "_reaches.json", "_pellet_outcomes.json")


def _triage_orch():
    from unittest import mock
    from mousereach.watcher.orchestrator import BaseOrchestrator
    o = BaseOrchestrator.__new__(BaseOrchestrator)
    o.db = mock.Mock()
    return o


def _fake_triage(**kw):
    return SimpleNamespace(
        verdict="auto_approved", flags=[], n_critical=0, n_warnings=0,
        save=lambda path: Path(path).write_text("{}", encoding="utf-8"))


def _stage_files(folder):
    out = {}
    for s in STAGES:
        p = folder / f"{VID}{s}"
        _write(p, {"stage": s, "rows": list(range(500))})
        out[p] = p.read_text(encoding="utf-8")
    return out


def test_quality_verdict_stamp_never_tears_a_stage_output(tmp_path, failing_disk):
    before = _stage_files(tmp_path)
    failing_disk["on"] = True
    failing_disk["spare"] = ("_triage.json",)
    _triage_orch()._run_quality_triage(VID, tmp_path, tmp_path / "p.h5", _fake_triage)
    failing_disk["on"] = False

    for p, text in before.items():
        assert p.read_text(encoding="utf-8") == text, p.name
    assert _stray(tmp_path, {p.name for p in before} | {f"{VID}_triage.json"}) == []


def test_quality_verdict_stamp_clears_leftover_temps_and_stamps(tmp_path):
    before = _stage_files(tmp_path)
    leftovers = [tmp_path / f".{VID}{s}.dead01.tmp" for s in STAGES]
    for p in leftovers:
        p.write_text("{", encoding="utf-8")
    verdict = _triage_orch()._run_quality_triage(VID, tmp_path, tmp_path / "p.h5",
                                                 _fake_triage)
    assert verdict == "auto_approved"
    assert not any(p.exists() for p in leftovers)
    for p in before:
        doc = json.loads(p.read_text(encoding="utf-8"))
        assert doc["validation_status"] == "auto_approved"
        assert doc["rows"] == list(range(500))


# ---------------------------------------------------------------------------
# MRR-02c: the corpus index and flagged_sessions.json, one writer at a time
# ---------------------------------------------------------------------------

def _race(monkeypatch, file_name, first, second, hold_s=1.5):
    """Run ``first`` in one thread and, while it sits between reading
    ``file_name`` and writing it back, run ``second`` in another thread --
    two review windows saving at the same moment."""
    from mousereach.review import causal_review_io as cio
    real_read = cio._read_json
    read_done, go = threading.Event(), threading.Event()
    errors = []

    def slow_read(path):
        doc = real_read(path)
        if (Path(path).name == file_name
                and threading.current_thread().name == "first"
                and not read_done.is_set()):
            read_done.set()
            go.wait(hold_s)           # the other window saves meanwhile
        return doc

    monkeypatch.setattr(cio, "_read_json", slow_read)

    def run(fn, after=None):
        try:
            if after is not None:
                after.wait(10)
            fn()
        except Exception as e:      # pragma: no cover - reported below
            errors.append(e)
        finally:
            go.set()

    t1 = threading.Thread(target=run, args=(first,), name="first")
    t2 = threading.Thread(target=run, args=(second, read_done), name="second")
    t1.start(); t2.start()
    t1.join(30); t2.join(30)
    assert not errors, errors


def test_two_saves_at_once_keep_both_index_entries(review_roots, monkeypatch):
    from mousereach.review import causal_review_io as cio
    monkeypatch.setattr(cio, "INDEX_LOCK_POLL_S", 0.02, raising=False)
    other = "PRJ0102_20990101_P1"

    def save(stem):
        return lambda: cio.update_corpus_index(
            stem, review_roots.bundle / f"{stem}_causal_review.json",
            [_seg_record(1)], "r", "2099-01-01T00:00:00")

    _race(monkeypatch, "causal_review_index.json", save(VID), save(other))
    entries = json.loads((review_roots.records / "causal_review_index.json")
                         .read_text(encoding="utf-8"))["entries"]
    assert set(entries) == {f"{VID}__seg1", f"{other}__seg1"}
    assert _stray(review_roots.records, {"causal_review_index.json"}) == []


def test_two_session_flags_at_once_keep_both(review_roots, monkeypatch):
    from mousereach.review import causal_review_io as cio
    monkeypatch.setattr(cio, "INDEX_LOCK_POLL_S", 0.02, raising=False)
    root = review_roots.records
    cio.flag_session("PRJ0109_20990101_P1", root, reason="earlier")
    _race(monkeypatch, "flagged_sessions.json",
          lambda: cio.flag_session(VID, root, reason="a"),
          lambda: cio.flag_session("PRJ0102_20990101_P1", root, reason="b"))
    doc = json.loads((root / "flagged_sessions.json").read_text(encoding="utf-8"))
    assert set(doc["sessions"]) == {cio.session_key(VID),
                                    cio.session_key("PRJ0102_20990101_P1"),
                                    cio.session_key("PRJ0109_20990101_P1")}


def test_a_live_lock_is_waited_for_then_given_up_loudly(review_roots, monkeypatch):
    from mousereach.review import causal_review_io as cio
    monkeypatch.setattr(cio, "INDEX_LOCK_WAIT_S", 0.2)
    monkeypatch.setattr(cio, "INDEX_LOCK_POLL_S", 0.02)
    review_roots.records.mkdir(parents=True, exist_ok=True)
    lock = review_roots.records / "causal_review_index.json.lock"
    lock.write_text(json.dumps({"token": "someone-else"}), encoding="utf-8")
    with pytest.raises(cio.IndexLockBusy):
        cio.update_corpus_index(VID, Path("x"), [_seg_record(1)], "r", "t")
    assert lock.exists()                      # never broken while live
    assert not (review_roots.records / "causal_review_index.json").exists()


def test_a_dead_holders_lock_is_broken_by_its_age(review_roots, monkeypatch):
    from mousereach.review import causal_review_io as cio
    monkeypatch.setattr(cio, "INDEX_LOCK_WAIT_S", 0.2)
    monkeypatch.setattr(cio, "INDEX_LOCK_POLL_S", 0.02)
    review_roots.records.mkdir(parents=True, exist_ok=True)
    lock = review_roots.records / "causal_review_index.json.lock"
    lock.write_text(json.dumps({"token": "dead"}), encoding="utf-8")
    old = time.time() - cio.INDEX_LOCK_STALE_S - 60
    os.utime(lock, (old, old))
    cio.update_corpus_index(VID, Path("x"), [_seg_record(1)], "r", "t")
    assert not lock.exists()
    entries = json.loads((review_roots.records / "causal_review_index.json")
                         .read_text(encoding="utf-8"))["entries"]
    assert f"{VID}__seg1" in entries
    assert _stray(review_roots.records, {"causal_review_index.json"}) == []
