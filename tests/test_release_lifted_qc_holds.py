"""A quality hold the current check would not make leaves a person's queue.

WHY: the quality check demanded an outcomes file Easy and Flat trays never get,
so 469 Easy/Flat bundles sat in deep review on that false alarm (2026-10-22).
Once the check is corrected, such a hold -- made by the machine alone, with no
file a person wrote -- is re-checked, and a bundle the check now passes goes
back to the posed folder for a fresh run in its priority turn. Anything a
person touched, any other reason for the hold, and any hold the current check
still makes stays where it is.
"""
import json
from datetime import datetime

import mousereach.watcher.review_gate as rg
import mousereach.watcher.review_return as rr
from mousereach.pipeline.triage import EXPECTED_SEGMENTS_MIN

POSE = "DLC_resnet101_MPSAOct27shuffle3_100000"


class FakeDB:
    def __init__(self, fail=False):
        self.forced = []
        self.fail = fail

    def get_video(self, vid):
        return {"video_id": vid, "state": "deep_review"}

    def force_state(self, vid, state, reason=None, **kw):
        if self.fail:
            raise RuntimeError("database is locked")
        self.forced.append((vid, state, kw))

    def update_state(self, *a, **k):
        pass


def _wire(monkeypatch, tmp_path):
    review = tmp_path / "Review"
    deep, triage, posed = review / "deep_review", review / "triage", tmp_path / "Posed"
    for d in (deep, triage, posed):
        d.mkdir(parents=True)
    monkeypatch.setattr(rr.Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(rr.Paths, "DEEP_REVIEW", deep)
    monkeypatch.setattr(rr.Paths, "REVIEW_ROOT", review)
    monkeypatch.setattr(rr.Paths, "DLC_STAGING", posed, raising=False)
    monkeypatch.setattr(rr.Paths, "ANALYZED_OUTPUT", None, raising=False)
    monkeypatch.setattr(rr, "_qc_recheck_cache", {})
    monkeypatch.setattr(rr, "_return_retry_after", {})
    monkeypatch.setattr(rr, "_declared_scorer", lambda: POSE)
    monkeypatch.setattr(rg, "route_to_queue", lambda *a, **k: None)
    return deep, posed, review


def _bundle(deep, stem, reason="qc_needs_review",
            flags=({"category": "structural", "severity": "critical",
                    "description": "Missing outcomes file: x"},),
            outcomes=False, person=False):
    b = deep / stem
    b.mkdir()
    (b / f"{stem}.mp4").write_bytes(b"video")
    for ext in (".h5", ".csv", "_meta.pickle"):
        (b / f"{stem}{POSE}{ext}").write_bytes(b"pose")
    (b / f"{stem}_segments.json").write_text(json.dumps(
        {"boundaries": list(range(0, 1000 * EXPECTED_SEGMENTS_MIN, 1000))}), encoding="utf-8")
    (b / f"{stem}_reaches.json").write_text(json.dumps({"segments": []}), encoding="utf-8")
    if outcomes:
        (b / f"{stem}_pellet_outcomes.json").write_text(json.dumps({"segments": []}),
                                                       encoding="utf-8")
    (b / f"{stem}_triage.json").write_text(json.dumps(
        {"video_id": stem, "verdict": "needs_review", "flags": list(flags)}), encoding="utf-8")
    (b / f"{stem}_routing.json").write_text(json.dumps(
        {"video_id": stem, "routed_reason": reason}), encoding="utf-8")
    if person:
        (b / f"{stem}_causal_review.json").write_text("{}", encoding="utf-8")
    return b


def test_a_lifted_hold_goes_back_to_the_posed_folder(tmp_path, monkeypatch):
    deep, posed, review = _wire(monkeypatch, tmp_path)
    stem = "20261012_CNT0615_F4"
    b = _bundle(deep, stem)
    db = FakeDB()

    summary = rr.scan_review_queues(db, tmp_path / "proc")

    assert summary["qc_released"] == 1
    assert not b.exists()
    assert sorted(p.name for p in posed.iterdir()) == sorted(
        [f"{stem}.mp4"] + [f"{stem}{POSE}{e}" for e in (".h5", ".csv", "_meta.pickle")])
    kept = review / "_Problematic" / f"_released_qc_holds_{datetime.now():%Y-%m-%d}" / stem
    assert (kept / f"{stem}_triage.json").is_file()       # machine outputs set aside
    assert (kept / f"{stem}_release_note.json").is_file()
    assert db.forced == [(stem, "dlc_complete", {
        "current_path": str(posed / f"{stem}.mp4"),
        "dlc_output_path": str(posed / f"{stem}{POSE}.h5")})]
    assert not (tmp_path / "proc").exists()                # never straight into processing


def test_holds_that_still_stand_or_belong_to_a_person_stay(tmp_path, monkeypatch):
    deep, posed, _ = _wire(monkeypatch, tmp_path)
    # Pillar videos: Easy/Flat are never held at all (see the tests below).
    kept = [
        _bundle(deep, "20261012_CNT0601_P2", person=True, outcomes=True),  # a person's
        _bundle(deep, "20261012_CNT0602_P2", outcomes=True, flags=(        # tracking hold
            {"category": "dlc_coherence", "severity": "critical",
             "description": "Missing SABL or SABR — cannot compute ruler"},)),
        _bundle(deep, "20261012_CNT0603_P1"),                              # Pillar: still missing
        _bundle(deep, "20261012_CNT0604_P3", reason="segmentation_failed"),  # other reason
    ]
    db = FakeDB()

    summary = rr.scan_review_queues(db, tmp_path / "proc")

    assert summary["qc_released"] == 0
    assert all(b.exists() for b in kept) and not db.forced
    assert list(posed.iterdir()) == []


def test_nothing_in_the_posed_folder_is_replaced(tmp_path, monkeypatch):
    deep, posed, _ = _wire(monkeypatch, tmp_path)
    stem = "20261012_CNT0615_F4"
    b = _bundle(deep, stem)
    (posed / f"{stem}.mp4").write_bytes(b"another copy")
    db = FakeDB()

    assert rr.scan_review_queues(db, tmp_path / "proc")["qc_released"] == 0

    assert (posed / f"{stem}.mp4").read_bytes() == b"another copy"
    assert (b / f"{stem}.mp4").read_bytes() == b"video" and not db.forced


def test_a_release_that_cannot_finish_leaves_the_bundle_as_it_was(tmp_path, monkeypatch):
    deep, posed, review = _wire(monkeypatch, tmp_path)
    stem = "20261012_CNT0615_F4"
    b = _bundle(deep, stem)
    before = sorted(p.name for p in b.iterdir())

    assert rr.scan_review_queues(FakeDB(fail=True), tmp_path / "proc")["qc_released"] == 0

    assert sorted(p.name for p in b.iterdir()) == before
    assert list(posed.iterdir()) == []
    assert not list((review / "_Problematic").rglob("*.*"))


def test_each_bundle_is_rechecked_once_until_it_changes(tmp_path, monkeypatch):
    deep, _, _ = _wire(monkeypatch, tmp_path)
    _bundle(deep, "20261012_CNT0603_P1")          # stays held: Pillar, outcomes missing
    import mousereach.pipeline.triage as tri
    calls = []
    real = tri.triage_video
    monkeypatch.setattr(tri, "triage_video", lambda *a, **k: calls.append(a) or real(*a, **k))

    for _ in range(3):
        rr.scan_review_queues(FakeDB(), tmp_path / "proc")

    assert len(calls) == 1


def test_identical_copies_already_in_the_posed_folder_stay_the_one_copy(tmp_path, monkeypatch):
    """Intake leaves a video's posed files in the posed folder until it is
    archived, so a held video usually still has them there (67 of the first 74
    releases were refused for it, 2026-10-22). Identical copies there are the
    one live copy; the bundle's duplicates go, its outputs are set aside."""
    deep, posed, review = _wire(monkeypatch, tmp_path)
    stem = "20261012_CNT0615_F4"
    b = _bundle(deep, stem)
    (posed / f"{stem}.mp4").write_bytes(b"video")
    for ext in (".h5", ".csv", "_meta.pickle"):
        (posed / f"{stem}{POSE}{ext}").write_bytes(b"pose")
    db = FakeDB()

    assert rr.scan_review_queues(db, tmp_path / "proc")["qc_released"] == 1

    assert not b.exists()
    assert len(list(posed.iterdir())) == 4
    kept = review / "_Problematic" / f"_released_qc_holds_{datetime.now():%Y-%m-%d}" / stem
    names = {p.name for p in kept.iterdir()}
    assert f"{stem}.mp4" not in names and f"{stem}{POSE}.h5" not in names   # no duplicates
    assert f"{stem}_triage.json" in names and f"{stem}_release_note.json" in names
    assert db.forced[0][1] == "dlc_complete"


def _copy_times(src_dir, dst_dir, stem):
    import os
    for f in src_dir.iterdir():
        if f.name == f"{stem}.mp4" or f.name.startswith(f"{stem}DLC"):
            st = f.stat()
            os.utime(dst_dir / f.name, (st.st_atime, st.st_mtime))


def test_same_size_and_time_needs_no_hashing(tmp_path, monkeypatch):
    """The quick check rsync and robocopy use: every copy in the chain keeps
    the modified time, so matching size and time is the same file -- no
    reading of either copy (that cost ~10 s a release over the share)."""
    deep, posed, _ = _wire(monkeypatch, tmp_path)
    stem = "20261012_CNT0615_F4"
    b = _bundle(deep, stem)
    (posed / f"{stem}.mp4").write_bytes(b"video")
    for ext in (".h5", ".csv", "_meta.pickle"):
        (posed / f"{stem}{POSE}{ext}").write_bytes(b"pose")
    _copy_times(b, posed, stem)
    hashed = []
    monkeypatch.setattr(rr, "_file_sha256", lambda p: hashed.append(p) or "x")

    assert rr.scan_review_queues(FakeDB(), tmp_path / "proc")["qc_released"] == 1
    assert hashed == [] and not b.exists()


def test_same_size_but_different_bytes_is_left_for_a_person(tmp_path, monkeypatch):
    deep, posed, _ = _wire(monkeypatch, tmp_path)
    stem = "20261012_CNT0615_F4"
    b = _bundle(deep, stem)
    (posed / f"{stem}.mp4").write_bytes(b"VIDEO")          # same size, other bytes
    import os
    t = (b / f"{stem}.mp4").stat().st_mtime - 3600
    os.utime(posed / f"{stem}.mp4", (t, t))                 # and another time

    assert rr.scan_review_queues(FakeDB(), tmp_path / "proc")["qc_released"] == 0
    assert b.exists() and (posed / f"{stem}.mp4").read_bytes() == b"VIDEO"


def test_a_ruler_variation_hold_is_released(tmp_path, monkeypatch):
    # 2026-10-23 (Logan): a ruler that varies within a video is recorded, never
    # held for -- the scale is an estimate, every kinematic exists in pixels,
    # and nothing is concluded from one reach. Older holds of that kind go back.
    deep, posed, _ = _wire(monkeypatch, tmp_path)
    stem = "20261012_CNT0601_P1"
    b = _bundle(deep, stem, outcomes=True, flags=(
        {"category": "dlc_coherence", "severity": "critical",
         "description": "Ruler (SABL-SABR) highly unstable: CV=0.210, std=9.1px, median=43.0px"},))
    db = FakeDB()

    summary = rr.scan_review_queues(db, tmp_path / "proc")

    assert summary["qc_released"] == 1
    assert not b.exists()
    assert (posed / f"{stem}.mp4").is_file()


def test_the_check_records_ruler_variation_without_holding():
    import numpy as np
    import pandas as pd
    from mousereach.pipeline import triage
    n = 600
    rng = np.random.default_rng(0)
    cols, data = [], []
    for bp, x0 in (("SABL", 100.0), ("SABR", 140.0), ("BOXL", 50.0), ("BOXR", 200.0)):
        x = np.full(n, x0)
        if bp == "SABR":
            x = x0 + rng.normal(0, 8.0, n)       # ruler length varies ~20%
        for coord, v in (("x", x), ("y", np.full(n, 300.0)), ("likelihood", np.full(n, 0.99))):
            cols.append(("scorer", bp, coord))
            data.append(v)
    df = pd.DataFrame(np.array(data).T, columns=pd.MultiIndex.from_tuples(cols))
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        h5 = os.path.join(d, "x.h5")
        df.to_hdf(h5, key="df")
        flags = triage.check_dlc_coherence(h5, [0, n])
    ruler = [f for f in flags if f.description.startswith(triage.RULER_UNSTABLE_DESC)]
    assert ruler and all(f.severity == "warning" for f in ruler)


def test_an_easy_or_flat_machine_hold_is_released_even_with_a_persons_file(tmp_path, monkeypatch):
    # 2026-10-23 (Logan): Easy and Flat are never held for a person; what a
    # person can answer is a pellet outcome, and those do not matter for them.
    # The person's file is set aside with the other outputs, not lost.
    deep, posed, review = _wire(monkeypatch, tmp_path)
    released = [
        _bundle(deep, "20261012_CNT0611_F1", person=True),
        _bundle(deep, "20261012_CNT0612_E2", reason="segmentation_failed"),
        _bundle(deep, "20261012_CNT0613_F3", flags=(
            {"category": "dlc_coherence", "severity": "critical",
             "description": "Missing SABL or SABR — cannot compute ruler"},)),
    ]
    flagged_by_hand = _bundle(deep, "20261012_CNT0614_F4",
                              reason="reviewer flagged session 20261012_CNT0614")
    db = FakeDB()

    summary = rr.scan_review_queues(db, tmp_path / "proc")

    assert summary["qc_released"] == 3
    assert not any(b.exists() for b in released)
    assert flagged_by_hand.exists()                     # a person's own flag stays
    kept = review / "_Problematic" / f"_released_qc_holds_{datetime.now():%Y-%m-%d}"
    assert (kept / "20261012_CNT0611_F1" / "20261012_CNT0611_F1_causal_review.json").is_file()
