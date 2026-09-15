"""Return-scan safety: never break a route in flight, never lose provenance,
never re-run a returned video on the wrong pose.

WHY (2026-09-14, first real run of an integrator routing Analyzed videos into
the triage queue):
  * the return scan retired a just-created, still-empty bundle folder to
    _Problematic while the router was copying files into it -- every remaining
    move failed. Empty folders are now residue only once they are old, and a
    future mtime (share clock ahead of this machine) counts as young;
  * bundles are now assembled in <queue>/.incoming/<stem> -- never a bundle to
    the scan, and a stuck one is named to a person, never moved;
  * the queue-metadata match was a suffix, so {stem}_processing_manifest.json
    (the provenance record) was deleted from every returned bundle;
  * dlc_output_path was "whichever .h5 moved last", so a bundle holding two
    pose models re-ran on the wrong one.
"""
import json
import logging
import os
import time

import mousereach.pipeline.manifest as pm
import mousereach.watcher.review_gate as rg
import mousereach.watcher.review_return as rr

OLD_S = 3 * rr.EMPTY_DIR_GRACE_SECONDS


class FakeDB:
    def __init__(self, state="triage"):
        self._state = state
        self.state_writes = []

    def get_video(self, vid):
        return {"video_id": vid, "state": self._state}

    def update_state(self, vid, state, **kw):
        self.state_writes.append((vid, state, kw))

    def force_state(self, vid, state, reason=None, **kw):
        self.state_writes.append((vid, state, kw))


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _wire(monkeypatch, tmp_path):
    triage = tmp_path / "Review" / "triage"
    deep = tmp_path / "Review" / "deep_review"
    triage.mkdir(parents=True)
    deep.mkdir(parents=True)
    monkeypatch.setattr(rr.Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(rr.Paths, "DEEP_REVIEW", deep)
    monkeypatch.setattr(rr.Paths, "REVIEW_ROOT", tmp_path / "Review")
    monkeypatch.setattr(rr.Paths, "ANALYZED_OUTPUT", None, raising=False)
    monkeypatch.setattr(rr.Paths, "DLC_STAGING", None, raising=False)
    monkeypatch.setattr(rr.time, "sleep", lambda s: None)
    monkeypatch.setattr(rr, "_WARNED_INCOMING", set())
    return triage, deep


def _age(path, seconds):
    """Set ``path``'s mtime ``seconds`` in the past (negative = future)."""
    t = time.time() - seconds
    os.utime(path, (t, t))


def _no_routes(monkeypatch):
    routed = []
    monkeypatch.setattr(rg, "route_to_queue", lambda *a, **k: routed.append(a))
    return routed


# --- (a) empty-dir grace ------------------------------------------------------

def test_young_empty_dir_is_not_retired(tmp_path, monkeypatch):
    triage, _ = _wire(monkeypatch, tmp_path)
    routed = _no_routes(monkeypatch)
    stem = "20250101_CNT0101_P1"
    (triage / stem).mkdir()                    # a router just created it

    summary = rr.scan_review_queues(FakeDB(), tmp_path / "proc")

    assert (triage / stem).is_dir()
    assert not summary.get("empty_retired")
    assert not routed
    assert not (tmp_path / "Review" / "_Problematic").exists()


def test_old_empty_dir_is_still_retired(tmp_path, monkeypatch):
    triage, _ = _wire(monkeypatch, tmp_path)
    routed = _no_routes(monkeypatch)
    stem = "20250101_CNT0102_P1"
    (triage / stem).mkdir()
    _age(triage / stem, OLD_S)

    summary = rr.scan_review_queues(FakeDB(), tmp_path / "proc")

    assert summary.get("empty_retired") == 1
    assert not (triage / stem).exists()
    assert not routed


def test_unreadable_age_counts_as_young(tmp_path):
    """A folder whose age cannot be read (a share hiccup during stat) must
    never be taken for old residue and retired under a running route."""
    assert rr._younger_than_grace(tmp_path / "does_not_exist")


def test_future_mtime_empty_dir_is_not_retired(tmp_path, monkeypatch):
    """The share's clock can run ahead of this machine's: a folder stamped an
    hour in the future is treated as brand new, not as ancient residue."""
    triage, _ = _wire(monkeypatch, tmp_path)
    _no_routes(monkeypatch)
    stem = "20250101_CNT0103_P1"
    (triage / stem).mkdir()
    _age(triage / stem, -3600)

    summary = rr.scan_review_queues(FakeDB(), tmp_path / "proc")

    assert (triage / stem).is_dir()
    assert not summary.get("empty_retired")


# --- (b) .incoming staging folders ---------------------------------------------

def test_incoming_is_never_treated_as_a_bundle(tmp_path, monkeypatch):
    triage, deep = _wire(monkeypatch, tmp_path)
    routed = _no_routes(monkeypatch)
    returned = []
    monkeypatch.setattr(rr, "_return_to_processing",
                        lambda *a, **k: returned.append(a) or True)
    stem = "20250101_CNT0104_P1"
    # Half-built triage bundle that WOULD divert if read (seg failed), and a
    # half-built deep bundle that WOULD return if read (clear marker present).
    t_build = triage / rr.INCOMING_DIR_NAME / stem
    t_build.mkdir(parents=True)
    (t_build / f"{stem}_segments.json").write_text(
        json.dumps({"overall_confidence": 0.0, "boundaries": []}),
        encoding="utf-8")
    d_build = deep / rr.INCOMING_DIR_NAME / stem
    d_build.mkdir(parents=True)
    (d_build / f"{stem}_deep_review_cleared.json").write_text(
        "{}", encoding="utf-8")
    empty_build = triage / rr.INCOMING_DIR_NAME / "20250101_CNT0105_P1"
    empty_build.mkdir()
    for p in (t_build, d_build, empty_build,
              triage / rr.INCOMING_DIR_NAME, deep / rr.INCOMING_DIR_NAME):
        _age(p, OLD_S)

    summary = rr.scan_review_queues(FakeDB(), tmp_path / "proc")

    assert not routed and not returned
    assert not summary.get("empty_retired")
    assert (t_build / f"{stem}_segments.json").is_file()
    assert (d_build / f"{stem}_deep_review_cleared.json").is_file()
    assert empty_build.is_dir()
    assert not (tmp_path / "Review" / "_Problematic").exists()


def test_stale_incoming_warns_once_and_is_left_alone(tmp_path, monkeypatch):
    triage, _ = _wire(monkeypatch, tmp_path)
    _no_routes(monkeypatch)
    stale = "20250101_CNT0106_P1"
    fresh = "20250101_CNT0107_P1"
    stale_dir = triage / rr.INCOMING_DIR_NAME / stale
    stale_dir.mkdir(parents=True)
    (stale_dir / f"{stale}.mp4").write_bytes(b"x")
    _age(stale_dir, OLD_S)
    (triage / rr.INCOMING_DIR_NAME / fresh).mkdir()   # a route in progress

    h = _Records()
    rr.logger.addHandler(h)
    try:
        rr.scan_review_queues(FakeDB(), tmp_path / "proc")
        rr.scan_review_queues(FakeDB(), tmp_path / "proc")
    finally:
        rr.logger.removeHandler(h)

    warnings = [r.getMessage() for r in h.records if r.levelno == logging.WARNING]
    assert len([m for m in warnings if stale in m]) == 1
    assert "a person should check" in [m for m in warnings if stale in m][0]
    assert not [m for m in warnings if fresh in m]
    assert (stale_dir / f"{stale}.mp4").is_file()      # never moved or deleted


# --- (c) queue metadata is matched by exact name --------------------------------

def test_is_queue_metadata_exact_names():
    stem = "20250101_CNT0108_P1"
    assert rr._is_queue_metadata(f"{stem}_manifest.json", stem)
    assert rr._is_queue_metadata(f"{stem}_routing.json", stem)
    assert rr._is_queue_metadata("manifest.json", stem)
    assert not rr._is_queue_metadata(f"{stem}_processing_manifest.json", stem)
    assert not rr._is_queue_metadata(f"{stem}_crop_manifest.json", stem)


def _pose_name(stem, model, snapshot):
    return f"{stem}DLC_resnet50_{model}shuffle1_{snapshot}.h5"


def test_processing_manifest_survives_a_return(tmp_path, monkeypatch):
    triage, _ = _wire(monkeypatch, tmp_path)
    stem = "20250101_CNT0109_P1"
    bundle = triage / stem
    bundle.mkdir()
    keep = [f"{stem}.mp4", _pose_name(stem, "ModelA", 100),
            f"{stem}_segments.json", f"{stem}_processing_manifest.json"]
    drop = [f"{stem}_manifest.json", f"{stem}_routing.json", "manifest.json"]
    for name in keep + drop:
        (bundle / name).write_text("{}", encoding="utf-8")
    proc = tmp_path / "proc"

    db = FakeDB()
    assert rr._return_to_processing(bundle, stem, proc, db, "triage_cleared")

    for name in keep:
        assert (proc / name).is_file(), name
    for name in drop:
        assert not (proc / name).exists(), name
    assert not bundle.exists()


# --- (d) dlc_output_path records the SELECTED pose -----------------------------

def test_declared_pose_is_recorded_not_the_last_moved(tmp_path, monkeypatch):
    triage, _ = _wire(monkeypatch, tmp_path)
    stem = "20250101_CNT0110_P1"
    bundle = triage / stem
    bundle.mkdir()
    (bundle / f"{stem}.mp4").write_bytes(b"x")
    declared = _pose_name(stem, "ModelA", 200)
    old = _pose_name(stem, "ModelZ", 100)       # lists after the declared one
    (bundle / declared).write_bytes(b"new")
    (bundle / old).write_bytes(b"old")
    _age(bundle / declared, 600)                 # old pose is also the newest
    monkeypatch.setattr(pm, "declared_dlc_scorer",
                        lambda *a, **k: "DLC_resnet50_ModelAshuffle1_200")
    proc = tmp_path / "proc"

    db = FakeDB()
    assert rr._return_to_processing(bundle, stem, proc, db, "triage_cleared")

    assert (proc / declared).is_file() and (proc / old).is_file()
    assert len(db.state_writes) == 1
    _, state, kw = db.state_writes[0]
    assert state == "processing"
    assert kw["dlc_output_path"] == str(proc / declared)


def test_return_refused_while_build_folder_still_holds_files(tmp_path, monkeypatch):
    """A fallback publish can leave a locked file (typically the mp4) in
    <queue>/.incoming/<stem> beside the published bundle, where nothing reads
    it. Returning then would re-run the video without it, so the return waits
    for a person and touches nothing."""
    triage, _ = _wire(monkeypatch, tmp_path)
    stem = "20250101_CNT0112_P1"
    bundle = triage / stem
    bundle.mkdir()
    (bundle / _pose_name(stem, "ModelA", 100)).write_bytes(b"pose")
    (bundle / f"{stem}_segments.json").write_text("{}", encoding="utf-8")
    leftover = triage / rr.INCOMING_DIR_NAME / stem
    leftover.mkdir(parents=True)
    (leftover / f"{stem}.mp4").write_bytes(b"x")
    proc = tmp_path / "proc"

    db = FakeDB()
    assert rr._return_to_processing(bundle, stem, proc, db, "triage_cleared") is False

    assert db.state_writes == []
    assert (bundle / f"{stem}_segments.json").is_file()
    assert (leftover / f"{stem}.mp4").is_file()
    assert not (proc / f"{stem}_segments.json").exists()


def test_pose_outside_the_bundle_is_recorded_in_place(tmp_path, monkeypatch):
    """A legacy bundle names its pose in Analyzed; that pose is used where it
    is, so dlc_output_path stays the resolved source path."""
    triage, _ = _wire(monkeypatch, tmp_path)
    stem = "20250101_CNT0111_P1"
    analyzed = tmp_path / "Analyzed"
    analyzed.mkdir()
    pose = analyzed / _pose_name(stem, "ModelA", 200)
    pose.write_bytes(b"pose")
    bundle = triage / stem
    bundle.mkdir()
    (bundle / f"{stem}.mp4").write_bytes(b"x")
    (bundle / f"{stem}_segments.json").write_text("{}", encoding="utf-8")
    (bundle / f"{stem}_manifest.json").write_text(
        json.dumps({"canonical_dlc_h5_path": str(pose)}), encoding="utf-8")
    proc = tmp_path / "proc"

    db = FakeDB()
    assert rr._return_to_processing(bundle, stem, proc, db, "triage_cleared")

    _, _, kw = db.state_writes[0]
    assert kw["dlc_output_path"] == str(pose)
    assert pose.is_file()                        # read in place, not moved
