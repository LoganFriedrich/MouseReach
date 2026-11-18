"""Recording issues: a failed recording is labeled and set aside, never cut.

WHY: a recording that stopped after a fraction of a second (a real case: 74 KB,
4 frames, 0.08 s) used to pass the intake -- nothing checked its length -- and
would have been cut into eight empty single-animal videos for a reviewer.

Everything lives under tmp_path with config.Paths pointed at it; the probe is
injected except in the tests that make a real (tiny) video with ffmpeg. All
ids are invented.
"""
import json
import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import mousereach.watcher.orchestrator as orch
from mousereach.watcher import recording_issue as ri
from mousereach.watcher import recording_outbox as ro
from mousereach.watcher.db import WatcherDB
from mousereach.watcher.orchestrator import DLCOrchestrator
from mousereach.watcher.state import WatcherStateManager

pytestmark = pytest.mark.real_recording_check

HOST = "NODE-A"
COLLAGE = ("20250101_TST0101,TST0102,TST0001,TST0002,TST0003,TST0004,"
           "TST0005,TST0006_P2.mkv")
KIDS = ["20250101_TST0101_P2", "20250101_TST0102_P2"]

LAYOUT = {
    "NAS_ROOT": "share",
    "MULTI_ANIMAL_SOURCE": "share/Unanalyzed/Multi-Animal",
    "SINGLE_ANIMAL_OUTPUT": "share/Unanalyzed/Single_Animal",
    "DLC_STAGING": "share/Processing/Posed",
    "REVIEW_ROOT": "share/Processing/Review",
    "TRIAGE_REVIEW": "share/Processing/Review/triage",
    "DEEP_REVIEW": "share/Processing/Review/deep_review",
    "ANALYZED_OUTPUT": "share/Analyzed",
    "PROCESSING_ROOT": "node",
    "DLC_QUEUE": "node/DLC_Queue",
    "PROCESSING": "node/Processing",
}

FAILED_PROBE = {"readable": True, "duration": 0.08, "fps": 60.0, "frames": 4,
                "frames_counted": False, "error": None}
GOOD_PROBE = {"readable": True, "duration": 616.0, "fps": 60.0, "frames": 36960,
              "frames_counted": False, "error": None}
UNREADABLE_PROBE = {"readable": False, "duration": None, "fps": None, "frames": None,
                    "frames_counted": False,
                    "error": "Invalid data found when processing input"}
DECODES = lambda path, at_end: (True, "")


def settings():
    return ri.CheckSettings()          # the defaults


# ------------------------------------------------------------------ the check

def test_a_four_frame_recording_is_a_recording_issue_in_plain_words(tmp_path):
    f = tmp_path / COLLAGE
    f.write_bytes(b"x" * 74386)
    v = ri.assess(f, settings(), probe_fn=lambda p: FAILED_PROBE, decode_fn=DECODES)
    assert v.outcome == "issue"
    assert v.reason.startswith("recording issue: 4 frames, 0.08 s -- the recording failed")
    assert v.measurements["frames"] == 4 and v.measurements["bytes"] == 74386


def test_a_normal_recording_passes(tmp_path):
    f = tmp_path / COLLAGE
    f.write_bytes(b"x" * 1000)
    big = dict(GOOD_PROBE)
    v = ri.assess(f, ri.CheckSettings(min_kb_per_second=0.0),
                  probe_fn=lambda p: big, decode_fn=DECODES)
    assert v.outcome == "ok" and v.reason == ""
    assert v.measurements["decodes_at_start"] and v.measurements["decodes_at_end"]


def test_an_unreadable_recording_is_a_recording_issue(tmp_path):
    f = tmp_path / COLLAGE
    f.write_bytes(b"")
    v = ri.assess(f, settings(), probe_fn=lambda p: UNREADABLE_PROBE, decode_fn=DECODES)
    assert v.outcome == "issue"
    assert "cannot be read as a video (0 bytes" in v.reason


def test_too_few_frames_too_small_and_undecodable_ends_are_issues(tmp_path):
    f = tmp_path / COLLAGE
    f.write_bytes(b"x" * 2000)
    few = dict(GOOD_PROBE, frames=100, frames_counted=True)
    v = ri.assess(f, ri.CheckSettings(min_kb_per_second=0.0), probe_fn=lambda p: few,
                  decode_fn=DECODES)
    assert v.is_issue and "100 frames" in v.reason and "recording_min_frames" in v.reason
    v = ri.assess(f, settings(), probe_fn=lambda p: GOOD_PROBE, decode_fn=DECODES)
    assert v.is_issue and "KB per second" in v.reason      # 2 KB for ten minutes
    v = ri.assess(f, ri.CheckSettings(min_kb_per_second=0.0),
                  probe_fn=lambda p: GOOD_PROBE,
                  decode_fn=lambda p, at_end: (not at_end, "" if not at_end else "EOF"))
    assert v.is_issue and "end of the recording cannot be decoded" in v.reason


def test_a_check_that_cannot_run_is_never_called_an_issue(tmp_path):
    f = tmp_path / COLLAGE
    f.write_bytes(b"x")

    def no_tool(path):
        raise ri.CannotCheck("neither ffprobe nor ffmpeg is installed on this machine")
    v = ri.assess(f, settings(), probe_fn=no_tool, decode_fn=DECODES)
    assert v.outcome == "cannot_check" and not v.is_issue
    v = ri.assess(tmp_path / "gone.mkv", settings())
    assert v.outcome == "cannot_check"


def _ffmpeg():
    exe = ri.find_tool("ffmpeg")
    if not exe:
        pytest.skip("no ffmpeg on this machine")
    return exe


def test_real_tiny_and_real_garbage_files_are_issues(tmp_path):
    """The real tools, on a real 4-frame video and on bytes that are no video."""
    ff = _ffmpeg()
    tiny = tmp_path / "tiny.mkv"
    subprocess.run([ff, "-hide_banner", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc=size=64x48:rate=60", "-frames:v", "4", str(tiny)], check=True)
    v = ri.assess(tiny, settings())
    assert v.outcome == "issue", v
    assert "4 frames" in v.reason or "0.0" in v.reason
    junk = tmp_path / "junk.mkv"
    junk.write_bytes(b"\x00" * 4096)
    v = ri.assess(junk, settings())
    assert v.outcome == "issue" and "cannot be read as a video" in v.reason


def test_real_recording_long_enough_passes(tmp_path):
    ff = _ffmpeg()
    vid = tmp_path / "ok.mkv"
    subprocess.run([ff, "-hide_banner", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc=size=64x48:rate=10", "-t", "8", str(vid)], check=True)
    v = ri.assess(vid, ri.CheckSettings(min_seconds=5, min_frames=50, min_kb_per_second=0))
    assert v.outcome == "ok", v


# ------------------------------------------------------------------ the intake

class FakeCoordinator:
    def __init__(self):
        self.updates, self.released = [], []

    def try_claim_collage(self, filename, hostname):
        return True

    def release_collage_claim(self, filename, hostname):
        self.released.append(filename)
        return True

    def update_collage_state(self, filename, state, **kw):
        self.updates.append((filename, state))
        return True

    def get_collage_claim(self, filename):
        return None


@pytest.fixture
def env(tmp_path, monkeypatch):
    dirs = {}
    for name, sub in LAYOUT.items():
        d = tmp_path / sub
        d.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(orch.Paths, name, d, raising=False)
        dirs[name.lower()] = d
    monkeypatch.setattr(orch, "require_processing_root", lambda: dirs["processing_root"])
    o = object.__new__(DLCOrchestrator)
    o.db = WatcherDB(db_path=tmp_path / "watcher.db")
    o.hostname = HOST
    o.working_dir = dirs["processing_root"] / "watcher_working"
    o.working_dir.mkdir()
    o.config = SimpleNamespace(also_process=False, dlc_gpu_device=0, dlc_config_path=None,
                               max_retries=3, work_priority=None, pause_while_running=None)
    o.coordinator = FakeCoordinator()
    o._sync_to_connectome = lambda *a, **k: None
    o._backup_local_db = lambda: None
    o._hold_collage_claim = lambda name: True
    cuts = []
    import mousereach.video_prep.core.cropper as cropper
    monkeypatch.setattr(cropper, "crop_collage",
                        lambda **kw: cuts.append(kw) or [])
    return SimpleNamespace(o=o, db=o.db, tmp=tmp_path, cuts=cuts,
                           problematic=dirs["nas_root"] / "Processing" / "Review" / "_Problematic",
                           **dirs)


def _stable_collage(env, data=b"x" * 74386):
    src = env.multi_animal_source / COLLAGE
    src.write_bytes(data)
    env.db.register_collage(COLLAGE, str(src))
    for s in ("validated", "stable"):
        env.db.update_collage_state(COLLAGE, s)
    return {"type": "collage", "id": COLLAGE, "data": env.db.get_collage(COLLAGE)}


def _probe_says(monkeypatch, result):
    real = ri.assess
    monkeypatch.setattr(ri, "assess", lambda path, s=None, **kw: real(
        path, s, probe_fn=lambda p: result, decode_fn=DECODES))


def test_short_collage_is_labeled_and_moved_never_cut(env, monkeypatch):
    _probe_says(monkeypatch, FAILED_PROBE)
    work = _stable_collage(env)
    assert env.o._process_collage(work) is True
    assert env.cuts == [], "a failed recording must never be cut"
    assert not (env.multi_animal_source / COLLAGE).exists()
    assert (env.problematic / COLLAGE).read_bytes() == b"x" * 74386, "never deleted"
    mark = json.loads((env.problematic / (COLLAGE + ri.REASON_SUFFIX)).read_text())
    assert mark["reason"].startswith("recording issue: 4 frames, 0.08 s")
    assert mark["says"].startswith("recording failed: " + COLLAGE)
    assert mark["flagged_by"].startswith("intake check on ")
    assert mark["sessions"] == KIDS
    assert mark["measurements"]["frames"] == 4 and mark["flagged_at"]
    assert env.db.get_collage(COLLAGE)["state"] == "recording_issue"
    assert (COLLAGE, "recording_issue") in env.o.coordinator.updates
    # Terminal: nothing offers it for work again.
    assert env.db.get_collages_in_state("stable") == []


def test_unreadable_collage_is_labeled_and_moved(env, monkeypatch):
    _probe_says(monkeypatch, UNREADABLE_PROBE)
    work = _stable_collage(env, b"")
    assert env.o._process_collage(work) is True
    assert env.cuts == []
    assert (env.problematic / COLLAGE).exists()
    assert env.db.get_collage(COLLAGE)["state"] == "recording_issue"


def test_normal_collage_goes_on_to_be_cut(env, monkeypatch):
    _probe_says(monkeypatch, dict(GOOD_PROBE))
    work = _stable_collage(env, b"x" * 3_200_000)     # > 5 KB per second of 616 s
    assert env.o._set_aside_if_recording_issue(COLLAGE, work["data"]) is False
    assert (env.multi_animal_source / COLLAGE).exists()
    assert not env.problematic.exists() or not list(env.problematic.iterdir())


def test_check_that_cannot_run_cuts_as_usual(env, monkeypatch):
    def no_tool(path):
        raise ri.CannotCheck("neither ffprobe nor ffmpeg is installed on this machine")
    real = ri.assess
    monkeypatch.setattr(ri, "assess", lambda path, s=None, **kw: real(path, s, probe_fn=no_tool))
    work = _stable_collage(env)
    assert env.o._set_aside_if_recording_issue(COLLAGE, work["data"]) is False
    assert (env.multi_animal_source / COLLAGE).exists()


# ------------------------------------------------------------ the manual mark

def test_manual_mark_is_honored_by_a_node_with_a_stable_row(env, monkeypatch, caplog):
    """A person marks the recording on ANY machine. Another node, whose own
    watcher.db holds the collage as 'stable', must record it as a recording
    issue -- and never log a crop failure for the file that vanished."""
    work = _stable_collage(env)
    monkeypatch.setattr(ri, "_claim_refusal", lambda name: None)
    monkeypatch.setattr("mousereach.watcher.cli._resolve_db_path",
                        lambda: env.tmp / "other_machine.db")
    monkeypatch.setattr(ri, "assess", lambda path, s=None, **kw: ri.Verdict("issue", "r", {}))
    assert ri.main([COLLAGE, "--reason", "OBS stopped after a second", "--by", "tester"]) == 0
    assert not (env.multi_animal_source / COLLAGE).exists()
    mark = json.loads((env.problematic / (COLLAGE + ri.REASON_SUFFIX)).read_text())
    assert mark["reason"] == "OBS stopped after a second"
    assert mark["flagged_by"].startswith("person: tester")

    caplog.set_level(logging.DEBUG)
    state = WatcherStateManager(env.db, SimpleNamespace(stability_wait_seconds=0, max_retries=3))
    assert state.apply_recording_issue_marks(env.multi_animal_source) == [COLLAGE]
    assert env.db.get_collage(COLLAGE)["state"] == "recording_issue"
    assert env.db.get_collages_in_state("stable") == []
    # Even work picked before the scan (a stale dict) is not a crop failure.
    assert env.o._process_collage(work) is True
    assert env.cuts == []
    assert env.db.get_collage(COLLAGE)["state"] == "recording_issue"
    text = caplog.text.lower()
    assert "crop failed" not in text and "collage not found" not in text
    assert "does not exist" not in text


def test_manual_mark_needs_a_reason_and_refuses_a_cut_collage(env, monkeypatch, capsys):
    _stable_collage(env)
    assert ri.main([COLLAGE]) == 1
    assert (env.multi_animal_source / COLLAGE).exists()
    monkeypatch.setattr(ri, "_claim_refusal", lambda name: "already cut by NODE-B")
    assert ri.main([COLLAGE, "--reason", "x"]) == 1
    assert (env.multi_animal_source / COLLAGE).exists()


def test_an_interrupted_move_is_finished_by_the_next_scan(env):
    _stable_collage(env)
    env.problematic.mkdir(parents=True)
    (env.problematic / (COLLAGE + ri.REASON_SUFFIX)).write_text(
        json.dumps({"collage": COLLAGE, "reason": "recording issue: 4 frames, 0.08 s"}))
    state = WatcherStateManager(env.db, SimpleNamespace(stability_wait_seconds=0, max_retries=3))
    state.apply_recording_issue_marks(env.multi_animal_source)
    assert (env.problematic / COLLAGE).exists()
    assert not (env.multi_animal_source / COLLAGE).exists()
    assert env.db.get_collage(COLLAGE)["state"] == "recording_issue"


def test_a_released_recording_is_checked_again(env):
    """A person moved it back and took the reason file away."""
    _stable_collage(env)
    env.db.update_collage_state(COLLAGE, "recording_issue")
    state = WatcherStateManager(env.db, SimpleNamespace(stability_wait_seconds=0, max_retries=3))
    assert COLLAGE in state.discover_new_collages(env.multi_animal_source)
    assert env.db.get_collage(COLLAGE)["state"] == "validated"


def test_a_name_clash_in_problematic_moves_nothing(env, monkeypatch):
    _probe_says(monkeypatch, FAILED_PROBE)
    work = _stable_collage(env)
    env.problematic.mkdir(parents=True)
    (env.problematic / COLLAGE).write_bytes(b"another file")
    assert env.o._process_collage(work) is True
    assert env.cuts == []
    assert (env.multi_animal_source / COLLAGE).exists(), "never overwritten, never renamed"
    assert (env.problematic / COLLAGE).read_bytes() == b"another file"


# --------------------------------------------------------- outbox and intake agree

def test_outbox_and_intake_give_the_same_verdict(env, monkeypatch, tmp_path):
    """One check, one set of settings: the recording PC labels a file exactly
    as the intake will."""
    from mousereach.config import WatcherConfig
    cfg = WatcherConfig({"recording_outbox": str(tmp_path / "out"),
                         "recording_min_minutes": 4, "recording_min_frames": 500,
                         "recording_min_kb_per_second": 2})
    out = ro.OutboxSettings.from_config(cfg)
    intake = ri.CheckSettings.from_config(cfg)
    assert out.check == intake
    _probe_says(monkeypatch, FAILED_PROBE)
    f = tmp_path / COLLAGE
    f.write_bytes(b"x" * 74386)
    sender = ro.RecordingSender(out, env.nas_root)
    assert sender._assess(f).reason == ri.assess(f, intake).reason
    assert sender._assess(f).reason.startswith("recording issue: 4 frames, 0.08 s")


# ----------------------------------------------------------- where it shows

def test_census_counts_the_sessions_as_recording_failed_not_missing(env):
    from mousereach.census import runner
    env.problematic.mkdir(parents=True)
    (env.problematic / COLLAGE).write_bytes(b"x")
    (env.problematic / (COLLAGE + ri.REASON_SUFFIX)).write_text(json.dumps({
        "collage": COLLAGE, "reason": "recording issue: 4 frames, 0.08 s -- the recording failed",
        "flagged_at": "2025-01-01T10:00:00", "sessions": KIDS}))
    c = runner.run_census()
    for k in KIDS:
        assert c["sessions"][k]["element"] == "recording_issue"
        assert c["sessions"][k]["recording_issue"]["collage"] == COLLAGE
    assert c["recording_issues"][0]["collage"] == COLLAGE
    assert "4 frames" in c["recording_issues"][0]["reason"]
    assert c["by_element"]["recording_issue"] == 2


def test_status_lists_recording_issues(env, capsys):
    from mousereach.watcher import cli
    env.problematic.mkdir(parents=True)
    (env.problematic / (COLLAGE + ri.REASON_SUFFIX)).write_text(json.dumps({
        "collage": COLLAGE, "reason": "recording issue: 4 frames, 0.08 s",
        "says": "recording failed: %s, recording issue: 4 frames, 0.08 s" % COLLAGE}))
    cli._print_held_and_failed(env.db)
    out = capsys.readouterr().out
    assert "RECORDING ISSUES (1)" in out and "recording failed: " + COLLAGE in out
    items = ri.attention_items(env.nas_root)
    assert items[0]["kind"] == "recording_issue" and items[0]["needs_person"] is False
