"""Sending named recordings from a recording PC to the shared intake.

Every test runs against temporary folders: an "outbox" (the recording folder)
and a fake shared drive with the intake. Time, the duration probe and the check
for a running recording program are injected, so nothing here depends on the
machine, a real share or a recording program.
"""
import json
import os
import types
from datetime import datetime
from pathlib import Path

import pytest

from mousereach.watcher import recording_outbox as ro
from mousereach.watcher import collage_index as ci
from mousereach.watcher import recording_issue as ri
from mousereach.watcher import node_status

NAMED = "20240102_TST0101,TST0102,TST0103,TST0104,TST0105,TST0106,TST0107,TST0108_P1.mkv"
NAMED2 = "20240102_TST0101,TST0102,TST0103,TST0104,TST0105,TST0106,TST0107,TST0108_P2.mkv"
RAW = "2024-01-02 09-30-00.mkv"
SATURDAY_NOON = datetime(2024, 1, 6, 12, 0)       # outside Mon-Fri
WEDNESDAY_10 = datetime(2024, 1, 3, 10, 0)        # inside Mon-Fri 08:00-18:00
WEDNESDAY_19 = datetime(2024, 1, 3, 19, 0)        # outside the hours
CONTENT = b"recording-bytes-" * 4096             # 64 KiB


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, minutes):
        self.t += minutes * 60.0


@pytest.fixture
def env(tmp_path):
    outbox = tmp_path / "outbox"
    outbox.mkdir()
    nas = tmp_path / "nas"
    (nas / "Unanalyzed" / "Multi-Animal").mkdir(parents=True)
    return types.SimpleNamespace(tmp=tmp_path, outbox=outbox, nas=nas,
                                 intake=nas / "Unanalyzed" / "Multi-Animal")


OK = ri.Verdict("ok")


def issue(reason="recording issue: 4 frames, 0.08 s -- the recording failed"):
    return ri.Verdict("issue", reason)


def cannot(reason="neither ffprobe nor ffmpeg is installed on this machine"):
    return ri.Verdict("cannot_check", reason)


def make_sender(env, now=SATURDAY_NOON, clock=None, assess=None, open_check=None,
                recorder_running=None, should_stop=None, chunk_size=8192, **settings):
    s = ro.OutboxSettings(outbox=env.outbox, stable_minutes=5, warn_after_failures=3,
                          pass_minutes=30,
                          recording_hours="08:00-18:00", recording_days="Mon-Fri")
    for k, v in settings.items():
        setattr(s, k, v)
    return ro.RecordingSender(
        s, env.nas, state_path=env.tmp / "state.json",
        transfer_log=env.tmp / "logs" / ro.TRANSFER_LOG_NAME,
        clock=clock or Clock(), local_now=(lambda: now),
        assess=assess or (lambda p: OK),
        open_check=open_check or (lambda p: False),
        recorder_running=recorder_running or (lambda: False),
        should_stop=should_stop or (lambda: False),
        chunk_size=chunk_size)


def two_passes(sender, clock):
    """First look records the size; the second, after the stable time, may send."""
    sender.run_pass()
    clock.advance(6)
    return sender.run_pass()


def put(folder, name, data=CONTENT):
    p = Path(folder) / name
    p.write_bytes(data)
    return p


def write_index(env):
    """What the processing server does every 10 minutes."""
    (env.nas / "Processing").mkdir(parents=True, exist_ok=True)
    return ci.write(env.nas)


def log_events(env):
    path = env.tmp / "logs" / ro.TRANSFER_LOG_NAME
    if not path.exists():
        return []
    return [json.loads(l)["event"] for l in path.read_text().splitlines() if l.strip()]


# --------------------------------------------------------------------------- names

def test_only_a_valid_session_name_is_sent_and_a_bad_name_is_held_with_reason(env):
    bad = put(env.outbox, "20240102_TST0101_P1.mkv")
    good = put(env.outbox, NAMED)
    clock = Clock()
    rep = two_passes(make_sender(env, clock=clock), clock)
    assert bad.exists(), "a file whose name fails the check must never move"
    assert not (env.intake / bad.name).exists()
    held = {h["name"]: h["reason"] for h in rep["held"]}
    assert bad.name in held and "not a valid session name" in held[bad.name]
    assert (env.intake / NAMED).read_bytes() == CONTENT
    assert not good.exists()


def test_raw_recorder_name_is_never_touched(env):
    raw = put(env.outbox, RAW)
    clock = Clock()
    rep = two_passes(make_sender(env, clock=clock), clock)
    assert raw.exists() and raw.read_bytes() == CONTENT
    assert list(env.intake.iterdir()) == []
    assert rep["waiting_for_name"] == [RAW]
    assert rep["held"] == []


# ------------------------------------------------------------ still being written

def test_growing_file_is_not_sent_until_unchanged_for_the_stable_time(env):
    p = put(env.outbox, NAMED)
    clock = Clock()
    sender = make_sender(env, clock=clock)
    rep = sender.run_pass()
    assert any("may still be written" in w["reason"] for w in rep["waiting"])
    clock.advance(6)
    with open(p, "ab") as f:
        f.write(b"more")
    rep = sender.run_pass()
    assert p.exists() and not (env.intake / NAMED).exists()
    clock.advance(3)
    rep = sender.run_pass()
    assert p.exists(), "only 3 of 5 minutes unchanged"
    clock.advance(3)
    rep = sender.run_pass()
    assert not p.exists()
    assert (env.intake / NAMED).read_bytes() == CONTENT + b"more"


@pytest.mark.skipif(os.name != "nt", reason="exclusive-open test is Windows only")
def test_file_open_for_writing_is_not_sent(env):
    p = put(env.outbox, NAMED)
    clock = Clock()
    sender = make_sender(env, clock=clock, open_check=ro.is_open_for_writing)
    with open(p, "ab") as writer:
        writer.write(b"")
        rep = two_passes(sender, clock)
        assert any(w["reason"] == "open in another program" for w in rep["waiting"])
        assert not (env.intake / NAMED).exists()
    clock.advance(1)
    sender.run_pass()
    assert (env.intake / NAMED).exists() and not p.exists()


def test_newest_file_is_left_while_the_recording_program_runs(env):
    older = put(env.outbox, NAMED)
    os.utime(older, (1_000_000, 1_000_000))
    newest = put(env.outbox, NAMED2)
    clock = Clock()
    rep = two_passes(make_sender(env, clock=clock, recorder_running=lambda: True), clock)
    assert newest.exists() and not (env.intake / NAMED2).exists()
    assert any("current file" in w["reason"] for w in rep["waiting"])
    assert not older.exists() and (env.intake / NAMED).exists()


# --------------------------------------------------------------- empty or broken

def test_failed_recording_is_sent_and_labeled_not_held(env):
    """A recording that only documents a mistake still goes to the shared
    drive; the intake sets it aside there as a recording issue."""
    p = put(env.outbox, NAMED, b"")
    clock = Clock()
    rep = two_passes(make_sender(env, clock=clock, assess=lambda path: issue()), clock)
    assert not p.exists() and (env.intake / NAMED).exists()
    assert rep["held"] == []
    assert [r["name"] for r in rep["recording_issues"]] == [NAMED]
    assert "0.08 s" in rep["recording_issues"][0]["reason"]
    assert "recording issue" in rep["sent_this_pass"][0]["how"]
    assert "recording_issue" in log_events(env)


def test_check_that_cannot_run_holds_the_file_in_red(env):
    p = put(env.outbox, NAMED)
    q = put(env.outbox, NAMED2)
    clock = Clock()
    assess = lambda path: cannot() if path.name == NAMED else OK
    rep = two_passes(make_sender(env, clock=clock, assess=assess), clock)
    assert p.exists() and not (env.intake / NAMED).exists()
    assert (env.intake / NAMED2).exists()
    reasons = {h["name"]: h["reason"] for h in rep["held"]}
    assert "could not run" in reasons[NAMED] and "ffmpeg" in reasons[NAMED]


def test_failed_recording_already_set_aside_is_not_sent_again(env):
    """Once the intake set it aside, the recording PC finds it there (same
    content) and only deletes its local copy."""
    folder = env.nas / "Processing" / "Review" / "_Problematic"
    folder.mkdir(parents=True)
    put(folder, NAMED, b"tiny")
    p = put(env.outbox, NAMED, b"tiny")
    clock = Clock()
    rep = two_passes(make_sender(env, clock=clock, assess=lambda path: issue()), clock)
    assert not p.exists()
    assert not (env.intake / NAMED).exists(), "never sent a second time"
    assert rep["sent_this_pass"][0]["how"].startswith("already on the shared drive")


# ------------------------------------------------------------------ recording time

def test_nothing_is_sent_during_recording_hours(env):
    p = put(env.outbox, NAMED)
    clock = Clock()
    rep = two_passes(make_sender(env, now=WEDNESDAY_10, clock=clock), clock)
    assert p.exists() and not (env.intake / NAMED).exists()
    assert "recording time" in rep["not_sending"]


def test_weekday_after_the_hours_sends(env):
    put(env.outbox, NAMED)
    clock = Clock()
    two_passes(make_sender(env, now=WEDNESDAY_19, clock=clock), clock)
    assert (env.intake / NAMED).exists()


def test_without_recording_hours_or_days_nothing_is_ever_sent(env):
    p = put(env.outbox, NAMED)
    clock = Clock()
    rep = two_passes(make_sender(env, clock=clock, recording_hours=None,
                                 recording_days=None), clock)
    assert p.exists() and not (env.intake / NAMED).exists()
    assert "not set" in rep["not_sending"]


def test_copy_stops_when_recording_time_begins_mid_copy(env):
    p = put(env.outbox, NAMED)
    clock = Clock()
    now = {"t": SATURDAY_NOON}
    sender = make_sender(env, clock=clock)
    sender._local_now = lambda: now["t"]
    sender.run_pass()
    clock.advance(6)
    calls = {"n": 0}
    real = sender._abort_reason

    def abort():
        calls["n"] += 1
        if calls["n"] > 3:          # a few blocks in, recording time starts
            now["t"] = WEDNESDAY_10
        return real()
    sender._abort_reason = abort
    rep = sender.run_pass()
    assert p.exists() and p.read_bytes() == CONTENT
    assert not (env.intake / NAMED).exists()
    assert not (env.intake / (NAMED + ".partial")).exists()
    assert "copy_stopped" in log_events(env)


# --------------------------------------------------------------- the copy itself

def test_verified_copy_then_rename_then_local_original_deleted(env):
    p = put(env.outbox, NAMED)
    clock = Clock()
    rep = two_passes(make_sender(env, clock=clock), clock)
    final = env.intake / NAMED
    assert final.read_bytes() == CONTENT
    assert not (env.intake / (NAMED + ".partial")).exists()
    assert not p.exists()
    assert [s["name"] for s in rep["sent_this_pass"]] == [NAMED]
    assert "sent" in log_events(env)


def test_original_kept_when_verification_fails_and_red_after_n_failures(env, monkeypatch):
    p = put(env.outbox, NAMED)
    clock = Clock()
    sender = make_sender(env, clock=clock)
    real = ro.sha256_file

    def lying_hash(path, abort=None):
        if str(path).endswith(".partial"):
            return "0" * 64        # the copy read back does not match
        return real(path, abort)
    monkeypatch.setattr(ro, "sha256_file", lying_hash)
    rep = two_passes(sender, clock)
    assert p.exists() and p.read_bytes() == CONTENT
    assert not (env.intake / NAMED).exists()
    assert not (env.intake / (NAMED + ".partial")).exists(), "the bad copy is removed"
    assert rep["held"] == [] and any("differs" in w["reason"] for w in rep["waiting"])
    for _ in range(2):
        clock.advance(1)
        rep = sender.run_pass()
    assert p.exists()
    assert [h["name"] for h in rep["held"]] == [NAMED]
    assert "3 times" in rep["held"][0]["reason"]
    # Fixed: the next pass sends it.
    monkeypatch.setattr(ro, "sha256_file", real)
    clock.advance(1)
    sender.run_pass()
    assert not p.exists() and (env.intake / NAMED).read_bytes() == CONTENT


def test_same_name_different_content_is_held_and_nothing_overwritten(env):
    p = put(env.outbox, NAMED)
    other = CONTENT[:-1] + b"X"                    # same size, other content
    put(env.intake, NAMED, other)
    clock = Clock()
    rep = two_passes(make_sender(env, clock=clock), clock)
    assert p.read_bytes() == CONTENT
    assert (env.intake / NAMED).read_bytes() == other
    assert [h["name"] for h in rep["held"]] == [NAMED]
    assert "different file of the same name" in rep["held"][0]["reason"]


def test_different_size_on_share_is_a_conflict_too(env):
    p = put(env.outbox, NAMED)
    put(env.intake, NAMED, b"short")
    clock = Clock()
    rep = two_passes(make_sender(env, clock=clock), clock)
    assert p.exists() and (env.intake / NAMED).read_bytes() == b"short"
    assert [h["name"] for h in rep["held"]] == [NAMED]


def test_already_sent_and_since_filed_is_not_sent_again(env):
    """A collage leaves the intake once its videos are filed; the server's
    collage index says where, the PC opens that one file, and the same
    content means it was sent: only the local copy goes."""
    p = put(env.outbox, NAMED)
    filed = env.nas / "Analyzed" / "ProjA" / "Group1" / "Multi-Animal"
    filed.mkdir(parents=True)
    put(filed, NAMED)
    write_index(env)
    clock = Clock()
    rep = two_passes(make_sender(env, clock=clock), clock)
    assert not p.exists()
    assert not (env.intake / NAMED).exists(), "never copied into the intake again"
    assert rep["sent_this_pass"][0]["how"] == "already on the shared drive"
    assert "deleted_local_already_on_share" in log_events(env)


# -------------------------------------------------------------------- crash safety

def test_crash_mid_copy_leaves_partial_that_the_next_passes_clean_up(env):
    p = put(env.outbox, NAMED)
    put(env.intake, NAMED + ".partial", CONTENT[:1000])   # what a killed copy leaves
    clock = Clock()
    sender = make_sender(env, clock=clock)
    rep = two_passes(sender, clock)
    assert p.exists(), "first sight of a partial: maybe still being written; wait"
    assert any("unfinished copy" in w["reason"] for w in rep["waiting"])
    clock.advance(1)
    rep = sender.run_pass()
    assert not (env.intake / (NAMED + ".partial")).exists()
    assert (env.intake / NAMED).read_bytes() == CONTENT
    assert not p.exists()


def test_crash_after_verify_before_delete_is_resolved_next_pass(env):
    """The rename happened, the delete did not: the next pass finds the
    identical file and deletes only the local copy."""
    p = put(env.outbox, NAMED)
    put(env.intake, NAMED)
    before = (env.intake / NAMED).stat().st_mtime
    clock = Clock()
    rep = two_passes(make_sender(env, clock=clock), clock)
    assert not p.exists()
    assert (env.intake / NAMED).read_bytes() == CONTENT
    assert (env.intake / NAMED).stat().st_mtime == before, "the share copy is not rewritten"
    assert rep["held"] == []


def test_original_never_deleted_before_the_verified_copy_exists(env, monkeypatch):
    """A failure at the rename (after the hash check) keeps the original."""
    p = put(env.outbox, NAMED)
    clock = Clock()

    def broken_rename(src, dst):
        raise OSError("share went away")
    monkeypatch.setattr(ro, "_rename_no_overwrite", broken_rename)
    two_passes(make_sender(env, clock=clock), clock)
    assert p.exists() and p.read_bytes() == CONTENT
    assert not (env.intake / NAMED).exists()
    assert not (env.intake / (NAMED + ".partial")).exists()


def test_stop_request_mid_copy_keeps_source_and_removes_partial(env):
    p = put(env.outbox, NAMED)
    clock = Clock()
    flag = {"stop": False}
    sender = make_sender(env, clock=clock, should_stop=lambda: flag["stop"])
    sender.run_pass()
    clock.advance(6)
    calls = {"n": 0}
    orig = sender._abort_reason

    def abort():
        calls["n"] += 1
        if calls["n"] == 4:         # pass gate, pre-send check, first block, then stop
            flag["stop"] = True
        return orig()
    sender._abort_reason = abort
    rep = sender.run_pass()
    assert p.exists() and p.read_bytes() == CONTENT
    assert list(env.intake.iterdir()) == []
    assert "copy_stopped" in log_events(env)


def test_holds_stay_reported_during_recording_time(env):
    """A node that is not paused during recording hours still shows its holds."""
    empty = put(env.outbox, NAMED, b"")
    short = put(env.outbox, NAMED2)
    clock = Clock()
    now = {"t": SATURDAY_NOON}
    sender = make_sender(env, clock=clock, assess=lambda path: cannot())
    sender._local_now = lambda: now["t"]
    two_passes(sender, clock)
    now["t"] = WEDNESDAY_10
    probes = []
    sender._assess = lambda path: probes.append(path) or cannot()
    rep = sender.run_pass()
    assert "recording time" in rep["not_sending"]
    assert sorted(h["name"] for h in rep["held"]) == sorted([NAMED, NAMED2])
    assert probes == [], "nothing is read from a file during recording time"
    assert empty.exists() and short.exists()


def test_repeated_failures_stay_red_during_recording_time(env, monkeypatch):
    p = put(env.outbox, NAMED)
    clock = Clock()
    now = {"t": SATURDAY_NOON}
    sender = make_sender(env, clock=clock)
    sender._local_now = lambda: now["t"]
    monkeypatch.setattr(ro, "_rename_no_overwrite",
                        lambda a, b: (_ for _ in ()).throw(OSError("share went away")))
    two_passes(sender, clock)
    for _ in range(2):
        clock.advance(1)
        sender.run_pass()
    now["t"] = WEDNESDAY_10
    rep = sender.run_pass()
    assert [h["name"] for h in rep["held"]] == [NAMED]
    assert "3 times" in rep["held"][0]["reason"]
    assert p.exists()


@pytest.mark.skipif(os.name != "nt", reason="thread background mode is Windows only")
def test_copy_runs_at_low_io_priority():
    with ro.low_io_priority() as took:
        assert took is True


def test_preview_copies_and_deletes_nothing(env):
    p = put(env.outbox, NAMED)
    sender = make_sender(env)
    rep = sender.run_pass(preview=True)
    assert p.exists() and list(env.intake.iterdir()) == []
    assert "would be copied" in rep["waiting"][0]["reason"]


# ---------------------------------------------------------------- report and config

def test_held_items_appear_in_red_in_the_node_status(env):
    put(env.outbox, NAMED, b"")
    clock = Clock()
    rep = two_passes(make_sender(env, clock=clock, assess=lambda path: cannot()), clock)
    assert node_status.write(env.nas, node_status.WORKING, hostname="pc1",
                             extra={"doing": None, "done_last_hour": 0,
                                    ro.STATUS_KEY: rep})
    plain = node_status.describe(env.nas)
    assert "[!] HELD " + NAMED in plain and "\033[31m" not in plain
    colored = node_status.describe(env.nas, color=True)
    assert "\033[31m" in colored
    assert node_status.held_recordings(env.nas)[0].startswith("pc1: ")


def test_work_report_carries_the_last_send_report():
    from mousereach.watcher.orchestrator import BaseOrchestrator
    fake = types.SimpleNamespace(_outbox_report={"held": []})
    out = BaseOrchestrator._work_report(fake)
    assert out[ro.STATUS_KEY] == {"held": []}
    out = BaseOrchestrator._work_report(types.SimpleNamespace())
    assert ro.STATUS_KEY not in out


def test_terminal_color_is_off_when_not_a_terminal():
    import io
    assert node_status.terminal_color(io.StringIO()) is False


def _watcher_stub(config):
    from mousereach.watcher.orchestrator import BaseOrchestrator
    stub = types.SimpleNamespace(config=config, _stop_requested=lambda: False,
                                 _OUTBOX_MIN_INTERVAL_S=0.0)
    return stub, (lambda: BaseOrchestrator._maybe_send_recordings(stub))


def test_watcher_turn_without_outbox_does_nothing():
    from mousereach.config import WatcherConfig
    stub, turn = _watcher_stub(WatcherConfig({}))
    turn()
    assert stub._outbox_report is None


def test_watcher_turn_sends_and_keeps_the_report(env, monkeypatch):
    from mousereach.config import WatcherConfig
    from mousereach.watcher import orchestrator
    monkeypatch.setattr(orchestrator.Paths, "NAS_ROOT", env.nas)
    cfg = WatcherConfig({"recording_outbox": str(env.outbox), "recording_hours": "08:00-18:00",
                         "recording_days": "Mon-Fri"})
    stub, turn = _watcher_stub(cfg)
    turn()
    assert stub._outbox_report["outbox"] == str(env.outbox)
    assert isinstance(stub._outbox_sender, ro.RecordingSender)
    # The settings are taken fresh on every turn: an edited hour applies at once.
    cfg.recording_hours = "00:00-24:00"
    turn()
    assert stub._outbox_sender.settings.recording_hours == "00:00-24:00"


def test_watcher_turn_survives_a_failing_pass(env, monkeypatch):
    from mousereach.config import WatcherConfig
    from mousereach.watcher import orchestrator
    monkeypatch.setattr(orchestrator.Paths, "NAS_ROOT", env.nas)

    def boom(self, preview=False):
        raise RuntimeError("disk gone")
    monkeypatch.setattr(ro.RecordingSender, "run_pass", boom)
    stub, turn = _watcher_stub(WatcherConfig({"recording_outbox": str(env.outbox)}))
    turn()                                   # must not raise
    assert stub._outbox_report["held"][0]["reason"] == "disk gone"


def test_config_keys_and_defaults():
    from mousereach.config import WatcherConfig
    off = WatcherConfig({})
    assert off.recording_outbox is None
    assert ro.OutboxSettings.from_config(off).outbox is None
    assert "recording_outbox" not in off.to_dict()
    on = WatcherConfig({"recording_outbox": "D:/rec", "recording_outbox_stable_minutes": 2,
                        "recording_min_minutes": "bad", "recording_min_frames": 100,
                        "recording_hours": "08:00-18:00"})
    s = ro.OutboxSettings.from_config(on)
    assert s.outbox == Path("D:/rec") and s.stable_minutes == 2
    # The outbox reads the intake's own check settings: one set, never two.
    assert s.check.min_seconds == 300.0 and s.check.min_frames == 100
    assert s.check == ri.CheckSettings.from_config(on)
    assert s.recording_hours == "08:00-18:00"
    d = on.to_dict()
    assert d["recording_outbox"] and d["recording_outbox_stable_minutes"] == 2


def test_sender_off_when_no_outbox(env):
    rep = ro.RecordingSender(ro.OutboxSettings(), env.nas).run_pass()
    assert "not set" in rep["not_sending"]
    assert ro.summary_line(rep) == "sending recordings: off"


# ------------------------------------------- the shared drive is never walked from a PC

def _populate_share(env):
    """A share with every kind of resting place, several levels deep."""
    for d in (env.nas / "Unanalyzed" / "Unsupported_Tray_Type" / "sub",
              env.nas / "Processing" / "Quarantine" / "a" / "b",
              env.nas / "Processing" / "Review" / "_Problematic",
              env.nas / "Analyzed" / "ProjA" / "Group1" / "Multi-Animal",
              env.nas / "Analyzed" / "ProjA" / "Group1" / "Single_Animal",
              env.nas / "Archive" / "Old" / "Multi-Animal"):
        d.mkdir(parents=True, exist_ok=True)
    put(env.nas / "Analyzed" / "ProjA" / "Group1" / "Multi-Animal", NAMED2)
    put(env.nas / "Processing" / "Quarantine" / "a" / "b", "x.mkv", b"q")


@pytest.fixture
def share_listings(env, monkeypatch):
    """Every folder listing whose path is on the fake shared drive."""
    import glob as globmod
    calls = []
    nas = os.path.normcase(os.path.abspath(str(env.nas)))

    def on_share(p):
        try:
            q = os.path.normcase(os.path.abspath(os.fspath(p)))
        except TypeError:
            return False
        return q == nas or q.startswith(nas + os.sep)

    def wrap(label, fn, arg=lambda a, k: a[0] if a else k.get("path", ".")):
        def inner(*a, **k):
            if on_share(arg(a, k)):
                calls.append((label, str(arg(a, k))))
            return fn(*a, **k)
        return inner

    monkeypatch.setattr(os, "walk", wrap("os.walk", os.walk))
    monkeypatch.setattr(os, "scandir", wrap("os.scandir", os.scandir))
    monkeypatch.setattr(os, "listdir", wrap("os.listdir", os.listdir))
    monkeypatch.setattr(globmod, "glob", wrap("glob.glob", globmod.glob))
    monkeypatch.setattr(globmod, "iglob", wrap("glob.iglob", globmod.iglob))
    for meth in ("iterdir", "glob", "rglob"):
        monkeypatch.setattr(Path, meth, wrap("Path." + meth, getattr(Path, meth),
                                             arg=lambda a, k: a[0]))
    return calls


def test_a_pass_never_lists_a_folder_on_the_shared_drive(env, share_listings):
    """The defect this guards: every pass walked the share's folder trees
    from the recording PC. Now a pass reads one index file and opens named
    files only -- not one listing of any folder on the share."""
    _populate_share(env)
    share_listings.clear()
    p = put(env.outbox, NAMED)
    clock = Clock()
    rep = two_passes(make_sender(env, clock=clock), clock)
    assert not p.exists() and (env.intake / NAMED).read_bytes() == CONTENT
    assert rep["sent_this_pass"][0]["how"] == "copied and verified"
    assert share_listings == [], share_listings


def test_a_preview_never_lists_a_folder_on_the_shared_drive(env, share_listings):
    _populate_share(env)
    share_listings.clear()
    put(env.outbox, NAMED)
    put(env.outbox, NAMED2)
    rep = make_sender(env).run_pass(preview=True)
    assert len(rep["waiting"]) == 2
    assert share_listings == [], share_listings


def test_already_on_share_found_through_the_index_without_listing(env, share_listings):
    _populate_share(env)
    write_index(env)                              # the server's job, before the guard counts
    share_listings.clear()
    p = put(env.outbox, NAMED2)                   # filed under Analyzed/.../Multi-Animal
    clock = Clock()
    rep = two_passes(make_sender(env, clock=clock), clock)
    assert not p.exists(), "same name and SHA-256 at the listed path: local copy deleted"
    assert not (env.intake / NAMED2).exists(), "never sent again"
    assert rep["sent_this_pass"][0]["how"] == "already on the shared drive"
    assert share_listings == [], share_listings


# ---------------------------------------- the index is a pointer, never proof

def test_index_listing_a_different_file_is_a_conflict_and_nothing_deleted(env):
    filed = env.nas / "Analyzed" / "ProjA" / "Group1" / "Multi-Animal"
    filed.mkdir(parents=True)
    other = CONTENT[:-1] + b"X"
    put(filed, NAMED, other)
    write_index(env)
    p = put(env.outbox, NAMED)
    clock = Clock()
    rep = two_passes(make_sender(env, clock=clock), clock)
    assert p.read_bytes() == CONTENT and (filed / NAMED).read_bytes() == other
    assert not (env.intake / NAMED).exists()
    assert [h["name"] for h in rep["held"]] == [NAMED]
    assert "different file of the same name" in rep["held"][0]["reason"]


def test_index_listing_a_path_that_is_gone_waits_neither_sent_nor_deleted(env):
    """Moved since the index was built: the PC cannot confirm a copy (so it
    keeps its file) and must not send it again (it would be cut twice)."""
    filed = env.nas / "Analyzed" / "ProjA" / "Group1" / "Multi-Animal"
    filed.mkdir(parents=True)
    put(filed, NAMED)
    write_index(env)
    (filed / NAMED).unlink()
    p = put(env.outbox, NAMED)
    clock = Clock(t=ci.read(env.nas).built_at_epoch + 60)   # noticed after the index
    rep = two_passes(make_sender(env, clock=clock), clock)
    assert p.read_bytes() == CONTENT
    assert not (env.intake / NAMED).exists()
    assert rep["held"] == []
    assert "not there now" in rep["waiting"][0]["reason"]
    assert "nothing is sent or deleted" in rep["waiting"][0]["reason"]


def test_still_not_found_after_a_newer_index_is_held_in_red(env):
    p = put(env.outbox, NAMED)
    (env.nas / "Processing").mkdir()
    built = {"epoch": 0.0}

    def reader(root):
        return ci.Index({}, built["epoch"], "then", "ok")
    clock = Clock(t=5000.0)
    sender = make_sender(env, clock=clock)
    sender._read_index = reader
    sender._claim_lookup = lambda name: {"hostname": "nodeA", "state": "archived"}
    built["epoch"] = 1000.0                       # older than the first notice
    rep = two_passes(sender, clock)
    assert "already has this recording" in rep["waiting"][0]["reason"]
    built["epoch"] = clock() + 1                  # a newer index still does not list it
    clock.advance(1)
    rep = sender.run_pass()
    assert [h["name"] for h in rep["held"]] == [NAMED]
    assert "a person must find where it went" in rep["held"][0]["reason"]
    assert p.read_bytes() == CONTENT and not (env.intake / NAMED).exists()


def test_without_an_index_a_name_the_pipeline_knows_is_not_sent_again(env):
    p = put(env.outbox, NAMED)
    clock = Clock()
    asked = []

    def lookup(name):
        asked.append(name)
        return {"hostname": "nodeA", "state": "cropped"}
    sender = make_sender(env, clock=clock)
    sender._claim_lookup = lookup
    rep = two_passes(sender, clock)
    assert asked == [NAMED], "one row, one name"
    assert p.exists() and not (env.intake / NAMED).exists()
    assert "claimed by nodeA, cropped" in rep["waiting"][0]["reason"]
    assert rep["collage_index"]["status"] == "missing"


def test_without_an_index_a_new_file_is_still_sent_and_verified(env):
    p = put(env.outbox, NAMED)
    clock = Clock()
    sender = make_sender(env, clock=clock)
    sender._claim_lookup = lambda name: None
    rep = two_passes(sender, clock)
    assert not p.exists() and (env.intake / NAMED).read_bytes() == CONTENT
    assert rep["sent_this_pass"][0]["how"] == "copied and verified"
    assert rep["collage_index"]["status"] == "missing"
    assert "sent" in log_events(env)


def test_a_claim_lookup_that_fails_does_not_stop_a_new_send(env):
    p = put(env.outbox, NAMED)
    clock = Clock()
    sender = make_sender(env, clock=clock)

    def broken(name):
        raise RuntimeError("database is locked")
    sender._claim_lookup = broken
    two_passes(sender, clock)
    assert not p.exists() and (env.intake / NAMED).read_bytes() == CONTENT


def test_a_name_found_by_name_never_asks_the_claim_table(env):
    put(env.intake, NAMED)
    p = put(env.outbox, NAMED)
    clock = Clock()
    sender = make_sender(env, clock=clock)
    sender._claim_lookup = lambda name: pytest.fail("asked the shared table needlessly")
    two_passes(sender, clock)
    assert not p.exists()


def test_index_is_read_once_per_pass(env):
    put(env.outbox, NAMED)
    put(env.outbox, NAMED2)
    clock = Clock()
    reads = []

    def reader(root):
        reads.append(root)
        return ci.Index(None, None, None, "missing")
    sender = make_sender(env, clock=clock)
    sender._read_index = reader
    sender.run_pass()
    clock.advance(6)
    reads.clear()
    sender.run_pass()
    assert len(reads) == 1


# ------------------------------------------------------------ the server's index

def test_index_lists_every_resting_place_with_relative_paths(env):
    _populate_share(env)
    put(env.intake, NAMED)
    put(env.nas / "Analyzed" / "ProjA" / "Group1" / "Single_Animal", "single.mp4")
    put(env.nas / "Processing" / "Review" / "_Problematic", "bad.mkv")
    put(env.nas / "Unanalyzed" / "Unsupported_Tray_Type" / "sub", "u.mkv")
    put(env.nas / "Archive" / "Old" / "Multi-Animal", "old.mkv")
    write_index(env)
    idx = ci.read(env.nas)
    assert idx.status == "ok" and idx.usable
    paths = {k: [r["path"] for r in v] for k, v in idx.entries.items()}
    assert paths[NAMED.lower()] == ["Unanalyzed/Multi-Animal/" + NAMED]
    assert paths[NAMED2.lower()] == ["Analyzed/ProjA/Group1/Multi-Animal/" + NAMED2]
    assert paths["x.mkv"] == ["Processing/Quarantine/a/b/x.mkv"]
    assert paths["bad.mkv"] == ["Processing/Review/_Problematic/bad.mkv"]
    assert paths["u.mkv"] == ["Unanalyzed/Unsupported_Tray_Type/sub/u.mkv"]
    assert paths["old.mkv"] == ["Archive/Old/Multi-Animal/old.mkv"]
    assert "single.mp4" not in paths, "only collage resting places"
    assert idx.entries[NAMED.lower()][0]["bytes"] == len(CONTENT)
    assert idx.paths(env.nas, NAMED) == [env.intake / NAMED]


def test_index_ignores_a_path_that_leaves_the_shared_root(env):
    idx = ci.Index({"a.mkv": [{"path": "../elsewhere/a.mkv"}, {"path": "/abs/a.mkv"},
                              {"path": "Unanalyzed/Multi-Animal/a.mkv"}]}, 1.0, "x", "ok")
    assert idx.paths(env.nas, "A.MKV") == [env.nas / "Unanalyzed" / "Multi-Animal" / "a.mkv"]


def test_index_missing_or_unreadable_is_reported_not_raised(env):
    assert ci.read(env.nas).status == "missing"
    (env.nas / "Processing").mkdir()
    ci.index_path(env.nas).write_text("{not json")
    idx = ci.read(env.nas)
    assert idx.status == "unreadable" and not idx.usable
    assert ci.read(None).status == "missing"


def test_index_rebuilt_at_most_every_interval(env, monkeypatch):
    t = {"now": 0.0}
    writes = []
    monkeypatch.setattr(ci, "write", lambda root: writes.append(root))
    r = ci.Rebuilder(every_s=600, clock=lambda: t["now"], background=False)
    assert r.maybe_write(env.nas) is True
    t["now"] = 599
    assert r.maybe_write(env.nas) is False
    t["now"] = 601
    assert r.maybe_write(env.nas) is True
    assert r.maybe_write(None) is False
    assert len(writes) == 2


def test_rebuilder_never_raises(env):
    r = ci.Rebuilder(background=False)
    assert r.maybe_write(env.nas) is False        # no Processing folder: logged, not raised


def test_rebuilder_builds_off_the_loop_thread_and_never_doubles(env, monkeypatch):
    import threading
    gate = threading.Event()
    ran = []

    def slow_write(root):
        ran.append(threading.current_thread().name)
        gate.wait(5)
    monkeypatch.setattr(ci, "write", slow_write)
    t = {"now": 0.0}
    r = ci.Rebuilder(every_s=600, clock=lambda: t["now"])
    assert r.maybe_write(env.nas) is True          # returns at once, build still running
    t["now"] = 10_000
    assert r.maybe_write(env.nas) is False         # due, but one is still running
    gate.set()
    r._thread.join(5)
    assert ran == ["collage-index"]


def test_problematic_folder_name_in_step_with_recording_issue():
    assert ci._PROBLEMATIC == ri.FOLDER_NAME


def test_only_the_processing_server_builds_the_index(env, monkeypatch):
    """The scan step that writes the index runs only in processing_server mode."""
    import inspect
    from mousereach.watcher import orchestrator
    src = inspect.getsource(orchestrator.ProcessingOrchestrator._scan_phase)
    assert "collage_index" in src and '"processing_server"' in src
    assert "collage_index" not in inspect.getsource(orchestrator.DLCOrchestrator)
    assert "build_index" not in inspect.getsource(ro), "the PC-side walk is gone"
    assert "os.walk" not in inspect.getsource(ro)
