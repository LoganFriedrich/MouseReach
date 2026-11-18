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


def make_sender(env, now=SATURDAY_NOON, clock=None, probe=None, open_check=None,
                recorder_running=None, should_stop=None, chunk_size=8192, **settings):
    s = ro.OutboxSettings(outbox=env.outbox, stable_minutes=5, min_bytes=100,
                          min_seconds=10, warn_after_failures=3, pass_minutes=30,
                          recording_hours="08:00-18:00", recording_days="Mon-Fri")
    for k, v in settings.items():
        setattr(s, k, v)
    return ro.RecordingSender(
        s, env.nas, state_path=env.tmp / "state.json",
        transfer_log=env.tmp / "logs" / ro.TRANSFER_LOG_NAME,
        clock=clock or Clock(), local_now=(lambda: now),
        probe=probe or (lambda p: (600.0, None)),
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

def test_empty_file_is_held_and_reported(env):
    p = put(env.outbox, NAMED, b"")
    clock = Clock()
    rep = two_passes(make_sender(env, clock=clock), clock)
    assert p.exists() and not (env.intake / NAMED).exists()
    assert [h["name"] for h in rep["held"]] == [NAMED]


def test_tiny_or_unreadable_duration_is_held(env):
    p = put(env.outbox, NAMED)
    q = put(env.outbox, NAMED2)
    clock = Clock()
    probe = lambda path: (2.0, None) if path.name == NAMED else (None, "no duration")
    rep = two_passes(make_sender(env, clock=clock, probe=probe), clock)
    assert p.exists() and q.exists()
    assert list(env.intake.iterdir()) == []
    reasons = {h["name"]: h["reason"] for h in rep["held"]}
    assert "2.0 s long" in reasons[NAMED]
    assert "cannot be read" in reasons[NAMED2]


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
    """A collage leaves the intake once its videos are filed; finding it there
    with the same content means it was sent: only the local copy goes."""
    p = put(env.outbox, NAMED)
    filed = env.nas / "Analyzed" / "ProjA" / "Group1" / "Multi-Animal"
    filed.mkdir(parents=True)
    put(filed, NAMED)
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
    probe = lambda path: (2.0, None)
    sender = make_sender(env, clock=clock, probe=probe)
    sender._local_now = lambda: now["t"]
    two_passes(sender, clock)
    now["t"] = WEDNESDAY_10
    probes = []
    sender._probe = lambda path: probes.append(path) or (2.0, None)
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



def test_config_keys_and_defaults():
    from mousereach.config import WatcherConfig
    off = WatcherConfig({})
    assert off.recording_outbox is None
    assert ro.OutboxSettings.from_config(off).outbox is None
    assert "recording_outbox" not in off.to_dict()
    on = WatcherConfig({"recording_outbox": "D:/rec", "recording_outbox_stable_minutes": 2,
                        "recording_outbox_min_mb": "bad", "recording_hours": "08:00-18:00"})
    s = ro.OutboxSettings.from_config(on)
    assert s.outbox == Path("D:/rec") and s.stable_minutes == 2
    assert s.min_bytes == 1_000_000 and s.recording_hours == "08:00-18:00"
    d = on.to_dict()
    assert d["recording_outbox"] and d["recording_outbox_stable_minutes"] == 2


def test_sender_off_when_no_outbox(env):
    rep = ro.RecordingSender(ro.OutboxSettings(), env.nas).run_pass()
    assert "not set" in rep["not_sending"]
    assert ro.summary_line(rep) == "sending recordings: off"
