"""Pausing for the whole recording window, program or not.

WHY: an open recording program turned out to be a poor signal -- it stays open
after filming and kept the behaviour-room nodes idle for the rest of the
window. With watcher.pause_during_recording_hours the window itself is the
rule: no work inside it whatever is open, full work outside it whatever is open.
"""
from datetime import datetime

import pytest

from mousereach.config import WatcherConfig
from mousereach.watcher import recording_guard as rg

# 2026-10-19 is a Monday.
MON_0859 = datetime(2026, 10, 19, 8, 59)
MON_0900 = datetime(2026, 10, 19, 9, 0)
MON_1459 = datetime(2026, 10, 19, 14, 59)
MON_1500 = datetime(2026, 10, 19, 15, 0)
SAT_10 = datetime(2026, 10, 24, 10, 0)


def _guard(when, running=(), names=("recorder.exe",), hours="09:00-15:00", days="Mon-Fri"):
    return rg.RecordingGuard(list(names), scan=lambda n: list(running), cache_seconds=0,
                             recording_hours=hours, recording_days=days,
                             local_now=lambda: when, whole_window=True)


@pytest.mark.parametrize("when,paused", [
    (MON_0859, False), (MON_0900, True), (MON_1459, True), (MON_1500, False), (SAT_10, False)])
def test_the_window_alone_decides_with_no_program_open(when, paused):
    assert (_guard(when).reason() is not None) == paused


@pytest.mark.parametrize("when,paused", [(MON_0859, False), (MON_1000 := datetime(2026, 10, 19, 10), True),
                                         (MON_1500, False), (SAT_10, False)])
def test_an_open_program_changes_nothing(when, paused):
    assert (_guard(when, running=("recorder.exe",)).reason() is not None) == paused


def test_it_works_with_no_program_listed():
    g = _guard(MON_1459, names=())
    assert g.enabled
    assert "recording hours" in g.reason()


def test_with_no_window_set_it_pauses_nothing():
    assert _guard(MON_1459, hours=None, days=None).reason() is None


def test_the_setting_is_read_and_written_and_off_by_default():
    assert WatcherConfig({}).pause_during_recording_hours is False
    cfg = WatcherConfig({"pause_during_recording_hours": True,
                         "recording_hours": "09:00-15:00", "recording_days": "Mon-Fri"})
    assert cfg.pause_during_recording_hours is True
    assert cfg.to_dict()["pause_during_recording_hours"] is True
    assert "pause_during_recording_hours" not in WatcherConfig({}).to_dict()


def test_pause_reason_uses_it(tmp_path, monkeypatch):
    cfg = WatcherConfig({"pause_during_recording_hours": True,
                         "recording_hours": "00:00-23:59", "recording_days": "Mon-Sun"})
    assert "recording hours" in rg.pause_reason(processing_root=tmp_path, config=cfg)
