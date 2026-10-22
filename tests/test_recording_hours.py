"""Recording hours and days: outside them an open recording program does not
pause the node.

WHY (G12, 2026-10-22): the recording program is routinely left open after the
day's recording, and a node that pauses whenever it is open then idled until a
person closed it -- once for two days. The lab knows when recording can happen
at all (in the configuration: watcher.recording_hours, watcher.recording_days);
outside that, an open program says nothing. Both empty must behave exactly as
before (an open program pauses at any time).
"""
from datetime import datetime

import pytest

from mousereach.config import WatcherConfig
from mousereach.watcher import recording_guard as rg

# 2026-10-19 is a Monday.
MON_10 = datetime(2026, 10, 19, 10, 0)
MON_0759 = datetime(2026, 10, 19, 7, 59)
MON_1800 = datetime(2026, 10, 19, 18, 0)
SAT_10 = datetime(2026, 10, 24, 10, 0)
SUN_10 = datetime(2026, 10, 25, 10, 0)


def _guard(when, hours=None, days=None, running=("recorder.exe",)):
    return rg.RecordingGuard(["recorder.exe"], scan=lambda names: list(running),
                             cache_seconds=0, recording_hours=hours,
                             recording_days=days, local_now=lambda: when)


def test_without_hours_or_days_an_open_program_pauses_at_any_time():
    for when in (MON_10, MON_0759, SAT_10, SUN_10):
        assert _guard(when).reason() == "recorder.exe is running"


@pytest.mark.parametrize("when,paused", [
    (MON_10, True), (MON_0759, False), (MON_1800, False), (SAT_10, False), (SUN_10, False)])
def test_a_weekday_window(when, paused):
    reason = _guard(when, hours="08:00-18:00", days="Mon-Fri").reason()
    assert (reason == "recorder.exe is running") is paused


def test_outside_the_window_the_process_list_is_not_even_read():
    def boom(names):
        raise AssertionError("scanned outside recording time")
    g = rg.RecordingGuard(["recorder.exe"], scan=boom, recording_days="Mon-Fri",
                          local_now=lambda: SAT_10)
    assert g.reason() is None


def test_hours_may_cross_midnight():
    w = rg.parse_hours("22:00-04:00")
    assert rg.within_hours(w, datetime(2026, 10, 19, 23, 0))
    assert rg.within_hours(w, datetime(2026, 10, 19, 3, 59))
    assert not rg.within_hours(w, datetime(2026, 10, 19, 4, 0))


def test_day_lists_ranges_and_wraps():
    assert rg.parse_days("Mon-Fri") == frozenset({0, 1, 2, 3, 4})
    assert rg.parse_days("monday, wed ,FRI") == frozenset({0, 2, 4})
    assert rg.parse_days(["Sat", "Sun"]) == frozenset({5, 6})
    assert rg.parse_days("Sat-Mon") == frozenset({5, 6, 0})


@pytest.mark.parametrize("bad", ["", "6-15", "06:00", "25:00-26:00", "06:00-06:00", "06:61-15:00"])
def test_unreadable_hours_mean_any_hour(bad):
    assert rg.parse_hours(bad) is None
    assert _guard(SAT_10, hours=bad).reason() == "recorder.exe is running"


def test_unreadable_days_mean_every_day():
    assert rg.parse_days("Funday") is None
    assert _guard(SAT_10, days="Funday").reason() == "recorder.exe is running"


def test_the_config_reads_and_writes_both_keys():
    cfg = WatcherConfig({"pause_while_running": ["recorder.exe"],
                         "recording_hours": " 08:00-18:00 ",
                         "recording_days": ["Mon", "Tue", "Wed", "Thu", "Fri"]})
    assert cfg.recording_hours == "08:00-18:00"
    assert cfg.recording_days == "Mon, Tue, Wed, Thu, Fri"
    d = cfg.to_dict()
    assert d["recording_hours"] == "08:00-18:00" and d["recording_days"] == "Mon, Tue, Wed, Thu, Fri"
    plain = WatcherConfig({"pause_while_running": ["recorder.exe"]}).to_dict()
    assert "recording_hours" not in plain and "recording_days" not in plain
