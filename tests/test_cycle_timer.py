"""Every step of a watcher cycle is timed, and a slow cycle says where its time went.

WHY: a processing node spent most of its time in scans between work items
(about 20 s of work, then 8-10 minutes until the next item) and nothing in the
log said which scan. These tests pin the line that now says so.
"""
from mousereach.watcher.cycle_timer import CycleTimer


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _timer(lines):
    clock = FakeClock()
    return CycleTimer(slow_seconds=5.0, log=lines.append, clock=clock), clock


def test_a_slow_cycle_names_every_step_and_the_parts_of_the_scan():
    lines = []
    timer, clock = _timer(lines)
    with timer.step("scan"):
        with timer.step("scan/posed discovery"):
            clock.now += 10
        with timer.step("scan/review return"):
            clock.now += 470
    with timer.step("pick"):
        clock.now += 0.5
    with timer.step("work"):
        clock.now += 20
    assert timer.total() == 500.5            # the parts are not counted twice
    line = timer.log_if_slow()
    assert line == ("Cycle 500.5 s: scan 480.0 s [posed discovery 10.0, review return 470.0], "
                    "pick 0.5 s, work 20.0 s")
    # each slow step was also logged as it finished, so a hang shows its last step
    assert "Cycle step scan/review return took 470.0 s" in lines


def test_a_fast_cycle_writes_nothing():
    lines = []
    timer, clock = _timer(lines)
    with timer.step("scan"):
        clock.now += 1
    with timer.step("work"):
        clock.now += 2
    assert timer.log_if_slow() is None and lines == []


def test_a_step_that_raises_is_still_timed():
    lines = []
    timer, clock = _timer(lines)
    try:
        with timer.step("scan/review return"):
            clock.now += 7
            raise RuntimeError("share went away")
    except RuntimeError:
        pass
    assert timer.steps == [("scan/review return", 7.0)]
    assert lines == ["Cycle step scan/review return took 7.0 s"]
