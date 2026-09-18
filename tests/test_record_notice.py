"""The node tells the operator when it is safe to record (watcher/record_notice.py).

WHY: pausing for a recording program is not instant -- the pose or crop running
at that moment is killed first -- and the operator cannot see when the GPU and
disk are actually free. One message when work starts being stopped, one when it
has stopped, once per recording episode, and never a crash from a message that
cannot be shown.
"""
from mousereach.watcher import record_notice as rn


def _recorder():
    said = []
    return said, (lambda title, text: said.append(title) or True)


def test_one_of_each_message_per_episode():
    said, fake = _recorder()
    n = rn.RecordNotices(enabled=True, notify_fn=fake)

    assert n.stopping() and not n.stopping()
    assert n.safe_to_record() and not n.safe_to_record()
    assert said == [rn.STOPPING_TITLE, rn.SAFE_TITLE]

    n.back_to_work()                 # the recording ended; the next one is told again
    assert n.stopping() and n.safe_to_record()
    assert said.count(rn.SAFE_TITLE) == 2


def test_turned_off_says_nothing():
    said, fake = _recorder()
    n = rn.RecordNotices(enabled=False, notify_fn=fake)
    assert not n.stopping() and not n.safe_to_record()
    assert said == []


def test_a_message_that_cannot_be_shown_never_raises():
    def boom(title, text):
        raise OSError("no desktop here")

    # notify() runs the box in a thread; a failure there must not reach the watcher.
    assert rn.notify("t", "x", show=boom) in (True, False)
    n = rn.RecordNotices(enabled=True, notify_fn=lambda *a: (_ for _ in ()).throw(OSError("nope")))
    try:
        n.stopping()
    except Exception as e:                       # pragma: no cover - the point of the test
        raise AssertionError(f"a failed message must not escape: {e}")


def test_words_say_what_the_operator_needs():
    assert "safe to record" in rn.SAFE_TEXT.lower()
    assert "stopping" in rn.STOPPING_TEXT.lower()
    assert "wait" in rn.STOPPING_TEXT.lower()
