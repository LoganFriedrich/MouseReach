"""The suite never draws a real message box on the machine it runs on: tests
start watchers configured to pause for a recording program, and the watcher's
"stopping" / "safe to record" boxes appeared on the screen of whoever was at
the machine. This check itself draws nothing."""
import threading

from mousereach.watcher import record_notice


def test_the_real_message_box_is_replaced_in_every_test():
    # The real function lives in record_notice; the stand-in from conftest
    # does not.
    assert record_notice._show_windows_box.__module__ != record_notice.__name__
    assert record_notice._close_windows_box.__module__ != record_notice.__name__


def test_a_watcher_notice_in_a_test_goes_to_the_stand_in(_no_real_message_boxes):
    notices = record_notice.RecordNotices(enabled=True)
    assert notices.stopping() is True
    # notify() shows the box from a background thread; wait for it.
    for t in threading.enumerate():
        if t.name == "mousereach-record-notice":
            t.join(timeout=10)
    assert record_notice.STOPPING_TITLE in _no_real_message_boxes
