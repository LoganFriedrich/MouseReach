"""Tell the person at the machine when it is safe to record.

WHY THIS EXISTS
---------------
On a node that both records behaviour and processes video, the watcher stands
aside while the recording program is open. But standing aside is not instant:
the pose or crop running at that moment has to be stopped first, and until it
is, the machine is still busy with the GPU and disk the recording needs. The
operator cannot see any of that -- they open the recording program and have to
guess whether the machine is theirs yet.

So the watcher says it, on the screen of that machine: one message when it
starts standing aside, and one when everything has actually stopped. The second
message is the one that matters: it means the GPU and disk are free.

HOW IT IS SHOWN
---------------
A plain Windows message box, from a background thread so the watcher never
waits for anybody to click it, and never more than one of each kind per
episode. On anything that is not Windows, or where no desktop is available
(a service, a machine nobody is logged in to), nothing is shown and nothing
fails -- the log line is always written either way.

Turned off with ``watcher.notify_safe_to_record: false``.
"""

import logging
import threading

logger = logging.getLogger(__name__)

SAFE_TITLE = "Safe to record"
SAFE_TEXT = ("MouseReach has stopped all work on this PC.\n\n"
             "The GPU and disk are free -- it is safe to record now.\n\n"
             "Processing starts again by itself a couple of minutes after the "
             "recording program is closed.")
STOPPING_TITLE = "MouseReach is stopping"
STOPPING_TEXT = ("A recording program was opened, so MouseReach is stopping the work "
                 "running on this PC.\n\n"
                 "Wait for the 'Safe to record' message before you start recording.")

# Windows MessageBox flags: information icon, OK only, in front of other
# windows, and shown even when this process has no window of its own.
_MB_OK = 0x00000000
_MB_ICONINFORMATION = 0x00000040
_MB_SETFOREGROUND = 0x00010000
_MB_TOPMOST = 0x00040000
_MB_SERVICE_NOTIFICATION = 0x00200000


def _show_windows_box(title: str, text: str) -> None:
    import ctypes
    ctypes.windll.user32.MessageBoxW(
        None, text, title,
        _MB_OK | _MB_ICONINFORMATION | _MB_SETFOREGROUND | _MB_TOPMOST)


def notify(title: str, text: str, show=None) -> bool:
    """Show one message without waiting for it. True if a box was started.

    Never raises: a message that cannot be shown must not stop a watcher, and
    the same words are always in the log.
    """
    import sys
    show = show or (_show_windows_box if sys.platform == "win32" else None)
    if show is None:
        return False
    try:
        thread = threading.Thread(target=_guarded, args=(show, title, text),
                                  name="mousereach-record-notice", daemon=True)
        thread.start()
        return True
    except Exception as e:
        logger.debug(f"could not show the '{title}' message: {e}")
        return False


def _guarded(show, title, text) -> None:
    try:
        show(title, text)
    except Exception as e:
        logger.debug(f"could not show the '{title}' message: {e}")


class RecordNotices:
    """One episode's worth of messages: 'stopping' once, then 'safe' once.

    An episode starts when the watcher pauses for a recording program and ends
    when it goes back to work, so an operator who records all morning is told
    once, not every thirty seconds. ``enabled`` False says nothing at all.
    """

    def __init__(self, enabled: bool = True, notify_fn=notify):
        self.enabled = bool(enabled)
        self._notify = notify_fn
        self._said_stopping = False
        self._said_safe = False

    def _say(self, title: str, text: str) -> bool:
        """Show one message. A message that cannot be shown is never allowed to
        reach the caller: this is called from the watcher's pause check and from
        the middle of a pose, and neither may die because a box failed."""
        try:
            return bool(self._notify(title, text))
        except Exception as e:
            logger.debug(f"could not show the '{title}' message: {e}")
            return False

    def stopping(self) -> bool:
        """Work is being stopped for a recording program."""
        if not self.enabled or self._said_stopping:
            return False
        self._said_stopping = True
        logger.info("Telling the operator that MouseReach is stopping its work.")
        return self._say(STOPPING_TITLE, STOPPING_TEXT)

    def safe_to_record(self) -> bool:
        """Nothing is running on this node any more."""
        if not self.enabled or self._said_safe:
            return False
        self._said_safe = True
        logger.info("Telling the operator it is safe to record: no work is running here.")
        return self._say(SAFE_TITLE, SAFE_TEXT)

    def back_to_work(self) -> None:
        """The pause is over; the next recording gets its own messages."""
        self._said_stopping = False
        self._said_safe = False
