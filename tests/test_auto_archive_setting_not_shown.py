"""auto_archive_approved is not shown: nothing in the code reads it.

WHY: the watcher's start banner said "Auto Archive: No", the Watcher Control
form had a checkbox for it and the setup summary printed it, yet no code acts
on it -- a video is filed when it passes every check, whatever the setting
says. A switch that does nothing misleads the person who flips it. A value
already in a settings file is left there (the form no longer owns the key, so
saving keeps it as it is).
"""
import pytest

cw = pytest.importorskip("mousereach.watcher.control_widget")


def test_the_form_neither_shows_nor_writes_it():
    from test_control_widget_recording import _form_self
    W = cw.WatcherControlWidget
    d = W._form_to_dict(_form_self())
    assert "auto_archive_approved" not in d
    assert "auto_archive_approved" not in W._FORM_MANAGED_KEYS


def test_no_screen_prints_it():
    from pathlib import Path
    import mousereach.watcher.cli as cli
    import mousereach.setup.wizard as wz
    for mod in (cli, wz):
        assert "Auto Archive" not in Path(mod.__file__).read_text(encoding="utf-8")
