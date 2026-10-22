"""Hand-parsed watcher commands refuse an option their usage does not list.

WHY (2026-10-22): an unknown option was silently ignored, so a typo such as
``mousereach-watch --dryrun`` started the REAL watcher instead of the read-only
preview. The usage text each command already prints for --help is the list of
options it accepts; anything else stops the command before it does anything.
"""
import sys

import pytest

from mousereach.watcher import cli

USAGE = """
usage: mousereach-watch [--once] [--dry-run] [--verbose] [--quiet]
"""


def _run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["mousereach-watch", *args])
    cli._maybe_print_help(USAGE)


def test_a_mistyped_option_stops_the_command(monkeypatch, capsys):
    with pytest.raises(SystemExit) as e:
        _run(monkeypatch, "--dryrun")
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "unknown option(s): --dryrun" in err and "Nothing was changed." in err
    err.encode("ascii")


def test_listed_options_and_plain_values_pass(monkeypatch):
    _run(monkeypatch, "--dry-run", "--once")
    _run(monkeypatch)
    _run(monkeypatch, "CNT0107")
    _run(monkeypatch, "--dry-run", "-5")      # a negative value is not an option


def test_help_still_prints_and_exits_0(monkeypatch, capsys):
    with pytest.raises(SystemExit) as e:
        _run(monkeypatch, "--help", "--dryrun")
    assert e.value.code == 0
    assert "usage: mousereach-watch" in capsys.readouterr().out


@pytest.mark.parametrize("entry", ["main_watch", "main_status", "main_reprocess",
                                   "main_quarantine", "main_process_animal",
                                   "main_version_check", "main_crystallize",
                                   "main_uncrystallize", "main_unresolvable"])
def test_every_hand_parsed_command_refuses_before_acting(monkeypatch, capsys, entry):
    monkeypatch.setattr(sys, "argv", [entry, "--no-such-option"])
    with pytest.raises(SystemExit) as e:
        getattr(cli, entry)()
    assert e.value.code == 2


@pytest.mark.parametrize("word", ["stop", "start"])
def test_the_watcher_refuses_a_word_and_says_how_to_stop(monkeypatch, capsys, word):
    """An install script ran "mousereach-watch stop" meaning to stop it; the
    word was ignored and the command started the watcher (2026-10-22)."""
    monkeypatch.setattr(sys, "argv", ["mousereach-watch", word])
    with pytest.raises(SystemExit) as e:
        cli.main_watch()
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "takes no words" in err and "watcher_stop.flag" in err
    err.encode("ascii")
