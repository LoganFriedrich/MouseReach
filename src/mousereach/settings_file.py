"""Reading a MouseReach settings file, with no side effect on import.

WHY a module of its own: ``mousereach.config`` reads ~/.mousereach/config.json
the moment it is imported and stops on a damaged file. The commands that must
SAY what is wrong -- ``mousereach-setup`` and the ``mousereach`` launcher --
need the error class and the reader without importing ``mousereach.config``
first, or the damage would stop them with a traceback before their own
handler ran. ``mousereach.config`` re-exports everything here, so code that
imports these names from there keeps working.

ASCII only in every message: it is printed to the Windows console.
"""

import json
from pathlib import Path


class ConfigurationError(Exception):
    """Raised when required configuration is missing."""
    pass


class DamagedSettingsFile(ConfigurationError, ValueError):
    """A settings file exists but cannot be read as a JSON object.

    A ValueError too, so every caller that already stops on a malformed
    setting (the filing rules raise ValueError) stops on this the same way."""


def read_settings_file(path) -> dict:
    """The JSON object in ``path``; {} when the file does not exist.

    Raises DamagedSettingsFile, naming the file, the line and column of a JSON
    error and what to do, when the file exists but cannot be read, is not
    UTF-8 text (naming the byte position), is not valid JSON, or does not
    hold an object. Accepts a byte-order mark (a file
    saved by Notepad starts with one).

    WHY strict: a damaged settings file used to read as an empty one. Every
    path was then "not configured", the tools told the person to run
    mousereach-setup, and a save from setup rewrote the file with only the few
    keys it manages -- every other setting gone. "Not there" and "there but
    broken" must never look the same."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    except UnicodeDecodeError as e:
        # WHY its own message: a file saved in a non-UTF-8 encoding (an accent
        # saved as "ANSI" by an old editor) is damaged in a way the JSON
        # message cannot describe, and an error of a type no caller expects
        # would get past every handler that turns damage into "fix the file"
        # (the [FAIL] line, a running watcher keeping its last good copy).
        raise DamagedSettingsFile(
            "The settings file %s is not saved as UTF-8 text: byte number %d (counting from 1) "
            "cannot be read. Open it in a text editor, save it again choosing the "
            "UTF-8 encoding, and try again." % (path, e.start + 1))
    except OSError as e:
        raise DamagedSettingsFile(
            "The settings file %s could not be read (%s). Check that the drive it is "
            "on is reachable and that nothing has it locked, then try again." % (path, e))
    if not text.strip():
        return {}               # an empty file holds no setting to lose
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise DamagedSettingsFile(
            "The settings file %s is damaged: %s at line %d, column %d. Open it in a "
            "text editor and fix that spot (a missing comma or quote is the usual "
            "cause), or restore it from a backup. MouseReach will not run on it until "
            "it is fixed, so that no setting is silently lost." % (path, e.msg, e.lineno, e.colno))
    if not isinstance(data, dict):
        raise DamagedSettingsFile(
            "The settings file %s does not hold a JSON object ({ ... }); it holds a %s. "
            "Fix it in a text editor or restore it from a backup."
            % (path, type(data).__name__))
    return data
