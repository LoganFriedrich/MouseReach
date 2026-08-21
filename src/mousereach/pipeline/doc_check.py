"""Refuse to let the documentation drift away from the code without saying so.

WHY THIS EXISTS
---------------
Every convention this project has written down about keeping documents current
has been broken, including by the people who wrote it. That is not a discipline
problem; a rule nothing enforces is a wish. In one week, four separate values the
pipeline computes were found never to reach the data, each discovered by
accident, because no document described what the code produced and none could be
trusted if it had.

So this is mechanical. A file maps directories of code to the document that
describes them. A pre-commit hook refuses a commit that changes mapped code
without touching its document. Skipping is allowed -- plenty of changes do not
affect a description -- but it must be said out loud in the commit message, where
it stays in the history instead of being invisible.

WHAT IT CANNOT DO
-----------------
It cannot tell you a document is CORRECT, only that somebody touched it. A
one-word edit satisfies it. That is the honest limit of any check on prose, and
it is why anything genuinely load-bearing belongs in the field audit instead,
which compares claims against data and cannot be satisfied by typing.

A local hook can also be bypassed with --no-verify. This project already forbids
that, and the escape hatch exists precisely so nobody needs it.

USAGE
-----
  mousereach-doc-check                 report which documents the code has moved past
  mousereach-doc-check --staged        check what is staged (used by the hook)
  mousereach-doc-check --install-hook  install the commit-msg hook in this repo

ASCII-only console output (Windows consoles cannot print Unicode).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

def _repo_root() -> Path:
    """The repository being worked in.

    A pre-commit hook runs with the working directory inside the repository
    being committed to, which is the repository whose documents matter. Anchoring
    to where this module happens to live gets that wrong whenever the code is
    installed from a different copy than the checkout -- which is the normal
    arrangement here, where the runtime copy lives on a different drive than the
    git working copy.
    """
    import subprocess as _sp
    try:
        out = _sp.run(["git", "rev-parse", "--show-toplevel"],
                      capture_output=True, text=True, check=False).stdout.strip()
        if out:
            return Path(out)
    except Exception:
        pass
    return Path(__file__).resolve().parents[3]


REPO = _repo_root()
COVERAGE = REPO / "docs" / "DOC_COVERAGE.json"

# The line a document carries saying which commit it was last checked against.
STAMP = re.compile(r"^Verified against:\s*([0-9a-f]{7,40})", re.M | re.I)

# Saying this in a commit message means "I looked, and this change does not
# alter any description." It is recorded in the history, which is the point.
SKIP = re.compile(r"^Doc-Impact:\s*none\b", re.M | re.I)

# The interpreter is written in at install time. A hook runs with a bare
# environment -- "python" was not on the PATH here, so the hook died with
# "command not found" and refused every commit for the wrong reason. Failing
# closed by accident is not the same as working.
# This is a commit-msg hook, not a pre-commit one. pre-commit runs BEFORE git
# writes the commit message, so a pre-commit hook cannot read it -- the
# "Doc-Impact: none" escape was silently impossible to use until this was tested.
# commit-msg runs after the message exists and is handed its path as $1.
HOOK_TEMPLATE = """#!/bin/sh
# Refuse a commit that changes documented code without touching its document.
# Installed by: mousereach-doc-check --install-hook
PY="{python}"
if [ ! -x "$PY" ]; then
    echo "doc-check: interpreter missing at $PY"
    echo "Re-install the hook with: mousereach-doc-check --install-hook"
    exit 1
fi
"$PY" -m mousereach.pipeline.doc_check --staged --message "$1" || exit 1
"""


def git(*args, cwd: Path = REPO) -> str:
    try:
        return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                              text=True, check=False).stdout.strip()
    except Exception:
        return ""


def load_coverage(path: Path = COVERAGE) -> List[dict]:
    """The map from code paths to the document that describes them."""
    if not Path(path).is_file():
        return []
    doc = json.loads(Path(path).read_text())
    return doc.get("documents", [])


def doc_stamp(doc_path: Path) -> Optional[str]:
    """The commit a document says it was last verified against."""
    p = REPO / doc_path
    if not p.is_file():
        return None
    m = STAMP.search(p.read_text(encoding="utf-8", errors="replace")[:4000])
    return m.group(1) if m else None


def covers(entry: dict, changed: List[str]) -> List[str]:
    """Which of the changed files this document is responsible for."""
    hit = []
    for path in entry.get("covers", []):
        norm = path.replace("\\", "/").rstrip("/")
        for f in changed:
            g = f.replace("\\", "/")
            if g == norm or g.startswith(norm + "/"):
                hit.append(f)
    return hit


def commits_since(stamp: str, paths: List[str]) -> List[str]:
    """Commits touching these paths since the document was verified."""
    if not stamp or not paths:
        return []
    out = git("log", "--oneline", "%s..HEAD" % stamp, "--", *paths)
    return [l for l in out.splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# the two modes
# ---------------------------------------------------------------------------

def check_staged(message_path: Optional[str] = None) -> int:
    """Is any documented code changing without its document?

    ``message_path`` is the file git hands a commit-msg hook. Without it there
    is no reliable way to read the message being written -- COMMIT_EDITMSG holds
    the PREVIOUS message when a pre-commit hook runs, so the escape hatch would
    silently never fire.
    """
    entries = load_coverage()
    if not entries:
        return 0
    changed = [l for l in git("diff", "--cached", "--name-only").splitlines() if l.strip()]
    if not changed:
        return 0

    message = ""
    if message_path and Path(message_path).is_file():
        message = Path(message_path).read_text(encoding="utf-8", errors="replace")
    excused = bool(SKIP.search(message))

    problems = []
    for e in entries:
        touched = covers(e, changed)
        if not touched:
            continue
        if e["doc"].replace("\\", "/") in [c.replace("\\", "/") for c in changed]:
            continue  # the document is being updated in the same commit
        problems.append((e, touched))

    if not problems:
        return 0

    print("")
    print("This commit changes code that a document describes, and does not")
    print("touch the document:")
    print("")
    for e, touched in problems:
        print("  %s" % e["doc"])
        for t in touched[:6]:
            print("      %s" % t)
        if len(touched) > 6:
            print("      ... and %d more" % (len(touched) - 6))
    print("")
    if excused:
        print("Allowed: the commit message says Doc-Impact: none.")
        print("")
        return 0
    print("Either update the document in this commit, or say why it does not need")
    print("updating by putting a line like this in the commit message:")
    print("")
    print("    Doc-Impact: none -- refactor only, behaviour unchanged")
    print("")
    print("It is recorded in the history either way, which is the point.")
    print("")
    return 1


def report() -> int:
    """How far the code has moved since each document was last verified."""
    entries = load_coverage()
    if not entries:
        print("No coverage map at %s" % COVERAGE)
        return 0

    stale, fine, unstamped, missing = [], [], [], []
    for e in entries:
        path = REPO / e["doc"]
        if not path.is_file():
            missing.append(e)
            continue
        stamp = doc_stamp(Path(e["doc"]))
        if not stamp:
            unstamped.append(e)
            continue
        commits = commits_since(stamp, e.get("covers", []))
        (stale if commits else fine).append((e, stamp, commits))

    print("Documents: %d" % len(entries))
    if fine:
        print("\nUp to date (%d):" % len(fine))
        for e, stamp, _ in fine:
            print("   %-42s verified at %s" % (e["doc"], stamp))
    if stale:
        print("\nCODE HAS MOVED SINCE THESE WERE VERIFIED (%d):" % len(stale))
        for e, stamp, commits in stale:
            print("\n   %s   (verified at %s)" % (e["doc"], stamp))
            for c in commits[:8]:
                print("      %s" % c)
            if len(commits) > 8:
                print("      ... and %d more commits" % (len(commits) - 8))
    if unstamped:
        print("\nNo 'Verified against:' line, so drift cannot be measured (%d):"
              % len(unstamped))
        for e in unstamped:
            print("   %s" % e["doc"])
    if missing:
        print("\nMapped but not present on disk (%d):" % len(missing))
        for e in missing:
            print("   %s" % e["doc"])
    if stale:
        print("\nRe-read the code, correct each document, and update its")
        print("'Verified against:' line to %s." % (git("rev-parse", "--short", "HEAD") or "HEAD"))
    return 0


def install_hook() -> int:
    hooks = REPO / ".git" / "hooks"
    if not hooks.is_dir():
        print("[FAIL] no .git/hooks in %s" % REPO)
        return 1
    target = hooks / "commit-msg"
    if target.exists():
        existing = target.read_text(encoding="utf-8", errors="replace")
        if "doc_check" in existing:
            # Re-install anyway: the interpreter path may have moved, and a hook
            # pointing at a missing interpreter refuses every commit.
            print("Re-installing over the existing hook (interpreter path refreshed)")
            existing = None
        if existing is not None:
            backup = hooks / "commit-msg.before-doc-check"
            backup.write_text(existing)
            print("Existing hook saved as %s" % backup.name)
    import sys as _sys
    target.write_text(HOOK_TEMPLATE.format(python=_sys.executable.replace("\\", "/")))
    try:
        target.chmod(0o755)
    except OSError:
        pass
    print("Installed %s" % target)
    print("A commit that changes documented code without its document is now refused.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Keep the documents and the code from drifting apart silently.")
    ap.add_argument("--staged", action="store_true",
                    help="Check what is staged for commit (used by the commit-msg hook)")
    ap.add_argument("--message", default=None,
                    help="Path to the commit message file, as git passes it to a "
                         "commit-msg hook")
    ap.add_argument("--install-hook", action="store_true",
                    help="Install the pre-commit hook into this repository")
    args = ap.parse_args(argv)

    if args.install_hook:
        return install_hook()
    if args.staged:
        return check_staged(args.message)
    return report()


if __name__ == "__main__":
    sys.exit(main())
