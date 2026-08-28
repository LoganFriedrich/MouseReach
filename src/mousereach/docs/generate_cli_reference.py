"""Generate docs/CLI_REFERENCE.md from the installed console scripts.

Every ``mousereach-*`` command's documentation is harvested from the command
itself (``--help``), so the reference cannot drift from the code: rerun this
after adding or changing a CLI and commit the regenerated file.

    python -m mousereach.docs.generate_cli_reference
    python -m mousereach.docs.generate_cli_reference --out <path>

Sections and one-line purposes come from pyproject.toml's own comment
structure in ``[project.scripts]``. Commands whose --help cannot be captured
(GUI-only entry points with no argument parsing, or a broken command) are
listed with the failure stated rather than silently dropped -- a command that
cannot print help is itself a finding.

ASCII-only console output (Windows cp1252 consoles cannot print Unicode).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

HELP_TIMEOUT_S = 45


def find_repo_root() -> Path:
    """The repo this module was imported from (pyproject.toml lives there)."""
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("pyproject.toml not found above %s" % p)


def parse_scripts(pyproject: Path):
    """[(section_title, [(name, target, inline_comment)])] in file order."""
    lines = pyproject.read_text(encoding="utf-8").splitlines()
    in_scripts = False
    sections = []
    current = ("Commands", [])
    pending_comment = []
    for line in lines:
        s = line.strip()
        if s == "[project.scripts]":
            in_scripts = True
            continue
        if in_scripts and s.startswith("[") and s != "[project.scripts]":
            break
        if not in_scripts:
            continue
        if s.startswith("#"):
            pending_comment.append(s.lstrip("# ").rstrip())
            continue
        m = re.match(r'^([A-Za-z0-9_.-]+)\s*=\s*"([^"]+)"\s*(#\s*(.*))?$', s)
        if m:
            if pending_comment:
                if current[1]:
                    sections.append(current)
                current = (" ".join(pending_comment), [])
                pending_comment = []
            current[1].append((m.group(1), m.group(2), (m.group(4) or "").strip()))
        elif not s:
            # blank line just separates; the next comment starts a new section
            pass
    if current[1]:
        sections.append(current)
    return sections



def _mask_machine_paths(text: str) -> str:
    """Replace this machine's own locations in harvested help text with placeholders.

    WHY: some commands print where THEIR configuration lives (home directory,
    the repository, the configured NAS/processing roots) in --help. The generated
    reference goes into a public repository, and a username or a share path is
    not part of the tool. Order matters: the most specific strings first, then a
    generic sweep of any remaining drive-letter path.
    """
    import json
    import re
    repl = []
    try:
        repl.append((str(find_repo_root()), "<repo>"))
    except Exception:
        pass
    try:
        cfg = json.loads((Path.home() / ".mousereach" / "config.json").read_text(encoding="utf-8"))
        for key, val in cfg.items():
            if isinstance(val, str) and len(val) > 3 and (":" in val or val.startswith("/")):
                repl.append((val, "<%s>" % key))
    except Exception:
        pass
    repl.append((str(Path.home()), "~"))
    for old, new in repl:
        for variant in (old, old.replace("\\", "/"), old.replace("\\", "\\\\")):
            text = text.replace(variant, new)
    # generic sweep: any drive-letter path that survived
    text = re.sub(r"[A-Z]:[\\/][^\s\"'<>|]+", "<path>", text)
    return text


def capture_help(exe_dir: Path, name: str) -> tuple:
    """(status, text). status in {ok, no_help, timeout, missing, error}."""
    for suffix in (".exe", ".cmd", ""):
        exe = exe_dir / (name + suffix)
        if exe.is_file():
            break
    else:
        return ("missing", "executable not found in %s" % exe_dir)
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"  # a GUI entry point must not open a window
    try:
        r = subprocess.run([str(exe), "--help"], capture_output=True, text=True,
                           timeout=HELP_TIMEOUT_S, env=env)
    except subprocess.TimeoutExpired:
        return ("timeout", "--help did not return within %ds (likely a GUI "
                           "entry point with no argument parsing)" % HELP_TIMEOUT_S)
    except Exception as e:
        return ("error", str(e))
    text = (r.stdout or "").strip() or (r.stderr or "").strip()
    if r.returncode not in (0, 2) or not text:
        return ("no_help", (text or "no output") [:2000])
    return ("ok", _mask_machine_paths(text))


def generate(out_path: Path) -> dict:
    root = find_repo_root()
    sections = parse_scripts(root / "pyproject.toml")
    # Console scripts live in Scripts/ next to python.exe on Windows conda
    # envs (bin/ on posix); sys.executable's own dir is the env ROOT there.
    exe_dir = Path(sys.executable).parent
    for cand in (exe_dir / "Scripts", exe_dir / "bin", exe_dir):
        if (cand / "mousereach.exe").is_file() or (cand / "mousereach").is_file():
            exe_dir = cand
            break

    counts = {"ok": 0, "failed": 0, "total": 0}
    lines = []
    add = lines.append
    add("# MouseReach CLI Reference")
    add("")
    add("Generated %s by `python -m mousereach.docs.generate_cli_reference` --"
        % date.today().isoformat())
    add("every entry below is the command's own `--help` output, harvested from")
    add("the installed executables, so this file cannot say something the code")
    add("does not. **Do not edit by hand; rerun the generator.**")
    add("")
    add("All commands exist only inside the mousereach conda environment:")
    add("")
    add("```")
    add("conda activate mousereach    # or the full path of the env if it was created with --prefix")
    add("mousereach-<command> --help")
    add("```")
    add("")
    add("`mousereach` alone launches the full napari GUI with every tab; the")
    add("commands here run one specific piece without the GUI (or launch one")
    add("specific tool window). See docs/REVIEW_TOOLS.md for the operator")
    add("walkthroughs of the review tools themselves.")
    add("")

    # table of contents
    add("## Contents")
    add("")
    for title, entries in sections:
        anchor = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        add("- [%s](#%s) -- %s" % (title, anchor,
                                   ", ".join("`%s`" % n for n, _, _ in entries)))
    add("")

    seen_targets = {}
    for title, entries in sections:
        add("## %s" % title)
        add("")
        for name, target, inline in entries:
            counts["total"] += 1
            add("### `%s`" % name)
            add("")
            if inline:
                add("*%s*" % inline)
                add("")
            if target in seen_targets:
                add("Alias of `%s` (same entry point: `%s`)."
                    % (seen_targets[target], target))
                add("")
                counts["ok"] += 1
                print("  %-40s alias of %s" % (name, seen_targets[target]))
                continue
            seen_targets[target] = name
            status, text = capture_help(Path(exe_dir), name)
            print("  %-40s %s" % (name, status))
            if status == "ok":
                counts["ok"] += 1
                add("```")
                add(text)
                add("```")
            else:
                counts["failed"] += 1
                add("**No `--help` available** (%s): %s" % (status, text))
                add("")
                add("Entry point: `%s`. Read its module docstring for usage."
                    % target)
            add("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output path (default: <repo>/docs/CLI_REFERENCE.md)")
    args = ap.parse_args()
    out = args.out or (find_repo_root() / "docs" / "CLI_REFERENCE.md")
    print("Generating %s" % out)
    counts = generate(out)
    print("Done: %d commands, %d with help, %d without"
          % (counts["total"], counts["ok"], counts["failed"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
