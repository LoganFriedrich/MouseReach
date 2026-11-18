# The test guard: tests never write into your real pipeline folders

MouseReach's automated tests (the files in `tests/`) build every folder they
need inside a temporary folder. The **test guard** makes sure that stays true:
while the tests run, nothing may be created, changed, moved or deleted inside
the folders your lab actually works in. You do not have to turn it on; it is
loaded every time the tests run.

This page is for anyone who runs the tests, including people who do not read
code.

## What it protects ("real roots")

A *real root* is a folder that holds real lab data or state. The guard
collects them when the tests start, from every place MouseReach itself would
look on that machine:

| Source | What is taken from it |
|---|---|
| `config.json` in the `.mousereach` folder of your home folder | every folder or file path in it (`nas_root`, `processing_root`, `nas_drive`, the watcher database, backup folders ...). A file counts as its folder. |
| The same file in the account's *own* home folder, as Windows (or the operating system) records it | the same. This is read even when the test run has moved the home folder elsewhere -- see "Isolated runs" below. |
| The shared lab settings file `mousereach_lab_config.json` in the `nas_root` each of those names | every path in it (for example the hand-scores file's folder). |
| Environment variables whose names start with `MOUSEREACH_`, `MOUSEDB_` or `MOUSEBRAIN_`, or are `CONNECTOME_ROOT` -- **in any mix of capital and small letters** | every absolute path in them. (A machine may set `MouseReach_PROCESSING_ROOT` and `MouseReach_NAS_DRIVE`; MouseReach reads those, so they count.) |
| The same variables as Windows keeps them for your account and for the machine (System Properties > Environment Variables) | the same. An isolated run removes them from its own copy of the environment, but Windows still has them, so the guard still reads them. |
| What MouseReach's own settings resolved when the tests started | the shared pipeline folder, the processing folder and the network drive. |
| A list **you** give (optional, see below) | every folder you list. |

Each real root is recognized however a path to it is written: capital or small
letters, `/` or `\`, a path relative to the current folder, the `\\?\` long-path
form, Windows short names (`PROGRA~1`), and -- for a mapped network drive --
both the drive letter (`Q:\...`) and the network form (`\\server\share\...`).

## What it refuses

Inside a real root, these changes are refused **before** they reach the disk
(see "What it cannot see" below for the limits):

- opening a file to write, append, create or update it (also the hidden
  temporary files programs write first and then rename into place);
- creating a folder (even one that already exists), creating a temporary file
  or folder;
- renaming or moving a file in or out, replacing a file, deleting a file or a
  folder, changing a file's time or permissions;
- copying a file **into** it (copying a file *out of* it is reading, which is
  allowed);
- opening a database file in it (opening one read-only is reading, and is
  allowed).

Reading is always allowed.

## What it cannot see

The guard watches what the Python code of the test run asks for. It cannot
see:

- writes made *inside* compiled libraries that open files on their own (for
  example a video writer, an HDF5 library or a graphics-card library);
- writes made by **other programs** a test starts (for example `ffmpeg`, or a
  second Python).

A test that uses such a library or program must still point it at its own
temporary folder. The guard is a safety net for the common case, not a proof
that nothing outside Python wrote anything.

## What it does not guard

- **The MouseReach folder itself** (its source code, tests, documents and the
  `__pycache__` folders Python writes there), even when that folder sits inside
  a real root. The folders *next to* it stay guarded.
- **The computer's temporary folder**, where the tests build their own
  folders, and the Python installation running the tests.

An exception wins only when it lies *inside* the real root. If a real root is
itself inside the temporary folder, it is still guarded.

## What a refusal looks like

The test that tried to write fails, and pytest prints:

```
this test tried to write into this machine's REAL pipeline folders (the test
guard refused it); point config.Paths (and the processing root) at tmp_path
instead:
  os.mkdir: q:\lab\pipeline\processing\review\.return_claims
```

Each line names the kind of change and the exact folder or file. Nothing was
written. If the attempt happened outside any test (while the test files were
being loaded), the run ends with a red **test guard** section listing it, and
the whole run counts as failed.

What to do: the test (not the guard) is wrong. It must point every folder it
uses at its own temporary folder (`tmp_path`). The line tells you which folder
it forgot.

## Isolated runs

The safest way to run the tests is *isolated*: the home folder variables
(`USERPROFILE`, `HOME`, `APPDATA`, `LOCALAPPDATA`) point at a scratch folder,
and every `MOUSEREACH_`, `MOUSEDB_`, `MOUSEBRAIN_` and `CONNECTOME_ROOT`
variable is removed -- in any letter case. Then the tests cannot even find the
real settings.

The guard still protects an isolated run: it reads the settings file in the
account's real home folder (Windows keeps that location separately from the
`USERPROFILE` variable), so the real roots are known and refused even though
the tests themselves no longer see them.

## Telling the guard about more folders

If a folder holds real data but none of the sources above names it (for
example a run on a machine without MouseReach configured, or a copy of the lab
drive mounted somewhere unusual), list it yourself before running the tests.
Either or both of:

- `REACH_TEST_GUARD_ROOTS` -- folders separated by `;` on Windows (`:` on
  macOS and Linux);
- `REACH_TEST_GUARD_ROOTS_FILE` -- the path of a plain text file with one
  folder per line; anything after `#` on a line is a comment.

These names deliberately do not start with `MOUSEREACH_`, so an isolated run,
which removes those, keeps them. If the file cannot be read, the guard says so
at the start of the run.

Example (Windows Command Prompt):

```
set REACH_TEST_GUARD_ROOTS_FILE=C:\path\to\my_real_roots.txt
python -m pytest tests
```

## pytest's cache

pytest normally keeps a `.pytest_cache` folder (and, when that fails, leftover
`pytest-cache-files-*` folders) in the folder it runs in. MouseReach turns that
off (`-p no:cacheprovider` in `pyproject.toml`), so running the tests leaves no
such folder in your MouseReach folder.

## Where it is in the code

`tests/real_root_guard.py` (what is collected and refused, and why) and
`tests/conftest.py` (where it is loaded and how a refusal fails a test). Its
own tests: `tests/test_real_share_guard.py`,
`tests/test_guard_refuses_real_writes.py` and
`tests/test_guard_holds_in_an_isolated_run.py`.
