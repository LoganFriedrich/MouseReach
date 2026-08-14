"""Bring ASPA multi-animal collages into the MouseReach pipeline.

What this does
--------------
Copies ASPA collages out of the read-only historical archive into the watcher's
intake (``Unanalyzed/Multi-Animal``), renaming each one into pipeline form on the
way (see :mod:`mousereach.aspa.identity`). Originals and the old ASPA analyses are
never touched -- this only ever READS the archive.

Scope: cohorts D and later
--------------------------
Only ``OptD`` .. ``OptG`` plus ``H`` .. ``M`` are in scope. Excluded:

- ``ABS1/2/3``, ``GI-A,B``, ``OptA/B/C`` -- the earlier generation, whose collages
  are named with animal RANGES (``20210924_ABS2_13-15_f``) rather than the eight
  ids the pipeline needs.
- ``N1``/``N2`` -- despite the later letters these are Dec 2021 / Jan 2022,
  i.e. EARLIER than OptD, and share the older malformed naming.

("Opt" in these directory names is a known misnomer; the cohorts are just D..G.)

Source selection: one file per session
--------------------------------------
A session can have several files on disk that differ only in trailing text::

    20220217_D01,...,D08_P1 uncropped.mkv     1.70 GB   Feb 2022  <- original
    20220217_D01,...,D08_P1.mkv               1.21 GB   Jan 2025  <- trimmed
    20220217_D01,...,D08_P1-proj.llc          347 B     Jan 2025  <- LosslessCut project

Those encode to the SAME name, so exactly one must be chosen. "uncropped" here
means *not trimmed in time*: someone opened the original in LosslessCut and cut it
down to the actual session. That matters -- cameras left running past the end of a
session produce junk frames that wreck DLC reference coverage and segmentation --
so the trimmed file is the one we want.

Preference: the clean ``_P{n}.mkv`` wins. If a session has no trimmed version, the
original is the only candidate and is used. LosslessCut ``-seg`` exports, ``(2)``
duplicates and ``.llc`` project files are always skipped.

Usage
-----
    python -m mousereach.aspa.import_collages            # dry run (default)
    python -m mousereach.aspa.import_collages --apply    # actually copy

Dry run is the default on purpose: it prints exactly what would be copied,
skipped, and why, plus every session where more than one source competed.

ASCII-only console output (Windows cp1252).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .identity import encode_collage_stem

# Cohort directories in scope (see module docstring for the exclusions).
IN_SCOPE_COHORTS = ["OptD", "OptE", "OptF", "OptG", "H", "I", "J", "K", "L", "M"]

VIDEO_EXTENSIONS = {".mkv", ".mp4"}

# {date}_{animals}_{tray}{position} followed by any trailing junk.
_SESSION = re.compile(r"^(\d{8}_[^_]+_[A-Za-z]\d+)(.*)$")

# Trailing-junk classification -> (kind, rank). Lower rank wins.
_CLEAN, _UNCROPPED, _REJECT = 0, 1, 99


def default_source_root() -> Optional[Path]:
    from ..config import Paths
    root = Paths.NAS_ROOT
    return (Path(root) / "Archive" / "historical" / "ASPA") if root else None


def default_dest() -> Optional[Path]:
    from ..config import Paths
    return Path(Paths.MULTI_ANIMAL_SOURCE) if Paths.MULTI_ANIMAL_SOURCE else None


def classify(trailing: str) -> Tuple[str, int]:
    """Classify a filename's trailing text into (kind, preference rank).

    Everything after ``{date}_{animals}_{tray}{position}`` is trailing text. It
    carries no information the pipeline needs, but it DOES decide which of several
    competing files is the right source for the session.
    """
    t = (trailing or "").strip().lower()
    if t == "":
        return "clean", _CLEAN
    if "uncropped" in t:
        return "uncropped", _UNCROPPED
    if re.search(r"-\d{2}\.\d{2}\.\d{2}", t) or "seg" in t:
        return "losslesscut_segment", _REJECT
    if re.match(r"^\s*\(\d+\)$", t):
        return "duplicate", _REJECT
    return "other", _REJECT


def scan(source_root: Path, cohorts: List[str]) -> Tuple[Dict[str, List[dict]], List[dict]]:
    """Group candidate source files by the session they belong to.

    Returns ``(by_session, unparseable)``. ``by_session`` maps the ENCODED stem to
    its competing candidates; ``unparseable`` holds files whose names do not have
    the expected shape (reported, never guessed at).
    """
    by_session: Dict[str, List[dict]] = defaultdict(list)
    unparseable: List[dict] = []

    for cohort in cohorts:
        multi = source_root / cohort / "Multi-Animal"
        if not multi.is_dir():
            continue
        for p in sorted(multi.iterdir()):
            if not p.is_file() or p.suffix.lower() not in VIDEO_EXTENSIONS:
                continue  # .llc project files and anything non-video
            m = _SESSION.match(p.stem)
            if not m:
                unparseable.append({"cohort": cohort, "path": p, "reason": "name shape"})
                continue
            session_stem, trailing = m.group(1), m.group(2)
            encoded = encode_collage_stem(session_stem)
            if not encoded:
                unparseable.append({"cohort": cohort, "path": p, "reason": "animal ids"})
                continue
            kind, rank = classify(trailing)
            by_session[encoded].append({
                "cohort": cohort, "path": p, "kind": kind, "rank": rank,
                "size": p.stat().st_size, "source_stem": p.stem,
            })
    return by_session, unparseable


def choose(candidates: List[dict]) -> Optional[dict]:
    """Pick the one source file for a session; None if every candidate is rejected.

    Preference is by rank (clean beats uncropped); ties break on the larger file,
    which is only reachable when two equally-clean files exist.
    """
    usable = [c for c in candidates if c["rank"] != _REJECT]
    if not usable:
        return None
    usable.sort(key=lambda c: (c["rank"], -c["size"]))
    return usable[0]


def build_plan(source_root: Path, dest: Path, cohorts: List[str]) -> dict:
    by_session, unparseable = scan(source_root, cohorts)
    plan, contested, dropped = [], [], []

    for encoded_stem in sorted(by_session):
        candidates = by_session[encoded_stem]
        pick = choose(candidates)
        if pick is None:
            dropped.append({"stem": encoded_stem, "candidates": candidates})
            continue
        if len(candidates) > 1:
            contested.append({"stem": encoded_stem, "chosen": pick, "candidates": candidates})
        plan.append({
            "encoded_stem": encoded_stem,
            "src": pick["path"],
            "dst": dest / f"{encoded_stem}{pick['path'].suffix.lower()}",
            "cohort": pick["cohort"],
            "kind": pick["kind"],
            "size": pick["size"],
            "source_stem": pick["source_stem"],
        })
    return {"plan": plan, "contested": contested, "dropped": dropped,
            "unparseable": unparseable}


def write_mapping(plan: List[dict], dest_root: Path) -> Path:
    """Write the versioned source -> encoded mapping next to the imported videos.

    The encoding is a rule and reconstructs itself, so nothing depends on this
    file -- it exists so the provenance of every imported video is recorded
    explicitly rather than inferred.
    """
    out = dest_root / "ASPA_import_mapping.json"
    existing = {}
    if out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    entries = existing.get("entries", {})
    for item in plan:
        entries[item["encoded_stem"]] = {
            "source_path": str(item["src"]),
            "source_stem": item["source_stem"],
            "cohort_dir": item["cohort"],
            "selected_because": item["kind"],
            "bytes": item["size"],
        }
    out.write_text(json.dumps({
        "schema_version": "1.0",
        "description": "ASPA historical collages imported into the MouseReach pipeline. "
                       "Encoded ids decode by rule: cohort number = alphabet position.",
        "updated_at": datetime.now().isoformat(),
        "entries": entries,
    }, indent=2), encoding="utf-8")
    return out


def _fmt_gb(n: int) -> str:
    return f"{n / (1024 ** 3):.1f} GB"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="mousereach-aspa-import",
        description=(
            "Copy ASPA historical collages (cohorts D and later) into the watcher's "
            "intake, renamed into pipeline form. The archive is only ever READ -- "
            "originals and old ASPA analyses are never modified. Dry run by default."),
        epilog="Examples:\n"
               "  python -m mousereach.aspa.import_collages\n"
               "  python -m mousereach.aspa.import_collages --apply\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true",
                        help="Actually copy. Without this, prints the plan and changes nothing.")
    parser.add_argument("--source", default=None, help="ASPA archive root (default: NAS Archive/historical/ASPA).")
    parser.add_argument("--dest", default=None, help="Intake dir (default: Unanalyzed/Multi-Animal).")
    parser.add_argument("--cohorts", default=None,
                        help="Comma-separated cohort dirs to import (default: %s)." % ",".join(IN_SCOPE_COHORTS))
    parser.add_argument("--limit", type=int, default=0, help="Only act on the first N sessions (testing).")
    args = parser.parse_args(argv)

    source_root = Path(args.source) if args.source else default_source_root()
    dest = Path(args.dest) if args.dest else default_dest()
    cohorts = args.cohorts.split(",") if args.cohorts else IN_SCOPE_COHORTS

    if not source_root or not source_root.is_dir():
        print(f"[FAIL] ASPA archive not found: {source_root}")
        return 1
    if not dest:
        print("[FAIL] Could not resolve the intake dir (is this machine configured?).")
        return 1

    print(f"Source: {source_root}")
    print(f"Dest:   {dest}")
    print(f"Cohorts: {', '.join(cohorts)}")
    print("Scanning...")
    result = build_plan(source_root, dest, cohorts)
    plan = result["plan"]
    if args.limit:
        plan = plan[:args.limit]

    total = sum(p["size"] for p in plan)
    print("")
    print(f"Sessions to import: {len(plan)}  ({_fmt_gb(total)})")
    by_cohort = defaultdict(int)
    for p in plan:
        by_cohort[p["cohort"]] += 1
    for c in cohorts:
        if by_cohort.get(c):
            print(f"    {c:<6} {by_cohort[c]:5d}")

    if result["contested"]:
        print("")
        print(f"Sessions with more than one candidate: {len(result['contested'])}")
        for c in result["contested"][:10]:
            print(f"  {c['stem']}")
            for cand in sorted(c["candidates"], key=lambda x: x["rank"]):
                mark = "->" if cand is c["chosen"] else "  "
                print(f"    {mark} {cand['kind']:<20} {_fmt_gb(cand['size']):>9}  {cand['source_stem'][:60]}")
        if len(result["contested"]) > 10:
            print(f"  ... and {len(result['contested']) - 10} more")

    if result["dropped"]:
        print("")
        print(f"[!] Sessions with NO usable candidate: {len(result['dropped'])}")
        for d in result["dropped"][:5]:
            print(f"  {d['stem']}  ({', '.join(c['kind'] for c in d['candidates'])})")

    if result["unparseable"]:
        print("")
        print(f"[!] Files whose names could not be parsed: {len(result['unparseable'])}")
        for u in result["unparseable"][:10]:
            print(f"  [{u['reason']}] {u['path'].name[:80]}")
        if len(result["unparseable"]) > 10:
            print(f"  ... and {len(result['unparseable']) - 10} more")

    already = [p for p in plan if p["dst"].exists()]
    if already:
        print("")
        print(f"Already present at the destination (will skip): {len(already)}")

    if not args.apply:
        print("")
        print("DRY RUN -- nothing copied. Re-run with --apply to perform the import.")
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    copied = skipped = failed = 0
    for n, item in enumerate(plan, 1):
        if item["dst"].exists():
            skipped += 1
            continue
        try:
            tmp = item["dst"].with_suffix(item["dst"].suffix + ".partial")
            shutil.copy2(item["src"], tmp)
            tmp.rename(item["dst"])
            copied += 1
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {item['encoded_stem']}: {e}")
        if n % 25 == 0 or n == len(plan):
            print(f"  {n}/{len(plan)}  copied={copied} skipped={skipped} failed={failed}")

    mapping = write_mapping(plan, dest)
    print("")
    print(f"[OK] copied={copied} skipped={skipped} failed={failed}")
    print(f"Mapping written: {mapping}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
