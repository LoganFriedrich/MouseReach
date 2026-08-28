"""Follow every field the pipeline claims to produce, and find the ones that vanish.

WHY THIS EXISTS
---------------
Four times in one day, a value the pipeline computes turned out never to reach
the data: the segment number, the kinematics version, the assignment version,
and -- worst -- which reach caused each pellet's fate. Each was found by
accident, one at a time. Then a sweep of the database columns found a second
family: measurements that are declared in the schema and never computed at all,
including reach extent and grasp aperture.

They all look the same from the outside. A field exists everywhere you look --
in the schema, in the file, as a database column -- and is empty in all of them.
Nothing errors. Nothing is logged. The analysis just quietly has a hole in it.

This walks every field from the stage that should produce it, through the
features file, into the database, and says which of those steps it survives.
Run it after any pipeline change: a field that drops out of "populated" is a
regression, and this is the only check that can see it.

WHAT IT CANNOT DO
-----------------
It cannot tell you a field is CORRECT, only that it is present. A field full of
plausible wrong numbers passes. It also cannot tell a deliberate absence from a
defect -- a stage that legitimately declines to decide looks identical to one
that forgot to. Both need a human to say which.

READING THE DATABASE
--------------------
It reads the parquet snapshot, never connectome.db. The database lives on a
network share in a mode where a writer blocks readers outright, so while the
watcher is running any read fails partway through.

USAGE
-----
  python -m mousereach.pipeline.field_audit
  python -m mousereach.pipeline.field_audit --limit 200      # quick pass
  python -m mousereach.pipeline.field_audit --json out.json  # machine-readable

ASCII-only console output (Windows consoles cannot print Unicode).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

def _analyzed_default():
    try:
        from mousereach.config import Paths
        return Path(Paths.ANALYZED_OUTPUT) if Paths.ANALYZED_OUTPUT else None
    except Exception:
        return None


ANALYZED = _analyzed_default()  # configured Analyzed tree (mousereach-setup); --root overrides
# An external database snapshot (parquet) is OPTIONAL and belongs to whatever
# integrator produced it; point at it with MOUSEREACH_SNAPSHOT_DIR or --snapshot.
import os as _os
SNAPSHOT = Path(_os.environ["MOUSEREACH_SNAPSHOT_DIR"]) if _os.environ.get("MOUSEREACH_SNAPSHOT_DIR") else None

# Which stage output holds which level of thing, and how to reach the items.
# "reach" level items end up as rows in reach_data; "segment" level items
# describe the pellet presentation those reaches belong to.
STAGE_OUTPUTS = [
    ("segmentation", "_segments.json", "segment"),
    ("reach detection", "_reaches.json", "reach"),
    ("outcome detection", "_pellet_outcomes.json", "segment"),
    ("reach assignment", "_reach_assignments.json", "reach"),
    ("kinematics", "_features.json", "reach"),
]

# A value that is present but means "nothing here".
EMPTY = (None, "", [], {})

# Values that are present, not empty, and still carry no information: a boolean
# that is always False, a count that is always 0. Counting these as "populated"
# is how a field can look healthy while being useless -- causal_reach is False on
# every reach in the corpus and would otherwise audit as 100% present. What
# matters is whether a field ever VARIES.
NO_INFORMATION = (False, 0, 0.0)

# How many distinct values to remember per field before giving up counting; we
# only need to know "one" from "more than one".
_DISTINCT_CAP = 5


def items_of(doc: dict, level: str) -> List[dict]:
    """Pull the per-reach or per-segment records out of a stage output file.

    The files disagree about shape: some put reaches at the top level, others
    nest them under segments. Handle both rather than assuming.
    """
    segs = doc.get("segments")
    if level == "segment":
        return [s for s in (segs or []) if isinstance(s, dict)]

    top = doc.get("reaches")
    if isinstance(top, list) and top:
        return [r for r in top if isinstance(r, dict)]
    out = []
    for s in (segs or []):
        if isinstance(s, dict):
            out.extend([r for r in (s.get("reaches") or []) if isinstance(r, dict)])
    return out


def tally(items: List[dict], present: Dict[str, int], populated: Dict[str, int],
          total: Dict[str, int], informative: Dict[str, int] = None,
          distinct: Dict[str, set] = None) -> None:
    """Count, per field, how often the key exists, holds a value, and says anything.

    "Populated" is not enough. A boolean that is always False is populated on
    every row and tells you nothing. "Informative" excludes those, and the
    distinct-value set catches the rest: a field with exactly one distinct value
    across the whole corpus is either genuinely constant or never filled in.
    """
    seen_keys = set()
    for it in items:
        seen_keys.update(it.keys())
    for k in seen_keys:
        total[k] += len(items)
    for it in items:
        for k, v in it.items():
            present[k] += 1
            if v not in EMPTY:
                populated[k] += 1
                if informative is not None and not any(
                        v is n or (type(v) is type(n) and v == n) for n in NO_INFORMATION):
                    informative[k] += 1
            if distinct is not None and len(distinct[k]) < _DISTINCT_CAP:
                try:
                    distinct[k].add(v if isinstance(v, (str, int, float, bool, type(None)))
                                    else "<complex>")
                except TypeError:
                    distinct[k].add("<unhashable>")


def scan_files(root: Path, limit: Optional[int] = None,
               only: Optional[set] = None) -> dict:
    """Walk the finished videos and tally every field of every stage output."""
    result = {}
    for stage, suffix, level in STAGE_OUTPUTS:
        present: Dict[str, int] = defaultdict(int)
        populated: Dict[str, int] = defaultdict(int)
        informative: Dict[str, int] = defaultdict(int)
        distinct: Dict[str, set] = defaultdict(set)
        total: Dict[str, int] = defaultdict(int)
        n_files = 0
        paths = sorted(root.rglob("*" + suffix))
        for p in paths:
            stem = p.name[: -len(suffix)]
            if only is not None and stem not in only:
                continue
            if limit and n_files >= limit:
                break
            try:
                doc = json.loads(p.read_text())
            except Exception:
                continue
            n_files += 1
            tally(items_of(doc, level), present, populated, total,
                  informative, distinct)
        result[stage] = {
            "suffix": suffix, "level": level, "files": n_files,
            "fields": {k: {"present": present[k], "populated": populated[k],
                           "informative": informative[k],
                           "distinct": len(distinct[k]), "of": total[k]}
                       for k in present},
        }
    return result


def scan_database(snapshot: Path, only: Optional[set] = None) -> dict:
    """Populated fraction of every reach_data column, from the parquet snapshot."""
    import pandas as pd

    f = snapshot / "reach_data.parquet"
    if not f.exists():
        return {}
    rd = pd.read_parquet(f)
    if only is not None and "video_name" in rd.columns:
        rd = rd[rd.video_name.isin(only)]
    out = {}
    for c in rd.columns:
        col = rd[c]
        nn = int(col.notna().sum())
        if nn and col.dtype == object:
            nn = int((col.notna() & (col.astype(str) != "")).sum())
        # Informative = present AND not one of the values that mean "nothing
        # here" even though they are not null. A column of all False or all 0
        # is fully populated and completely uninformative.
        try:
            inf = int((col.notna() & (col != 0) & (col != False)).sum())
        except Exception:
            inf = nn
        out[c] = {"populated": nn, "informative": inf, "of": int(len(rd)),
                  "distinct": int(col.nunique(dropna=True))}
    return out


def classify(field: str, producers: List[tuple], in_features: Optional[dict],
             in_db: Optional[dict]) -> tuple:
    """Decide what happened to one field. Returns (verdict, explanation)."""
    def says_something(d):
        """Does this field ever carry information, rather than merely exist?"""
        if not d:
            return False
        if d.get("informative") is not None:
            return d["informative"] > 0
        return d["populated"] > 0

    made = [(stage, d) for stage, d in producers if says_something(d)]
    declared = [(stage, d) for stage, d in producers if not says_something(d)]

    feat_ok = says_something(in_features)
    db_ok = says_something(in_db)
    known_to_db = in_db is not None

    if not made and declared:
        where = ", ".join(s for s, _ in declared)
        # Distinguish "the key is absent/null" from "the key is there and always
        # says the same nothing" -- they need different fixes.
        always_same = any(d.get("populated", 0) > 0 for _, d in declared)
        how = ("written by %s and always carrying the same empty value "
               "(False / 0)" % where) if always_same else               ("written by %s and left null on every item" % where)
        return ("NEVER COMPUTED", "the key is " + how)

    if not made and not declared:
        # No upstream stage carries it. If the features file has it, kinematics
        # computed it itself -- that is normal, not a defect.
        if feat_ok:
            if known_to_db and not db_ok:
                return ("LOST IN TRANSIT",
                        "kinematics computes it, the database column is empty")
            if not known_to_db:
                return ("NOT IN THE DATABASE",
                        "kinematics computes it, no database column holds it")
            return ("OK", "computed by kinematics through to the database")
        if in_features is not None and not feat_ok:
            return ("NEVER COMPUTED",
                    "declared in the features schema and never given a value")
        if known_to_db and not db_ok:
            return ("NEVER COMPUTED",
                    "a database column that nothing ever fills")
        if known_to_db and db_ok:
            return ("OK", "filled by the sync itself (identifiers and provenance)")
        return ("NOT PRODUCED", "no stage output carries this field")

    src = ", ".join(s for s, _ in made)
    if not feat_ok and any(s != "kinematics" for s, _ in made):
        return ("LOST IN TRANSIT",
                "%s produces it, the features file does not carry it" % src)
    if feat_ok and known_to_db and not db_ok:
        return ("LOST IN TRANSIT",
                "the features file carries it, the database column is empty")
    if feat_ok and not known_to_db:
        return ("NOT IN THE DATABASE",
                "computed and written to file, but no database column holds it")
    if db_ok:
        return ("OK", "populated from %s through to the database" % src)
    return ("UNCLEAR", "produced by %s; could not follow it further" % src)


def audit(root: Path = ANALYZED, snapshot: Path = SNAPSHOT,
          limit: Optional[int] = None, only: Optional[set] = None) -> dict:
    files = scan_files(root, limit=limit, only=only)
    db = scan_database(snapshot, only=only)

    feature_fields = files.get("kinematics", {}).get("fields", {})
    producers_by_field: Dict[str, List[tuple]] = defaultdict(list)
    for stage, info in files.items():
        if stage == "kinematics":
            continue
        for field, d in info["fields"].items():
            producers_by_field[field].append((stage, d))

    names = set(producers_by_field) | set(feature_fields) | set(db)
    findings = {}
    for field in sorted(names):
        verdict, why = classify(field, producers_by_field.get(field, []),
                                feature_fields.get(field), db.get(field))
        findings[field] = {
            "verdict": verdict, "why": why,
            "producers": {s: d for s, d in producers_by_field.get(field, [])},
            "features": feature_fields.get(field),
            "database": db.get(field),
        }
    return {"files": files, "findings": findings}


def _pct(d: Optional[dict]) -> str:
    """Percentage of items where the field says something, not merely exists."""
    if not d or not d.get("of"):
        return "    -"
    n = d.get("informative")
    if n is None:
        n = d.get("populated", 0)
    return "%5.1f%%" % (n / d["of"] * 100)


VERDICT_NOTES = {
    "NEVER COMPUTED":
        "The field exists in the schema and nothing ever gives it a value. "
        "Nothing errors; the analysis simply has a hole where this measurement "
        "should be.",
    "LOST IN TRANSIT":
        "A stage computes this and writes it to its own output file, and the "
        "next stage reads it from somewhere else, so it never arrives. The "
        "answer exists on disk and is thrown away.",
    "NOT IN THE DATABASE":
        "Computed and written to the features file, but no database column "
        "holds it. Fine if deliberate -- a problem if you expected to query it.",
    "OK": "Populated end to end.",
}


def to_markdown(res: dict, corpus: str, _doc_dir=None) -> str:
    """Render the audit as the document that goes in docs/.

    Generated rather than written by hand, for the same reason the figures'
    companion is: a description of what the pipeline produces has to be
    regenerated from the pipeline, or it drifts and starts lying.
    """
    from collections import defaultdict as _dd
    by = _dd(list)
    for field, f in res["findings"].items():
        by[f["verdict"]].append((field, f))

    import subprocess as _sp

    def _git(args, cwd=None):
        try:
            return _sp.run(["git", *args], cwd=cwd, capture_output=True,
                           text=True, check=False).stdout.strip()
        except Exception:
            return ""

    # Stamp the repository the DOCUMENT lives in, not whatever directory this
    # happens to be run from. Run from mousedb, the old code stamped mousedb's
    # commit -- so regenerating after a MouseReach change produced a
    # byte-identical file and the documentation check read it as never updated.
    _doc_repo = str(_doc_dir) if _doc_dir else None
    _sha = _git(["rev-parse", "--short", "HEAD"], cwd=_doc_repo) or "unknown"
    _when = _git(["log", "-1", "--format=%ad", "--date=short"], cwd=_doc_repo)

    L = ["# What the pipeline actually produces, field by field", "",
         # The stamp the documentation check reads. Emitted here because a
         # generated document that loses its stamp on every regeneration would
         # register as permanently out of date.
         "Describes: every stage output plus src/mousereach/sync/database.py",
         "Verified against: %s (%s)" % (_sha, _when), "",
         "Generated by `python -m mousereach.pipeline.field_audit --markdown`.",
         "Do not edit by hand -- re-run it.", "",
         corpus, "",
         "A field is counted as present only if it ever VARIES. A boolean that is "
         "False on every row, or a count that is always zero, is fully populated "
         "and says nothing; the first version of this audit called those healthy, "
         "which is exactly the mistake it exists to catch.", "",
         "Percentages are the share of items where the field carries information, "
         "at the stage that should produce it, in the features file, and in the "
         "database.", ""]

    for verdict in ["NEVER COMPUTED", "LOST IN TRANSIT", "NOT IN THE DATABASE", "OK"]:
        rows = by.get(verdict)
        if not rows:
            continue
        L += ["## %s (%d)" % (verdict.title(), len(rows)), "",
              VERDICT_NOTES.get(verdict, ""), "",
              "| field | stage | features | database | note |",
              "|---|---|---|---|---|"]
        for field, f in sorted(rows):
            prod = next(iter(f["producers"].values()), None)
            note = "" if verdict == "OK" else f["why"]
            L.append("| `%s` | %s | %s | %s | %s |"
                     % (field, _pct(prod).strip(), _pct(f["features"]).strip(),
                        _pct(f["database"]).strip(), note))
        L.append("")
    return chr(10).join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Follow every pipeline field from the stage that produces it "
                    "into the database, and report the ones that vanish.")
    ap.add_argument("--root", type=Path, default=ANALYZED,
                    help="Tree of finished videos to read (default: Analyzed)")
    ap.add_argument("--snapshot", type=Path, default=SNAPSHOT,
                    help="Directory holding reach_data.parquet")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only read this many files per stage (quick pass)")
    ap.add_argument("--only-videos", type=Path, default=None,
                    help="Restrict to the video ids listed in this text file (one per "
                         "line); an integrator can produce such a list, MouseReach "
                         "does not depend on one")
    ap.add_argument("--json", type=Path, default=None,
                    help="Also write the full result as JSON")
    ap.add_argument("--markdown", type=Path, default=None,
                    help="Also write the report as a markdown document")
    args = ap.parse_args(argv)

    only = None
    if args.only_videos:
        only = {ln.strip() for ln in args.only_videos.read_text(encoding="utf-8").splitlines() if ln.strip()}
        print("restricting to %d listed videos" % len(only))

    res = audit(args.root, args.snapshot, limit=args.limit, only=only)

    print("\nfiles read per stage:")
    for stage, info in res["files"].items():
        print("   %-18s %-26s %6d files" % (stage, info["suffix"], info["files"]))

    order = ["NEVER COMPUTED", "LOST IN TRANSIT", "NOT IN THE DATABASE",
             "NOT PRODUCED", "UNCLEAR", "OK"]
    by_verdict = defaultdict(list)
    for field, f in res["findings"].items():
        by_verdict[f["verdict"]].append((field, f))

    for verdict in order:
        rows = by_verdict.get(verdict)
        if not rows:
            continue
        print("\n" + "=" * 78)
        print("%s  (%d)" % (verdict, len(rows)))
        print("=" * 78)
        print("   %-34s %7s %7s %7s" % ("field", "stage", "feature", "db"))
        for field, f in sorted(rows):
            prod = next(iter(f["producers"].values()), None)
            print("   %-34s %7s %7s %7s   %s"
                  % (field[:34], _pct(prod), _pct(f["features"]),
                     _pct(f["database"]), f["why"] if verdict != "OK" else ""))

    counts = {v: len(by_verdict.get(v, [])) for v in order if by_verdict.get(v)}
    print("\nsummary: " + ", ".join("%s %d" % (k.lower(), n) for k, n in counts.items()))

    if args.json:
        args.json.write_text(json.dumps(res, indent=2, default=str))
        print("wrote %s" % args.json)
    if args.markdown:
        n_files = max((i["files"] for i in res["files"].values()), default=0)
        corpus = ("Read over **%d videos** that are finished and current at every "
                  "stage. The database side comes from the parquet snapshot, never "
                  "from connectome.db while a watcher is running." % n_files)
        args.markdown.write_text(
            to_markdown(res, corpus, _doc_dir=Path(args.markdown).resolve().parent),
            encoding="utf-8")
        print("wrote %s" % args.markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
