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

ANALYZED = Path(r"Y:/LAB_ROOT/Behavior/MouseReach_Pipeline/Analyzed")
SNAPSHOT = Path(r"C:/LAB_ROOT/_analysis_snapshot")

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
          total: Dict[str, int]) -> None:
    """Count, per field, how often the key exists and how often it holds a value."""
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


def scan_files(root: Path, limit: Optional[int] = None,
               only: Optional[set] = None) -> dict:
    """Walk the finished videos and tally every field of every stage output."""
    result = {}
    for stage, suffix, level in STAGE_OUTPUTS:
        present: Dict[str, int] = defaultdict(int)
        populated: Dict[str, int] = defaultdict(int)
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
            tally(items_of(doc, level), present, populated, total)
        result[stage] = {
            "suffix": suffix, "level": level, "files": n_files,
            "fields": {k: {"present": present[k], "populated": populated[k],
                           "of": total[k]} for k in present},
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
        s = rd[c]
        nn = int(s.notna().sum())
        if nn and s.dtype == object:
            nn = int((s.notna() & (s.astype(str) != "")).sum())
        out[c] = {"populated": nn, "of": int(len(rd)),
                  "distinct": int(s.nunique(dropna=True))}
    return out


def classify(field: str, producers: List[tuple], in_features: Optional[dict],
             in_db: Optional[dict]) -> tuple:
    """Decide what happened to one field. Returns (verdict, explanation)."""
    made = [(stage, d) for stage, d in producers if d["populated"] > 0]
    declared = [(stage, d) for stage, d in producers if d["populated"] == 0]

    feat_ok = bool(in_features and in_features["populated"] > 0)
    db_ok = bool(in_db and in_db["populated"] > 0)
    known_to_db = in_db is not None

    if not made and declared:
        where = ", ".join(s for s, _ in declared)
        return ("NEVER COMPUTED",
                "the key is written by %s and left empty on every item" % where)

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
    if not d or not d.get("of"):
        return "    -"
    return "%5.1f%%" % (d["populated"] / d["of"] * 100)


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
    ap.add_argument("--finished-only", action="store_true",
                    help="Restrict to videos that are finished and current at "
                         "every stage (needs mousedb)")
    ap.add_argument("--json", type=Path, default=None,
                    help="Also write the full result as JSON")
    args = ap.parse_args(argv)

    only = None
    if args.finished_only:
        try:
            from mousedb.analyzable import finished_videos
            only = finished_videos()
            print("restricting to %d finished, current videos" % len(only))
        except Exception as e:
            print("[!] could not load the finished-video list (%s); reading everything" % e)

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
