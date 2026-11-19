#!/usr/bin/env python3
"""Quick summary: the filed results per session, plus overall totals.

Redirected 2026-11-19 to ``mousereach-results --level session``
(export/results.py). WHY: this command read only the local Processing folder,
which filing empties, so after filing it found nothing; and it counted a
session's success rate over every segment, a segment with no pellet
included. It now writes the session-level export of the FILED results
(retrieved / pellets presented) and prints the overall totals from it.
"""
import sys


def main(argv=None):
    import argparse
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    from mousereach.export.results import export_results
    try:
        res = export_results("session")
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    for p in res["problems"]:
        print(f"[!] {p}")
    if not res["path"]:
        print(f"No filed results under {res['root']}.")
        return 1
    import pandas as pd
    df = pd.read_csv(res["path"])
    pres = df["session_pellets_presented"].fillna(0).sum()
    got = df["session_retrieved"].fillna(0).sum()
    print("=" * 60)
    print("MOUSEREACH SUMMARY (filed results)")
    print("=" * 60)
    print(f"Sessions (trays): {len(df)}")
    print(f"Pellets presented: {int(pres)}")
    print(f"Retrieved: {int(got)}")
    print(f"Reaches: {int(df['session_n_reaches'].fillna(0).sum())}")
    if pres:
        print(f"OVERALL SUCCESS RATE: {100.0 * got / pres:.1f}% (retrieved / presented)")
    print(f"Per-session CSV: {res['path']}")
    print("Other levels: mousereach-results --level reach|pellet|day|animal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
