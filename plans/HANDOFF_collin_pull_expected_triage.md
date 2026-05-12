# Handoff: Pull `expected_triage` toggle into collin's MouseReach tools

Audience: a Claude Code session running on collin (schultzc)'s machine.
Goal: bring 3 files in collin's MouseReach runtime up to date with the new
`expected_triage` toggle that's now committed in the shared NAS git repo.
Scope: STRICTLY LIMITED. Pull only the 3 listed files at the listed commit.
Do NOT merge or pull any other in-progress work from the same branch.

Originator handoff (parent doc, same dir, optional read for context):
`HANDOFF_gt_tool_expected_triage.md`

Date: 2026-05-04.

---

## What this update adds (summary)

A new "Expected Triage" toggle on outcome segments, surfaced in two places:

- **GroundTruthWidget** -- powers the "GT Tool" tab AND the "3 - Review Tool"
  tab in the bare `mousereach` launcher (one class, two `review_mode` modes).
- **PelletOutcomeAnnotatorWidget** -- powers the "Step 3b - Review Outcomes"
  tab.

When ON, the toggle stamps `expected_triage_flagged_by` (current user, uppercase)
and `expected_triage_flagged_at` (UTC ISO timestamp matching the corpus
convention `YYYY-MM-DDTHH:MM:SS.ffffff+00:00`). When OFF, the boolean clears
but the `_flagged_by`/`_flagged_at` audit trail stays in place. An optional
free-text reason field (`expected_triage_reason`) appears only when the toggle
is ON.

Field is backward-compatible: missing on read -> False. The cascade evaluator
already reads `seg.get("expected_triage", False)`, so no downstream changes
needed there.

---

## Where the change lives

- **Repo (shared NAS)**: `Y:\2_Connectome\Behavior\MouseReach\` (Y: is shared --
  collin sees the same drive).
- **Commit**: `632281f` ("GT/Review tools: expected_triage toggle on outcome
  segments") on branch `feature/v8-loocv-output-extension`.
- **Verify the commit exists** on Y::

  ```bash
  cd Y:/2_Connectome/Behavior/MouseReach
  git show --stat 632281f
  ```

  Expected output: 3 files changed, 174 insertions(+), 7 deletions(-).
  If the commit isn't there, stop and tell collin -- something's out of sync
  on Y: and the originator (loge) needs to push or otherwise reconcile.

---

## Files to update (exactly these 3, no others)

Relative to the MouseReach repo root:

1. `src/mousereach/review/unified_gt.py` -- adds 4 fields to `OutcomeGT` plus
   `get_triage_timestamp()` helper.
2. `src/mousereach/review/ground_truth_widget.py` -- adds toggle button + reason
   field to outcome rows.
3. `src/mousereach/outcomes/review_widget.py` -- adds triage checkbox + reason
   field inside the existing flag group.

---

## CRITICAL: limited scope

The branch `feature/v8-loocv-output-extension` contains a LOT of other
in-progress work (v8 LOOCV reach detector, outcome v6 cascade, restart_phase
scripts, improvement framework changes, etc.) -- much of it uncommitted on Y:'s
working tree.

**DO NOT** do any of these:
- `git pull`, `git merge`, or `git checkout feature/v8-loocv-output-extension`
- copy files from Y:'s working tree using plain `cp` (Y:'s working tree may
  diverge from the commit -- the user has been editing locally)
- migrate or rewrite any GT files
- touch unified_gt.py's other fields, the cascade evaluator, or any
  improvement scripts

**DO** extract the 3 files at the exact commit SHA, regardless of what Y:'s
working tree currently looks like.

---

## Steps

### 1. Locate collin's runtime copy of MouseReach

Collin runs MouseReach from a local-drive editable pip install (mirrors loge's
C: setup, but his local drive letter may differ -- could be C:, A:, D:, etc.).
Likely candidates:

```
<local-drive>:\2_Connectome\Behavior\MouseReach\src\mousereach\
```

Find it by running:

```bash
python -c "import mousereach, pathlib; print(pathlib.Path(mousereach.__file__).parent)"
```

The directory printed is collin's runtime `mousereach/` package root. The 3
target files live under it at:
- `review/unified_gt.py`
- `review/ground_truth_widget.py`
- `../outcomes/review_widget.py` (sibling -- one level up from `review/`)

If the python command doesn't work because mousereach isn't installed in the
active env, ask collin which conda env he uses for MouseReach (likely something
like `mousereach` or `MouseBrain`) and re-run with that env activated.

### 2. Extract the 3 files from the commit and write them to the runtime path

Set `RUNTIME` to the directory printed in step 1 (the `mousereach/` package
root). Then:

```bash
cd Y:/2_Connectome/Behavior/MouseReach

git show 632281f:src/mousereach/review/unified_gt.py            > "$RUNTIME/review/unified_gt.py"
git show 632281f:src/mousereach/review/ground_truth_widget.py   > "$RUNTIME/review/ground_truth_widget.py"
git show 632281f:src/mousereach/outcomes/review_widget.py       > "$RUNTIME/outcomes/review_widget.py"
```

(PowerShell equivalent: `git show 632281f:<path> | Out-File -Encoding utf8 "$RUNTIME\..."`
-- but bash via Git Bash usually works fine on Windows.)

This pulls the EXACT committed contents and overwrites collin's local copy of
just those 3 files. Other in-progress work on Y: or in collin's runtime is not
touched.

### 3. Verify the files compile

```bash
python -m py_compile "$RUNTIME/review/unified_gt.py" \
                     "$RUNTIME/review/ground_truth_widget.py" \
                     "$RUNTIME/outcomes/review_widget.py"
```

No output = success. If any file fails to compile, stop and surface the error.

### 4. Verify the schema round-trips on a known corpus file

```bash
python -c "
from pathlib import Path
import json
from mousereach.review.unified_gt import _dict_to_unified_gt, _unified_gt_to_dict

gt_path = Path(r'Y:/2_Connectome/Behavior/MouseReach_Improvement/validation_runs/DLC_2026_03_27/gt/20250710_CNT0215_P4_unified_ground_truth.json')
with open(gt_path) as f:
    gt = _dict_to_unified_gt(json.load(f))
flagged = [o for o in gt.outcomes if o.expected_triage]
print(f'{len(flagged)} segment(s) with expected_triage=True')
for o in flagged:
    print(f'  seg {o.segment_num}: by={o.expected_triage_flagged_by} reason={o.expected_triage_reason!r:.80}')
"
```

Expected: `1 segment(s) with expected_triage=True` and seg 8 line showing
`FRIEDRICHL` and a fast-displacement-out reason.

### 5. Restart napari and visually confirm

- Close any open napari windows.
- Launch `mousereach` (or `mousereach-gt`) and load any video with existing
  GT, ideally `20250710_CNT0215_P4`.
- Navigate to outcome segment 8. The new toggle button on the segment's row
  should read `[!] TRIAGE: ON` with an orange background. Hovering shows a
  tooltip with FRIEDRICHL + the timestamp + the reason.
- Click the toggle. It should flip to `Triage: off` (gray). Click again, it
  should flip back to ON and re-stamp metadata with collin's username + a fresh
  UTC timestamp (not FRIEDRICHL anymore -- collin is now the editor).
- Switch to the "3 - Review Tool" tab. Same toggle button appears on outcome
  rows there too (it's the same widget class, just a different mode).
- Switch to the "Step 3b - Review Outcomes" tab. A new checkbox
  "[!] Expected triage (algo should have triaged)" appears inside the existing
  "Flag Segment" group, with a "Triage reason:" line edit below it.

If any of those visual checks fail, surface them -- something didn't sync.

---

## What if collin doesn't have a local runtime copy and runs directly from Y:

Some users edit and run from Y: directly with no separate local copy. If
`python -c "import mousereach; print(...)"` shows a path under `Y:/...`, that's
the case. Then `RUNTIME` = `Y:/2_Connectome/Behavior/MouseReach/src/mousereach`.

Still extract via `git show 632281f:<path> > <runtime>/<path>` -- this is
equivalent to `git checkout 632281f -- src/mousereach/...` but doesn't touch
the index, leaving the rest of Y:'s in-progress state alone.

After this, Y:'s working tree for those 3 files matches the commit. The other
in-progress mods on Y: (improvement/, outcomes/pillar_geometry_widget.py, etc.)
are still uncommitted -- LEAVE THEM ALONE, they belong to loge.

---

## Out of scope (do NOT do these in this session)

- Don't push, pull, fetch, merge, rebase, or branch.
- Don't migrate existing GT files.
- Don't touch the cascade evaluator (`scripts/restart_phase_e_stage45_validate.py`)
  or any other downstream consumer -- they already handle the new field.
- Don't add the displaced_out -> retrieved+triage shortcut convention; that's
  a separate downstream change.
- Don't add a Co-Authored-By Claude line to anything (this update doesn't
  create new commits; just file syncs).

---

## If something goes wrong

- Compile fails on any file: stop, paste the traceback to collin.
- Round-trip test in step 4 returns 0 segments instead of 1: the GT file
  on Y: was modified since the originator session; not a code issue, but flag
  it.
- napari launches but the toggle doesn't appear: confirm the runtime path in
  step 1 actually points to where mousereach is installed; check that the 3
  files contain the new code (`grep -n expected_triage <runtime>/review/unified_gt.py`
  should show field declarations).
- napari throws AttributeError on `outcome.expected_triage`: unified_gt.py
  wasn't updated -- redo step 2 for that file specifically.
