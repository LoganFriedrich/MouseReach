# Handoff: GT Tool — `expected_triage` toggle + displaced_out convention

Audience: a Claude Code session working on the MouseReach repository (specifically the GT tool).
Goal: add a UI toggle to the napari ground-truth tool that lets the human scorer mark an outcome segment as "expected_triage = true/false," and formalize a small change to how displaced_out cases are scored.

Originator: LoganFriedrich (collaborating with schultzc on GT scoring).
Date: 2026-05-04.

---

## Background — what's already in the JSON schema

Some GT files in the corpus already carry `expected_triage` on outcome segments. The fields are written by the auto-cascade evaluator (and recently by some hand-edits) and look like this on an outcome segment:

```json
{
  "segment_num": 8,
  "outcome": "displaced_outside",
  "interaction_frame": 13639,
  "outcome_known_frame": 13644,
  "causal_reach_id": 89,
  "determined": true,
  "determined_by": "schultzc",
  "determined_at": "2026-02-23T14:54:45.376590",
  "comment": null,
  "expected_triage": true,
  "expected_triage_reason": "Fast-displacement-out: pellet exits the SA in 1-3 frames as a motion-blur streak ...",
  "expected_triage_flagged_by": "FRIEDRICHL",
  "expected_triage_flagged_at": "2026-05-04T00:00:00.000000+00:00"
}
```

When the field is omitted, treat it as `false`.

Existing examples (already in this state on disk; do NOT regenerate them):
- `20250710_CNT0215_P4_unified_ground_truth.json` (segment 8)
- `20250806_CNT0311_P2_unified_ground_truth.json` (multiple segments — short-segment apparatus failures + the displaced_outside one)
- `20250806_CNT0312_P2_unified_ground_truth.json`
- `20250821_CNT0110_P4_unified_ground_truth.json`
- `20251007_CNT0316_P2_unified_ground_truth.json`
- `20251023_CNT0401_P4_unified_ground_truth.json`
- `20251031_CNT0413_P2_unified_ground_truth.json`

Both canonical (`Y:\2_Connectome\Validation_Runs\DLC_2026_03_27\gt\`) and quarantine (`Y:\2_Connectome\Validation_Runs\DLC_2026_03_27\iterations\2026-04-28_outcome_v4.0.0_dev_walkthrough\gt\`) trees were updated. New scoring sessions should produce the same shape via the new toggle.

---

## What to build

### 1. Outcome-panel toggle in the GT widget

The GT widget lives at `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\review\ground_truth_widget.py`. CLI entry point is `mousereach-gt`. The outcome-scoring panel is what users press buttons in to set `outcome` ∈ {untouched, retrieved, displaced_sa, displaced_outside, abnormal_exception}.

Add a single toggle widget alongside that panel — call it **"Expected Triage"** — with the following behavior.

#### State model

The toggle reads/writes the boolean `expected_triage` on the **currently-displayed outcome segment**. Two states:

- **OFF (false / absent)** — default. Visually neutral (e.g., gray pill or empty-checkbox style).
- **ON (true)** — visually distinct and unambiguous. Suggested: yellow/orange filled pill or a filled checkbox plus a label like "Expected Triage: ON" so the user can tell at a glance from across the screen.

It must be **obvious which state it is in** without hovering, clicking, or reading small text. The user explicitly called this out: the visual must read at a glance.

#### Toggle behavior

When the user clicks the button:

- If currently OFF → set ON.
  - Set `seg["expected_triage"] = true`
  - Stamp `seg["expected_triage_flagged_by"] = <current scorer>` (uppercase username, matching the existing `determined_by` convention)
  - Stamp `seg["expected_triage_flagged_at"] = <ISO 8601 UTC, microsecond precision, with `+00:00` offset>` matching the format already in the corpus
  - Optionally prompt for / accept a `expected_triage_reason` string. Implementation choice: a small QLineEdit shown when ON. If empty, store `null` or just omit. See "Reason field" below.
- If currently ON → set OFF.
  - Set `seg["expected_triage"] = false` (or remove the field — pick one and be consistent; recommend setting to `false` so the audit trail of `_flagged_by`/`_flagged_at` stays intact as a record).
  - Leave `expected_triage_flagged_by` / `expected_triage_flagged_at` in place (these are a record of when the flag was last set).
  - Leave or clear `expected_triage_reason`. Either is fine, document the choice.

The toggle should mark the GT file dirty so the existing save flow persists the change.

#### Initial state on load

When a GT file loads, for the segment currently in view, the toggle should reflect `seg.get("expected_triage", False)`. Switching segments updates the toggle to that segment's value.

#### Reason field (optional but recommended)

A small `QLineEdit` placed next to or under the toggle, only visible when toggle is ON. Bound to `expected_triage_reason`. When the user types and tabs out / focuses elsewhere, save the value. Empty string → store `null` or omit.

The reason field exists so the scorer can write a 1-2 sentence note on why this segment is being flagged ("pellet streaked out of SA, DLC can't see it"). This is read by humans only — the algos don't depend on it.

### 2. Convention change: displaced_out → retrieved + expected_triage (deferred)

Logan and schultzc are also planning a scoring convention change: when the pellet leaves the SA entirely (currently `outcome: "displaced_outside"`), record it as **`outcome: "retrieved"` + `expected_triage: true` + reason string** instead. Rationale: this collapses a category that the DLC-based outcome detector cannot see anyway, getting the algo and GT to agree on the easy answer (retrieved) while still flagging that the case was an edge.

**Don't make this convention change part of the toggle's required behavior** — the scorer is the one who picks `outcome`. But:
- Make the toggle work cleanly with `outcome == "retrieved"` as well (no special-casing needed).
- A future helper (separate from this handoff) might offer a "this is a fast-streak displaced_out → retrieve+triage" composite shortcut. Out of scope for now.

### 3. Two source-tree copies to keep in sync

The GT widget exists in two places that need to stay in sync:
- `Y:\2_Connectome\Behavior\MouseReach\src\mousereach\review\ground_truth_widget.py` — git working copy on Y:.
- `C:\2_Connectome\Behavior\MouseReach\src\mousereach\review\ground_truth_widget.py` — runtime copy on C: that the editable pip install picks up. After editing on Y:, copy the changed file(s) to C: so the running tool gets the change.

(See `Y:\2_Connectome\CLAUDE.md` "Git Workflow" section for details.)

### 4. Branch + commit hygiene

Per `CLAUDE.md`:
- Branch off `master` for this change. Name like `feature/gt-tool-expected-triage`.
- Edit on Y:, commit on Y:, push to GitHub from Y:.
- One commit = one change. Suggested split: (a) JSON-schema bump + reader/writer, (b) Qt widget + wiring, (c) any docs.
- After merge, delete the branch.

---

## Validation steps after implementing

1. Launch `mousereach-gt` against any GT file from the corpus list above. Navigate to a segment that already has `expected_triage: true` in its JSON. Confirm the toggle reads ON for that segment, OFF for neighbors.
2. Toggle a fresh segment (one with `expected_triage` absent). Confirm the JSON now has the four fields stamped with the current user + ISO timestamp. Save and reload — confirm persistence.
3. Toggle it back OFF. Confirm `expected_triage` flips to `false` (or is removed); confirm save+reload behaves consistently.
4. Test with both schemas if the tool reads the older `type: "ground_truth"` format (with `segments[]` at root) — at minimum `20251031_CNT0413_P2_outcome_ground_truth.json`. The toggle should work or at least fail gracefully.
5. After this lands, the cascade-evaluation runner (`Y:\2_Connectome\Behavior\MouseReach\scripts\restart_phase_e_stage45_validate.py`) automatically picks up `expected_triage` from GT — no changes needed there. The runner already reads `seg.get("expected_triage", False)`.

---

## Open question for downstream (not in scope here, but flag it)

The cascade trust framework currently treats `expected_triage: true` as "algo MUST triage; commits fail trust." That makes sense for apparatus-failure expected_triage cases (e.g., short-segment cycling aborts). But for the displaced_out → retrieved convention, the user said "algo committing retrieved here is acceptable" — i.e., commits should NOT fail trust on those.

This means there are conceptually two flavors of expected_triage:
- **must-triage** (apparatus failure): the only correct algo decision is triage.
- **acceptable-either-way** (DLC-can't-see fast-streak displaced_out): algo can commit retrieved or triage, both are fine.

How to distinguish them in the JSON is a downstream design question — not part of this handoff. Options when it comes up:
- An `expected_triage_acceptable_decisions: ["triage", "retrieved"]` array on the segment.
- A separate flag like `algo_outcome_acceptable_either_way: true`.
- Or just decide that all expected_triage = "any decision OK" and update the trust framework accordingly.

Leave this for a future round; for now, the toggle just reads/writes the single boolean as documented.

---

## What NOT to do

- Don't migrate or rewrite existing GT files in this change. The schema is already additive (extra fields on a few segments); the new tool just needs to handle reading them and producing them.
- Don't change `outcome` values automatically based on the toggle. The scorer picks the outcome class; the toggle is independent metadata.
- Don't bundle this with broader GT-tool refactors. Keep the diff small and focused so it can be merged cleanly.
- Don't write any "Co-Authored-By Claude" lines in commit messages unless the user has set that as their convention; the project's `CLAUDE.md` doesn't require it.
