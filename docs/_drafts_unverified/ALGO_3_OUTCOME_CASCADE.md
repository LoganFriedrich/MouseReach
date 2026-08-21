# Outcome detection: what happened to each pellet

Describes: `src/mousereach/outcomes/v6_cascade/` (all files), `src/mousereach/outcomes/core/` (`batch.py`, `pellet_outcome.py`, `triage.py`, `advance.py`, `geometry.py`, `__init__.py`), `src/mousereach/outcomes/cli.py`, `src/mousereach/outcomes/_review.py`, `src/mousereach/lib/pillar_geometry.py`

Verified against: 61d98b9 (2026-08-21)

---

## What this step does

The video has already been cut into **segments** (one per pellet presentation) and **reaches** (windows where the mouse pushed a paw out) have already been found. This step answers one question per segment: *what happened to that pellet?*

It writes one file per video, `{video}_pellet_outcomes.json`, with one record per segment.

It does not look at the video pixels except in two narrow places (described below). Everything else is computed from the DeepLabCut pose file — the per-frame x, y and confidence value for each tracked body part.

---

## Which detector actually runs

There are two detectors in this directory. Only one runs in production.

**v6 cascade** (`v6_cascade/`, version string `6.1.0`, set at `v6_cascade/__init__.py:16`). This is the default everywhere:

- `outcomes/core/batch.py:154` `process_single()` runs v6 unless you pass `legacy=True`. Every production caller uses it with the default: `watcher/orchestrator.py:977` and `:1848`, `pipeline/run_all.py:57`, `pipeline/reprocess_to_current.py:143`.
- `mousereach-detect-outcomes` runs v6 unless you pass `--legacy` (`outcomes/cli.py:96-124`).

**Legacy detector** (`core/pellet_outcome.py`, version string `4.0.0_step2` at line 108, despite the file's own documentation calling it "v2.3"). Reachable only via `--legacy` or `legacy=True`. Nothing in the repo passes either. Its output has a completely different shape (see "What is not in the file"). Archived outcome files on the NAS written before the switch carry `"detector_version": "2.4.4"` and that older shape; anything reading outcome files has to cope with both.

The rest of this document describes v6 unless it says otherwise.

**Outcome detection is skipped entirely for tray types E and F.** `pipeline/run_all.py:89-91`, `watcher/orchestrator.py:1062` and `:1971` all check the tray letter parsed from the filename and skip the step, because those trays have no reliable pellet. No outcomes file is produced at all for those videos.

---

## Inputs, and what happens when one is missing

`detect_outcomes_v6_cascade(dlc_df, segments, reaches, video_id, video_dir)` (`v6_cascade/detector.py:183`).

| Input | Where it comes from | If absent |
|---|---|---|
| Pose data (`dlc_df`) | the DeepLabCut `.h5`, loaded by `mousereach.reach.v8.features.load_dlc_h5` | hard failure; the whole video fails |
| `segments` | `{video}_segments.json`. Boundary frames are converted to inclusive pairs `(b[j], b[j+1]-1)` at `core/batch.py:206-207`. Segment numbers are 1-based array positions (`detector.py:224`), **not** read from the segments file | hard failure |
| `reaches` | `{video}_reaches.json` | silently becomes an empty list (`core/batch.py:208-210`). The cascade still runs. Every reach-dependent stage defers, so almost everything ends up triaged |
| `video_dir` | searched for `{video}.avi/.mp4/.mkv` next to the pose file (`core/batch.py:145-151`) | the two pixel-reading checks become no-ops. No warning is printed |

A reach belongs to a segment if its **start** frame falls inside the segment (`detector.py:176-181`). A reach that runs past the segment end is still assigned to the segment it started in.

### The reach-loading bug that still exists in the CLI

`core/batch.py:107` `_extract_reaches()` reads the current reach-file format, where reaches are nested per segment under `segments[].reaches`. It also falls back to an older flat top-level `reaches` list. Its own comment records that reading only the flat form used to return an empty list against real reach files, starving the cascade of every reach; that was fixed in August 2026.

**`outcomes/cli.py:242` has a second copy of `_extract_reaches` that was never fixed.** It reads only the flat top-level `reaches` key. Against current reach files it returns `[]`. So `mousereach-detect-outcomes` produces materially different (and much worse) outcomes than the pipeline does on the same inputs. It also uses `r.get("start_frame") or r.get("start")`, so a reach starting at frame 0 would be dropped.

---

## The vocabulary the stages share

Nearly every stage is built from the same handful of measurements. Understanding these five is most of understanding the cascade.

**Clean zone.** Almost every stage ignores the last 5 frames of the segment: `clean_end = seg_end - 5`. The reasoning (written out at `stage_2_pellet_stable_untouched.py:38-46`) is that the segmenter's boundary marks "the old segment is clearly ending", and the frames around it are a no-man's-land where the tray is moving. If a segment is 5 frames or shorter, most stages defer with `too_short`.

**Pillar circle.** The pillar is the small post the pellet sits on. Its position is *not* tracked directly for this purpose — it is computed from the two front corners of the scoring-area tray, `SABL` and `SABR` (`lib/pillar_geometry.py:41-67`):

- `ruler` = pixel distance between SABL and SABR (physically 9 mm)
- pillar centre = midpoint of SABL/SABR, moved `0.944 * ruler` upward in the image
- pillar radius = `0.10 * ruler`

Because it is derived from the tray corners, it moves with the tray. The two upper corners `SATL`/`SATR` are not used, and **the confidence values of SABL and SABR are not checked at all** — if DeepLabCut misplaces a corner, the pillar circle silently moves with it. The corner positions are smoothed over 3 frames before the geometry is computed.

**Radii.** Distances are expressed as multiples of the pillar radius. "On pillar" is usually within 1.0 radii; "clearly in the scoring area" is usually more than 3.0 radii.

**Slit line.** `pillar_cy + pillar_r` — a horizontal line just below the pillar. A body part above it (smaller y) is inside the mouse's reaching space; below it is out in the tray. Used two ways: a paw above the line counts as "paw is out reaching", and a *pellet* above the line is taken as evidence the pellet is in the mouse's mouth or paw rather than lying in the tray.

**Paw parts.** Four body parts are pooled as "the paw": `RightHand`, `RHLeft`, `RHOut`, `RHRight`.

**"Sustained" / "run".** `guards.lrun()` returns the longest unbroken stretch of consecutive frames where a condition holds. Almost every threshold in the cascade is a run length, not a total count — a real event lasts many frames, tracking noise does not.

**Cleaning.** Most stages pass the pose data through `mousereach.lib.dlc_cleaning.clean_dlc_bodyparts` first, which repairs the scoring-area corner positions and the pellet position. Confidence values are always read from the *raw*, uncleaned data.

---

## The labels it can write

Only four values ever appear in the `outcome` field of a v6 file. Counted across every `committed_class=` in `v6_cascade/`: 10 stages commit `displaced_sa`, 10 commit `retrieved`, 8 commit `untouched`, and nothing else exists.

| Label | Meaning |
|---|---|
| `untouched` | No reach in this segment moved the pellet. Note this includes the case where the pellet arrived already lying in the tray from a previous segment and the mouse pushed it around within the tray — it was never on the pillar to be taken off it (see `stage_5_pellet_off_pillar_throughout.py:10-22`) |
| `retrieved` | The pellet left the apparatus: the mouse grasped it and ate it. The evidence is always some form of "the pellet stopped being visible after a reach and never came back" |
| `displaced_sa` | The pellet was knocked off the pillar and came to rest in the scoring area (the tray) |
| `triaged` | The cascade declined to decide. This is not an outcome; it is a request for a human to look |

Three further labels exist in the documentation, the review tool's key bindings and the legacy detector, but **v6 never writes them**: `displaced_outside`, `no_pellet`, `uncertain`. `outcomes/cli.py:26` and `outcomes/core/__init__.py:19-24` both advertise `displaced_outside (O)` as a live category. It is not.

---

## How the cascade works

`detector.py:112-146` builds a fixed list of 33 stage objects, in a fixed order. For each segment, `detector.py:234-243` calls each stage's `decide()` in turn. Each returns one of three things:

- **commit** — "I am confident; the answer is X". The loop stops. Nothing after this stage is consulted.
- **triage** — "I recognise this as a case nobody should score automatically". The loop also stops.
- **continue** — "not my case", with a short reason. Move on.

First one to commit or triage wins. There is no voting, no confidence score, no comparison between stages. Ordering is the entire arbitration mechanism: the early stages are the safe, obvious ones, and the late ones are progressively more speculative rescues of what is left.

The stage list is rebuilt from scratch for every video. All stages are stateless.

Every stage also fills a `features` dictionary with the numbers it computed. **None of it is ever written anywhere.** `StageDecision.features` (`stage_base.py:57`) is read by no production code path; `detector.py:248-268` builds the output record from `committed_class`, `whens` and `reason` only. The `features` payload — every distance, count and intermediate that would explain a decision — is discarded at the end of each segment.

### The three frames a commit emits

A commit must supply `whens["outcome_known_frame"]`, and (for `retrieved` and `displaced_sa`) `whens["interaction_frame"]`.

- **`outcome_known_frame`** — the earliest frame from which the outcome is determinable. For every `untouched` stage this is simply `seg_end - 5`, the last clean frame: you only know nothing happened once the segment is over. For touched outcomes it is anchored to the event — the reach end plus a small settling offset, or the frame the pellet first appears at its resting place.
- **`interaction_frame`** — when the paw was over the pellet. Always `null` for `untouched`. For touched outcomes it is a point inside the causal reach window: the middle of the reach in most stages, 40% of the way in for stages 7 and 8 (`IFR_POSITION_IN_BOUT = 0.4`).

`stage_base.py:44-52` states this contract. It is not enforced anywhere — a stage that committed without setting `outcome_known_frame` would emit `null` and nothing would complain (`detector.py:250`, `.get()` with no default).

---

## Guards wrapped around every stage

After the stage list is built, `detector.py:154-163` wraps three extra checks around **every** stage's `decide()`, plus one attribute override. Each only does anything when the stage underneath commits `displaced_sa`. They run in this order (vanish first, then presence, then pixels):

**1. Vanish guard** (`guards.py:127`). If the pellet's confidence drops below 0.5 for 60 or more consecutive frames in the clean zone, a `displaced_sa` commit is converted to **continue**. A displaced pellet stays visible in the tray; a pellet that disappears for two seconds was retrieved.

`guards.py` also defines `DISPLACED_VANISH_GUARD_CLASSES`, a set of four stage class names, and its own module docstring (`guards.py:10`) says the guard is applied only to those. `detector.py:88` imports the set and **never uses it**. The guard goes on all 33 stages. The practical difference is that stages 8, 11, 14, 18, 20 and the two retry stages also get the guard, which the design note says they should not.

**2. Scoring-area presence guard** (`guards.py:144`). If the pellet was never held at more than 3 pillar-radii from the pillar, at confidence ≥ 0.7, for 30 or more consecutive frames, a `displaced_sa` commit is converted to **continue**. "Displaced" means the pellet ended up somewhere, and somewhere is a place it stays.

**3. Pixel check on the landing spot** (`cv_artifact_gate.py:216`). This is the only guard that can *create* a triage. It opens the video, finds where the pellet supposedly landed, and measures brightness there before and after. It computes `(after − before) / pellet_brightness`. A real pellet arriving makes a dark spot bright. If the change is below **0.10**, the tracker had latched onto something already bright at that location — dust, a corner marker, a reflection — and the commit becomes **triage** with reason `cv_artifact_landing_no_pellet_arrival`.

This guard is a no-op if `video_dir` is `None`, if the file is not found, or if any of the sampled brightness measurements come back unusable. It accepts `.mp4` and `.avi` (`cv_artifact_gate.py:196-203`).

**4. Paw-confidence override.** `detector.py:157-158` raises `paw_lk_threshold` from 0.5 to 0.9 on exactly two stages, `Stage16DisplacedViaMaxDisplacement` and `Stage17DisplacedViaDominantMaxDisplacement` (`guards.py:117-120`), because DeepLabCut 4.0 detects an approaching paw past the slit far more often than 3.x did. This override is set on the top-level stage objects only. Stages 22 and 25 construct **their own private copies** of stages 16 and 17 inside `__init__`, and those copies keep the 0.5 threshold. The three wrapping guards do still apply to 22 and 25, because they wrap the outer object's `decide` (confirmed in archived output: stage 22 appears with `cv_artifact` triage reasons).

---

## The stages, in order

Names in the first column are exactly the strings that appear in the `stage` field of the output file (they come from `detector.py:112-146`, and several of them differ from the stage class's own `name` attribute — e.g. the file `stage_2_pellet_stable_untouched.py` reports as `stage_2_stable_on_pillar`).

### Untouched stages (0–6b)

| Stage | Asks | Commits when |
|---|---|---|
| `stage_0_short_segment_triage` | — | **Never does anything.** See "Stages that can never fire" |
| `stage_1_position_never_changed` | Is the pellet still sitting on the pillar at the end? | In the last 30 clean frames, ≥ 50% have the pellet at confidence ≥ 0.9 and within 1.0 radii of the pillar centre — **and** the shared touched-guard does not fire |
| `stage_2_stable_on_pillar` | Was the pellet parked inside the pillar circle all segment, and is the last 11-frame window clean? | Pellet inside the circle for ≥ 95% of non-reach frames, and in the final window either every frame is clean or ≥ 60% are. "Clean" = pellet confidence ≥ 0.7, inside the circle, and no confident paw within 2 radii |
| `stage_3_paw_never_in_pellet_area` | Could any reach even have touched it? | The paw never got within 2.5 radii of the pillar centre (best 3-frame average), **and** the pellet was not sustained out in the tray for 15+ frames |
| `stage_4_pellet_returns_to_pillar` | After the last reach, is the pellet back on the pillar? | 3 consecutive frames with the pellet within 1.2 radii, at least 15 frames after the last reach ends, with no paw out; blocked if the pillar body part becomes visible in that window (confidence ≥ 0.5) or if the pellet vanished for 60+ frames |
| `stage_5_pellet_off_pillar_throughout` | Did the pellet arrive already lying in the tray, and stay there? | Three checks: (a) after the tray settles, 5 consecutive frames with the pellet confidently more than 3 radii out, before the first reach; (b) it never comes back to the pillar; (c) it stays observable afterwards |
| `stage_6_predominantly_on_pillar` | Was it visible and on the pillar essentially the whole time? | Pellet visible in ≥ 99% of non-reach frames **and** inside 1.0 radii in ≥ 99% of those, and the shared touched-guard does not fire |
| `stage_6b_never_entered_sa` | Same question, tolerating more tracking noise | Confident in ≥ 70% of frames, present in ≥ 50% of the last 30, median radius when confident < 1.8, never absent for more than 10 frames in a row, never sustained beyond 3 radii for 15+ frames |

Stages 1, 3, 4, 5 and 6b were rewritten for DeepLabCut 4.0. Their docstrings say so and explain what the old signal was and why it died. Stages 1, 3, 4 and 6 all carry a version of the same safety net — the pellet must not have been sustained out in the tray (>3 radii for 15+ frames) and must not have vanished (confidence < 0.5 for 60+ frames). `guards.pellet_displaced_or_vanished` is the shared implementation.

Stage 5 is the only untouched stage that can triage: if the pellet started off-pillar and then apparently vanished, that is physically impossible (nothing can be retrieved out of the tray), so the segment goes to a human with reason `pellet_appears_retrieved_from_off_pillar_state`. Across 2,159 archived files this path fired zero times.

### Touched stages (7–29)

| Stage | Commits | Asks |
|---|---|---|
| `stage_7_settled_off_pillar_late` | `displaced_sa` | In the last half of the segment, is the pellet parked at one spot off the pillar, inside the tray quadrilateral, at confidence ≥ 0.95, for ≥ 100 frames? Then it identifies which reach did it, by walking back from the first sustained off-pillar sighting, and requires that reach to also be the reach with the largest pellet displacement, and requires the pellet never to have been at that resting spot before it |
| `stage_8_pellet_displaced_to_sa` | `displaced_sa` | Same idea with a different route: pellet confidently on the pillar before the first reach, then at rest off-pillar in the tray for ≥ 40 frames somewhere in the segment. Pellet confidence threshold 0.95, paw threshold raised to 0.95 for 4.0 |
| `stage_9_pellet_vanished_after_reach` | `retrieved` | Given the segment is touched, did the pellet disappear rather than land in the tray? Late-zone visibility ≤ 10%; from the causal reach onward, at most 5 sustained frames of any pellet sighting; from the *first* reach onward, at most 5 sustained frames of the pellet off-pillar inside the tray. Reaches less than 20 frames apart are chained into one retrieval action. Has a second attempt (`_recheck`, line 176) that re-picks the causal reach using a sustained on-pillar test when the first pass deferred with `no_candidate_reach_for_retrieval` |
| `stage_10_pillar_revealed_after_reach` | `displaced_sa` | *(name is stale — this is not about the pillar)* One run of ≥ 5 confident frames (≥ 0.95) with the pellet more than 2.0 radii out and inside the tray, after a reach; requires ≥ 30 such frames in total afterwards and **zero** sustained sightings back near the pillar |
| `stage_11_single_reach_clean_displacement` | `retrieved` | *(name is stale — this commits retrieved)* Did tracking lose the pellet completely? ≤ 100 sustained frames of pellet sighting in the whole clean zone, ≤ 5 in the late half, and at least one paw-out bout |
| `stage_12_retrieved_pellet_above_slit` | `retrieved` | Are the post-reach pellet sightings *above* the slit line (in the mouse's face) rather than below it (in the tray)? Needs ≥ 3 sustained above-slit frames and ≤ 5 below-slit ones. Single-reach segments only |
| `stage_13_retrieved_via_pillar_lk_transition` | `retrieved` | In a single-reach segment, does the pillar body part go from hidden (confidence < 0.3 for ≥ 5 frames) to revealed (> 0.5 for ≥ 30 frames) across the reach, with the pellet gone afterwards? |
| `stage_14` | — | **Disabled** |
| `stage_15` | — | **Disabled** |
| `stage_16_displaced_via_max_displacement` | `displaced_sa` | Across the segment's reaches, is there exactly one where the pellet's median position shifts by ≥ 1.5 radii? Requires ≥ 50 off-pillar sightings in the last 30% of the segment |
| `stage_17_displaced_via_dominant_max_displacement` | `displaced_sa` | Same, but when several reaches show displacement and one is at least 3× the next largest |
| `stage_18`, `stage_19`, `stage_20` | — | **Disabled** |
| `stage_21_causal_reach_via_on_off_transition` | `displaced_sa` **or** `retrieved` | The physics test: exactly one reach has the pellet confidently on the pillar in a paw-free window immediately before (≥ 2 of ≥ 3 paw-clear frames within 10) and *no* on-pillar frame immediately after. If more than one reach shows this, the detection is noisy and it defers. Class then follows: pellet later parked in the tray (≥ 30 frames within a 15 px cluster, > 1.5 radii out) → displaced; late zone essentially empty (≤ 3 sightings) → retrieved |
| `stage_22_retry_with_stabilized_dlc` | whatever the inner stage says | Re-runs stages 21, 9, 16, 17 against pose data where short gaps in pellet tracking (≤ 5 frames, endpoints within 10 px) have been filled in. Never touches confident detections, never smooths across a real position jump |
| `stage_23`, `stage_24` | — | **Disabled** |
| `stage_25_retry_with_strict_pellet_confidence` | whatever the inner stage says | Re-runs the same four stages with every pellet detection below confidence 0.85 zeroed out, on the theory that most false "pellet seen in the tray" evidence is the tracker firing on the wrong object at 0.7–0.85 |
| `stage_26_retrieved_via_unique_vanish_reach` | `retrieved` | Per reach, in windows capped by the neighbouring reaches (30 before, 60 after): pellet confidently seen before, essentially absent after (< 20% of post frames). Commits if **exactly one** reach shows this, no earlier reach already moved the pellet ≥ 10 px, no run of ≥ 10 frames afterwards with the pellet confidently (≥ 0.85) out in the tray, and no unannotated paw activity of ≥ 10 frames before it |
| `stage_27_displaced_sa_via_unique_high_displacement` | `displaced_sa` | Exactly one reach moves the pellet ≥ 10 px, no reach shows the vanish signal, that reach is the first to move it at all (≥ 5 px), and the pellet stays visible off-pillar afterwards. **Also triages**: if the chosen reach starts within 30 frames of the segment start or ends within 60 frames of the segment end, boundary noise makes the pick unreliable and it goes to a human |
| `stage_28_retrieved_via_pillar_visibility_transition` | `retrieved` | Exactly one reach where the pillar goes from hidden (< 0.4) to clearly visible (> 0.8, a rise of > 0.5) **and** the same reach shows the pellet-vanish signal |
| `stage_29_displaced_sa_pillar_disambiguated_multi_disp` | `displaced_sa` | Two or more reaches move the pellet ≥ 10 px, but exactly one of them is the reach that reveals the pillar. That one is causal — the later ones were bouncing an already-displaced pellet |
| `stage_98_lost_in_shadow_triage` | triage only | See below |
| `stage_99_residual_triage` | triage only | Always triages. Reached by anything left |

The touched half of the cascade has a clear shape: 7–17 are the broad rules, 21–25 are careful re-examinations of what those missed, and 26–29 are late per-reach rules built specifically on the physical constraints that the pellet cannot return to the pillar, cannot be retrieved out of the tray, and cannot move without a reach moving it.

### What actually fires

Over the 2,159 archived files carrying `detector_version: 6.1.0` in `MouseReach_Pipeline/Analyzed` (43,180 segments), the stage field breaks down as:

```
16212  stage_7_settled_off_pillar_late          2137  stage_99_residual_triage
14649  stage_1_position_never_changed            427  stage_3_paw_never_in_pellet_area
 2361  stage_8_pellet_displaced_to_sa            247  stage_25_retry_with_strict_pellet_confidence
 1569  stage_9_pellet_vanished_after_reach       215  stage_22_retry_with_stabilized_dlc
 1137  stage_21_causal_reach_via_on_off_...      139  stage_4_pellet_returns_to_pillar
 1104  stage_26_retrieved_via_unique_vanish      100  stage_6b_never_entered_sa
  790  stage_28_retrieved_via_pillar_vis...       72  stage_13, 59 stage_5, 42 stage_12
  777  stage_27_displaced_sa_via_unique_...       37  stage_11, 26 stage_10, 14 stage_17
  684  stage_16_displaced_via_max_disp            13  stage_2, 6 stage_6, 3 stage_98
  360  stage_29_displaced_sa_pillar_disamb...
```

Labels: `displaced_sa` 20,548, `untouched` 15,393, `retrieved` 4,334, `triaged` 2,905. Two stages account for 71% of all decisions. Six stages (2, 6, 10, 11, 17, 98) each account for well under a tenth of a percent. This is an observation of the archive, not a property of the code, but it is the fastest way to see which parts of this subsystem matter.

---

## Stages that can never fire

Eight of the 33 stages cannot produce any decision. They defer 100% of the time. All are still constructed, still called once per segment, and still cost time.

| Stage | Why |
|---|---|
| `stage_0_short_segment_triage` | `decide()` returns `continue` on the first line with reason `stage0_bypassed` (`stage_0_short_segment_triage.py:63`). Segment length is algorithm 1's problem now. Its 40-line docstring still describes the threshold-300 triage it no longer performs |
| `stage_14_single_reach_moderate_displacement` | `MIN_SUSTAINED_DISPLACEMENT_FRAMES = 100000` (line 53) |
| `stage_15_multi_reach_retrieved_above_slit` | `MIN_POST_ABOVE_SLIT = 10000` (line 60) |
| `stage_18_displaced_via_first_significant_displacement` | `DISPLACEMENT_RADII_MIN = 100000` (line 59) |
| `stage_19_retrieved_via_pillar_lk_first_reach` | `MIN_PRE_LOW_FRAMES = 100000` (line 62) |
| `stage_20_per_bout_classifier_displaced` | `MIN_PRE_BOUT_ON_PILLAR = 100000` (line 73) |
| `stage_23_retrieved_with_pillar_tip_noise` | `CLUSTER_STD_PX_MAX = 0` (line 61) — a standard deviation can never be below zero |
| `stage_24_transition_triangulation` | `MIN_TRANSITION_STRENGTH = 100.0` (line 58) |

Each of the seven threshold-disabled ones carries a dated comment explaining what was tried and why it did not reach acceptable accuracy, and says the file is kept for documentation. That is a defensible choice, but it means the docstrings at the top of those files describe behaviour that does not happen, and the archived data above confirms none of them has ever committed anything.

Two helper modules are dead as a result: `pellet_calibration.py` is imported only by stage 20, and `transition_detector.py` only by stage 24. A third, `trust_calibrator.py` (225 lines, computes per-stage agreement with ground truth), is **imported by nothing at all**. Its docstring says "the trust score is what determines triage at runtime" — no runtime code reads a trust score. Triage is decided entirely by the ordering and the hard-coded thresholds.

---

## When it declines to decide

A segment comes out as `"outcome": "triaged"` with `"flagged_for_review": true` and a `flag_reason` string, from exactly four places:

1. **`stage_99_residual_triage`** — nothing committed. Reason `fell_through_all_committing_stages`. This is by far the largest source (2,137 of 2,905 triages in the archive).
2. **The pixel guard on displaced commits** — `cv_artifact_landing_no_pellet_arrival`. A stage was willing to say "displaced", but the video shows nothing arrived at the landing spot. 690 in the archive, attributed to whichever stage tried to commit (mostly 7, 16, 27, 21).
3. **`stage_27`, causal reach too near a segment edge** — `causal_reach_too_near_segment_start` / `..._end`. 357 in the archive.
4. **`stage_98_lost_in_shadow_triage`** — the deliberate "this is unscorable and here is why" case. 3 in the archive.
5. **`stage_5`**, when a pellet that started in the tray appears to vanish. Zero occurrences in the archive.

`detector.py:257-268` also has a fallback for a segment where every stage returned `continue`, producing `stage: "residual (auto-triage)"` and reason `fell_through_all_stages`. Since stage 99 always triages, this cannot happen.

### Stage 98 in detail

Stage 98 is the only stage whose purpose is to give a triage a *specific* reason instead of the generic one. It fires when tracking lost the pellet **and** the video shows the tray is genuinely too dark to track in. Its gates, in order (`stage_98_lost_in_shadow_triage.py:128-405`):

- pellet detected (≥ 0.7) in less than 10% of clean-zone frames — otherwise tracking worked, defer
- no in-mouth signal: fewer than 3 sustained frames of the pellet above the slit line — otherwise this is a retrieval, defer
- at least 3 sustained frames of the pellet off-pillar in the tray — otherwise a pure retrieval, defer
- then it opens the video, samples 10 evenly spaced frames, takes the bounding box of the four tray corners, and measures the fraction of pixels darker than intensity 60. If no frame reaches 30% dark, defer

**It only looks for `{video_id}.mp4`** (`stage_98_lost_in_shadow_triage.py:120-126`). The directory search that supplies it accepts `.avi`, `.mp4` and `.mkv` (`core/batch.py:147`). For a video stored as `.avi`, `video_dir` is found, this stage reports `video_unavailable`, and the segment falls to stage 99 with the generic reason. The pixel guard on displaced commits does not have this problem — it checks both extensions.

Every failure inside stage 98 is swallowed into a `continue`: `cv2` not installed, video will not open, tray box degenerate. Nothing is logged. The result looks identical to "not a shadow case".

---

## Exactly what lands in `{video}_pellet_outcomes.json`

Written by `core/batch.py:210-211` (pipeline) or `cli.py:186-188` (command line). Built at `detector.py:246-273`. Here is a real file, unedited:

```json
{
  "video_id": "20250624_CNT0103_P1",
  "detector": "v6_cascade",
  "detector_version": "6.1.0",
  "segments": [
    {
      "segment_num": 1,
      "outcome": "displaced_sa",
      "outcome_known_frame": 1684,
      "interaction_frame": 1664,
      "stage": "stage_7_settled_off_pillar_late",
      "flagged_for_review": false
    }
  ]
}
```

Top level — exactly four keys, no more:

| Key | Value |
|---|---|
| `video_id` | whatever the caller passed; the pipeline derives it from the pose filename by splitting on `"DLC"` (`core/batch.py:193`) |
| `detector` | the constant string `"v6_cascade"` |
| `detector_version` | `"6.1.0"`. Read by `pipeline/manifest.py:155` to decide whether a video needs reprocessing |
| `segments` | the list |

Per segment, committed case — exactly six keys:

| Key | Value |
|---|---|
| `segment_num` | 1-based position in the segment list |
| `outcome` | `retrieved`, `displaced_sa` or `untouched` |
| `outcome_known_frame` | integer frame, absolute in the video |
| `interaction_frame` | integer frame, or `null` for `untouched` |
| `stage` | the build label of the deciding stage |
| `flagged_for_review` | always `false` here |

Per segment, triaged case — the same six keys plus `flag_reason`, with `outcome` set to `"triaged"` and both frame fields set to `null`. The human-readable reason string is the **only** surviving trace of why the cascade behaved as it did.

### What is not in the file

This is the part most worth writing down. Every one of the following is either declared in the documentation, produced by the code, or expected by a consumer — and none of it is written by v6.

**Computed and thrown away:**
- `StageDecision.features` — every measurement each stage made. Roughly 40 stages' worth of diagnostics, discarded per segment.
- `StageDecision.reason` for committed segments. The reason string is built (often with the actual numbers interpolated into it) and then dropped; only triage reasons survive.

**Documented but never produced:**
- `confidence` — advertised per segment in `outcomes/core/__init__.py:70`, in `cli.py:38-42`, and used as the triage gate in `core/triage.py:128-135`. v6 has no notion of confidence at all.
- `causal_reach_id` / `causal_reach_frame` — advertised at `outcomes/core/__init__.py:71` and in `AGENTS.md`. v6 identifies a causal reach internally in most touched stages and does not record which one it was. Causal reach attribution is a **separate step** (algorithm 4) that writes `{video}_reach_assignments.json` (`assignment/run.py:70-72`). It never writes back into the outcomes file.
- `summary` — the per-video counts block. Not written. `core/triage.py:81` and `_review.py:84` both index it directly.
- `n_segments`, `video_name`, `total_frames`, `detected_at`, `validated`, `validated_by`, `corrections_made`, `segments_flagged` — all fields of the legacy `VideoOutcomes` dataclass (`core/pellet_outcome.py:143-159`). v6 writes none of them. Note `video_name` vs `video_id`: the key changed and readers that use `data.get('video_name')` now get `None`.
- `validation_status` — documented in `cli.py:41-58` as the field that drives the whole review workflow (`auto_approved` / `needs_review` / `validated`). Nothing writes it at detection time. `pipeline/manifest.py:229` reads it and records `"unknown"` for every video.
- `pellet_visible_start` / `pellet_visible_end` / `distance_from_pillar_start` / `distance_from_pillar_end` — legacy-only.

**Present only in the legacy detector's output.** Archived files with `detector_version: 2.4.4` do carry `confidence`, `causal_reach_id`, `summary`, `n_segments`, `distance_from_pillar_*` — and always `"outcome_known_frame": null`, because the legacy detector documents that field as "set manually by annotator" and never fills it. So a consumer reading a mixed corpus sees the *older* files carrying more fields than the newer ones.

### What later steps add

The napari triage-clearing tool writes back into this file when a human resolves a segment (`review/triage_clearing.py:564-594`). It sets, on that segment only: `flagged_for_review: false`, `triage_cleared: true`, `human_verified: true`, `cleared_by`, `cleared_at`, `original_triage_reason`, `original_outcome` (if the human changed it), `causal_reach_id`, and overwrites `outcome` / `interaction_frame` / `outcome_known_frame`. So `causal_reach_id` appears in an outcomes file **only** for human-cleared segments.

`flagged_for_review` is what routes a segment into the human worklist (`review/triage_queue.py:196`). That module also independently triages any segment whose outcome is touched but for which algorithm 4 committed no causal reach, even though `flagged_for_review` is false — so the file's own flag is not the complete picture of what needs review.

---

## Things that are broken or misleading

**Three of the four installed command-line tools do not work against v6 output.**

- `mousereach-advance-outcomes` crashes immediately. `cli.py:340` calls `advance_videos(args.input, require_validation=not args.force)`; the function signature is `advance_videos(input_dir, output_dir, verbose=True)` (`core/advance.py:38`). There is no `require_validation` parameter and `output_dir` is required. Every invocation raises `TypeError`.
- `mousereach-review-pellet-outcomes` crashes on any v6 file. `_review.py:84` does `data['summary']` on the raw JSON; v6 files have no `summary` key. It also offers `displaced_outside`, `no_pellet` and `uncertain` as corrections, which the detector never produces.
- `mousereach-triage-outcomes` does not crash but does nothing useful. `core/triage.py:80-81` requires `data['n_segments']` and `data['summary']`; on a v6 file the `KeyError` is caught and the video is recorded as a load error, then marked `needs_review` with reason set to the exception text. Every video, always. The function is additionally marked deprecated and raises a `DeprecationWarning` when called.
- `mousereach-detect-outcomes` works but silently loads zero reaches from current reach files (see "Inputs" above).

**Docstrings that describe something other than what the code does.** These have been checked individually against the code at this commit:

- `v6_cascade/__init__.py:5` and `detector.py:9` say "30 stages". There are 33 (stage 6b is extra, and 0/98/99 are outside the 0–29 numbering).
- `detector.py:29` says `detector_version` is `"6.0.0"`. It is `6.1.0`.
- Every stage file numbered 1, 2, 4, 5, 8 opens with a *different* stage number in its first line ("Stage 1: Pellet-stable-untouched" heads `stage_2_...py`; "Stage 4:" heads `stage_5_...py`; "Stage 5:" heads `stage_8_...py`). The cascade was renumbered and the headers were not.
- `stage_10_pillar_revealed_after_reach.py` has nothing to do with the pillar being revealed; the filename is left over from an earlier design. `stage_11_single_reach_clean_displacement.py` commits `retrieved`, not a displacement.
- `stage_5_pellet_off_pillar_throughout.py:33-44` says co-detection of pellet and pillar at the same place is triaged. The code defers instead (line 306-320), with an inline comment explaining the change. The docstring above was not updated.
- `stage_2`'s commit comment (line 186-193) says the emitted frame is `seg_end - 10`; the code emits `seg_end - 5` (line 168).
- `stage_11`'s docstring says "< 10 sustained frames" and "≤ 2 late"; the constants are 100 and 5.
- `stage_26` and `stage_27` docstrings say the displacement threshold is 12 px; both constants are 10.0.
- `guards.py:10` says the vanish guard applies to four named stage classes; it applies to all of them.
- `outcomes/AGENTS.md` and `outcomes/core/__init__.py` describe the legacy geometric algorithm, ruler units, `displaced_outside`, confidence scores and the `validation_status` workflow — the entire v6 cascade is absent from both.

**Parameters that have no effect.**
- `Stage2PelletStableUntouched` is constructed with `commit_distance_radii=1.5` (`detector.py:115`). The attribute is stored (`stage_2:88`) and never read. Its `start_end_window` parameter is likewise unused, and the feature its docstring names as an input, `pellet_position_start_end_distance_in_radii`, is never computed.
- `stage_2` imports `detect_tray_motion_onset` and never calls it.
- `DISPLACED_VANISH_GUARD_CLASSES` is imported by `detector.py:88` and never used.

**Failures that are swallowed.** `process_batch` (`core/batch.py:295-299`) and the v6 CLI loop (`cli.py:190-193`) both catch every exception per video, print `FAILED: <message>`, and continue. Nothing is raised and no file is written; a video with a corrupt pose file simply has no outcomes file afterwards. Inside stage 98 and the pixel guard, every video-access failure becomes a silent `continue`.

---

## Configuration

There is almost none. This subsystem has **no config file, no environment variables and no tunable settings**. Every threshold in the cascade is a module-level constant in the stage file that uses it. Changing behaviour means editing code.

The only things that vary at runtime:

| What | Where | Effect |
|---|---|---|
| `legacy=True` / `--legacy` | `core/batch.py:177`, `cli.py:96` | Runs the old geometric detector instead. Different algorithm, different output shape, different version string. No production caller sets it |
| `video_dir` | `core/batch.py:151`, defaults to searching the pose file's directory | If no video file is found, the pixel-based artefact guard and stage 98 both become no-ops. Displaced commits that would have been triaged are committed instead. Nothing reports that this happened |
| Tray type | `pipeline/run_all.py:89`, `watcher/orchestrator.py:1062`, `:1971` | Tray `E` or `F` → the whole step is skipped, no file written |
| Whether reaches exist | `core/batch.py:208` | No reach file → empty reach list → most touched stages defer → heavy triage |

The stage list itself (`detector.py:112-146`) is hard-coded and identical for every video. There is no way to enable, disable or reorder a stage without editing `detector.py` or the stage's constants.
