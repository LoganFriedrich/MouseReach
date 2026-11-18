# MouseReach Algorithm Reference

> **Update 2026-11-18 -- read this first.** Only the "Pipeline Overview" and the
> "Version Summary" below describe the code that runs today. The detailed sections
> 1-3 (Segmentation v2.1.0, Reach Detection v3.5.0, Outcome Classification v2.4.4)
> and "Algorithm Evolution and Evidence" are HISTORY: they describe detectors that
> have been replaced and no longer run. They are kept for the record, not as a
> description of current behavior. For the current algorithms read:
>
> - `docs/ALGO_1_SEGMENTATION.md` -- segmentation
> - `docs/ALGO_2_REACH_DETECTION.md` -- reach detection (where each reach starts and ends)
> - `docs/ALGO_3_OUTCOME_CASCADE.md` -- outcome classification
> - `docs/ALGO_4_REACH_ASSIGNMENT.md` -- which reach caused each pellet's outcome
> - `docs/KINEMATICS_FIELDS.md` -- the kinematic values computed for each reach

## Pipeline Overview

The MouseReach pipeline analyzes video of mice performing a single-pellet reaching task. A DeepLabCut model tracks body parts frame by frame, producing x/y coordinates and a confidence score (likelihood) for each point. Four algorithms then run in order on this tracking data:

1. **Segmentation** finds the 21 pellet presentations on a pillar tray. Easy and Flat tray videos are not segmented: the whole video is one span.
2. **Reach Detection** finds every reach in the whole video. It is a trained classifier (a scikit-learn gradient-boosted model) that scores every frame; it is not a set of hand-written rules.
3. **Outcome Classification** decides what happened to each pellet (retrieved, displaced in the scoring area, or untouched). It is an ordered cascade of 33 rule-based stages; a pellet no stage can decide is sent to a person for review (triage).
4. **Reach Assignment** decides which reach caused each pellet's outcome.

Kinematics (speeds, distances, timing) are then computed for each reach. The kinematics read the frame rate from each video (or from the pose tool's record of it) and assume no fixed rate. (Segmentation still records a fixed 60 frames per second in its output file; it does not measure the rate.)

All spatial measurements use a "ruler unit" defined as the distance between SABL and SABR (scoring area bottom-left and bottom-right), which corresponds to 9mm in physical space.

---

## 1. Segmentation (v2.1.0)

**Source:** `src/mousereach/segmentation/core/segmenter_robust.py`

### What It Detects

Each session presents 21 pellets via an automated dispensing tray. The tray advances forward to position each pellet at the pillar. The algorithm detects the 21 moments when the tray advances by tracking the motion of scoring area anchor points.

### Primary Detection Method: SABL-Centered Crossing

The SABL (Scoring Area Bottom Left) point is physically attached to the tray and moves with it. The algorithm:

1. Smooths the SABL x-position with a 5-frame median filter to remove tracking noise
2. Computes frame-to-frame velocity
3. Detects frames where SABL crosses the box center (midpoint of BOXL and BOXR) with positive velocity exceeding 0.03 ruler units/frame
4. Enforces a minimum interval of 300 frames (~5 seconds) between detections to prevent duplicates

The box center crossing is evaluated within a window of -5 to +10 pixels, and the velocity threshold starts at 1.2 but is lowered to 0.8 if fewer than 21 candidates are found.

### Secondary Validation

Each primary detection is validated by checking whether other scoring area points (SABR, SATL, SATR) show correlated motion within a 3-frame alignment window. Validated detections receive higher confidence scores.

### Fallback Detection: Motion Peak Finding

If the primary method fails to find 21 boundaries (due to poor SABL tracking), a fallback method activates:

1. Computes velocity peaks across the full SABL signal
2. Finds peaks with minimum spacing of 20 seconds and prominence above the 85th percentile of the velocity distribution
3. These detections receive a confidence score of 0.5 (vs. ~0.95 for primary detections)

### Grid Fitting

If the combined primary + fallback detections still don't yield exactly 21 boundaries, the algorithm fits the candidates to an expected interval grid (1839 frames / ~30.65 seconds between presentations). It interpolates missing boundaries and removes extras based on deviation from the grid.

### Quality Assessment

Reference point stability determines overall confidence:
- BOXL/BOXR standard deviation < 5 pixels: "good" quality
- < 15 pixels: "suspect" quality
- Above 15 pixels: "bad" quality (camera vibration or tracking failure)

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Velocity threshold | 0.03 ruler/frame | Minimum tray advance speed |
| Smoothing window | 5 frames | Median filter for noise |
| Minimum interval | 300 frames | ~5 sec, prevents duplicates |
| Maximum interval | 1200 frames | ~20 sec, flags gaps |
| Expected boundaries | 21 | Standard pellet count |
| Expected interval | 1839 frames | ~30.65 sec between pellets |
| Crossing range | -5 to +10 pixels | Window around box center |
| Alignment window | 3 frames | Multi-anchor validation |

---

## 2. Reach Detection (v3.5.0)

**Source:** `src/mousereach/reach/core/reach_detector.py`

### What It Detects

Within each pellet presentation segment, the mouse may make zero or more reaching attempts through the slit. Each reach is defined by a start frame (hand emerges through slit), an apex (maximum extension), and an end frame (hand retracts). The algorithm identifies these events by tracking hand visibility while the mouse's nose is engaged at the slit.

### Detection Logic: State Machine

The detector operates as a three-state machine for each segment:

**State 1: IDLE**
The mouse is not engaged. Transitions to ENGAGED when the nose comes within 25 pixels of the slit center (midpoint of BOXL and BOXR).

**State 2: ENGAGED**
The mouse's nose is at the slit. The detector watches for hand emergence. Transitions to REACHING when any of the four hand tracking points (RightHand, RHLeft, RHOut, RHRight) reaches a DLC confidence of 0.5 or higher. This is the reach start frame.

**State 3: REACHING**
A reach is in progress. The detector tracks hand position to determine the apex (frame of maximum extension from slit) and watches for the reach end. The reach ends when the first of three conditions is met:

1. **Hand disappearance:** All four hand points drop below 0.5 confidence for 2 or more consecutive frames. This catches the common case where the hand retracts back through the slit and DLC loses tracking.

2. **Hand retraction:** The hand moves back toward the slit by more than 40% of its maximum extension distance. This catches slow retractions where DLC maintains tracking.

3. **Hand return:** The hand position returns within 5 pixels of the slit center after having been extended more than 5 pixels beyond it.

After a reach ends, the state returns to IDLE (or ENGAGED if the nose is still at the slit).

### Post-Processing

After initial detection, three post-processing steps refine the results:

**Merging:** Reaches separated by 2 or fewer frames are merged into a single reach. This handles brief DLC tracking dropouts mid-reach that would otherwise split one reach into two.

**Splitting:** Reaches longer than 25 frames (the 95th percentile of ground truth reach duration) are examined for confidence valleys. If hand confidence drops from above 0.5 to below 0.35 then rises again, the reach is split at the valley. This handles cases where the algorithm concatenates two distinct reaches.

**Filtering:** Reaches with maximum extent below -15 pixels (hand behind the slit, toward the mouse) are removed. These are grooming or paw-adjustment movements, not reaching attempts. Reaches shorter than 4 frames are also removed (ground truth analysis showed 42% of false positives were 3 frames or shorter).

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Hand confidence threshold | 0.5 | Minimum DLC confidence for hand detection |
| Nose engagement distance | 25 pixels | Maximum nose-to-slit distance |
| Minimum reach duration | 4 frames | Eliminates brief false positives |
| Hand disappearance threshold | 2 frames | Consecutive frames below confidence |
| Retraction fraction | 0.40 | 40% of max extension triggers end |
| Hand return threshold | 5 pixels | Distance from slit for "returned" |
| Gap tolerance (merge) | 2 frames | Maximum gap to merge across |
| Split threshold | 25 frames | Duration above which splitting is attempted |
| Confidence valley | 0.35 | Low confidence that triggers split |
| Minimum extent | -15 pixels | Filters behind-slit detections |

### Hand Tracking Points

The four DLC-tracked hand points and their typical visibility:
- **RightHand** — dorsal surface of the paw, most reliably tracked
- **RHLeft** — left edge of paw (from camera perspective)
- **RHOut** — outer/distal edge of paw
- **RHRight** — right edge of paw (from camera perspective)

A reach starts when ANY of these four exceeds 0.5 confidence. The reach is active as long as at least one remains above threshold.

### Known Limitations

- **DLC tracking dropout:** When the hand moves quickly or is partially occluded, DLC confidence drops below 0.5 and the algorithm cannot see the reach. This is the primary source of missed reaches (~7% false negative rate).
- **End frame bias:** DLC confidence often dips momentarily 1-2 frames before the hand fully retracts, causing the algorithm to end reaches slightly early. This introduces a systematic ~1 frame bias in kinematic measurements.
- **Ambiguous paw-at-slit:** When the mouse rests its paw at the slit edge without extending through, all four hand points may briefly exceed 0.5 confidence. The minimum duration (4 frames) and minimum extent (-15 pixels) filters catch most but not all of these.

---

## 3. Outcome Classification (v2.4.4)

**Source:** `src/mousereach/outcomes/core/pellet_outcome.py`

### What It Classifies

For each pellet presentation segment, the algorithm determines what happened to the pellet:

| Outcome | Definition |
|---------|------------|
| **retrieved** | Mouse successfully grasped and consumed the pellet |
| **displaced_sa** | Mouse knocked the pellet off the pillar; pellet remains in the scoring area |
| **displaced_outside** | Mouse knocked the pellet outside the scoring area |
| **untouched** | Pellet remained on the pillar throughout the segment |
| **no_pellet** | No pellet was detected (operator error or dispensing failure) |
| **uncertain** | Tracking data is ambiguous; requires human review |

### Detection Logic: Four-Stage Progressive Validation

The classifier evaluates each segment through four stages of increasing specificity. Each stage can produce a final classification or pass to the next stage for further analysis.

**Stage 0: Early Retrieval Detection**

Checks whether the pellet disappeared while a paw was nearby:
- Pellet visibility (DLC confidence) must drop from above 0.8 to below 0.3
- At least one hand point must be within 30 pixels of the pellet position at the time of disappearance
- If the pellet was more than 0.4 ruler units from the pillar when grabbed, it's classified as `displaced_outside` rather than `retrieved`

**Stage 1: Initial Hypothesis**

Evaluates the overall pellet trajectory:
- **Pellet disappeared** (visibility below 10% of segment): Likely `retrieved`
- **Pellet never appeared** (no frames above 0.5 confidence): `no_pellet`
- **Pellet started far from pillar** (>0.3 ruler units at segment start): Flagged as potential operator error, classified as `untouched`
- **Otherwise:** Passes to Stage 2 for displacement analysis

**Stage 2: Feature Validation**

Two key checks:

*Sustained Displacement Detection:* The pellet's distance from the pillar must exceed 0.30 ruler units (~2.7mm) for at least 10 consecutive frames. Brief excursions (e.g., from tracking jitter) are ignored. The algorithm also identifies the displacement onset frame by looking backward for the moment when pellet confidence dropped (indicating paw occlusion).

*Paw Proximity Verification:* For any detected displacement, the algorithm looks back 30 frames from the displacement onset to confirm a hand point was within 30 pixels of the pellet. Displacement without paw proximity suggests tray wobble (the tray advancing caused apparent pellet motion) rather than a real interaction. If no paw was nearby and the pellet ends near the pillar, the classification is `untouched`.

**Stage 3: Temporal Consistency**

Resolves ambiguous cases from Stage 2:
- **Displaced then disappeared:** If the pellet was displaced but then its visibility drops to zero, it was likely retrieved after displacement. Checks for eating signature (see below).
- **Pellet moved with tray:** If the pellet's position relative to the scoring area anchor points remained constant while its absolute position changed, the motion was from tray advancement, not mouse interaction. Classified as `untouched`.
- **Pellet displaced then returned:** If the pellet moves away from pillar then returns, checks for eating signature before classifying.

*Eating Signature Detection:* A strong indicator of retrieval. The algorithm looks for sustained periods (30+ frames) where the nose is retracted from the slit (>30 pixels from BOXR) AND a hand point is near the nose/mouth (<30 pixels). This pattern indicates the mouse is holding and eating the pellet.

**Stage 4: Confidence and Interaction Frame**

Assigns a confidence score and identifies the interaction frame (the frame when the pellet was first affected by the mouse). The interaction frame is used to determine the causal reach — which reaching attempt caused the pellet outcome. The causal reach is the most recent reach whose temporal window contains the interaction frame.

### Spatial Reference: Pillar Geometry

The pellet sits on a pillar whose position is calculated geometrically from the scoring area points. The pillar is located at the apex of a 55-degree isosceles triangle whose base is the SABL-SABR line, at a perpendicular distance of 0.944 ruler units (~8.5mm) from that line. All "distance from pillar" measurements reference this calculated position.

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Pellet confidence threshold | 0.5 | Minimum DLC confidence for pellet tracking |
| SA point confidence threshold | 0.8 | Minimum for scoring area reference points |
| On-pillar threshold | 0.20 ruler | ~1.8mm, pellet considered on pillar |
| Displaced threshold | 0.25 ruler | ~2.25mm, minimum for displacement |
| Sustained displacement duration | 10 frames | Minimum for real displacement |
| Sustained displacement distance | 0.30 ruler | ~2.7mm, minimum distance |
| Paw proximity threshold | 30 pixels | Hand near pellet for grab detection |
| Eating distance threshold | 30 pixels | Hand-to-nose for eating signature |
| Eating duration | 30 frames | Minimum frames for eating behavior |
| Pillar perpendicular distance | 0.944 ruler | ~8.5mm, geometric constant |
| Skip start frames | 20 | Ignore first 20 frames of segment |
| Skip end frames | 30 | Ignore last 30 frames of segment |

### Known Limitations

- **Tray wobble false positives:** When the tray advances, the pellet moves in absolute coordinates even if it stays on the pillar. The tray-relative motion filter (Stage 3) catches most cases but subtle wobble during slow tray movement can trigger false `displaced_sa` classifications.
- **Pellet occlusion by paw:** When the mouse's paw covers the pellet, DLC confidence drops, which can look like pellet disappearance (retrieval). The paw proximity check (Stage 2) helps distinguish, but fast grabs where the paw is near the pellet for only 1-2 frames can be missed.
- **Multiple reaches per segment:** When several reaches occur before the outcome, attributing the correct causal reach is ambiguous. The algorithm assigns the most recent reach, which may not be correct if an earlier reach displaced the pellet.

---

## Physical Geometry Reference

All spatial measurements use ruler units (1 ruler = SABL-to-SABR distance = 9mm).

| Measurement | Ruler Units | Millimeters | Description |
|-------------|-------------|-------------|-------------|
| Ruler (SABL-SABR) | 1.000 | 9.0 | Reference distance |
| Pillar diameter | 0.458 | 4.125 | Pellet platform |
| Pellet diameter | 0.278 | ~2.5 | Sucrose pellet |
| Pillar to SA edge | 1.069 | 9.618 | Pillar center to scoring area |
| Pillar perpendicular | 0.944 | 8.5 | Pillar distance from SA baseline |
| SABR to SATR | 1.667 | 15.0 | Scoring area height |

---

## Evaluation Results

Accuracy figures measured against a lab's own hand-labeled videos describe that lab's
data, not the tool, so they are not kept in this repository. Keep your evaluation
results with your own data. (Update 2026-11-18: the table that stood here described
detector versions that no longer run.)

---

## Algorithm Evolution and Evidence (history, pre-v8)

> Update 2026-11-18: everything in this section describes detectors that no longer run (reach detection v1-v3.5, segmentation v1-v2.1, outcome classification v1-v2.4.4). Where it says "current", read "current at the time". The rules found here shaped the retired rule-based reach detectors; the v8 detector learns reach frames from annotated videos and applies no nose gate.

This section documents why the algorithm versions of that time were what they were — what was tried, what failed, and what the data showed.

### Reach Detection: From Naive to Data-Driven

Reach detection underwent the most significant evolution, driven by a ground truth analysis of a few hundred manually-labeled reaches from a single exhaustively labeled video.

**Version progression and measured performance:**

| Version | Approach | Precision | Recall | F1 | What Changed |
|---------|----------|-----------|--------|-----|-------------|
| v1.0.0 | Extent > 5px filter | 87.2% | 45.2% | 0.59 | First attempt: only count reaches extending >5px past slit. Missed all short reaches. |
| v2.0.0 | Extent > 0px filter | 82.4% | 62.8% | 0.71 | Relaxed threshold: any hand past slit counts. Still missed reaches where hand doesn't cross slit plane. |
| v3.0.0 | Extent >= 5px + duration >= 10 | 91.3% | 48.1% | 0.63 | Strict filtering: high precision but missed half of real reaches. Too conservative. |
| v3.3.0 | Extent >= 0px (BUG) | 93.1% | **14.7%** | 0.25 | **Catastrophic regression.** A filter intended to remove negative-extent reaches (`extent >= 0`) inadvertently rejected valid reaches where the hand approaches but doesn't fully cross the slit. Lost 85% of detected reaches. |
| v3.4.0 | Duration >= 2 only | 78.9% | **98.3%** | 0.88 | **Recovery.** Removed all extent filtering. Kept only minimum duration. Recall jumped from 15% to 98%. Accepted the precision trade-off. |
| v3.5.0 | Duration >= 4, extent >= -15px | 81.0% | 93.1% | 0.85 | Re-introduced a conservative negative extent filter (-15px) after GT analysis showed 44% of false positives had extent below -10px. Added minimum duration of 4 frames after finding 42% of FPs were 3 frames or shorter. |

**The critical lesson from v3.3.0:** Negative extent values are scientifically valid. A mouse can reach toward the slit without its hand crossing the slit plane. Filtering these out destroys recall. The current v3.5.0 only filters reaches where the hand is far behind the slit (< -15 pixels), preserving approach-but-no-cross reaches.

**Ground truth analysis findings (DISCOVERED_RULES.md):**

The three core detection rules were empirically derived:

1. **Nose Engagement Gate** — 95th percentile of nose-to-slit distance at labeled reach starts was 21.3 pixels. Threshold set to 25px for margin. This rule ensures the mouse is actively oriented toward the slit, not grooming or exploring elsewhere.

2. **Reach Start = Hand Appearance** — 100% of the ground truth reaches had at least one hand point visible at start (RightHand: 99.7%, RHRight: 99.7%, RHLeft: 99.1%, RHOut: 91.9%). This justified the "any hand point above threshold" rule.

3. **Reach End: Two Modes** — 94.6% of reaches ended with hand disappearance (mean 1.6 frames after marked end). The remaining 5.4% ended with hand retraction (leftward movement) into the next reach. 90.5% of inter-reach gaps showed leftward hand motion.

**Key statistics from ground truth:**

| Metric | Value |
|--------|-------|
| Median reach duration | 12 frames |
| Mean reach duration | 22 frames |
| 95th percentile duration | 25 frames (used as split threshold) |
| Median inter-reach gap | 30 frames |
| Minimum inter-reach gap | 3 frames |
| Nose distance at start (median) | 14.5 pixels |
| Nose distance at start (95th percentile) | 21.3 pixels |

### Segmentation: Stable After v2.0

Segmentation has been stable since v2.0. The SABL-centered crossing method works reliably because it tracks a mechanical signal (tray movement) rather than mouse behavior. The main improvements were:

- **v1.0:** Initial SABL crossing detection. Worked for most videos but failed when SABL tracking quality was poor.
- **v2.0:** Added multi-anchor validation (checking SABR, SATL, SATR for correlated motion) and fallback peak-finding when the primary method failed. Also added the grid-fitting step to enforce exactly 21 boundaries.
- **v2.1.0:** Refined velocity thresholds and improved handling of late-start videos where the first boundary occurs after frame 1000.


### Outcome Classification: Progressive Refinement

Outcome classification evolved through iterative refinement of the pellet tracking logic:

- **v1.0:** Initial geometric classification based on pellet distance from pillar. High false positive rate for `displaced_sa` because tray wobble caused apparent pellet motion.
- **v2.0:** Added pillar geometry calculation using the 55-degree isosceles triangle model. Improved displacement threshold from pixel-based to ruler-unit-based, making it robust to different camera magnifications.
- **v2.3:** Added tray-relative motion filtering (Stage 3) to distinguish real displacement from tray wobble. Added causal reach attribution using the interaction frame.
- **v2.4.4:** Improved handling of pellet start/end position for cases where the pellet starts slightly off-pillar (operator error) or is retrieved very quickly. Fixed edge cases with pellets that disappear immediately on contact.

The remaining failure mode is over-calling `displaced_sa` on videos with unusual tray dynamics (notably one ground-truth video at 70% accuracy, vs. 95-100% for the others).

### Version Simulator Tool

A version simulator (`src/mousereach/eval/version_simulator.py`) can replay historical algorithm versions against current ground truth data. This allows direct comparison of how each version would perform on the same data:

```bash
python -m mousereach.eval.version_simulator Processing Processing
```

This produces a table showing precision, recall, and F1 for each historical version.

> **Update 2026-11-18 -- WARNING, run it only on a copy.** The simulator replays the
> filters of the old rule-based reach detectors (v0.5-v3.4) over existing reach files. While
> it runs it deletes each `{video}_reaches.json` in the folder you give it, puts a filtered
> file in its place, and writes the original back from memory afterwards. If it is stopped
> part-way (window closed, crash), the original file is left filtered or missing. Copy the
> reach and ground-truth files to a scratch folder and point it there -- never at a live
> pipeline folder such as `Processing`. Old versions that filter on reach extent stop with
> an error on reach files written by the current (v8) detector, which leaves extent empty.

---

## Version Summary

Versions that run today (Update 2026-11-18; each is the version constant in the code):

| Component | Version | Where it is described |
|-----------|---------|-----------------------|
| Segmentation | 2.2.4 | `docs/ALGO_1_SEGMENTATION.md` |
| Reach Detection | 8.1.0 (v8 trained classifier) | `docs/ALGO_2_REACH_DETECTION.md` |
| Outcome Classification | 6.1.0 (33-stage v6 cascade) | `docs/ALGO_3_OUTCOME_CASCADE.md` |
| Reach Assignment | 2.1.0 | `docs/ALGO_4_REACH_ASSIGNMENT.md` |
| Kinematics extractor | 2.6.0 | `docs/KINEMATICS_FIELDS.md` |

---

*This document describes the algorithms as implemented in code. For the source of each parameter value, see the files referenced at the top of each section. For ground truth analysis details, see `src/mousereach/reach/analysis/DISCOVERED_RULES.md`.*