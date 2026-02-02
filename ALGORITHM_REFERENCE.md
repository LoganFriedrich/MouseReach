# MouseReach Algorithm Reference

> Auto-generated reference for the MouseReach analysis pipeline algorithms.
> For the most current parameter values, see the source files referenced in each section.

## Pipeline Overview

The MouseReach pipeline analyzes high-speed video (60 fps) of mice performing a single-pellet reaching task. A DeepLabCut model tracks 18 body parts frame-by-frame, producing x/y coordinates and confidence scores for each point. Three sequential algorithms process this tracking data:

1. **Segmentation** identifies the 21 pellet presentation events per session
2. **Reach Detection** identifies individual reaching attempts within each presentation
3. **Outcome Classification** determines what happened to each pellet

Each algorithm is rule-based (no machine learning beyond the DLC pose estimation). All spatial measurements use a "ruler unit" defined as the distance between SABL and SABR (scoring area bottom-left and bottom-right), which corresponds to 9mm in physical space.

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

<!-- AUTO-GENERATED PERFORMANCE SECTION START -->
## Current Performance

Evaluated 2026-01-30 against 19 ground truth videos with human-verified annotations. Run `python -m mousereach.eval.report_cli` to regenerate these numbers and produce plots.

### Aggregate Metrics

| Metric | Value | Corpus Size |
|--------|-------|-------------|
| Segmentation boundary recall | 98.7% | 19 videos (399 boundaries) |
| Reach detection precision | 81.0% | 16 videos (1,071 matched reaches) |
| Reach detection recall | 93.1% | 16 videos |
| Reach detection F1 | 85.2% | 16 videos |
| Outcome classification accuracy | 95.8% | 19 videos (380 outcomes) |

Three videos were excluded from reach evaluation because they lack reach ground truth annotations (CNT0216_P3, CNT0213_P3, CNT0306_P2).

### Segmentation: Per-Video Boundary Recall

15 of 19 videos achieved 100% boundary detection. The 4 remaining videos each missed 1 boundary (20/21 detected = 95.2%).

| Video | Recall | Matched/Total |
|-------|--------|---------------|
| CNT0115_P2 | 95.2% | 20/21 |
| CNT0104_P2 | 95.2% | 20/21 |
| CNT0408_P1 | 95.2% | 20/21 |
| CNT0415_P1 | 95.2% | 20/21 |
| All others (15 videos) | 100% | 21/21 |

**Boundary timing accuracy** (394 matched boundaries):

| Tolerance | Count | Percentage |
|-----------|-------|------------|
| Exact match (0 frames) | 191 | 48.5% |
| Within 1 frame | 266 | 67.5% |
| Within 2 frames | 354 | 89.8% |
| Mean error | -0.64 frames | Slightly early |
| Standard deviation | 1.35 frames | |

The negative mean error indicates the algorithm tends to detect boundaries slightly before the human-marked frame — consistent with detecting the onset of tray motion rather than the moment the pellet reaches final position.

### Reach Detection: Per-Video Precision/Recall

Performance varies substantially across videos. Some videos achieve perfect detection; others have high false positive rates.

| Video | Precision | Recall | F1 | True Positives | False Positives | False Negatives |
|-------|-----------|--------|-----|----------------|-----------------|-----------------|
| CNT0104_P2 | 100% | 100% | 100% | 5 | 0 | 0 |
| CNT0102_P2 | 100% | 100% | 100% | 6 | 0 | 0 |
| CNT0216_P1 | 100% | 100% | 100% | 21 | 0 | 0 |
| CNT0301_P4 | 100% | 100% | 100% | 1 | 0 | 0 |
| CNT0305_P2 | 98% | 98% | 98% | 83 | 2 | 2 |
| CNT0401_P4 | 98% | 93% | 95% | 50 | 1 | 4 |
| CNT0415_P1 | 94% | 100% | 97% | 29 | 2 | 0 |
| CNT0103_P2 | 97% | 91% | 94% | 84 | 3 | 8 |
| CNT0408_P1 | 92% | 91% | 92% | 220 | 19 | 21 |
| CNT0115_P2 | 78% | 100% | 88% | 7 | 2 | 0 |
| CNT0413_P2 | 88% | 82% | 85% | 165 | 22 | 35 |
| CNT0309_P1 | 75% | 95% | 84% | 18 | 6 | 1 |
| CNT0312_P2 | 52% | 100% | 68% | 126 | 118 | 0 |
| CNT0307_P4 | 50% | 83% | 62% | 124 | 126 | 26 |
| CNT0110_P2 | 41% | 79% | 54% | 57 | 83 | 15 |
| CNT0311_P2 | 34% | 78% | 48% | 75 | 143 | 21 |

**Three problematic videos** (CNT0311_P2, CNT0110_P2, CNT0307_P4) have precision below 50%, meaning the algorithm detects more false reaches than real ones. These videos likely have challenging DLC tracking conditions (hand flicker, frequent paw-at-slit without reaching).

**Reach timing accuracy** (1,071 matched reaches):

| Metric | Start Frame | End Frame |
|--------|-------------|-----------|
| Exact match (0 frames) | 1,011 (94.4%) | 16 (1.5%) |
| Within 1 frame | 1,015 (94.8%) | 900 (84.0%) |
| Within 2 frames | 1,033 (96.5%) | 915 (85.4%) |
| Mean error | -0.10 frames | -1.30 frames |

**Start frames are highly accurate** — 94.4% exact match. This is because reach start is well-defined (hand first appears above confidence threshold).

**End frames have a systematic early bias** — mean error of -1.30 frames, with only 1.5% exact matches. The algorithm ends reaches ~1 frame before the human-marked end, consistent with DLC confidence dipping before the hand fully retracts. This is the most significant measurement concern because kinematic features (velocity, trajectory) are computed over the detected reach window. A 1-frame early end means the final phase of retraction is excluded from analysis.

### Outcome Classification: Per-Video Accuracy

13 of 19 videos achieved 100% accuracy. The remaining 6 videos:

| Video | Accuracy | Errors | Error Details |
|-------|----------|--------|---------------|
| CNT0311_P2 | 80% | 4 | 3 retrieved misclassified (2 as displaced_sa, 1 as uncertain), 1 displaced_outside as untouched |
| CNT0216_P3 | 80% | 4 | Mixed: displaced_sa/untouched confusion in both directions |
| CNT0408_P1 | 80% | 4 | 4 retrieved misclassified as displaced_sa |
| CNT0312_P2 | 95% | 1 | 1 retrieved as untouched |
| CNT0103_P2 | 95% | 1 | 1 untouched as displaced_sa |
| CNT0401_P4 | 95% | 1 | 1 untouched as displaced_sa |
| CNT0413_P2 | 95% | 1 | 1 displaced_outside as displaced_sa |

**Confusion matrix** (380 total classifications):

| GT \ Algorithm | retrieved | displaced_sa | displaced_outside | untouched |
|----------------|-----------|-------------|-------------------|-----------|
| **retrieved** | **33** | 6 | 0 | 1 |
| **displaced_sa** | 1 | **151** | 0 | 1 |
| **displaced_outside** | 0 | 1 | **1** | 1 |
| **untouched** | 0 | 4 | 0 | **179** |

**Key error pattern:** The dominant misclassification is `retrieved` being called `displaced_sa` (6 instances). This occurs when the mouse grabs the pellet quickly and the pellet tracker briefly shows displacement before the pellet disappears. The algorithm sees "pellet moved" before "pellet gone" and classifies the initial movement as displacement.

The secondary pattern is `untouched` being called `displaced_sa` (4 instances). This is the tray wobble problem — the pellet shifts slightly during tray advancement and the algorithm interprets it as mouse-caused displacement.

### Summary of Weaknesses

| Issue | Severity | Affected Videos | Impact |
|-------|----------|----------------|--------|
| Reach false positives (precision < 50%) | High | 3 of 16 | Corrupts kinematic analysis with non-reach data |
| Reach end frame 1-frame early bias | Medium | All | Systematic bias in kinematic measurements |
| retrieved → displaced_sa misclassification | Medium | 3 of 19 | Wrong outcome for 6/40 retrieval events |
| untouched → displaced_sa (tray wobble) | Low | 3 of 19 | 4 false displacement events |
<!-- AUTO-GENERATED PERFORMANCE SECTION END -->

---

## Algorithm Evolution and Evidence

This section documents why the current algorithm versions are what they are — what was tried, what failed, and what the data showed.

### Reach Detection: From Naive to Data-Driven

Reach detection underwent the most significant evolution, driven by a ground truth analysis of 321 manually-labeled reaches from video 20251021_CNT0405_P4 (37,109 frames, analyzed 2026-01-12).

**Version progression and measured performance:**

| Version | Approach | Precision | Recall | F1 | What Changed |
|---------|----------|-----------|--------|-----|-------------|
| v1.0.0 | Extent > 5px filter | 87.2% | 45.2% | 0.59 | First attempt: only count reaches extending >5px past slit. Missed all short reaches. |
| v2.0.0 | Extent > 0px filter | 82.4% | 62.8% | 0.71 | Relaxed threshold: any hand past slit counts. Still missed reaches where hand doesn't cross slit plane. |
| v3.0.0 | Extent >= 5px + duration >= 10 | 91.3% | 48.1% | 0.63 | Strict filtering: high precision but missed half of real reaches. Too conservative. |
| v3.3.0 | Extent >= 0px (BUG) | 93.1% | **14.7%** | 0.25 | **Catastrophic regression.** A filter intended to remove negative-extent reaches (`extent >= 0`) inadvertently rejected valid reaches where the hand approaches but doesn't fully cross the slit. Lost 85% of detected reaches. |
| v3.4.0 | Duration >= 2 only | 78.9% | **98.3%** | 0.88 | **Recovery.** Removed all extent filtering. Kept only minimum duration. Recall jumped from 15% to 98%. Accepted the precision trade-off. |
| v3.5.0 | Duration >= 4, extent >= -15px | 81.0% | 93.1% | 0.85 | **Current.** Re-introduced a conservative negative extent filter (-15px) after GT analysis showed 44% of false positives had extent below -10px. Added minimum duration of 4 frames after finding 42% of FPs were 3 frames or shorter. |

**The critical lesson from v3.3.0:** Negative extent values are scientifically valid. A mouse can reach toward the slit without its hand crossing the slit plane. Filtering these out destroys recall. The current v3.5.0 only filters reaches where the hand is far behind the slit (< -15 pixels), preserving approach-but-no-cross reaches.

**Ground truth analysis findings (DISCOVERED_RULES.md):**

The three core detection rules were empirically derived:

1. **Nose Engagement Gate** — 95th percentile of nose-to-slit distance at labeled reach starts was 21.3 pixels. Threshold set to 25px for margin. This rule ensures the mouse is actively oriented toward the slit, not grooming or exploring elsewhere.

2. **Reach Start = Hand Appearance** — 100% of 321 ground truth reaches had at least one hand point visible at start (RightHand: 99.7%, RHRight: 99.7%, RHLeft: 99.1%, RHOut: 91.9%). This justified the "any hand point above threshold" rule.

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
- **v2.1.0 (current):** Refined velocity thresholds and improved handling of late-start videos where the first boundary occurs after frame 1000.

Performance has been at 95-100% boundary recall since v2.0, now at 98.7% across 19 GT videos.

### Outcome Classification: Progressive Refinement

Outcome classification evolved through iterative refinement of the pellet tracking logic:

- **v1.0:** Initial geometric classification based on pellet distance from pillar. High false positive rate for `displaced_sa` because tray wobble caused apparent pellet motion.
- **v2.0:** Added pillar geometry calculation using the 55-degree isosceles triangle model. Improved displacement threshold from pixel-based to ruler-unit-based, making it robust to different camera magnifications.
- **v2.3:** Added tray-relative motion filtering (Stage 3) to distinguish real displacement from tray wobble. Added causal reach attribution using the interaction frame.
- **v2.4.4 (current):** Improved handling of pellet start/end position for cases where the pellet starts slightly off-pillar (operator error) or is retrieved very quickly. Fixed edge cases with pellets that disappear immediately on contact.

The remaining failure mode is over-calling `displaced_sa` on videos with unusual tray dynamics (notably CNT0311_P2 at 70% accuracy, vs. 95-100% for other videos).

### Version Simulator Tool

A version simulator (`src/mousereach/eval/version_simulator.py`) can replay historical algorithm versions against current ground truth data. This allows direct comparison of how each version would perform on the same data:

```bash
python -m mousereach.eval.version_simulator Processing Processing
```

This produces a table showing precision, recall, and F1 for each historical version, confirming that the current versions represent the best validated performance.

---

## Version Summary

| Component | Version | Key Changes |
|-----------|---------|-------------|
| Segmentation | v2.1.0 | SABL-centered crossing with multi-strategy fallback |
| Reach Detection | v3.5.0 | Data-driven thresholds from 321 ground truth reaches |
| Outcome Classification | v2.4.4 | Improved pellet start/end position handling |

---

*This document describes the algorithms as implemented in code. For the source of each parameter value, see the files referenced at the top of each section. For ground truth analysis details, see `src/mousereach/reach/analysis/DISCOVERED_RULES.md`.*