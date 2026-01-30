# Eval Findings: 2026-01-27

Algorithm evaluation session using 11 unified ground truth files against current algorithm versions: segmentation v2.1.0, reach detection v3.5.0, outcome classification v2.4.4.

---

## Section 1: What This Pipeline Exists To Do

This pipeline is a **measurement instrument** for skilled forelimb motor function after spinal cord injury. It is not application software where 85% accuracy is acceptable. It is a scientific instrument. If the instrument is wrong, the measurements are wrong, and any conclusions drawn from them are invalid.

### The Task

The single-pellet reaching task presents 20 pellets per session. Each pellet sits on a pillar outside a narrow slot. The mouse reaches through the slot with one paw to grab and retrieve the pellet. The mouse may reach any number of times per pellet — some pellets get 1 reach, others get 15+.

### Segmentation

Segmentation identifies which pellet presentation (1-20) the video is showing at any point. This allows linking all downstream events (reaches, outcomes) to a specific pellet number. It does not identify "trials" — it identifies pellet boundaries so that reaches and outcomes can be attributed to the correct pellet.

### Reach Detection

Reach detection identifies individual reaching movements by their exact start and end frames. **Frame-level accuracy is non-negotiable.** If start or end is off by even 1 frame, the kinematic features (velocity, trajectory, angle, extent, pronation, curvicity) are computed over the wrong temporal window and do not represent reality.

The entire system exists to produce accurate kinematics. If this step is wrong, everything built on top of it is wrong.

### Outcome Classification

Outcome classification determines what happened to each pellet:
- **Retrieved** — mouse grabbed and ate the pellet
- **Displaced SA** — pellet knocked into scoring area (not eaten)
- **Displaced Outside** — pellet knocked outside scoring area
- **Untouched** — pellet still on pillar at segment end
- **No Pellet** — no pellet visible at segment start

The `interaction_frame` identifies when the pellet was first physically affected by the mouse. This frame is used to determine which reach caused the outcome (`causal_reach_id`). Getting the causal reach wrong means attributing kinematic properties to the wrong movement.

### Why Causal Reach Identification Matters

The distinction between "missed (not yet interacted with pellet)" and "missed (already interacted / pellet gone)" reveals whether the mouse is executing a motor pattern from an action generator vs. performing goal-directed actions. Mice reaching after the pellet is already gone are reaching in vain — running a motor pattern that is no longer goal-relevant.

This behavior pattern is scientifically critical for understanding CST pyramidotomy recovery. The hypothesis is that mice run motor patterns produced by an action generator, not top-down reasoned actions. This could explain why pyramidotomy of the CST is entirely recoverable with identical kinematics — the action generator persists even when the corticospinal tract is severed.

If we ask "what were the kinematic properties surrounding the reach that produced the successful pellet outcome on pellet 3 of session P4 on 20240118 for mouse CNT0406 who had X residual connectivity after SCI" — we need to be certain we identified the correct reach that caused the interaction.

### The Standard

**Exact human match.** Zero tolerance on start frame, zero tolerance on end frame, correct causal reach identification, correct outcome classification. This is the standard because the science depends on it.

---

## Section 2: Current Eval Results

### Segmentation (v2.1.0) — Strong

| Metric | Value |
|--------|-------|
| Boundary match | 95-100% across 11 videos |
| Timing bias | Slightly early (-0.2 to -2.1 frames average) |
| Action needed | None |

Segmentation is reliable. Minor timing bias does not affect downstream processing since boundaries are used to define segment ranges, not exact event timing.

### Outcome Classification (v2.4.4) — Mostly Strong, One Failure

| Video | Accuracy | Errors | Notes |
|-------|----------|--------|-------|
| 8 of 11 videos | 100% | 0 | Perfect |
| CNT0213_P3 | 95% | 1 | Single misclassification |
| CNT0103_P2 | 95% | 1 | Single misclassification |
| **CNT0311_P2** | **70%** | **6** | **Human-verified by SCHULTZC on 2026-01-26. 13/20 outcomes were corrected from algo predictions. Algo over-calls `displaced_sa` when human says `untouched` or `retrieved`.** |

CNT0311_P2 is a verified failure. SCHULTZC reviewed all 20 outcomes and corrected 13 of them. The ground truth is legitimate — the algorithm genuinely failed on this video.

### Reach Detection (v3.5.0) — Insufficient

| Metric | Value | Problem |
|--------|-------|---------|
| Precision | 84-100% | Acceptable range |
| Recall | 82-91% | **10-18% of real reaches are missed entirely** |
| End frame bias | ~1 frame early (average) | **Kinematic features computed over wrong window** |
| GT coverage | 6 of 11 videos have reach GT | 5 lack reach GT due to annotation labor cost |

**Recall failure:** Missing 10-18% of reaches means those reaches are invisible to the system. Any kinematic analysis, outcome attribution, or behavioral pattern analysis based on these videos is working with incomplete data.

**End frame accuracy:** ~1 frame early on average sounds minor but means kinematic measurements are systematically computed over the wrong temporal window for a large proportion of detected reaches. Velocity at apex, trajectory straightness, hand angle — all of these change when the window shifts by even 1 frame.

---

## Section 3: Critical Gaps Identified

### Gap 1: Causal Reach Not Visible in Review Tools

The reach annotator widget (`reach/review_widget.py`) shows reaches as a list with start/end frames, duration, and verification status. It does **not** show which reach the algorithm thinks caused the pellet outcome.

The reviewer has no way to verify or correct causal reach assignment during reach review.

The `causal_reach_id` field exists in the outcome JSON and unified GT schema, but the GT files have it as `null` for all segments — meaning it was never populated during review because no tool exposed it.

### Gap 2: No Per-Reach Outcome Display

The reach list shows no information about what happened to the pellet. A reviewer looking at 15 reaches in a segment has no idea which one the algorithm considers the interacted reach, or what the outcome was. This context is essential for verifying the most important link in the analysis chain.

**What is needed:** Each reach in the list should show what the algorithm thinks that reach's role was — missed (before interaction), causal (produced the outcome), or post-interaction (reaching after pellet already affected). The causal reach should display the outcome it produced (retrieved, displaced, etc.) and the interaction frame.

### Gap 3: Reach End Frame Accuracy

The eval reports end frames as "~1 frame early on average" which sounds minor but means kinematic measurements are systematically computed over the wrong temporal window. This is a measurement instrument — systematic bias in the measurement window produces systematic bias in the measurements.

### Gap 4: CNT0311_P2 Outcome Failures

6/20 outcomes wrong on a human-verified video. Investigation needed:
- What makes this video different? (tray type, lighting, mouse behavior, pellet visibility)
- Is the algo's `displaced_sa` over-calling pattern specific to certain visual conditions?
- Are there other videos with similar characteristics that haven't been GT-annotated yet?

### Gap 5: Eval Reporting Should Exclude Missing GT Sections

5 of 11 videos have 0 GT reaches — these genuinely lack reach annotations due to annotation labor cost. They should be excluded from reach eval aggregates, not reported as 0% precision/recall. Similarly, any video lacking outcome GT should be excluded from outcome aggregates.

---

## Section 4: Requirements for Algorithm Improvement

### Reach Detection
- **Target:** 100% recall with exact frame match
- **Tolerance:** 0 frames on start, 0 frames on end
- **Rationale:** Every missed reach is invisible data. Every frame of error produces wrong kinematics.

### Outcome Classification
- **Target:** 100% accuracy on all videos including edge cases like CNT0311_P2
- **Investigation:** Determine what visual/behavioral pattern causes `displaced_sa` over-calling
- **Fix:** Address the root cause, not just tune thresholds

### Causal Reach Identification
- **Prerequisite:** Must be verifiable in review tools before it can be evaluated
- **Implementation:** Review tools must show per-reach outcome tags and identify the causal reach
- **GT population:** Causal reach must be reviewable and correctable during GT annotation

### Eval Reporting
- **Fix:** Exclude videos with missing GT sections from those section's aggregates
- **Report:** Clearly state how many GT files contributed to each metric

### Priority Order
1. Make causal reach visible in review tools (can't evaluate what can't be seen)
2. Improve reach detection recall to 100% with exact frame match
3. Investigate and fix CNT0311_P2 outcome pattern
4. Fix eval reporting to exclude missing GT sections

---

*Generated from eval session 2026-01-27. Algorithm versions: segmentation v2.1.0, reach detection v3.5.0, outcome classification v2.4.4. GT corpus: 11 unified ground truth files.*
