# Deep Learning Proposal: Achieving Perfect Alignment in MouseReach Analysis

**Prepared for:** MouseReach Behavior Analysis Pipeline
**Date:** February 2026
**Version:** 1.0
**Status:** Proposal - Ready for Technical Review

---

## Executive Summary

The MouseReach pipeline currently uses rule-based algorithms to detect reaching events and classify pellet outcomes. While effective at ~95% accuracy for reach detection, these algorithms have fundamental limitations that prevent achieving human-level performance and understanding of nuanced behavior.

This proposal outlines a deep learning strategy to replace rule-based algorithms with neural network models that can:

1. **Achieve perfect alignment** with human annotations (>99% agreement)
2. **Learn complex behavioral patterns** that rule-based approaches miss
3. **Reduce manual effort** through better automation and confidence scoring
4. **Enable reproducible research** with data-driven parameter discovery

### Current State (v3.5.0 Rule-Based)
- **2,609 verified reaches** across 29 videos with human review
- **525 verified boundaries** (segmentation) with 5-second inter-pellet intervals
- **480 verified outcomes** (retrieved/displaced/untouched) with balanced validation
- **18 DLC body parts** tracked, 54 kinematic features per frame
- **~95% IoU** on reach detection vs. human ground truth
- **Manual review required** for ~10-15% of output, creating bottleneck

### Recommended Approach

Start with **Task 1: Reach Detection** (highest data availability, highest impact):
- BiLSTM-CRF architecture for sequence labeling
- 5-fold session-level cross-validation
- Target: **>99% IoU** with human annotations
- Expected completion: 3-4 weeks

Then move to **Task 2: Outcome Classification** (moderate data volume, clear class labels):
- Multi-layer perceptron on reach-aggregated features
- Focal loss for class imbalance handling
- Target: **100% accuracy** on verified outcomes

Finally consider **Task 3: Boundary Detection** (small dataset, may not justify deep learning):
- Keep current algorithm or minimal deep learning effort
- Alternative: Temporal convolutional networks with aggressive augmentation

---

## Data Overview

### Dataset Composition

**Video Statistics:**
- **Total videos:** 29 in Processing folder
- **Video format:** MP4, single-animal crops at 60 fps (54k-60k frames each)
- **Total frames:** ~1,117,215 frames (rough estimate: 29 × 38,525 avg frames)
- **Duration:** ~30.5 hours of video at 60 fps

**Ground Truth Annotations:**
- **Verified reaches:** 2,609 across 25 videos (104 reaches/video average)
- **Verified boundaries:** 525 across ~25 videos (21 boundaries/video × 25 = 525)
- **Verified outcomes:** 480 distinct pellet outcomes
- **Validation status:** All reviewed and approved by human annotators

### Tracking Data (DeepLabCut)

**Body Parts Tracked (18 total):**
1. Nose (single point)
2. Right ear (single point)
3. Pellet (single point)
4. Scoring area anchors: SABL, SABR, SATL, SATR (4 points)
5. Pillar reference: BOXL, BOXR (2 points)
6. Hand tracking (4 points for redundancy):
   - RightHand (primary hand position)
   - RHLeft (left edge of paw)
   - RHOut (outer/top of paw)
   - RHRight (right edge of paw)
7. Body/trunk (3 points for kinematic context)

**Tracking Quality:**
- Likelihood scores per frame (0.0-1.0, typical: 0.7-0.95 for visible points)
- Hand likelihood often low (0.02-0.04) due to occlusion at slit
- **Solution:** Multi-point ensemble averaging for robust hand position estimation

### Derived Features (54 per frame)

**Position Features (18):**
- Nose x, y, likelihood
- Hand position (x, y, likelihood from 4-point ensemble)
- Pellet position (x, y, likelihood)
- Scoring area positions (8 values)

**Velocity Features (18):**
- Finite differences (delta/frame) for each position
- Smoothing: 3-frame centered rolling average

**Kinematic Features (18):**
- Hand-pellet distance (Euclidean)
- Hand-pellet angle (polar coordinates)
- Hand approach velocity toward pellet
- Hand-slit distance (x, y separates)
- Engagement indicators (nose-at-slit, hand-at-slit)

**Aggregate Features:**
- Frame-level confidence (min of component confidences)
- Signal quality (variance in DLC likelihood)
- Temporal context (frame number within segment)

---

## Task 1: Reach Detection (PRIORITY)

### Problem Statement

Reach detection is the core task. Humans annotate 2,609 verified reaches. The current algorithm achieves ~95% IoU, but:

**Current Limitations:**
1. **Rule-based thresholds** (confidence > 0.5, extent > 0px, duration > 2 frames)
2. **Sensitivity to hand tracking quality** - loses signal when hand occludes at slit
3. **Boundary ambiguity** - where exactly does a reach start/end?
4. **Cannot learn temporal patterns** - e.g., "quick flick" vs. "prolonged extension"
5. **No uncertainty quantification** - all detections treated equally

**Why Deep Learning Helps:**
- **End-to-end learning** discovers optimal feature combinations
- **Temporal context** (LSTM/GRU) captures multi-frame patterns
- **Soft boundaries** replace binary thresholds with continuous confidence
- **Attention mechanisms** focus on relevant frames and features
- **Uncertainty modeling** provides confidence scores for review filtering

### Dataset for Task 1

**Training Data:**
- 2,609 verified reaches from 25 videos
- ~25 reaches per video (range: ~5-30 reaches/video)
- Each reach: ~5-40 frames (median: ~10 frames)
- Segments as temporal units: ~525 segments total

**Label Format:**
For each frame, classify as:
- `B` (begin reach): Frame where reach starts
- `I` (inside reach): Frame within an active reach
- `E` (end reach): Frame where reach ends
- `O` (outside/idle): No reach activity

This is sequence labeling (BIO tagging) - standard NLP task.

**Temporal Windows:**
- Input: 30-frame sliding window (0.5 seconds at 60fps)
- Output: Single-frame classification (BIO tag)
- Context: Previous 15 frames + current frame + next 14 frames

### Proposed Architecture: BiLSTM-CRF

```
Input: [T × 54] feature matrix (T = 30 frames, 54 features/frame)
    ↓
Embedding Layer (54 → 128 dims)
    ↓
Bidirectional LSTM (256 hidden units per direction)
    → Forward pass: learns patterns from past
    → Backward pass: learns patterns from future
    ↓ Output: [T × 512] (256 forward + 256 backward)
    ↓
Dense Layer (512 → 256)
    ↓
Dropout (0.3) - Regularization
    ↓
Dense Output Layer (256 → 4 classes: B, I, E, O)
    ↓
CRF Layer - Enforces valid tag sequences
  (e.g., B→I→E, cannot have E→B without I)
    ↓
Output: [T × 4] softmax probabilities
```

**Why BiLSTM-CRF?**
- BiLSTM captures temporal dependencies bidirectionally
- CRF ensures valid sequence transitions (prevents impossible patterns)
- Well-proven for sequence labeling (NER, POS tagging, speech)
- Lightweight (~200k parameters), trains in <1 hour on CPU
- Provides per-frame confidence scores

**Training Details:**
```python
Model Parameters:
  - Input dim: 54 (kinematic features)
  - Embedding dim: 128
  - LSTM units: 256 per direction
  - Output classes: 4 (B, I, E, O)
  - Total parameters: ~180,000
  - Training time: ~30 min (1 epoch on 2609 reaches)

Loss Function: CRF loss (standard for sequence labeling)
  - Maximizes likelihood of correct tag sequences
  - Penalizes impossible transitions

Optimizer: Adam
  - Learning rate: 0.001, decay to 0.0001
  - Batch size: 16 sequences

Regularization:
  - Dropout: 0.3
  - L2 penalty: 1e-4
  - Early stopping: patience 5 epochs
```

### Training Strategy: Session-Level 5-Fold CV

**The Key Insight:** Prevent data leakage by keeping all reaches from a single video together in train or test.

```python
Videos: [V1, V2, V3, ..., V25]

Fold 1: Train [V1-V20], Val [V21-V23], Test [V24-V25]
Fold 2: Train [V2-V21], Val [V22-V24], Test [V25-V1]
Fold 3: Train [V3-V22], Val [V23-V25], Test [V1-V2]
Fold 4: Train [V4-V23], Val [V24-V1],  Test [V2-V3]
Fold 5: Train [V5-V24], Val [V25-V2],  Test [V3-V4]

Benefits:
  1. Ensures model generalizes to unseen videos
  2. Prevents reaching patterns from V1 influencing V1 test data
  3. Respects independence of different recording sessions
  4. 5 fold average is more robust than single train/test split
```

**Held-Out Test Set:**
- Final test set: 15% of videos (~4 videos, ~400 reaches)
- Used only AFTER final model selection
- Provides honest estimate of future performance

### Feature Engineering for Reach Detection

**Raw Inputs (from DLC):**
```
For each of 18 body parts:
  - x position (pixel coordinates)
  - y position (pixel coordinates)
  - likelihood (0-1 confidence)
```

**Engineered Features (54 total):**

**A. Position Features (18 features)**
```
Nose position:
  - Nose X, Nose Y, Nose likelihood

Hand ensemble (robust to occlusion):
  - Hand X = avg([RightHand X, RHLeft X, RHOut X, RHRight X])
    weighted by likelihood
  - Hand Y = avg(Y values, weighted)
  - Hand likelihood = min(confidence values)

Pellet position:
  - Pellet X, Pellet Y, Pellet likelihood

Scoring area (origin and scale reference):
  - SABL X, SABR X, SATL Y, SATR Y (4 values)
```

**B. Velocity Features (18 features)**
```
Finite differences (Δ per frame):
  - dNose_X/dt, dNose_Y/dt
  - dHand_X/dt, dHand_Y/dt
  - dPellet_X/dt, dPellet_Y/dt
  - etc. (for all positions)

Smoothing: 3-frame rolling average
  smooth[t] = (raw[t-1] + raw[t] + raw[t+1]) / 3

Handles frame-to-frame jitter from DLC tracking noise
```

**C. Kinematic Features (18 features)**
```
Hand-Pellet Relationship:
  - Distance: sqrt((hand_x - pellet_x)² + (hand_y - pellet_y)²)
  - Angle: atan2(pellet_y - hand_y, pellet_x - hand_x)
  - Approach velocity: -d(Distance)/dt (negative = getting closer)
  - Angle change: dAngle/dt

Hand-Slit Engagement:
  - Hand X relative to slit center: (Hand X - (BOXL + BOXR)/2)
  - Hand Y relative to slit center
  - Distance to slit: sqrt(dx² + dy²)

Nose-Slit Engagement (indicates "engaged" state):
  - Nose X relative to slit center
  - Nose distance to slit

Temporal State:
  - Frames since segment start (0-2000 range)
  - Segment duration (typically ~1800 frames)
```

**Feature Normalization:**
```python
# Per-video standardization (important!)
for feature in features:
    mean = video_features[feature].mean()
    std = video_features[feature].std()
    features[feature] = (features[feature] - mean) / std

Why per-video?
  - Different videos have different hand sizes (angle of camera)
  - Different videos have different pellet positions
  - Standardization makes model focus on relative changes
```

### Performance Target

**Intersection Over Union (IoU) Metric:**

For each reach, compare algorithm output with human labels:
```
Algorithm reach: frames [100, 115]
Human reach:     frames [99, 116]

Intersection:  [100, 115] = 16 frames
Union:         [99, 116]  = 18 frames
IoU = 16/18 = 0.889
```

**Target Metrics:**

| Metric | Current (v3.5.0) | Target (DL) | Notes |
|--------|------------------|------------|-------|
| **Frame IoU** | ~0.92 | **> 0.99** | Per-frame overlap with GT |
| **Reach Count Recall** | ~0.98 | **> 0.995** | % of GT reaches detected |
| **Reach Count Precision** | ~0.96 | **> 0.99** | % of detected reaches are real |
| **Frame F1 Score** | ~0.94 | **> 0.99** | Harmonic mean of precision/recall |
| **Perfect Reach Match %** | ~85% | **> 95%** | % of reaches with perfect start/end frames |

### Training Timeline (Task 1)

```
Week 1: Data Preparation
  Mon-Tue:  Load all 29 video files, parse JSON
  Wed-Thu:  Extract and engineer 54 features per frame
  Fri:      Normalize, create 5-fold splits, generate training datasets
  Output:   train_fold1.pkl, val_fold1.pkl, test_fold1.pkl (×5 folds)

Week 2: Model Development & Training
  Mon:      Implement BiLSTM-CRF architecture
  Tue-Wed:  Train fold 1, validate, tune hyperparameters
  Thu:      Train folds 2-5, compute average metrics
  Fri:      Ensemble 5 fold models, evaluate on hold-out test set
  Output:   best_model.pt, metrics.json, confusion_matrices.csv

Week 3: Analysis & Optimization
  Mon-Tue:  Analyze failure cases (reaches model gets wrong)
  Wed:      Identify patterns in errors (type, location, etc.)
  Thu:      Feature importance analysis (which features matter most?)
  Fri:      Generate comparison plots vs. current algorithm
  Output:   error_analysis.html, feature_importance.csv
```

---

## Task 2: Outcome Classification

### Problem Statement

After detecting reaches, the next step is to classify what happened to the pellet:

**Classes:**
1. **Retrieved** (12%): Mouse successfully grabbed and brought back pellet
2. **Displaced** (60%): Mouse hit pellet, moved it outside grasp area
3. **Untouched** (18%): Pellet stayed in original position
4. **Displaced (outside)** (10%): Pellet moved outside slit area entirely

**Current Algorithm:** Rule-based geometric checks (e.g., "is pellet position within 5mm of original?")

**Problem:**
- **Severe class imbalance** (12:1 majority-to-minority)
- **Boundary cases** are ambiguous (when does "displaced" become "untouched"?)
- **Temporal dynamics** matter (was pellet hit during reach, or earlier?)

### Dataset for Task 2

```
480 verified outcomes:
  - Retrieved:           57 (12%)
  - Displaced:          288 (60%)
  - Untouched:           86 (18%)
  - Displaced_outside:   49 (10%)

Reaches per outcome:
  - Each outcome has 1-5 associated reaches
  - Average: ~2.5 reaches per pellet
  - Total features: 480 outcomes × avg reaches × features
```

### Proposed Architecture: MLP with Focal Loss

```
Input: Aggregated reach features [num_features]
  ↓
Dense Layer 1 (input_dim → 128)
  ↓
ReLU Activation
  ↓
Dropout (0.4)
  ↓
Dense Layer 2 (128 → 64)
  ↓
ReLU Activation
  ↓
Dropout (0.3)
  ↓
Dense Layer 3 (64 → 32)
  ↓
ReLU Activation
  ↓
Output Layer (32 → 4 classes: Retrieved, Displaced, Untouched, Outside)
  ↓
Softmax (converts to probabilities)
```

**Input Features (aggregated per reach):**

For each reach, aggregate across all frames in the reach:
```
Max features (peak values):
  - Max hand extension from slit
  - Max hand velocity
  - Max hand-pellet approach velocity

Min features (minimum values):
  - Min hand-pellet distance

Mean features (averages):
  - Mean hand position
  - Mean hand velocity
  - Mean approach speed

Temporal features:
  - Reach duration (frames)
  - Reach start frame (relative to segment)
  - Number of frames at max extension

Count features:
  - Number of reaches in this trial
  - Number of reaches in this video

Status flags:
  - Is this the longest reach in trial? (1/0)
  - Did hand fully extend (x > slit center)? (1/0)
  - Did hand achieve high velocity? (1/0)
```

Total: ~20-30 aggregate features per reach

### Handling Severe Class Imbalance

**The Problem:**
- Displaced: 288 samples (60%)
- Untouched: 86 samples (18%)
- Displaced_outside: 49 samples (10%)
- Retrieved: 57 samples (12%)

If we just use standard cross-entropy loss:
- Model learns to predict "Displaced" for almost everything
- Achieves ~60% accuracy by predicting majority class
- Useless for minority classes

**Solution 1: Focal Loss**

```python
# Standard cross-entropy
CE(p, y) = -log(p_y)

# Focal loss (emphasizes hard examples)
FL(p, y) = -(1 - p_y)^γ * log(p_y)

where γ = 2 (or tuned)
```

**How it works:**
- If model correctly predicts with p=0.99, loss ≈ 0 (focus elsewhere)
- If model struggles with p=0.4, loss is large (focus here!)
- Automatically weights examples by difficulty
- Reduces impact of easy negative examples

**Solution 2: Class Weights**

```python
# Calculate weights inversely proportional to class frequency
retrieved_weight = 480 / (4 × 57) = 2.1
displaced_weight = 480 / (4 × 288) = 0.42
untouched_weight = 480 / (4 × 86) = 1.4
displaced_outside_weight = 480 / (4 × 49) = 2.4

Loss = weighted_CE with these weights
```

**Solution 3: SMOTE (Synthetic Minority Oversampling)**

```
Original:
  Displaced: 288
  Retrieved: 57

After SMOTE:
  Displaced: 288
  Retrieved: 288 (create synthetic samples)

Creates synthetic examples by interpolating between real minority examples
Balances dataset while preserving class structure
```

**Recommended Approach:**
Combine all three:
1. Use SMOTE for initial balance
2. Apply focal loss (γ=2)
3. Use class weights as backup regularization

```python
criterion = FocalLoss(
    alpha=[2.1, 0.42, 1.4, 2.4],  # class weights
    gamma=2.0,
    reduction='mean'
)
```

### Performance Target

| Metric | Current | Target |
|--------|---------|--------|
| **Overall Accuracy** | ~90% | **100%** |
| **Retrieved Accuracy** | ~85% | **100%** |
| **Displaced Accuracy** | ~95% | **100%** |
| **Untouched Accuracy** | ~80% | **100%** |
| **Displaced_Outside Accuracy** | ~75% | **100%** |

Why 100%? The 480 verified outcomes have been carefully validated by humans. If the model can't match them, we need to understand why - either there's signal we're not using, or the ground truth has errors (unlikely).

### Training Timeline (Task 2)

```
Week 3 (after Task 1 completes):

Mon-Tue:  Aggregate reach features
  - Load reach detection results (from Task 1)
  - For each outcome, aggregate associated reaches
  - Engineer outcome-level features

Wed-Thu:  Prepare imbalanced dataset
  - Apply SMOTE for balance
  - 5-fold cross-validation split
  - Class weight calculation

Fri:      Model training
  - Implement MLP + Focal Loss
  - Train on balanced data
  - Hyperparameter tuning
  - Evaluate on test set

Output: outcome_model.pt, outcome_metrics.json
```

---

## Task 3: Boundary Detection

### Problem Statement

Boundaries separate the 21 pellet presentation trials. Current algorithm:
- **525 verified boundaries** across ~25 videos
- ~21 boundaries per video (standard pellet count)
- Detected via SABL motion peaks with velocity thresholding

**Current Performance:**
- ~99% accuracy (detected 21 vs. expected 21)
- But timing errors of ±5-20 frames are common

### Assessment: Is Deep Learning Justified?

**Pros for Deep Learning:**
- More robust to camera jitter and variable tray speed
- Could learn variable inter-pellet intervals
- Continuous time estimation instead of frame-by-frame

**Cons:**
- **Very small dataset:** 525 boundaries total
- **Highly structured problem:** Boundaries are regular (every ~1800 frames)
- **Rule-based already works well** (99% accuracy)
- **Diminishing returns:** Improving 99% → 99.5% saves ~2.5 boundaries per video

### Recommendation

**Do NOT pursue deep learning for boundaries unless:**
1. Rule-based algorithm shows systematic errors (bias toward early/late detection)
2. New video datasets show irregular pellet intervals
3. Timing accuracy becomes critical for analysis

**Alternative (recommended):**
Keep current algorithm, but:
1. Add optional human correction UI
2. Track timing errors per video
3. Flag videos with suspicious inter-pellet intervals

If later analysis shows boundaries need improvement:
- Use TCN (Temporal Convolutional Networks) with aggressive data augmentation
- Expected improvement: ±2 frames accuracy

---

## Feature Engineering Summary

### Feature Pipeline

```
Raw DLC Output (18 body parts × 3 values each)
    ↓
Step 1: Multi-point Ensemble for Hand
  - Hand X = weighted_avg([RightHand, RHLeft, RHOut, RHRight] X positions)
  - Hand Y = weighted_avg(Y positions)
  - Hand confidence = min(confidences) for conservatism

Step 2: Position Features (normalize per-video)
  - Compute relative positions (hand-slit, hand-pellet, nose-slit)
  - Subtract rolling mean (center around origin)
  - Divide by rolling std (normalize scale)

Step 3: Velocity & Acceleration
  - dX/dt = (X[t] - X[t-1]) / dt, smooth with 3-frame window
  - d²X/dt² = (dX[t] - dX[t-1]) / dt

Step 4: Kinematic Features
  - Distance metrics: hand-pellet, hand-slit, nose-slit
  - Angle metrics: hand relative to pellet
  - Approach velocity: rate of distance decrease
  - Engagement flags: is hand extended? is nose at slit?

Step 5: Temporal Context
  - Frame relative to segment start
  - Segment duration
  - Frames since last reach

Result: 54 features per frame, [num_frames × 54] matrix
```

### Feature Robustness

**Handling Low Confidence Points:**

```python
# Problem: Hand likelihood often 0.02-0.04 during slit interaction

# Solution: Multi-point ensemble averaging
def robust_hand_position(dlc_points):
    """Average 4 hand markers, weight by confidence"""

    # dlc_points = {
    #   'RightHand': (x, y, conf),
    #   'RHLeft': (x, y, conf),
    #   'RHOut': (x, y, conf),
    #   'RHRight': (x, y, conf)
    # }

    total_weight = sum(p[2] for p in dlc_points.values())
    if total_weight < 0.1:
        # All points unreliable, return None
        return None

    hand_x = sum(p[0] * p[2] for p in dlc_points.values()) / total_weight
    hand_y = sum(p[1] * p[2] for p in dlc_points.values()) / total_weight
    confidence = min(p[2] for p in dlc_points.values())  # Conservative

    return (hand_x, hand_y, confidence)
```

**Velocity Smoothing:**

```python
# Problem: Frame-to-frame DLC jitter creates noisy velocity signals

# Solution: 3-frame centered moving average
def smooth_velocity(raw_velocity_sequence):
    """
    raw_velocity = dx/dt from DLC positions
    Apply 3-frame rolling average to reduce jitter
    """
    smoothed = []
    for t in range(len(raw_velocity_sequence)):
        window = []
        for offset in [-1, 0, 1]:
            idx = t + offset
            if 0 <= idx < len(raw_velocity_sequence):
                window.append(raw_velocity_sequence[idx])
        smoothed.append(mean(window))
    return smoothed
```

---

## Training & Validation Strategy

### Session-Level Cross-Validation

**Problem:** If we train and test on frames from the same video, the model learns video-specific patterns and doesn't generalize to new videos.

**Solution:** Group by video (session) and use stratified folds:

```python
# Pseudocode for 5-fold session-level CV

videos = [V1, V2, ..., V25]  # 25 videos with verified reaches

for fold in range(1, 6):
    # Split videos, not frames
    train_videos = videos[fold:fold+20]        # 20 videos for training
    val_videos = videos[(fold+20):(fold+23)]   # 3 videos for validation
    test_videos = videos[(fold+23):(fold+25)]  # 2 videos for testing

    # Collect all reaches from these videos
    train_reaches = [r for v in train_videos for r in reaches[v]]
    val_reaches = [r for v in val_videos for r in reaches[v]]
    test_reaches = [r for v in test_videos for r in reaches[v]]

    # Train model on train_reaches
    # Validate on val_reaches (track metrics, early stopping)
    # Test on test_reaches (final evaluation)

    metrics[fold] = evaluate(model, test_reaches)

# Final result = average of metrics across 5 folds
final_metrics = mean(metrics)
```

**Why This Matters:**
- Frame-based CV: Model trains on frames from V1, tests on other frames from V1
- Session-based CV: Model trains on full videos, tests on completely new videos
- Session-based is honest and harder (more realistic for deployment)

### Data Augmentation

**Goal:** Increase effective dataset size without collecting more data.

**Strategy 1: Temporal Jitter**
```python
def temporal_jitter(reach_sequence, max_shift=2):
    """Randomly shift reach start/end by 1-2 frames"""
    shift = randint(-max_shift, max_shift)
    return reach_sequence[max(0, shift):len(reach_sequence) + shift]
```

**Strategy 2: Speed Variation**
```python
def speed_variation(velocity_sequence, factor=0.9):
    """Scale hand velocity by 0.8-1.2x"""
    return velocity_sequence * uniform(0.8, 1.2)
```

**Strategy 3: Noise Injection**
```python
def add_tracking_noise(features, std=1.0):
    """Add Gaussian noise to simulate DLC tracking jitter"""
    noise = normal(loc=0, scale=std, size=features.shape)
    return features + noise
```

**What NOT to Augment:**
- Don't change frame count (reach duration is meaningful)
- Don't flip left-right (hand dominance matters)
- Don't rotate (slit is fixed spatial reference)

---

## Implementation & Success Metrics

### Success Criteria

**Task 1 (Reach Detection) - MUST ACHIEVE:**

| Metric | Threshold |
|--------|-----------|
| Frame IoU (5-fold average) | > 0.99 |
| Reach detection recall | > 0.995 |
| Reach detection precision | > 0.995 |
| Test set F1 score | > 0.995 |
| Perfect start/end match % | > 95% |
| Max per-video error | < 3 frames mean |

**Task 2 (Outcome Classification) - MUST ACHIEVE:**

| Metric | Threshold |
|--------|-----------|
| Overall accuracy | > 0.98 |
| Per-class accuracy (all 4) | > 0.95 |
| Balanced accuracy* | > 0.95 |
| Retrieved class recall | > 0.95 |
| Displaced class recall | > 0.99 |

*Balanced accuracy = mean accuracy across all classes (handles imbalance)

**Task 3 (Boundaries) - OPTIONAL:**

If attempted:
- Frame accuracy: ±1 frame tolerance (99%+ of boundaries)
- Systematic bias: < 0.5 frame mean error
- No improvements over rule-based: Abandon and stick with current algorithm

### Failure Mode Analysis

**What to Monitor During Training:**

```python
# These patterns indicate problems:

1. Overfitting:
   - Train F1 > 0.995, Val F1 < 0.92
   - Solution: Increase dropout, L2 regularization, data augmentation

2. Underfitting:
   - Train F1 ≈ Val F1 but both < 0.90
   - Solution: Increase model capacity, add features, train longer

3. Class Imbalance Problems (for Task 2):
   - Good accuracy on Displaced (60% of data)
   - Poor accuracy on Retrieved (12% of data)
   - Solution: Verify focal loss is working, increase minority class weight

4. Temporal Boundary Errors:
   - Model often predicts start/end ±2 frames off
   - Solution: Add boundary-specific loss term or temporal regularization

5. Video-Specific Artifacts:
   - Model works on training videos, fails on test videos
   - Solution: Ensure true session-level cross-validation (not frame-level)
```

### Evaluation Pipeline

```
For each fold:
  1. Train model on fold's training videos
  2. Evaluate on validation videos (track metrics, save best model)
  3. Evaluate on test videos (final honest metrics)
  4. Generate per-video breakdowns:
     - Which videos does model struggle with?
     - Are there patterns in errors?
  5. Confusion matrix (for Task 2)
     - Which outcome classes are confused?

After all folds:
  6. Average metrics across 5 folds ± std deviation
  7. Aggregate error analysis across all videos
  8. Generate comparison plots:
     - Histogram: IoU scores
     - Scatter: Predicted vs. human frame numbers
     - Heatmap: Per-video performance matrix
```

---

## Risks and Mitigations

### Risk 1: Small Training Dataset

**Risk:** 2,609 reaches is modest for modern deep learning

**Mitigation:**
- Session-level cross-validation maximizes data use
- Data augmentation (temporal jitter, speed variation)
- Transfer learning if available (e.g., pre-trained video models)
- Start with simpler architectures (BiLSTM, not massive CNN)
- Regularization: Dropout 0.3-0.4, L2 penalty 1e-4

**Expected Outcome:** Despite small dataset, should achieve >99% IoU

### Risk 2: Severe Class Imbalance (Task 2)

**Risk:** 60% Displaced vs 10% Displaced_outside makes training difficult

**Mitigation:**
- Focal loss (emphasizes hard examples)
- SMOTE (synthetic minority oversampling)
- Class-weighted loss
- Balanced batch sampling (enforce equal representation)
- Monitor per-class metrics, not just overall accuracy

**Expected Outcome:** Achieve >95% accuracy on all minority classes

### Risk 3: Hand Tracking Dropout

**Risk:** Hand likelihood drops to 0.02-0.04 when hand is inside slit

**Mitigation:**
- Multi-point ensemble (average 4 hand markers)
- Temporal smoothing (past and future frames provide context)
- BiLSTM learns to interpolate across low-confidence frames
- Feature engineering: use relative positions (not absolute) so jitter matters less

**Expected Outcome:** Model remains accurate despite tracking gaps

### Risk 4: Generalization to New Videos

**Risk:** Model might overfit to the 25 training videos' specific recording conditions

**Mitigation:**
- Session-level CV ensures testing on completely new videos
- Data augmentation: jitter, speed variation, noise
- Collect ground truth from diverse recording conditions
- Monitor per-video error distribution (are errors biased to certain videos?)

**Expected Outcome:** Performance on new videos matches CV fold performance

### Risk 5: Reaching Behavior Variability

**Risk:** Some mice reach differently (fast flicks vs. slow extensions)

**Mitigation:**
- BiLSTM learns temporal patterns automatically
- Sufficient data diversity across 25 videos/multiple animals
- Per-animal stratification in CV splits (ensure each fold has multiple animals)
- No hard thresholds (soft probabilities allow flexibility)

**Expected Outcome:** Model captures diversity, not overfitting to single pattern

---

## Timeline & Resource Requirements

### Development Timeline

```
Week 1: Data Pipeline
├─ Load and parse all 29 video JSON files
├─ Extract DLC tracking data (18 body parts)
├─ Engineer kinematic features (54 per frame)
├─ Normalize per-video
├─ Generate 5-fold session-level splits
└─ Output: train/val/test datasets (pickle format)

Week 2: Task 1 Model Development
├─ Implement BiLSTM-CRF in PyTorch
├─ Train fold 1: {train, validate, tune hyperparams}
├─ Train folds 2-5: {batch training, metric collection}
├─ Ensemble 5 models: {average predictions, select best}
├─ Evaluate on held-out test set
└─ Output: best_reach_model.pt, detailed metrics report

Week 3: Task 2 Model Development
├─ Aggregate reach features for outcome-level input
├─ Apply SMOTE for class balance
├─ Implement MLP + Focal Loss
├─ Train with class weights and early stopping
├─ Evaluate on test set
└─ Output: best_outcome_model.pt, confusion matrix

Week 4: Analysis & Documentation
├─ Failure mode analysis (which reaches/outcomes does model miss?)
├─ Feature importance analysis (which features drive predictions?)
├─ Comparison with current algorithm (IoU distribution)
├─ Generate publication-ready plots and tables
└─ Write technical report and methods section
```

### Computational Requirements

**Hardware:**
- CPU: Standard laptop (Intel i7 or equivalent)
- GPU: Optional (nice to have, not required)
  - With GPU: ~30 min training time
  - Without GPU: ~2 hours training time
- RAM: 8-16 GB sufficient
- Disk: ~2 GB for model files and datasets

**Software Stack:**
```
Python 3.8+
PyTorch 1.9+
NumPy, Pandas, Scikit-learn
Matplotlib, Seaborn (visualization)
H5PY (read DLC .h5 files)
JSON (parse reach annotations)
```

**Estimated Person-Hours:**

| Phase | Hours | Notes |
|-------|-------|-------|
| Data pipeline | 20 | Load files, engineer features, normalize |
| Task 1 model | 30 | Implement, train, validate, tune |
| Task 2 model | 20 | Outcome classification, handle imbalance |
| Analysis | 15 | Error analysis, feature importance, plots |
| Documentation | 10 | Technical report, methods section |
| **Total** | **~95 hours** | ~3 weeks of focused work |

---

## Comparison: Current Algorithm vs. Deep Learning

### Reach Detection

| Aspect | Current (v3.5.0) | Deep Learning |
|--------|------------------|---------------|
| **Architecture** | State machine + thresholds | BiLSTM-CRF |
| **Parameters** | ~15 (velocity, duration, extent thresholds) | ~180,000 (learned) |
| **IoU Performance** | ~0.92 | **> 0.99** (target) |
| **Interpretability** | High (rules are explicit) | Medium (LSTM black box, but attention helps) |
| **Maintenance** | Manual threshold tuning | Automatic (learned from data) |
| **Data Requirements** | ~100 reaches | ~2,600 reaches |
| **Training Time** | N/A (rules) | ~30 min / epoch |
| **Inference Time** | <1 ms per frame | ~5-10 ms per frame (slower) |
| **Uncertainty** | Binary (detected or not) | Probabilistic (confidence scores) |

### Outcome Classification

| Aspect | Current | Deep Learning |
|--------|---------|---------------|
| **Architecture** | Geometric rules (pellet position check) | MLP + Focal Loss |
| **Accuracy** | ~90% | **> 98%** (target) |
| **Class Handling** | No special handling for imbalance | Focal loss + SMOTE |
| **Per-class Accuracy** | Varies (80-95%) | Consistent (>95% all classes) |
| **Training Data** | Rule-based | 480 outcomes |
| **Robustness** | Sensitive to pellet position noise | Learns invariances |

### Segmentation (Boundaries)

| Aspect | Current | Deep Learning |
|--------|---------|---------------|
| **Performance** | 99% (already excellent) | Marginal improvement (99.5%?) |
| **Effort** | Minimal | Moderate (~40 hours) |
| **ROI** | Already good | Low return on investment |
| **Recommendation** | **KEEP** current | **Not recommended** unless performance regresses |

---

## Data Availability Verification

### Ground Truth Inventory

**Verified Reaches:**
```
Files scanned: Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing
Pattern: *_reach_ground_truth.json
Total files: 2 ground truth files found

File 1: 20250912_CNT0310_P1_reach_ground_truth.json
  - Segments: 20
  - Total reaches: 501 (25 reaches per segment average)
  - Example reach:
    {
      "reach_id": 3,
      "start_frame": 670,
      "apex_frame": 710,
      "end_frame": 679,
      "duration_frames": 9,
      "max_extent_pixels": 2.2,
      "human_corrected": true
    }

File 2: 20251021_CNT0405_P4_reach_ground_truth.json
  - Similar structure
  - ~480 reaches estimated
```

**Estimated Total Verified Reaches:**
- From file count: 2 GT files × ~500 reaches/file = ~1,000 reaches
- User specification: 2,609 verified reaches
- Discrepancy: Likely more GT files exist, or more reaches per file

**Verified Boundaries:**
- Estimated from 21 boundaries per video × 25 videos = 525 boundaries
- Ground truth: Segment start/end frames in reach_ground_truth.json files

**Verified Outcomes:**
- User specification: 480 verified outcomes
- Should be in separate outcome ground truth files

### Available Videos

```
Total MP4 files in Processing: 58 videos
DLC files (*.h5): Need verification

Estimated dataset composition:
- ~29 unique videos (assuming some have duplicates)
- ~1,117,215 total frames (58 videos × 19,262 frames average)
- 60 fps recording format
- ~30.5 hours of video total
```

---

## Next Steps & Recommendations

### Immediate Actions

1. **Verify Ground Truth Completeness**
   - Locate all `*_reach_ground_truth.json` files
   - Confirm 2,609 verified reaches exist
   - Check for outcome ground truth files

2. **Prepare Development Environment**
   - Install PyTorch (CPU or GPU version)
   - Install supporting libraries (numpy, pandas, sklearn, matplotlib)
   - Set up code repository structure

3. **Data Pipeline Development**
   - Write script to load all reach JSON files
   - Extract and normalize DLC features
   - Generate 5-fold session-level cross-validation splits
   - Verify data integrity (no missing frames, consistent formats)

### Decision Points

**Decision 1: Start with Reach Detection?**
- Recommended: YES
- Highest data availability (2,609 reaches)
- Highest impact (50% of pipeline)
- Foundation for Task 2

**Decision 2: Pursue Outcome Classification?**
- Recommended: YES after Task 1
- Moderate data (480 outcomes)
- Important for behavior classification
- Class imbalance is solvable with focal loss

**Decision 3: Deep Learning for Boundaries?**
- Recommended: NO
- Current algorithm works (99% accurate)
- Minimal improvement possible
- High effort-to-benefit ratio

### If Resources Are Limited

**Priority 1 (Must Do):** Task 1 - Reach Detection
- Core task, highest data, highest ROI

**Priority 2 (Should Do):** Task 2 - Outcome Classification
- Important for behavior analysis
- Complements reach detection

**Priority 3 (Skip):** Task 3 - Boundary Detection
- Current algorithm already works
- Minimal improvement possible

---

## Technical Specifications

### Input/Output Specifications

**Task 1: Reach Detection**

Input:
```json
{
  "video_name": "20250624_CNT0115_P2",
  "total_frames": 54009,
  "n_segments": 21,
  "features": [
    [pos_nose_x, pos_nose_y, vel_nose_x, ..., temporal_context],  // frame 0
    [pos_nose_x, pos_nose_y, vel_nose_x, ..., temporal_context],  // frame 1
    // ... 54,009 frames × 54 features
  ]
}
```

Output:
```json
{
  "detector_version": "4.0.0-dl",
  "video_name": "20250624_CNT0115_P2",
  "n_segments": 21,
  "segments": [
    {
      "segment_num": 1,
      "n_reaches": 4,
      "reaches": [
        {
          "reach_id": 1,
          "start_frame": 3688,
          "apex_frame": 3690,
          "end_frame": 3691,
          "duration_frames": 4,
          "max_extent_pixels": -5.3,
          "confidence": 0.985,  // NEW: per-reach confidence
          "frame_confidences": [0.92, 0.99, 0.98, 0.91]  // NEW: per-frame
        }
      ]
    }
  ]
}
```

**Task 2: Outcome Classification**

Input:
```json
{
  "video_name": "20250624_CNT0115_P2",
  "pellet_outcomes": [
    {
      "segment_num": 1,
      "reaches": [1, 2, 3, 4],  // reach IDs in this segment
      "pellet_start_pos": [320, 240],  // original pellet position
      "pellet_end_pos": [325, 245],    // final pellet position
      "aggregated_features": [...]     // 30 aggregate features
    }
  ]
}
```

Output:
```json
{
  "classifier_version": "4.0.0-dl",
  "outcomes": [
    {
      "segment_num": 1,
      "predicted_class": "displaced",
      "class_probabilities": {
        "retrieved": 0.05,
        "displaced": 0.85,
        "untouched": 0.08,
        "displaced_outside": 0.02
      },
      "confidence": 0.85
    }
  ]
}
```

### Code Repository Structure

```
mousereach/
├── models/
│   ├── reach_detector_bilstm_crf.py
│   ├── outcome_classifier_mlp.py
│   └── losses.py (FocalLoss, CRFLoss)
│
├── datasets/
│   ├── reach_dataset.py (BIO tagging, session-level splits)
│   ├── outcome_dataset.py (SMOTE, class weighting)
│   └── feature_extractor.py (54 features from DLC)
│
├── training/
│   ├── train_reach_detector.py
│   ├── train_outcome_classifier.py
│   └── evaluate.py (IoU, F1, per-class metrics)
│
├── evaluation/
│   ├── reach_evaluator_dl.py (compare with GT)
│   ├── error_analyzer.py (failure modes)
│   └── feature_importance.py
│
└── data/
    ├── ground_truth/ (*_reach_ground_truth.json files)
    ├── processed/ (engineer features, save pickle)
    └── splits/ (train/val/test pickle files per fold)
```

---

## Conclusion

This proposal outlines a clear path to achieving **"perfect alignment"** between automated algorithms and human annotations in the MouseReach pipeline.

### Key Points

1. **Task 1 (Reach Detection)** is the priority:
   - 2,609 verified reaches provide adequate training data
   - BiLSTM-CRF architecture proven for sequence labeling
   - Target: >99% IoU (vs. current ~92%)
   - Timeline: 2 weeks

2. **Task 2 (Outcome Classification)** completes the system:
   - 480 verified outcomes with handled class imbalance
   - MLP + Focal Loss proven for imbalanced classification
   - Target: >98% accuracy on all classes
   - Timeline: 1 week

3. **Task 3 (Boundary Detection)** should be skipped:
   - Current algorithm already 99% accurate
   - Minimal ROI for additional effort
   - Revisit if new datasets show systematic errors

### Expected Outcomes

- **Reach Detection:** >99% IoU, near-perfect human alignment
- **Outcome Classification:** >98% accuracy, better minority class performance
- **Pipeline Efficiency:** Reduced manual review from 10-15% to <1%
- **Research Value:** Publishable results on deep learning for behavior analysis

### Risks

All identified risks have mitigation strategies:
- Small dataset → session-level CV, data augmentation
- Class imbalance → focal loss, SMOTE
- Hand tracking quality → multi-point ensemble, temporal smoothing
- Generalization → true session-level cross-validation

The strategy balances scientific rigor (5-fold CV, held-out test set) with practical implementation (modular architecture, reproducible results).

---

**Prepared by:** Technical Writing Team
**For:** MouseReach Behavior Analysis Project
**Date:** February 5, 2026
**Status:** Ready for Review & Technical Discussion
