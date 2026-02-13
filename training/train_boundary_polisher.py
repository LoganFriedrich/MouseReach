"""
Train a boundary polishing model that corrects reach start/end frames.

Architecture:
    v5.1 rule-based detector -> coarse boundaries
    -> ML boundary polisher -> predicts frame corrections from DLC signals
    -> Final reaches (target: 99%+ accuracy)

Strategy: Conservative two-stage correction
    Stage 1: Classify whether boundary needs correction (|offset| > 2)
    Stage 2: For boundaries needing correction, predict the offset

    This avoids the naive regression problem where model noise corrupts
    the ~80% of boundaries that are already correct.

Features per frame (in the window):
    - 4 hand points: x relative to slit, likelihood (8 features)
    - Nose: x relative to slit, likelihood (2 features)
    - Mean hand x relative to slit (1 feature)
    - Number visible hand points (1 feature)
    - Hand x velocity (1 feature)
    Total: ~13 features per frame x (2*WINDOW+1) frames = ~403 features

Additional context features:
    - Reach duration, max extent, confidence, boundary type
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.metrics import mean_absolute_error, classification_report
from sklearn.model_selection import GroupKFold

DATA_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing")
ALGO_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach\Archive\Pipeline_5_1")
MODEL_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach\training\models")
CACHE_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach\training\cache")

RH_POINTS = ['RightHand', 'RHLeft', 'RHOut', 'RHRight']
HAND_THRESHOLD = 0.5
WINDOW = 20  # frames before and after boundary (larger window for better context)
MAX_CORRECTION = 30  # clamp targets to ±30 (covers most correctable errors)


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_dlc_cached(video):
    """Load DLC data with numpy array caching."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{video}_arrays.npz"

    if cache_file.exists():
        data = np.load(cache_file)
        arrays = {key: data[key] for key in data.files}
        return arrays, int(arrays['_n_frames'][0])

    dlc_files = list(DATA_DIR.glob(f"{video}DLC*.h5"))
    if not dlc_files:
        return None, 0

    t0 = time.time()
    df = pd.read_hdf(dlc_files[0])
    if isinstance(df.columns, pd.MultiIndex):
        scorer = df.columns.get_level_values(0)[0]
        df = df[scorer]
        df.columns = [f"{bp}_{coord}" for bp, coord in df.columns]

    n = len(df)
    arrays = {}
    for p in RH_POINTS:
        arrays[f'{p}_x'] = df[f'{p}_x'].values if f'{p}_x' in df.columns else np.full(n, np.nan)
        arrays[f'{p}_l'] = df[f'{p}_likelihood'].values if f'{p}_likelihood' in df.columns else np.zeros(n)
    arrays['nose_x'] = df['Nose_x'].values if 'Nose_x' in df.columns else np.full(n, np.nan)
    arrays['nose_l'] = df['Nose_likelihood'].values if 'Nose_likelihood' in df.columns else np.zeros(n)
    arrays['BOXL_x'] = df['BOXL_x'].values if 'BOXL_x' in df.columns else np.full(n, np.nan)
    arrays['BOXR_x'] = df['BOXR_x'].values if 'BOXR_x' in df.columns else np.full(n, np.nan)
    arrays['_n_frames'] = np.array([n])

    np.savez_compressed(cache_file, **arrays)
    elapsed = time.time() - t0
    print(f"    [cached in {elapsed:.1f}s]", flush=True)

    return arrays, n


def extract_boundary_features(arrays, n_frames, boundary_frame, slit_x, window=WINDOW):
    """Extract feature vector for a boundary using pre-computed arrays."""
    n_features_per_frame = 13
    total_frames = 2 * window + 1
    features = np.full(n_features_per_frame * total_frames + 4, np.nan)

    prev_best_x = None

    for i, offset in enumerate(range(-window, window + 1)):
        frame = boundary_frame + offset
        base = i * n_features_per_frame

        if frame < 0 or frame >= n_frames:
            continue

        best_l = 0
        best_x = None
        visible_xs = []

        for pi, p in enumerate(RH_POINTS):
            x = arrays[f'{p}_x'][frame]
            l = arrays[f'{p}_l'][frame]
            features[base + pi] = (x - slit_x) if not np.isnan(x) else np.nan
            features[base + 4 + pi] = l
            if l >= HAND_THRESHOLD:
                if not np.isnan(x):
                    visible_xs.append(x)
                if l > best_l:
                    best_l = l
                    best_x = x

        nose_x = arrays['nose_x'][frame]
        nose_l = arrays['nose_l'][frame]
        features[base + 8] = (nose_x - slit_x) if not np.isnan(nose_x) else np.nan
        features[base + 9] = nose_l
        features[base + 10] = (np.mean(visible_xs) - slit_x) if visible_xs else np.nan
        features[base + 11] = len(visible_xs)

        if best_x is not None and prev_best_x is not None:
            features[base + 12] = best_x - prev_best_x
        else:
            features[base + 12] = np.nan

        if best_x is not None:
            prev_best_x = best_x

    return features


def get_slit_center(arrays, seg_start, seg_end):
    bl = np.nanmedian(arrays['BOXL_x'][seg_start:seg_end])
    br = np.nanmedian(arrays['BOXR_x'][seg_start:seg_end])
    return (bl + br) / 2


def match_reaches(gt_reaches, algo_reaches, max_dist=30):
    candidates = []
    for gi, gr in enumerate(gt_reaches):
        for ai, ar in enumerate(algo_reaches):
            dist = abs(gr['start_frame'] - ar.get('start_frame', 0))
            if dist <= max_dist:
                candidates.append((dist, gi, ai))
    candidates.sort()
    gt_used, algo_used = set(), set()
    matches = []
    for dist, gi, ai in candidates:
        if gi not in gt_used and ai not in algo_used:
            gt_used.add(gi)
            algo_used.add(ai)
            matches.append((gi, ai))
    return matches


def build_dataset():
    """Build feature matrices and targets from all GT-matched reaches."""
    print("Building training dataset...", flush=True)
    t0 = time.time()

    all_start_features = []
    all_end_features = []
    y_start = []
    y_end = []
    groups = []
    metadata = []
    video_idx = 0

    gt_files = sorted(DATA_DIR.glob("*_unified_ground_truth.json"))
    gt_files = [f for f in gt_files if 'archive' not in str(f)]
    print(f"Found {len(gt_files)} GT files", flush=True)

    for gt_file in gt_files:
        gt = load_json(gt_file)
        if not gt:
            continue
        video = gt['video_name']

        gt_reaches = [r for r in gt.get('reaches', {}).get('reaches', [])
                      if r.get('start_determined') and r.get('end_determined')
                      and not r.get('exclude_from_analysis', False)]
        if not gt_reaches:
            continue

        algo_file = ALGO_DIR / f"{video}_reaches.json"
        algo_data = load_json(algo_file)
        if not algo_data:
            continue

        algo_reaches = [r for seg in algo_data.get('segments', [])
                        for r in seg.get('reaches', [])]

        print(f"  Loading {video}...", end='', flush=True)
        arrays, n_frames = load_dlc_cached(video)
        if arrays is None:
            print(" no DLC data", flush=True)
            continue

        seg_data = load_json(DATA_DIR / f"{video}_segments.json")
        segments = seg_data.get('segments', []) if seg_data else []
        algo_segments = algo_data.get('segments', [])

        def find_slit(frame):
            for seg in segments:
                if seg.get('start_frame', 0) <= frame <= seg.get('end_frame', n_frames):
                    return get_slit_center(arrays, seg['start_frame'], seg['end_frame'])
            for seg in algo_segments:
                if seg.get('start_frame', 0) <= frame <= seg.get('end_frame', n_frames):
                    return get_slit_center(arrays, seg['start_frame'], seg['end_frame'])
            return None

        matches = match_reaches(gt_reaches, algo_reaches)
        n_matched = 0
        for gi, ai in matches:
            gr = gt_reaches[gi]
            ar = algo_reaches[ai]

            algo_start = ar.get('start_frame', 0)
            algo_end = ar.get('end_frame', 0)
            gt_start = gr['start_frame']
            gt_end = gr['end_frame']

            slit_x = find_slit(gt_start)
            if slit_x is None:
                continue

            start_feats = extract_boundary_features(arrays, n_frames, algo_start, slit_x)
            ctx_idx = 13 * (2 * WINDOW + 1)
            start_feats[ctx_idx] = algo_end - algo_start + 1
            start_feats[ctx_idx + 1] = ar.get('max_extent_pixels', 0)
            start_feats[ctx_idx + 2] = ar.get('confidence', 0) or 0
            start_feats[ctx_idx + 3] = 0

            end_feats = extract_boundary_features(arrays, n_frames, algo_end, slit_x)
            end_feats[ctx_idx] = algo_end - algo_start + 1
            end_feats[ctx_idx + 1] = ar.get('max_extent_pixels', 0)
            end_feats[ctx_idx + 2] = ar.get('confidence', 0) or 0
            end_feats[ctx_idx + 3] = 1

            all_start_features.append(start_feats)
            all_end_features.append(end_feats)

            # Clamp targets to ±MAX_CORRECTION
            s_off = np.clip(gt_start - algo_start, -MAX_CORRECTION, MAX_CORRECTION)
            e_off = np.clip(gt_end - algo_end, -MAX_CORRECTION, MAX_CORRECTION)
            y_start.append(s_off)
            y_end.append(e_off)
            groups.append(video_idx)
            metadata.append({
                'video': video,
                'gt_start': gt_start, 'gt_end': gt_end,
                'algo_start': algo_start, 'algo_end': algo_end,
                'start_offset': gt_start - algo_start,
                'end_offset': gt_end - algo_end,
            })
            n_matched += 1

        if n_matched > 0:
            print(f" {n_matched} matched", flush=True)
            video_idx += 1
        else:
            print(f" 0 matched", flush=True)

    feat_names = []
    for offset in range(-WINDOW, WINDOW + 1):
        prefix = f'f{offset:+03d}'
        for p in RH_POINTS:
            feat_names.append(f'{prefix}_{p}_x')
        for p in RH_POINTS:
            feat_names.append(f'{prefix}_{p}_l')
        feat_names.extend([f'{prefix}_nose_x', f'{prefix}_nose_l',
                          f'{prefix}_mean_hand_x', f'{prefix}_n_visible',
                          f'{prefix}_velocity'])
    feat_names.extend(['ctx_duration', 'ctx_max_extent', 'ctx_confidence', 'ctx_boundary_type'])

    X_start = np.array(all_start_features)
    X_end = np.array(all_end_features)
    y_start = np.array(y_start)
    y_end = np.array(y_end)
    groups = np.array(groups)

    elapsed = time.time() - t0
    print(f"\nDataset: {len(y_start)} reaches, {video_idx} videos, {X_start.shape[1]} features ({elapsed:.1f}s)", flush=True)
    print(f"  Start offsets: mean={y_start.mean():.2f}, std={y_start.std():.2f}, [{y_start.min()}, {y_start.max()}]", flush=True)
    print(f"  End offsets:   mean={y_end.mean():.2f}, std={y_end.std():.2f}, [{y_end.min()}, {y_end.max()}]", flush=True)

    return X_start, X_end, y_start, y_end, groups, metadata, feat_names


def train_conservative_polisher(X, y, groups, feat_names, boundary_type="start"):
    """Train with conservative two-stage correction.

    Stage 1: Regression model predicts offset
    Stage 2: Apply correction only when |prediction| >= threshold

    Sweep thresholds to find optimal one that maximizes within-2 accuracy.
    """
    print(f"\n{'='*70}", flush=True)
    print(f"Training {boundary_type.upper()} boundary polisher (conservative)", flush=True)
    print(f"{'='*70}", flush=True)

    X_clean = np.nan_to_num(X, nan=-999)
    n = len(y)
    n_videos = len(np.unique(groups))
    n_folds = min(5, n_videos)
    gkf = GroupKFold(n_splits=n_folds)

    # Get raw offsets (unclamped) for evaluation
    raw_offsets = y.copy()  # these are already clamped, but that's fine for within-2 eval

    # Stage 1: Train regressor with cross-validation
    reg_predictions = np.zeros(n, dtype=float)
    t0 = time.time()
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_clean, y, groups)):
        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.06,
            subsample=0.8, colsample_bytree=0.6, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42,
            verbosity=0, tree_method='hist',
        )
        model.fit(X_clean[train_idx], y[train_idx])
        reg_predictions[test_idx] = model.predict(X_clean[test_idx])
        mae = mean_absolute_error(y[test_idx], reg_predictions[test_idx])
        print(f"  Fold {fold+1}/{n_folds} regressor (MAE={mae:.2f})", flush=True)
    reg_time = time.time() - t0

    # Stage 2: Also train classifier (is |offset| > 2?)
    y_cls = (np.abs(y) > 2).astype(int)
    cls_predictions = np.zeros(n, dtype=float)
    t0 = time.time()
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_clean, y_cls, groups)):
        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.06,
            subsample=0.8, colsample_bytree=0.6, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42,
            verbosity=0, tree_method='hist', eval_metric='logloss',
        )
        model.fit(X_clean[train_idx], y_cls[train_idx])
        cls_predictions[test_idx] = model.predict_proba(X_clean[test_idx])[:, 1]
        print(f"  Fold {fold+1}/{n_folds} classifier", flush=True)
    cls_time = time.time() - t0
    print(f"  Training: regressor {reg_time:.0f}s, classifier {cls_time:.0f}s", flush=True)

    # Evaluate multiple strategies
    base_exact = np.sum(y == 0)
    base_w2 = np.sum(np.abs(y) <= 2)

    print(f"\n  Baseline: exact={base_exact}/{n} ({base_exact/n*100:.1f}%), "
          f"w2={base_w2}/{n} ({base_w2/n*100:.1f}%)", flush=True)

    # Strategy A: Raw regression (round to nearest int)
    raw_pred = np.round(reg_predictions).astype(int)
    corrected_a = y - raw_pred
    a_exact = np.sum(corrected_a == 0)
    a_w2 = np.sum(np.abs(corrected_a) <= 2)
    print(f"\n  Strategy A (raw regression): exact={a_exact}/{n} ({a_exact/n*100:.1f}%), "
          f"w2={a_w2}/{n} ({a_w2/n*100:.1f}%)", flush=True)

    # Strategy B: Threshold sweep on regression predictions
    print(f"\n  Strategy B (regression + threshold):", flush=True)
    best_thresh = 0
    best_w2 = base_w2
    best_exact = base_exact
    for thresh in np.arange(0.5, 8.0, 0.5):
        correction = np.where(np.abs(reg_predictions) >= thresh,
                              np.round(reg_predictions).astype(int), 0)
        residual = y - correction
        t_exact = np.sum(residual == 0)
        t_w2 = np.sum(np.abs(residual) <= 2)
        delta_w2 = t_w2 - base_w2
        delta_exact = t_exact - base_exact
        marker = " <-- BEST" if t_w2 > best_w2 else ""
        print(f"    thresh={thresh:.1f}: exact={t_exact}/{n} ({t_exact/n*100:.1f}%) [{delta_exact:+d}], "
              f"w2={t_w2}/{n} ({t_w2/n*100:.1f}%) [{delta_w2:+d}]{marker}", flush=True)
        if t_w2 > best_w2:
            best_w2 = t_w2
            best_exact = t_exact
            best_thresh = thresh

    print(f"\n  Best regression threshold: {best_thresh} -> w2={best_w2}/{n} ({best_w2/n*100:.1f}%)", flush=True)

    # Strategy C: Classifier-gated regression
    print(f"\n  Strategy C (classifier-gated regression):", flush=True)
    best_c_thresh = 0.5
    best_c_w2 = base_w2
    for cls_thresh in np.arange(0.2, 0.95, 0.05):
        needs_correction = cls_predictions >= cls_thresh
        correction = np.where(needs_correction, np.round(reg_predictions).astype(int), 0)
        residual = y - correction
        t_exact = np.sum(residual == 0)
        t_w2 = np.sum(np.abs(residual) <= 2)
        n_corrected = np.sum(needs_correction)
        delta_w2 = t_w2 - base_w2
        # Also track cases made worse
        n_worse = np.sum((np.abs(y) <= 2) & (np.abs(residual) > 2))
        marker = " <-- BEST" if t_w2 > best_c_w2 else ""
        print(f"    cls_thresh={cls_thresh:.2f}: correcting {n_corrected}/{n}, "
              f"exact={t_exact}/{n} ({t_exact/n*100:.1f}%), "
              f"w2={t_w2}/{n} ({t_w2/n*100:.1f}%) [{delta_w2:+d}], worse={n_worse}{marker}", flush=True)
        if t_w2 > best_c_w2:
            best_c_w2 = t_w2
            best_c_thresh = cls_thresh

    print(f"\n  Best classifier threshold: {best_c_thresh} -> w2={best_c_w2}/{n} ({best_c_w2/n*100:.1f}%)", flush=True)

    # Strategy D: Combined - classifier gate + regression threshold
    print(f"\n  Strategy D (classifier + regression threshold):", flush=True)
    best_d_w2 = base_w2
    best_d_params = (0.5, 1.5)
    for cls_thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for reg_thresh in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
            needs_correction = (cls_predictions >= cls_thresh) & (np.abs(reg_predictions) >= reg_thresh)
            correction = np.where(needs_correction, np.round(reg_predictions).astype(int), 0)
            residual = y - correction
            t_exact = np.sum(residual == 0)
            t_w2 = np.sum(np.abs(residual) <= 2)
            if t_w2 > best_d_w2:
                best_d_w2 = t_w2
                best_d_params = (cls_thresh, reg_thresh)

    cls_t, reg_t = best_d_params
    needs_correction = (cls_predictions >= cls_t) & (np.abs(reg_predictions) >= reg_t)
    correction = np.where(needs_correction, np.round(reg_predictions).astype(int), 0)
    residual = y - correction
    d_exact = np.sum(residual == 0)
    d_w2 = np.sum(np.abs(residual) <= 2)
    n_corrected = np.sum(needs_correction)
    print(f"    Best: cls={cls_t:.1f}, reg={reg_t:.1f}, correcting {n_corrected}/{n}", flush=True)
    print(f"    exact={d_exact}/{n} ({d_exact/n*100:.1f}%), "
          f"w2={d_w2}/{n} ({d_w2/n*100:.1f}%) [{d_w2-base_w2:+d}]", flush=True)

    # Pick best overall strategy
    strategies = {
        'baseline': (base_w2, base_exact, 'no correction'),
        'raw_regression': (a_w2, a_exact, 'round(prediction)'),
        'threshold_regression': (best_w2, best_exact, f'threshold={best_thresh}'),
        'classifier_gated': (best_c_w2, 0, f'cls_thresh={best_c_thresh}'),
        'combined': (d_w2, d_exact, f'cls={cls_t}, reg={reg_t}'),
    }

    best_strategy = max(strategies.items(), key=lambda x: x[1][0])
    print(f"\n  BEST STRATEGY: {best_strategy[0]} -> w2={best_strategy[1][0]}/{n} "
          f"({best_strategy[1][0]/n*100:.1f}%), params: {best_strategy[1][2]}", flush=True)

    # Train final models on all data
    print(f"\n  Training final models on all data...", flush=True)
    final_reg = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.06,
        subsample=0.8, colsample_bytree=0.6, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42,
        verbosity=0, tree_method='hist',
    )
    final_reg.fit(X_clean, y)

    final_cls = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.06,
        subsample=0.8, colsample_bytree=0.6, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42,
        verbosity=0, tree_method='hist', eval_metric='logloss',
    )
    final_cls.fit(X_clean, y_cls)

    # Feature importance
    importance = final_reg.feature_importances_
    top_idx = np.argsort(importance)[-15:][::-1]
    print(f"\n  Top 15 regressor features:", flush=True)
    for idx in top_idx:
        name = feat_names[idx] if idx < len(feat_names) else f"feature_{idx}"
        print(f"    {name:<45} {importance[idx]:.4f}", flush=True)

    return (final_reg, final_cls, reg_predictions, cls_predictions,
            best_strategy[0], best_d_params)


def compute_combined_accuracy(metadata, y_start, y_end,
                              start_reg_pred, start_cls_pred, start_params,
                              end_reg_pred, end_cls_pred, end_params):
    """Compute combined accuracy using best strategy for each boundary type."""
    n = len(metadata)

    # Apply best strategy corrections
    s_cls_t, s_reg_t = start_params
    e_cls_t, e_reg_t = end_params

    start_needs = (start_cls_pred >= s_cls_t) & (np.abs(start_reg_pred) >= s_reg_t)
    start_correction = np.where(start_needs, np.round(start_reg_pred).astype(int), 0)

    end_needs = (end_cls_pred >= e_cls_t) & (np.abs(end_reg_pred) >= e_reg_t)
    end_correction = np.where(end_needs, np.round(end_reg_pred).astype(int), 0)

    print(f"\n{'='*70}", flush=True)
    print(f"COMBINED ACCURACY ({n} matched reaches, cross-validated)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Start params: cls_thresh={s_cls_t}, reg_thresh={s_reg_t}, "
          f"correcting {np.sum(start_needs)}/{n}", flush=True)
    print(f"  End params:   cls_thresh={e_cls_t}, reg_thresh={e_reg_t}, "
          f"correcting {np.sum(end_needs)}/{n}", flush=True)

    # Use UNCLAMPED offsets for evaluation (metadata has raw offsets)
    counts = {'base': {}, 'polished': {}}
    for key in ['start_exact', 'start_w2', 'end_exact', 'end_w2', 'both_exact', 'both_w2']:
        counts['base'][key] = 0
        counts['polished'][key] = 0

    worse_cases = []
    for i, m in enumerate(metadata):
        s_off = m['start_offset']  # raw (unclamped)
        e_off = m['end_offset']
        cs = s_off - start_correction[i]
        ce = e_off - end_correction[i]

        if s_off == 0: counts['base']['start_exact'] += 1
        if abs(s_off) <= 2: counts['base']['start_w2'] += 1
        if e_off == 0: counts['base']['end_exact'] += 1
        if abs(e_off) <= 2: counts['base']['end_w2'] += 1
        if s_off == 0 and e_off == 0: counts['base']['both_exact'] += 1
        if abs(s_off) <= 2 and abs(e_off) <= 2: counts['base']['both_w2'] += 1

        if cs == 0: counts['polished']['start_exact'] += 1
        if abs(cs) <= 2: counts['polished']['start_w2'] += 1
        if ce == 0: counts['polished']['end_exact'] += 1
        if abs(ce) <= 2: counts['polished']['end_w2'] += 1
        if cs == 0 and ce == 0: counts['polished']['both_exact'] += 1
        if abs(cs) <= 2 and abs(ce) <= 2: counts['polished']['both_w2'] += 1

        # Track cases made worse
        base_ok = abs(s_off) <= 2 and abs(e_off) <= 2
        pol_ok = abs(cs) <= 2 and abs(ce) <= 2
        if base_ok and not pol_ok:
            worse_cases.append(m)

    print(f"\n  {'Metric':<25} {'Baseline':>20} {'Polished':>20} {'Delta':>8}", flush=True)
    print(f"  {'-'*75}", flush=True)
    for key, label in [('start_exact', 'Start exact'), ('start_w2', 'Start within 2'),
                       ('end_exact', 'End exact'), ('end_w2', 'End within 2'),
                       ('both_exact', 'Both exact'), ('both_w2', 'Both within 2')]:
        b = counts['base'][key]
        p = counts['polished'][key]
        d = p - b
        print(f"  {label:<25} {b:>5}/{n} ({b/n*100:.1f}%)"
              f"  {p:>5}/{n} ({p/n*100:.1f}%) {d:>+5}", flush=True)

    print(f"\n  Cases made WORSE (was both-w2, now not): {len(worse_cases)}", flush=True)

    # Per-video breakdown
    print(f"\n  Per-video (both-within-2):", flush=True)
    videos = {}
    for i, m in enumerate(metadata):
        v = m['video']
        if v not in videos:
            videos[v] = {'n': 0, 'base': 0, 'pol': 0}
        videos[v]['n'] += 1
        s_off = m['start_offset']
        e_off = m['end_offset']
        cs = s_off - start_correction[i]
        ce = e_off - end_correction[i]
        if abs(s_off) <= 2 and abs(e_off) <= 2:
            videos[v]['base'] += 1
        if abs(cs) <= 2 and abs(ce) <= 2:
            videos[v]['pol'] += 1

    for v in sorted(videos.keys()):
        d = videos[v]
        delta = d['pol'] - d['base']
        print(f"    {v}: {d['base']}/{d['n']} -> {d['pol']}/{d['n']} [{delta:+d}]", flush=True)

    return counts


def main():
    total_t0 = time.time()
    print(f"Boundary Polisher Training (Conservative Strategy)", flush=True)
    print(f"{'='*70}\n", flush=True)

    X_start, X_end, y_start, y_end, groups, metadata, feat_names = build_dataset()

    if len(y_start) == 0:
        print("ERROR: No matched reaches found!", flush=True)
        return

    (start_reg, start_cls, start_reg_pred, start_cls_pred,
     start_best_name, start_params) = train_conservative_polisher(
        X_start, y_start, groups, feat_names, "start")

    (end_reg, end_cls, end_reg_pred, end_cls_pred,
     end_best_name, end_params) = train_conservative_polisher(
        X_end, y_end, groups, feat_names, "end")

    compute_combined_accuracy(metadata, y_start, y_end,
                              start_reg_pred, start_cls_pred, start_params,
                              end_reg_pred, end_cls_pred, end_params)

    # Save models and config
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    start_reg.save_model(str(MODEL_DIR / "boundary_polisher_start_reg.json"))
    start_cls.save_model(str(MODEL_DIR / "boundary_polisher_start_cls.json"))
    end_reg.save_model(str(MODEL_DIR / "boundary_polisher_end_reg.json"))
    end_cls.save_model(str(MODEL_DIR / "boundary_polisher_end_cls.json"))

    config = {
        'feature_names': feat_names,
        'window': WINDOW,
        'hand_points': RH_POINTS,
        'hand_threshold': HAND_THRESHOLD,
        'max_correction': MAX_CORRECTION,
        'start_params': {'cls_threshold': start_params[0], 'reg_threshold': start_params[1]},
        'end_params': {'cls_threshold': end_params[0], 'reg_threshold': end_params[1]},
        'n_training_reaches': len(y_start),
        'n_videos': int(len(np.unique(groups))),
    }
    with open(MODEL_DIR / "boundary_polisher_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    total_elapsed = time.time() - total_t0
    print(f"\nTotal time: {total_elapsed:.0f}s", flush=True)
    print(f"Models saved to {MODEL_DIR}", flush=True)


if __name__ == "__main__":
    main()
