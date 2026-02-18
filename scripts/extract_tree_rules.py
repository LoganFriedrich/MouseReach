"""Extract actual decision rules from the boundary polisher XGBoost models."""

import json
import sys
from pathlib import Path

try:
    import xgboost as xgb
    import numpy as np
except ImportError:
    print("Need xgboost and numpy")
    sys.exit(1)

MODEL_DIR = Path(r"y:\2_Connectome\Behavior\MouseReach\training\models")

# Load config for feature names
with open(MODEL_DIR / "boundary_polisher_config.json") as f:
    config = json.load(f)

feature_names = config['feature_names']

def load_model(filename, model_type='classifier'):
    if model_type == 'classifier':
        m = xgb.XGBClassifier()
    else:
        m = xgb.XGBRegressor()
    m.load_model(str(MODEL_DIR / filename))
    return m

def print_feature_importance(model, name, top_n=20):
    """Print top features by importance."""
    print(f"\n{'='*70}")
    print(f"  {name}: TOP {top_n} MOST IMPORTANT FEATURES")
    print(f"{'='*70}")

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    for rank, idx in enumerate(indices[:top_n], 1):
        feat = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        print(f"  {rank:3d}. {feat:40s}  importance={importances[idx]:.4f}")

def extract_tree_text(model, name, n_trees=3):
    """Extract and print actual tree rules in plain text."""
    print(f"\n{'='*70}")
    print(f"  {name}: FIRST {n_trees} TREES (ACTUAL DECISION RULES)")
    print(f"{'='*70}")

    booster = model.get_booster()
    dump = booster.get_dump(with_stats=True)

    for tree_idx in range(min(n_trees, len(dump))):
        print(f"\n--- Tree {tree_idx} ---")
        tree_text = dump[tree_idx]

        # Replace feature indices with names
        for i, name_str in enumerate(feature_names):
            tree_text = tree_text.replace(f"f{i}]", f"{name_str}]")
            tree_text = tree_text.replace(f"f{i}<", f"{name_str}<")

        # Print with readable indentation
        for line in tree_text.strip().split('\n'):
            # Parse the line to make it more readable
            depth = line.count('\t')
            line = line.strip()
            indent = "  " * depth

            if 'leaf=' in line:
                # Leaf node
                leaf_val = line.split('leaf=')[1].split(',')[0]
                cover = ""
                if 'cover=' in line:
                    cover = f"  (covers {line.split('cover=')[1].split(')')[0]} samples)"
                print(f"{indent}-> PREDICT: {float(leaf_val):+.4f}{cover}")
            elif '<' in line:
                # Split node
                parts = line.split('[')
                if len(parts) > 1:
                    condition = parts[1].split(']')[0]
                    yes_no = ""
                    if 'yes=' in line:
                        yes = line.split('yes=')[1].split(',')[0]
                        no = line.split('no=')[1].split(',')[0]
                        yes_no = f"  [yes->{yes}, no->{no}]"
                    cover = ""
                    if 'cover=' in line:
                        cover_val = line.split('cover=')[1].rstrip(')')
                        cover = f"  ({cover_val} samples)"
                    print(f"{indent}IF {condition}{yes_no}{cover}")

def summarize_split_thresholds(model, name, top_n=15):
    """Summarize the most common split conditions across all trees."""
    print(f"\n{'='*70}")
    print(f"  {name}: MOST COMMON SPLIT CONDITIONS (across all trees)")
    print(f"{'='*70}")

    booster = model.get_booster()
    dump = booster.get_dump()

    from collections import defaultdict
    split_counts = defaultdict(list)  # feature -> list of thresholds

    for tree_text in dump:
        for line in tree_text.strip().split('\n'):
            if '<' in line and '[' in line:
                try:
                    condition = line.split('[')[1].split(']')[0]
                    feat_idx_str, threshold = condition.split('<')
                    feat_idx = int(feat_idx_str.replace('f', ''))
                    feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"f{feat_idx}"
                    split_counts[feat_name].append(float(threshold))
                except (ValueError, IndexError):
                    continue

    # Sort by number of times a feature is used as a split
    sorted_features = sorted(split_counts.items(), key=lambda x: len(x[1]), reverse=True)

    for feat_name, thresholds in sorted_features[:top_n]:
        thresholds = np.array(thresholds)
        print(f"\n  {feat_name}")
        print(f"    Used in {len(thresholds)} splits across all trees")
        print(f"    Threshold range: {thresholds.min():.2f} to {thresholds.max():.2f}")
        print(f"    Median threshold: {np.median(thresholds):.2f}")
        print(f"    Most common split region: {np.percentile(thresholds, 25):.2f} - {np.percentile(thresholds, 75):.2f}")


# === MAIN ===

print("Loading models...")
start_cls = load_model("boundary_polisher_start_cls.json", "classifier")
start_reg = load_model("boundary_polisher_start_reg.json", "regressor")
end_cls = load_model("boundary_polisher_end_cls.json", "classifier")
end_reg = load_model("boundary_polisher_end_reg.json", "regressor")

print(f"Models loaded. {len(feature_names)} features, window={config['window']}")
print(f"Start classifier threshold: {config['start_params']['cls_threshold']}")
print(f"End classifier threshold: {config['end_params']['cls_threshold']}")
print(f"Max correction: {config['max_correction']} frames")

# Feature importances
print_feature_importance(start_cls, "START CLASSIFIER (needs correction?)")
print_feature_importance(start_reg, "START REGRESSOR (how many frames?)")
print_feature_importance(end_cls, "END CLASSIFIER (needs correction?)")
print_feature_importance(end_reg, "END REGRESSOR (how many frames?)")

# Most common split conditions
summarize_split_thresholds(start_cls, "START CLASSIFIER")
summarize_split_thresholds(end_cls, "END CLASSIFIER")
summarize_split_thresholds(start_reg, "START REGRESSOR")
summarize_split_thresholds(end_reg, "END REGRESSOR")

# First few actual trees
extract_tree_text(start_cls, "START CLASSIFIER", n_trees=2)
extract_tree_text(end_cls, "END CLASSIFIER", n_trees=2)
