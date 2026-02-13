"""Quick test to verify deep learning setup."""
import sys
from pathlib import Path

print("=" * 60)
print("DEEP LEARNING SETUP VERIFICATION")
print("=" * 60)

# Test config
try:
    from config import get_config
    c = get_config()
    print(f"\n[OK] Config loaded")
    print(f"  - Python: {c.paths.python_exe}")
    print(f"  - Processing dir: {c.paths.processing_dir}")
    print(f"  - Body parts: {len(c.dlc.body_parts)}")
    print(f"  - Hand parts: {c.dlc.hand_parts}")
    print(f"  - Outcome classes: {c.outcome_classifier.outcome_classes}")
except Exception as e:
    print(f"[FAIL] Config: {e}")

# Test feature extraction
try:
    from feature_extraction import DLCDataLoader, FeatureExtractor
    print(f"\n[OK] Feature extraction modules imported")
except Exception as e:
    print(f"[FAIL] Feature extraction: {e}")

# Test data loader
try:
    from data_loader import BoundaryDataset, ReachSequenceDataset, OutcomeDataset
    print(f"[OK] Data loader modules imported")
except Exception as e:
    print(f"[FAIL] Data loader: {e}")

# Test models
try:
    from models import create_boundary_detector, ReachDetector, OutcomeClassifier
    print(f"[OK] Model modules imported")
except Exception as e:
    print(f"[FAIL] Models: {e}")

# Test PyTorch
try:
    import torch
    print(f"\n[OK] PyTorch {torch.__version__}")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"[FAIL] PyTorch: {e}")

# Count data files
try:
    processing = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing")
    dlc_files = list(processing.glob("*DLC*.h5"))
    gt_files = list(processing.glob("*_unified_ground_truth.json"))
    print(f"\n[OK] Data files found")
    print(f"  - DLC files: {len(dlc_files)}")
    print(f"  - GT files: {len(gt_files)}")
except Exception as e:
    print(f"[FAIL] Data files: {e}")

print("\n" + "=" * 60)
print("SETUP VERIFICATION COMPLETE")
print("=" * 60)
