"""
GPU detection, validation, and CUDA environment setup for MouseReach.

This module handles the messy reality of GPU computing on Windows:
  - TensorFlow 2.10.x is the last version with native Windows GPU support
  - cuDNN must be on PATH (typically via conda's Library/bin)
  - CUDA toolkit bin must be on PATH
  - PyTorch ships CPU-only from default pip; CUDA builds need --index-url
  - TF_USE_LEGACY_KERAS=1 is required for DLC 2.3.x with TF 2.10+

Usage:
    from mousereach.gpu import check_gpu, setup_gpu_env, print_gpu_info

    # Quick boolean check
    if check_gpu().has_any_gpu:
        print("GPU available")

    # Ensure CUDA paths and env vars are set (idempotent)
    setup_gpu_env()

    # Full diagnostic printout
    print_gpu_info()
"""

import os
import sys
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# GPU STATUS
# =============================================================================

@dataclass
class GPUStatus:
    """Result of GPU detection and validation."""

    # Hardware
    nvidia_gpu_name: Optional[str] = None
    nvidia_driver_version: Optional[str] = None
    nvidia_smi_available: bool = False

    # TensorFlow
    tf_version: Optional[str] = None
    tf_gpu_available: bool = False
    tf_gpu_devices: list = field(default_factory=list)

    # PyTorch
    torch_version: Optional[str] = None
    torch_cuda_available: bool = False
    torch_cuda_version: Optional[str] = None
    torch_gpu_name: Optional[str] = None

    # cuDNN
    cudnn_available: bool = False
    cudnn_version: Optional[str] = None

    # Environment
    cuda_paths_on_path: list = field(default_factory=list)
    tf_use_legacy_keras: bool = False

    # Warnings
    warnings: list = field(default_factory=list)

    @property
    def has_any_gpu(self) -> bool:
        """True if any GPU backend (TF or PyTorch) can see a GPU."""
        return self.tf_gpu_available or self.torch_cuda_available

    @property
    def has_nvidia_hardware(self) -> bool:
        """True if NVIDIA hardware was detected (even if drivers/libs missing)."""
        return self.nvidia_gpu_name is not None


# =============================================================================
# DETECTION
# =============================================================================

def _detect_nvidia_smi() -> tuple:
    """Try nvidia-smi to get GPU name and driver version.

    Returns:
        (gpu_name, driver_version, available) tuple
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(',')
            gpu_name = parts[0].strip() if len(parts) > 0 else None
            driver_version = parts[1].strip() if len(parts) > 1 else None
            return gpu_name, driver_version, True
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return None, None, False


def _detect_tf_gpu() -> tuple:
    """Check TensorFlow GPU availability.

    Returns:
        (version, gpu_available, gpu_devices, warnings) tuple
    """
    warnings = []
    try:
        # Suppress TF's verbose startup logging
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
        import tensorflow as tf
        version = tf.__version__

        # Check version compatibility on Windows
        if sys.platform == 'win32':
            major_minor = tuple(int(x) for x in version.split('.')[:2])
            if major_minor > (2, 10):
                warnings.append(
                    f"TensorFlow {version} has no native Windows GPU support. "
                    f"Pin to tensorflow==2.10.* for GPU on Windows."
                )

        gpus = tf.config.list_physical_devices('GPU')
        gpu_names = [g.name for g in gpus]
        return version, len(gpus) > 0, gpu_names, warnings

    except ImportError:
        return None, False, [], ["TensorFlow not installed"]
    except Exception as e:
        return None, False, [], [f"TensorFlow GPU check failed: {e}"]


def _detect_torch_gpu() -> tuple:
    """Check PyTorch CUDA availability.

    Returns:
        (version, cuda_available, cuda_version, gpu_name, warnings) tuple
    """
    warnings = []
    try:
        import torch
        version = torch.__version__

        # Check for CPU-only build
        if '+cpu' in version or not torch.cuda.is_available():
            if '+cpu' in version:
                warnings.append(
                    f"PyTorch {version} is a CPU-only build. "
                    f"Install CUDA build: pip install torch --index-url "
                    f"https://download.pytorch.org/whl/cu121"
                )
            return version, False, None, None, warnings

        cuda_version = torch.version.cuda
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
        return version, True, cuda_version, gpu_name, warnings

    except ImportError:
        return None, False, None, None, ["PyTorch not installed"]
    except Exception as e:
        return None, False, None, None, [f"PyTorch GPU check failed: {e}"]


def _detect_cudnn() -> tuple:
    """Check if cuDNN is available via TF or PyTorch.

    Returns:
        (available, version) tuple
    """
    # Try PyTorch first (faster)
    try:
        import torch
        if torch.cuda.is_available() and torch.backends.cudnn.is_available():
            version = str(torch.backends.cudnn.version())
            # Format version number (e.g., 8907 -> 8.9.7)
            if len(version) >= 4:
                formatted = f"{version[0]}.{version[1]}.{version[2:]}"
                return True, formatted
            return True, version
    except Exception:
        pass

    # Try TF
    try:
        import tensorflow as tf
        build_info = tf.sysconfig.get_build_info()
        cudnn_version = build_info.get('cudnn_version')
        if cudnn_version:
            return True, str(cudnn_version)
    except Exception:
        pass

    return False, None


def _find_cuda_paths() -> list:
    """Find CUDA-related directories on PATH."""
    cuda_dirs = []
    path_dirs = os.environ.get('PATH', '').split(os.pathsep)
    for d in path_dirs:
        d_lower = d.lower()
        if 'cuda' in d_lower or ('library' in d_lower and 'bin' in d_lower):
            if Path(d).exists():
                cuda_dirs.append(d)
    return cuda_dirs


# =============================================================================
# MAIN CHECK FUNCTION
# =============================================================================

def check_gpu() -> GPUStatus:
    """Run full GPU detection and validation.

    Returns a GPUStatus dataclass with all findings. This is the main
    entry point for GPU diagnostics.
    """
    status = GPUStatus()

    # 1. NVIDIA hardware (nvidia-smi)
    gpu_name, driver_ver, smi_ok = _detect_nvidia_smi()
    status.nvidia_gpu_name = gpu_name
    status.nvidia_driver_version = driver_ver
    status.nvidia_smi_available = smi_ok

    # 2. TensorFlow
    tf_ver, tf_gpu, tf_devices, tf_warns = _detect_tf_gpu()
    status.tf_version = tf_ver
    status.tf_gpu_available = tf_gpu
    status.tf_gpu_devices = tf_devices
    status.warnings.extend(tf_warns)

    # 3. PyTorch
    torch_ver, torch_cuda, cuda_ver, torch_gpu, torch_warns = _detect_torch_gpu()
    status.torch_version = torch_ver
    status.torch_cuda_available = torch_cuda
    status.torch_cuda_version = cuda_ver
    status.torch_gpu_name = torch_gpu
    status.warnings.extend(torch_warns)

    # 4. cuDNN
    status.cudnn_available, status.cudnn_version = _detect_cudnn()

    # 5. Environment
    status.cuda_paths_on_path = _find_cuda_paths()
    status.tf_use_legacy_keras = os.environ.get('TF_USE_LEGACY_KERAS') == '1'

    # 6. Cross-checks
    if status.has_nvidia_hardware and not status.has_any_gpu:
        status.warnings.append(
            "NVIDIA GPU detected but no ML framework can use it. "
            "Check CUDA toolkit, cuDNN, and framework installations."
        )

    if status.tf_gpu_available and not status.cudnn_available:
        status.warnings.append(
            "TensorFlow reports GPU but cuDNN not detected. "
            "DLC may fail — install cuDNN: conda install cudnn=8.*"
        )

    if not status.tf_use_legacy_keras and status.tf_version:
        status.warnings.append(
            "TF_USE_LEGACY_KERAS not set. DLC 2.3.x requires this. "
            "Set TF_USE_LEGACY_KERAS=1 in environment."
        )

    return status


# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

def setup_gpu_env(conda_prefix: str = None):
    """Ensure CUDA paths and environment variables are set for GPU use.

    This is idempotent — safe to call multiple times. It:
      1. Sets TF_USE_LEGACY_KERAS=1 (required for DLC 2.3.x)
      2. Adds conda Library/bin to PATH (for cuDNN DLLs)
      3. Adds CUDA toolkit bin to PATH (if found)
      4. Suppresses TF verbose logging

    Args:
        conda_prefix: Path to conda environment root. Auto-detected from
                      CONDA_PREFIX env var or sys.prefix if not provided.
    """
    # TF_USE_LEGACY_KERAS — DLC 2.3.x needs Keras 2 APIs
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

    # Suppress TF startup spam
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    # Detect conda prefix
    if conda_prefix is None:
        conda_prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    conda_prefix = Path(conda_prefix)

    current_path = os.environ.get("PATH", "")
    additions = []

    # conda Library/bin — contains cuDNN DLLs (cudnn64_8.dll etc.)
    library_bin = conda_prefix / "Library" / "bin"
    if library_bin.exists() and str(library_bin) not in current_path:
        additions.append(str(library_bin))

    # CUDA toolkit — check common install locations
    cuda_candidates = [
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"),
    ]
    for cuda_root in cuda_candidates:
        if cuda_root.exists():
            # Find the most recent version
            versions = sorted(cuda_root.iterdir(), reverse=True)
            for v in versions:
                cuda_bin = v / "bin"
                if cuda_bin.exists() and str(cuda_bin) not in current_path:
                    additions.append(str(cuda_bin))
                    break
            break

    if additions:
        os.environ["PATH"] = os.pathsep.join(additions) + os.pathsep + current_path
        logger.debug(f"Added to PATH: {additions}")


# =============================================================================
# DISPLAY
# =============================================================================

def print_gpu_info():
    """Print a formatted GPU diagnostic summary."""
    status = check_gpu()

    print("\nGPU / CUDA Status:")

    # Hardware
    if status.nvidia_gpu_name:
        print(f"  NVIDIA GPU:        {status.nvidia_gpu_name}")
        print(f"  Driver:            {status.nvidia_driver_version or '?'}")
    else:
        print(f"  NVIDIA GPU:        Not detected")
        if not status.nvidia_smi_available:
            print(f"                     (nvidia-smi not on PATH or failed)")

    # TensorFlow
    if status.tf_version:
        tf_gpu_str = "Yes" if status.tf_gpu_available else "No"
        print(f"  TensorFlow:        {status.tf_version} (GPU: {tf_gpu_str})")
        if status.tf_gpu_devices:
            for dev in status.tf_gpu_devices:
                print(f"                     {dev}")
    else:
        print(f"  TensorFlow:        Not installed")

    # PyTorch
    if status.torch_version:
        if status.torch_cuda_available:
            print(f"  PyTorch:           {status.torch_version} (CUDA {status.torch_cuda_version})")
            if status.torch_gpu_name:
                print(f"                     {status.torch_gpu_name}")
        else:
            print(f"  PyTorch:           {status.torch_version} (CPU only)")
    else:
        print(f"  PyTorch:           Not installed")

    # cuDNN
    if status.cudnn_available:
        print(f"  cuDNN:             {status.cudnn_version or 'available'}")
    else:
        print(f"  cuDNN:             Not detected")

    # Environment
    print(f"  TF_USE_LEGACY_KERAS: {'Set' if status.tf_use_legacy_keras else 'NOT SET'}")
    if status.cuda_paths_on_path:
        print(f"  CUDA paths on PATH:")
        for p in status.cuda_paths_on_path:
            print(f"    {p}")

    # Warnings
    if status.warnings:
        print()
        for w in status.warnings:
            print(f"  [!] {w}")

    print()
