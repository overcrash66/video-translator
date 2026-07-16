#!/usr/bin/env python3
"""
Cross-platform installer for Video Translator.

Detects the operating system, GPU availability, and installs the appropriate
dependency set from pyproject.toml.

Usage:
    python scripts/install.py              # Auto-detect everything
    python scripts/install.py --cpu        # Force CPU-only
    python scripts/install.py --cuda 12.4  # Force specific CUDA version
    python scripts/install.py --cuda 12.8  # Force CUDA 12.8 (RTX 50 series)
    python scripts/install.py --minimal    # Core only (no ML models)
    python scripts/install.py --verify     # Only run verification, skip install
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


# =============================================================================
# CONSTANTS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = PROJECT_ROOT / "venv"

CUDA_INDEX_URLS = {
    "12.4": "https://download.pytorch.org/whl/cu124",
    "12.8": "https://download.pytorch.org/whl/cu128",
}


# =============================================================================
# HELPERS
# =============================================================================

def print_header(msg: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}\n")


def print_step(msg: str) -> None:
    """Print a step indicator."""
    print(f"  → {msg}")


def print_ok(msg: str) -> None:
    """Print a success message."""
    print(f"  ✓ {msg}")


def print_warn(msg: str) -> None:
    """Print a warning message."""
    print(f"  ⚠ {msg}")


def print_error(msg: str) -> None:
    """Print an error message."""
    print(f"  ✗ {msg}")


def run_cmd(cmd: list[str], check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"    $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, **kwargs)


def get_python_executable() -> str:
    """Get the correct Python executable path for the venv."""
    if sys.platform == "win32":
        python = VENV_DIR / "Scripts" / "python.exe"
    else:
        python = VENV_DIR / "bin" / "python"

    if python.exists():
        return str(python)
    return sys.executable


def get_pip_executable() -> str:
    """Get the correct pip executable path for the venv."""
    if sys.platform == "win32":
        pip = VENV_DIR / "Scripts" / "pip.exe"
    else:
        pip = VENV_DIR / "bin" / "pip"

    if pip.exists():
        return str(pip)
    return f"{get_python_executable()} -m pip"


# =============================================================================
# DETECTION
# =============================================================================

def detect_os() -> str:
    """Detect the operating system."""
    system = platform.system().lower()
    if system == "windows":
        return "windows"
    elif system == "darwin":
        return "macos"
    elif system == "linux":
        return "linux"
    else:
        return system


def detect_gpu() -> dict:
    """Detect GPU availability and type."""
    result = {
        "has_nvidia": False,
        "cuda_version": None,
        "gpu_name": None,
    }

    # Check for nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            output = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10
            )
            if output.returncode == 0 and output.stdout.strip():
                result["has_nvidia"] = True
                result["gpu_name"] = output.stdout.strip().split(",")[0].strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

    # Check CUDA version from nvcc
    nvcc = shutil.which("nvcc")
    if nvcc:
        try:
            output = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True, text=True, timeout=10
            )
            if output.returncode == 0:
                # Parse "release X.Y" from output
                for line in output.stdout.split("\n"):
                    if "release" in line.lower():
                        parts = line.split("release")[-1].strip().split(",")[0].strip()
                        result["cuda_version"] = parts
                        break
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

    return result


def detect_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    return shutil.which("ffmpeg") is not None


# =============================================================================
# INSTALLATION
# =============================================================================

def create_venv() -> None:
    """Create a virtual environment if it doesn't exist."""
    if VENV_DIR.exists():
        print_ok(f"Virtual environment already exists: {VENV_DIR}")
        return

    print_step(f"Creating virtual environment at {VENV_DIR}...")
    run_cmd([sys.executable, "-m", "venv", str(VENV_DIR)])
    print_ok("Virtual environment created")


def upgrade_pip() -> None:
    """Upgrade pip in the virtual environment."""
    print_step("Upgrading pip...")
    python = get_python_executable()
    run_cmd([python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
            check=False)


def install_pytorch(cuda_version: str | None, force_cpu: bool) -> None:
    """Install PyTorch with the correct CUDA version."""
    pip = get_pip_executable()
    python = get_python_executable()

    if force_cpu or cuda_version is None:
        print_step("Installing PyTorch (CPU-only)...")
        run_cmd([python, "-m", "pip", "install", "--no-cache-dir",
                 "torch>=2.5.1", "torchvision>=0.20.0", "torchaudio>=2.5.1",
                 "torchmetrics>=1.4.0"])
    else:
        # Determine index URL
        cuda_major_minor = cuda_version
        # Normalize: "12.4.1" -> "12.4"
        if cuda_version.count(".") > 1:
            cuda_major_minor = ".".join(cuda_version.split(".")[:2])

        index_url = CUDA_INDEX_URLS.get(cuda_major_minor)
        if not index_url:
            # Default to 12.4 if unknown version
            print_warn(f"Unknown CUDA version {cuda_version}, defaulting to CUDA 12.4")
            index_url = CUDA_INDEX_URLS["12.4"]

        min_torch = "2.7.0" if cuda_major_minor == "12.8" else "2.5.1"
        print_step(f"Installing PyTorch with CUDA {cuda_major_minor}...")
        run_cmd([python, "-m", "pip", "install", "--no-cache-dir",
                 f"torch>={min_torch}", f"torchvision>=0.20.0", f"torchaudio>={min_torch}",
                 "torchmetrics>=1.4.0",
                 "--extra-index-url", index_url])


def install_dependencies(extras: list[str], os_name: str) -> None:
    """Install project dependencies via pyproject.toml extras."""
    python = get_python_executable()

    # Install the project with selected extras
    extras_str = ",".join(extras)
    print_step(f"Installing dependencies [{extras_str}]...")

    # First install core requirements that might need special handling
    run_cmd([python, "-m", "pip", "install", "--no-cache-dir",
             "numpy<2.0.0", "cython"], check=False)

    # Then install the project with extras
    # Using pip install with the extras from the project directory
    run_cmd([python, "-m", "pip", "install", "--no-cache-dir",
             "-e", f".[{extras_str}]"],
            cwd=str(PROJECT_ROOT), check=False)


def install_platform_specific(os_name: str) -> None:
    """Install platform-specific packages that can't be handled by markers alone."""
    python = get_python_executable()

    if os_name == "macos":
        # macOS: Install CPU-only alternatives
        print_step("Installing macOS-specific alternatives...")
        run_cmd([python, "-m", "pip", "install", "--no-cache-dir",
                 "onnxruntime"], check=False)
    elif os_name == "windows":
        # Windows-specific: voicefixer may need special handling
        print_step("Checking Windows-specific packages...")
        try:
            run_cmd([python, "-m", "pip", "install", "--no-cache-dir",
                     "voicefixer>=0.1.2"], check=False)
        except Exception:
            print_warn("voicefixer installation failed (optional, continuing)")


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_installation(os_name: str) -> dict:
    """Run post-install verification checks."""
    results = {"passed": 0, "failed": 0, "warnings": 0, "details": []}
    python = get_python_executable()

    checks = [
        ("numpy", "import numpy; print(f'numpy {numpy.__version__}')"),
        ("torch", "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"),
        ("gradio", "import gradio; print(f'gradio {gradio.__version__}')"),
        ("soundfile", "import soundfile; print(f'soundfile {soundfile.__version__}')"),
        ("opencv", "import cv2; print(f'opencv {cv2.__version__}')"),
        ("transformers", "import transformers; print(f'transformers {transformers.__version__}')"),
        ("faster_whisper", "import faster_whisper; print('faster_whisper OK')"),
        ("edge_tts", "import edge_tts; print('edge_tts OK')"),
        ("deep_translator", "import deep_translator; print('deep_translator OK')"),
    ]

    for name, check_code in checks:
        try:
            result = subprocess.run(
                [python, "-c", check_code],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                results["passed"] += 1
                results["details"].append((name, "OK", result.stdout.strip()))
            else:
                results["failed"] += 1
                error_msg = result.stderr.strip().split("\n")[-1] if result.stderr else "Unknown error"
                results["details"].append((name, "FAIL", error_msg))
        except subprocess.TimeoutExpired:
            results["warnings"] += 1
            results["details"].append((name, "TIMEOUT", "Import timed out"))
        except Exception as e:
            results["failed"] += 1
            results["details"].append((name, "ERROR", str(e)))

    # Check ffmpeg
    if detect_ffmpeg():
        results["passed"] += 1
        results["details"].append(("ffmpeg", "OK", "Available in PATH"))
    else:
        results["failed"] += 1
        results["details"].append(("ffmpeg", "MISSING", "Not found in PATH"))

    return results


def print_verification_results(results: dict) -> None:
    """Print verification results in a formatted table."""
    print_header("Installation Verification")

    max_name = max(len(d[0]) for d in results["details"])
    for name, status, detail in results["details"]:
        icon = "✓" if status == "OK" else "✗" if status in ("FAIL", "ERROR", "MISSING") else "⚠"
        print(f"  {icon} {name:<{max_name + 2}} {detail}")

    print()
    total = results["passed"] + results["failed"] + results["warnings"]
    print(f"  Results: {results['passed']}/{total} passed", end="")
    if results["warnings"]:
        print(f", {results['warnings']} warnings", end="")
    if results["failed"]:
        print(f", {results['failed']} failed", end="")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    """Main installer entry point."""
    parser = argparse.ArgumentParser(
        description="Cross-platform installer for Video Translator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/install.py              # Auto-detect everything
  python scripts/install.py --cpu        # Force CPU-only (no CUDA)
  python scripts/install.py --cuda 12.8  # Force CUDA 12.8 (RTX 50 series)
  python scripts/install.py --minimal    # Core only (no ML models)
  python scripts/install.py --verify     # Only run verification
        """
    )
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU-only installation (no CUDA)")
    parser.add_argument("--cuda", type=str, default=None,
                        help="Force specific CUDA version (e.g., 12.4, 12.8)")
    parser.add_argument("--minimal", action="store_true",
                        help="Install core dependencies only (no ML models)")
    parser.add_argument("--verify", action="store_true",
                        help="Only run verification, skip installation")
    parser.add_argument("--no-venv", action="store_true",
                        help="Skip virtual environment creation")

    args = parser.parse_args()

    print_header("Video Translator — Cross-Platform Installer")

    # --- Detection Phase ---
    os_name = detect_os()
    gpu_info = detect_gpu()
    has_ffmpeg = detect_ffmpeg()

    print(f"  Platform:  {os_name} ({platform.machine()})")
    print(f"  Python:    {sys.version}")
    if gpu_info["has_nvidia"]:
        print(f"  GPU:       {gpu_info['gpu_name']}")
        if gpu_info["cuda_version"]:
            print(f"  CUDA:      {gpu_info['cuda_version']}")
    else:
        print(f"  GPU:       No NVIDIA GPU detected")
    print(f"  FFmpeg:    {'Available' if has_ffmpeg else 'NOT FOUND (required!)'}")

    if not has_ffmpeg:
        print_warn("FFmpeg is required but not found in PATH!")
        print_warn("Install it before running the application:")
        if os_name == "windows":
            print_warn("  winget install ffmpeg")
        elif os_name == "macos":
            print_warn("  brew install ffmpeg")
        else:
            print_warn("  sudo apt-get install ffmpeg")

    # --- Verify-only mode ---
    if args.verify:
        results = verify_installation(os_name)
        print_verification_results(results)
        return 0 if results["failed"] == 0 else 1

    # --- Installation Phase ---
    print_header("Installing Dependencies")

    # 1. Create venv
    if not args.no_venv:
        create_venv()

    # 2. Upgrade pip
    upgrade_pip()

    # 3. Determine GPU configuration
    force_cpu = args.cpu or os_name == "macos" or not gpu_info["has_nvidia"]
    cuda_version = args.cuda or gpu_info.get("cuda_version")

    if force_cpu:
        print_step("Using CPU-only PyTorch")
    elif cuda_version:
        print_step(f"Using CUDA {cuda_version}")
    else:
        print_step("No CUDA detected, using CPU-only PyTorch")
        force_cpu = True

    # 4. Install PyTorch
    install_pytorch(cuda_version if not force_cpu else None, force_cpu)

    # 5. Determine extras
    if args.minimal:
        extras = ["dev"]
    else:
        extras = ["tts", "ocr", "lipsync", "diarization", "translation-llm",
                   "audio-processing", "dev"]

    # 6. Install dependencies
    install_dependencies(extras, os_name)

    # 7. Platform-specific packages
    install_platform_specific(os_name)

    # --- Verification Phase ---
    results = verify_installation(os_name)
    print_verification_results(results)

    # --- Summary ---
    print_header("Installation Complete")
    if os_name == "windows":
        print("  To run:")
        print(f"    .\\venv\\Scripts\\activate")
        print(f"    python app.py")
    else:
        print("  To run:")
        print(f"    source venv/bin/activate")
        print(f"    python app.py")

    print(f"\n  Then open: http://127.0.0.1:7860")

    if results["failed"] > 0:
        print_warn(f"\n  {results['failed']} verification check(s) failed!")
        print_warn("  The application may still work, but some features might be unavailable.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
