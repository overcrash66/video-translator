import torch
import sys
import os
from pathlib import Path

def check_env():
    print("=== Environment Verification ===")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        try:
            print(f"Device Count: {torch.cuda.device_count()}")
            print(f"Device Name: {torch.cuda.get_device_name(0)}")
            x = torch.tensor([1.0]).cuda()
            print("CUDA Tensor Test: PASS")
        except Exception as e:
            print(f"CUDA Tensor Test: FAIL ({e})")
    else:
        print("WARNING: CUDA is NOT available. Inferences will be slow.")

    print("\n=== Dependency Verification ===")
    try:
        import face_alignment
        print(f"face_alignment: FOUND (v{getattr(face_alignment, '__version__', 'unknown')})")
    except ImportError:
        print("face_alignment: MISSING")

    try:
        import cv2
        print(f"OpenCV: FOUND (v{cv2.__version__})")
    except ImportError:
        print("OpenCV: MISSING")

    print("\n=== Model Verification ===")
    wav2lip_path = Path("models/wav2lip/wav2lip_gan.pth")
    if wav2lip_path.exists():
        print(f"Wav2Lip-GAN Model: FOUND ({wav2lip_path})")
    else:
        print(f"Wav2Lip-GAN Model: MISSING ({wav2lip_path})")

    try:
        from src.processing.lipsync import LipSyncer
        syncer = LipSyncer()
        print("LipSyncer Import: PASS")
        # syncer.load_model() # Don't load full model here to save time/memory, just checking import structure
    except Exception as e:
        print(f"LipSyncer Import: FAIL ({e})")
        
    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    try:
        check_env()
    except Exception as e:
        print(f"Verification Script Failed: {e}")
