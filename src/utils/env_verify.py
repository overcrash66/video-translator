
import sys

def check_import(package_name):
    try:
        __import__(package_name)
        print(f"[OK] {package_name} imported successfully.")
        return True
    except ImportError as e:
        print(f"[ERROR] Failed to import {package_name}: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Error importing {package_name}: {e}")
        return False

packages = [
    "numpy",
    "transformers",
    "pyannote.audio",
    "soundfile",
    "torch",
    "torchaudio",
    "cv2",
    "paddle",
    "paddleocr",
    "f5_tts",
    "nemo"
]

print(f"Python version: {sys.version}")

failed = []
for pkg in packages:
    if not check_import(pkg):
        failed.append(pkg)

if failed:
    print(f"\nVerification FAILED for: {', '.join(failed)}")
    sys.exit(1)
else:
    print("\nVerification PASSED: All key packages imported successfully.")
