
import os
import sys
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
import logging

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.synthesis.tts import TTSEngine
from src.utils import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_audio(filename, duration_sec, silence=False):
    sr = 24000
    samples = int(sr * duration_sec)
    if silence:
        # Absolute silence
        wav = np.zeros(samples, dtype=np.float32)
    else:
        # Constant small value or noise
        wav = np.random.randn(samples).astype(np.float32) * 0.1
        
    path = config.TEMP_DIR / filename
    config.TEMP_DIR.mkdir(exist_ok=True, parents=True)
    sf.write(str(path), wav, sr)
    return str(path)

def test_xtts_crash():
    print("Testing XTTS Crash Conditions...")
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping crash test (since it's a device-side assert).")
        return

    engine = TTSEngine()
    
    # 1. Very short audio (0.1s)
    short_audio = create_dummy_audio("short_ref.wav", 0.1, silence=False)
    print(f"Created short audio: {short_audio}")
    
    # 2. Silent audio (2.0s)
    silent_audio = create_dummy_audio("silent_ref.wav", 2.0, silence=True)
    print(f"Created silent audio: {silent_audio}")
    
    # 3. Valid length but near silence (0.8s)
    near_short = create_dummy_audio("near_short.wav", 0.8, silence=False)
    
    # 4. Medium Short (1.8s) - Should fail new check
    medium_short = create_dummy_audio("medium_short.wav", 1.8, silence=False)
    
    # 5. Valid Audio (5.0s) to test Short TEXT
    valid_audio = create_dummy_audio("valid_ref.wav", 5.0, silence=False)
    
    # 6. Large Audio (60.0s) - Test potential memory/buffer issues with large refs
    large_audio = create_dummy_audio("large_ref.wav", 60.0, silence=False)
    
    text = "This is a test of the emergency broadcast system."
    short_text = "Hi."
    dot_text = "."
    dots_text = "I'm not sure...."
    
    cases = [
        ("Short Audio (0.1s)", short_audio, text),
        ("Silent Audio (2.0s)", silent_audio, text),
        ("Near Short (0.8s)", near_short, text),
        ("Medium Short (1.8s)", medium_short, text),
        ("Short Text ('Hi.')", valid_audio, short_text),
        ("Dot Text ('.')", valid_audio, dot_text),
        ("Trailing Dots ('I'm not sure....')", valid_audio, dots_text),
        ("Large Audio (60s)", large_audio, text)
    ]
    
    for name, ref_audio, txt in cases:
        print(f"\n--- Running Case: {name} ---")
        try:
            output = engine.generate_audio(
                text=txt,
                speaker_wav_path=ref_audio,
                language="en",
                model="xtts",
                output_path=config.TEMP_DIR / f"output_{name.split()[0]}.wav"
            )
            print(f"Result: {output}")
            
        except Exception as e:
            print(f"Caught Exception: {e}")
        except RuntimeError as e:
            print(f"Caught RuntimeError: {e}")
            
    print("\nTest Complete.")

if __name__ == "__main__":
    test_xtts_crash()
