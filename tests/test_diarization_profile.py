
import os
import sys
import shutil
from pathlib import Path
import numpy as np
import soundfile as sf

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio.diarization import Diarizer
from src.utils import config

def test_profile_extraction():
    print("Testing Speaker Profile Extraction Logic...")
    
    # 1. Setup Test Audio (5 seconds long)
    sr = 16000
    audio_path = config.TEMP_DIR / "test_diarization_audio.wav"
    config.TEMP_DIR.mkdir(exist_ok=True, parents=True)
    
    # Create valid speech-like audio (random noise but correct format)
    wav = np.random.randn(sr * 5).astype(np.float32)
    sf.write(str(audio_path), wav, sr)
    
    diarizer = Diarizer()
    output_dir = config.TEMP_DIR / "test_profiles"
    if output_dir.exists(): shutil.rmtree(output_dir)
    
    # Case A: Only short segments (should be skipped)
    # 4 segments of 0.4s = 1.6s total (would pass old valid > 1.0s logic if stitched, but crash XTTS)
    segments_short = [
        {'speaker': 'SPEAKER_00', 'start': 0.0, 'end': 0.4},
        {'speaker': 'SPEAKER_00', 'start': 1.0, 'end': 1.4},
        {'speaker': 'SPEAKER_00', 'start': 2.0, 'end': 2.4},
        {'speaker': 'SPEAKER_00', 'start': 3.0, 'end': 3.4},
    ]
    
    print("\n--- Case A: Only Short Segments (<1.0s) ---")
    profiles = diarizer.extract_speaker_profiles(audio_path, segments_short, output_dir)
    print(f"Profiles created: {profiles}")
    
    if 'SPEAKER_00' not in profiles:
        print("PASS: SPEAKER_00 profile correctly skipped (all segments < 1.0s)")
    else:
        print("FAIL: SPEAKER_00 profile created despite short segments!")
        
    # Case B: Mixed segments (short + one long but total < 3.0s) -> Should be skipped
    # One 1.5s segment + some shorts. Total valid = 1.5s < 3.0s threshold
    segments_mixed_short = [
         {'speaker': 'SPEAKER_01', 'start': 0.0, 'end': 0.4}, # Skip
         {'speaker': 'SPEAKER_01', 'start': 1.0, 'end': 2.5}, # Keep (1.5s)
    ]
    
    print("\n--- Case B: Valid Segment but Total < 3.0s ---")
    profiles = diarizer.extract_speaker_profiles(audio_path, segments_mixed_short, output_dir)
    print(f"Profiles created: {profiles}")
    
    if 'SPEAKER_01' not in profiles:
        print("PASS: SPEAKER_01 profile correctly skipped (total valid duration 1.5s < 3.0s)")
    else:
        print("FAIL: SPEAKER_01 profile created despite short total duration!")

    # Case C: Good segments (total > 3.0s) -> Should succeed
    segments_good = [
        {'speaker': 'SPEAKER_02', 'start': 0.0, 'end': 2.0}, # 2.0s
        {'speaker': 'SPEAKER_02', 'start': 2.5, 'end': 4.5}, # 2.0s
    ]
    
    print("\n--- Case C: Good Segments (> 3.0s total) ---")
    profiles = diarizer.extract_speaker_profiles(audio_path, segments_good, output_dir)
    print(f"Profiles created: {profiles}")
    
    if 'SPEAKER_02' in profiles:
        print(f"PASS: SPEAKER_02 profile created: {profiles['SPEAKER_02']}")
        # Verify file existence
        if Path(profiles['SPEAKER_02']).exists():
             print("PASS: Profile file exists.")
    else:
        print("FAIL: SPEAKER_02 profile NOT created!")

if __name__ == "__main__":
    test_profile_extraction()
