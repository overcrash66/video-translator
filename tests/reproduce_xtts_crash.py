
import logging
import torch
import os
from src.synthesis.tts import TTSEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_xtts_crash():
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, cannot reproduce CUDA crash.")
        return

    logger.info("Initializing TTS Engine...")
    eng = TTSEngine()
    
    # Use a real reference wav if possible, or create a dummy one that is at least valid
    # XTTS needs a valid wav file for speaker cloning
    dummy_wav = "tests/data/test_audio.wav"
    if not os.path.exists(dummy_wav):
        # Create a dummy wav
        import soundfile as sf
        import numpy as np
        sr = 22050
        wav = np.random.uniform(-0.1, 0.1, sr * 3) # 3 seconds
        os.makedirs("tests/data", exist_ok=True)
        sf.write(dummy_wav, wav, sr)

    text = "Thank you."
    output_path = "tests/data/xtts_crash_test.wav"
    
    logger.info("Starting BAD AUDIO stress test...")
    
    # Test 1: Absolute Silence
    logger.info("Test 1: Absolute Silence Reference")
    silent_wav = "tests/data/silent.wav"
    import soundfile as sf
    import numpy as np
    sf.write(silent_wav, np.zeros(22050 * 3), 22050)
    
    try:
        eng.generate_audio(
            text="Testing silence.", 
            speaker_wav_path=silent_wav, 
            language="en", 
            output_path=output_path, 
            model="xtts"
        )
        logger.info("Silence test passed (Unexpected)")
    except Exception as e:
        logger.error(f"Silence test failed: {e}")

    # Test 2: Very Short Audio
    logger.info("Test 2: Short Audio Reference (0.1s)")
    short_wav = "tests/data/short.wav"
    sf.write(short_wav, np.random.uniform(-0.1, 0.1, 2205), 22050) # 0.1s
    
    try:
        eng.generate_audio(
            text="Testing short audio.", 
            speaker_wav_path=short_wav, 
            language="en", 
            output_path=output_path, 
            model="xtts"
        )
        logger.info("Short audio test passed")
    except Exception as e:
        logger.error(f"Short audio test failed: {e}")
        
    logger.info("Attempting to empty cache...")
    try:
        torch.cuda.empty_cache()
        logger.info("Cache emptied successfully.")
    except Exception as e:
        logger.critical(f"CRITICAL: Failed to empty cache (CUDA Poisoned?): {e}")

if __name__ == "__main__":
    test_xtts_crash()
