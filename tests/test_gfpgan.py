
import unittest
import os
from src.processing.wav2lip import Wav2LipSyncer

class TestGFPGAN(unittest.TestCase):
    def test_load_gfpgan(self):
        syncer = Wav2LipSyncer()
        syncer.load_gfpgan()
        
        # We can't guarantee download in CI/Test env without network, 
        # but if it fails it sets restorer to None.
        # We just want to ensure it doesn't crash on import/call.
        if syncer.restorer is not None:
             print("GFPGAN loaded successfully.")
        else:
             print("GFPGAN failed to load (check logs).")

if __name__ == "__main__":
    unittest.main()
