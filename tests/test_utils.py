import unittest
import torch
import numpy as np
import tempfile
import os
import soundfile as sf
import pytest
from src.utils import languages
from src.utils import audio_utils

class TestLanguages(unittest.TestCase):
    def test_get_language_code(self):
        self.assertEqual(languages.get_language_code("English"), "en")
        self.assertEqual(languages.get_language_code("Auto Detect"), "auto")
        self.assertEqual(languages.get_language_code("Chinese (Simplified)"), "zh")
        self.assertEqual(languages.get_language_code("NonExistent"), "en") # Fallback

    def test_voice_map_structure(self):
        # Ensure map has critical keys
        self.assertIn("en", languages.EDGE_TTS_VOICE_MAP)
        self.assertIn("Female", languages.EDGE_TTS_VOICE_MAP["en"])
        self.assertTrue(len(languages.EDGE_TTS_VOICE_MAP["en"]["Male"]) > 0)

@pytest.mark.requires_real_audio
class TestAudioUtils(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.wav_path = os.path.join(self.temp_dir.name, "test.wav")
        
        # Create a dummy wav file (1 sec, 24k, stereo)
        # Soundfile expects [Time, Channels]
        sr = 24000
        data = np.random.uniform(-1, 1, size=(24000, 2)).astype(np.float32)
        sf.write(self.wav_path, data, sr)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_audio_basic(self):
        # Load
        wav, sr = audio_utils.load_audio(self.wav_path)
        
        # Check shape [Channels, Time]
        self.assertEqual(wav.ndim, 2)
        self.assertEqual(wav.shape[0], 2) # Stereo
        self.assertEqual(wav.shape[1], 24000)
        self.assertEqual(sr, 24000)
        self.assertIsInstance(wav, torch.Tensor)

    def test_load_audio_mono(self):
        wav, sr = audio_utils.load_audio(self.wav_path, mono=True)
        self.assertEqual(wav.shape[0], 1)
        self.assertEqual(wav.shape[1], 24000)

    def test_load_audio_resample(self):
        wav, sr = audio_utils.load_audio(self.wav_path, target_sr=16000)
        self.assertEqual(sr, 16000)
        self.assertEqual(wav.shape[1], 16000)

    def test_save_audio(self):
        # Create tensor [Channels, Time]
        tensor = torch.randn(2, 24000)
        out_path = os.path.join(self.temp_dir.name, "out.wav")
        
        audio_utils.save_audio(out_path, tensor, 24000)
        
        self.assertTrue(os.path.exists(out_path))
        info = sf.info(out_path)
        self.assertEqual(info.samplerate, 24000)
        self.assertEqual(info.channels, 2)
