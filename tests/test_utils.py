import unittest
import torch
import numpy as np
import tempfile
import os
import soundfile as sf
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

class TestAudioUtils(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.wav_path = os.path.join(self.temp_dir.name, "test.wav")
        # Note: We are using global mocks from conftest.py, so sf.write inside setUp won't actually write to disk.
        # We need to rely on mocking return values in tests.

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_audio_basic(self):
        # Mock sf.read and torch.from_numpy
        with unittest.mock.patch("src.utils.audio_utils.sf.read") as mock_read, \
             unittest.mock.patch("src.utils.audio_utils.torch.from_numpy") as mock_from:
             
             # Setup Read
             mock_data = np.random.randn(24000, 2).astype(np.float32)
             mock_read.return_value = (mock_data, 24000)
             
             # Setup Tensor conversion
             # Create a mock that behaves like a tensor with correct shape logic
             def from_numpy_side_effect(arr):
                 m = unittest.mock.MagicMock()
                 m.ndim = arr.ndim
                 m.shape = arr.shape
                 # transpose logic
                 def transpose(d0, d1):
                     t = unittest.mock.MagicMock()
                     t.ndim = m.ndim
                     # Swap shape dim
                     s = list(m.shape)
                     s[d0], s[d1] = s[d1], s[d0]
                     t.shape = tuple(s)
                     # chain ...
                     t.mean.return_value = unittest.mock.MagicMock(shape=(1, s[1]))
                     return t
                 m.transpose.side_effect = transpose
                 return m
                 
             mock_from.side_effect = from_numpy_side_effect
             
             wav, sr = audio_utils.load_audio(self.wav_path)
             
             self.assertEqual(wav.ndim, 2)
             self.assertEqual(wav.shape[0], 2) # Stereo
             self.assertEqual(wav.shape[1], 24000)
             self.assertEqual(sr, 24000)

    def test_load_audio_mono(self):
        with unittest.mock.patch("src.utils.audio_utils.sf.read") as mock_read, \
             unittest.mock.patch("src.utils.audio_utils.torch.from_numpy") as mock_from:
             
             mock_data = np.random.randn(24000, 2).astype(np.float32)
             mock_read.return_value = (mock_data, 24000)
             
             # Copied side effect logic (or refactor helper)
             def from_numpy_side_effect(arr):
                 m = unittest.mock.MagicMock()
                 m.ndim = arr.ndim
                 m.shape = arr.shape
                 def transpose(d0, d1):
                     t = unittest.mock.MagicMock()
                     t.ndim = m.ndim
                     s = list(m.shape)
                     s[d0], s[d1] = s[d1], s[d0]
                     t.shape = tuple(s)
                     # convert to mono
                     t.mean.return_value = unittest.mock.MagicMock(shape=(1, s[1]))
                     return t
                 m.transpose.side_effect = transpose
                 return m
             mock_from.side_effect = from_numpy_side_effect
             
             wav, sr = audio_utils.load_audio(self.wav_path, mono=True)
             self.assertEqual(wav.shape[0], 1)
             self.assertEqual(wav.shape[1], 24000)

    def test_load_audio_resample(self):
        with unittest.mock.patch("src.utils.audio_utils.sf.read") as mock_read, \
             unittest.mock.patch("src.utils.audio_utils.torch.from_numpy") as mock_from, \
             unittest.mock.patch("src.utils.audio_utils.torchaudio.transforms.Resample") as MockResample:
             
             mock_data = np.random.randn(24000, 2).astype(np.float32)
             mock_read.return_value = (mock_data, 24000)
             
             def from_numpy_side_effect(arr):
                 m = unittest.mock.MagicMock()
                 m.ndim = arr.ndim
                 m.shape = arr.shape
                 def transpose(d0, d1):
                     t = unittest.mock.MagicMock()
                     t.ndim = m.ndim
                     s = list(m.shape)
                     s[d0], s[d1] = s[d1], s[d0]
                     t.shape = tuple(s)
                     return t
                 m.transpose.side_effect = transpose
                 return m
             mock_from.side_effect = from_numpy_side_effect
             
             # Resample Logic
             mock_resampler = MockResample.return_value
             def resample_side_effect(tensor):
                  # input (2, 24000). target 16000
                  new_m = unittest.mock.MagicMock()
                  new_m.shape = (2, 16000)
                  return new_m
             mock_resampler.side_effect = resample_side_effect
             
             wav, sr = audio_utils.load_audio(self.wav_path, target_sr=16000)
             self.assertEqual(sr, 16000)
             self.assertEqual(wav.shape[1], 16000)

    def test_save_audio(self):
        # Create tensor [Channels, Time]
        tensor = torch.randn(2, 24000)
        out_path = os.path.join(self.temp_dir.name, "out.wav")
        
        with unittest.mock.patch("src.utils.audio_utils.sf.write") as mock_write:
             audio_utils.save_audio(out_path, tensor, 24000)
             mock_write.assert_called_once()
             args, _ = mock_write.call_args
             self.assertEqual(args[0], out_path)
             # args[1] is data (numpy). args[2] is samplerate
             self.assertEqual(args[2], 24000)
