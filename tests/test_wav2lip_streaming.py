
import unittest.mock
import pytest
import os
import torch
import cv2
import numpy as np
from pathlib import Path
from src.processing.wav2lip import Wav2LipSyncer
from src.utils import config

# Mock classes to avoid full model loading in tests if possible, 
# but for memory test we might want real heavy loading or simulated heavy data.
# Since we want to test the *streaming* logic, we can verify that
# frames are not exhausted.

@pytest.fixture
def mock_video(tmp_path):
    """Create a dummy video file."""
    video_path = tmp_path / "test_video.mp4"
    # Create 5 chunks worth of video (e.g. 5 seconds at 30fps = 150 frames)
    # We want enough to trigger multiple chunks if chunk size is small.
    # But internal chunk size is hardcoded to 300.
    # Let's mock the internal chunk size for testing?
    
    fps = 30
    duration = 2 # seconds
    frames = int(fps * duration)
    
    out = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (100, 100))
    for _ in range(frames):
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    return video_path

@pytest.fixture
def mock_audio(tmp_path):
    """Create a dummy audio file."""
    audio_path = tmp_path / "test_audio.wav"
    # Create valid wav file
    # 2 seconds tone
    fs = 16000
    seconds = 2
    t = np.linspace(0, seconds, seconds * fs, False)
    audio = np.sin(440 * 2 * np.pi * t) * 32767
    
    import scipy.io.wavfile
    scipy.io.wavfile.write(str(audio_path), fs, audio.astype(np.int16))
    return audio_path

def test_wav2lip_streaming_logic(mock_video, mock_audio, tmp_path):
    """
    Test that sync_lips runs without error using the new streaming logic.
    We mock the heavy models (Wav2Lip, FaceAlignment) to isolate logic.
    """
    output_path = tmp_path / "output.mp4"
    
    syncer = Wav2LipSyncer()
    
    # Mock Models
    # 1. Detector
    class MockDetector:
        def get_landmarks(self, img):
            # Return a fake face in center
            h, w = img.shape[:2]
            # 68 points
            return [np.random.rand(68, 2) * [w, h]]
            
    syncer.detector = MockDetector()
    
    # 2. Wav2Lip Model
    class MockModel(torch.nn.Module):
        def forward(self, mel, img):
            # Return matching shape for transpose(0, 2, 3, 4, 1)
            # Expected: (B, 3, T, H, W) -> (B, T, H, W, C)
            # mel: (B, 1, 80, 16)
            # img: (B, 6, 96, 96) or (B, 5, 6, 96, 96)
            B = mel.shape[0]
            return torch.zeros((B, 3, 5, 96, 96))
            
    syncer.model = MockModel()
    
    # Patch load_model/unload to do nothing
    syncer.load_model = lambda: None
    syncer.unload_model = lambda: None
    
    # Patch load_gfpgan
    syncer.load_gfpgan = lambda: None
    
    # Reduce batch size for test
    syncer.batch_size = 2
    
    # Override CHUNK_SIZE via monkeypatch if possible, or just rely on logic
    # We can't easily monkeypatch local variable in method.
    # But 2 seconds video (60 frames) is < 300, so it will be 1 chunk.
    
    # Mock cv2 completely to avoid SmartMock flakiness/shape variances
    with unittest.mock.patch("src.processing.wav2lip.cv2") as mock_cv2, \
         unittest.mock.patch("src.processing.wav2lip.audio_utils") as mock_audio_utils:
        
        # Audio Utils Mock
        # Return a large enough mel
        mock_audio_utils.wav2mel.return_value = np.random.rand(500, 80).astype(np.float32)

        # Constants
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.COLOR_BGR2RGB = 1
        mock_cv2.INTER_LANCZOS4 = 2
        mock_cv2.NORMAL_CLONE = 3
        
        # Functions
        # resize: check dsize and return correct shape array
        def resize_side_effect(src, dsize, *args, **kwargs):
            return np.zeros((dsize[1], dsize[0], 3), dtype=np.uint8)
        mock_cv2.resize.side_effect = resize_side_effect
        
        mock_cv2.cvtColor.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Seamless clone and friends
        mock_cv2.seamlessClone.side_effect = lambda src, dst, *args: dst
        mock_cv2.GaussianBlur.side_effect = lambda src, *args: src
        mock_cv2.ellipse.return_value = None
        
        # VideoCapture Mock
        def create_mock_cap(*args):
            m = unittest.mock.MagicMock()
            
            # Metadata
            def get_side_effect(prop):
                if prop == mock_cv2.CAP_PROP_FPS: return 30.0
                if prop == mock_cv2.CAP_PROP_FRAME_COUNT: return 60
                if prop == mock_cv2.CAP_PROP_FRAME_WIDTH: return 100
                if prop == mock_cv2.CAP_PROP_FRAME_HEIGHT: return 100
                return 0
            m.get.side_effect = get_side_effect
            
            # Frames
            frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(60)]
            read_results = [(True, f) for f in frames] + [(False, None)]
            m.read.side_effect = read_results
            return m
            
        mock_cv2.VideoCapture.side_effect = create_mock_cap
        
        # We also need to mock VideoWriter
        mock_writer = unittest.mock.MagicMock()
        mock_cv2.VideoWriter.return_value = mock_writer

        # Run
        syncer.sync_lips(str(mock_video), str(mock_audio), str(output_path), enhance_face=False)
        
    assert output_path.exists()

