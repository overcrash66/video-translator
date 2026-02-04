
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
    
    # Run
    # try:
    syncer.sync_lips(str(mock_video), str(mock_audio), str(output_path), enhance_face=False)
    # except Exception as e:
    #     pytest.fail(f"Streaming sync failed: {e}")
        
    assert output_path.exists()

