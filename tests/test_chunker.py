import pytest
import subprocess
import shutil
from pathlib import Path
from src.utils.chunker import VideoChunker
from src.utils import config as app_config

# Mock config.TEMP_DIR for tests to avoid polluting real temp
@pytest.fixture(autouse=True)
def mock_temp_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(app_config, "TEMP_DIR", tmp_path)
    return tmp_path

@pytest.fixture
def temp_video(tmp_path):
    """Generates a 10s dummy video with audio"""
    video_path = tmp_path / "test_video.mp4"
    try:
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=10:size=640x360:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=1000:duration=10",
            "-c:v", "libx264", "-g", "30", "-c:a", "aac", str(video_path)
        ], check=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        pytest.skip("FFmpeg not available or failed to create test video")
    return video_path

@pytest.fixture
def temp_audio(tmp_path):
    """Generates a 10s dummy audio"""
    audio_path = tmp_path / "test_audio.wav"
    try:
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=1000:duration=10",
            "-c:a", "pcm_s16le", str(audio_path)
        ], check=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        pytest.skip("FFmpeg not available or failed to create test audio")
    return audio_path

def test_should_chunk(temp_video):
    # Video is 10s. Threshold 1.5x
    
    # Case 1: Max duration 2s (Threshold 3s) -> Should Chunk
    chunker = VideoChunker(max_duration_sec=2)
    assert chunker.should_chunk(temp_video) == True
    
    # Case 2: Max duration 10s (Threshold 15s) -> Should NOT Chunk
    chunker = VideoChunker(max_duration_sec=10)
    assert chunker.should_chunk(temp_video) == False

def test_split_video(temp_video):
    # Split 10s video into 3s chunks
    chunker = VideoChunker(max_duration_sec=3)
    chunks = chunker.split_video(temp_video)
    
    # Should produce approx 4 chunks (3, 3, 3, 1) or similar depending on keyframes
    # Relaxed assertion: 3 to 5 chunks
    assert 3 <= len(chunks) <= 5
    for chunk in chunks:
        assert chunk.exists()
        assert chunk.suffix == ".mp4"

def test_split_audio(temp_audio, temp_video):
    # We need video chunks first to define split points
    chunker = VideoChunker(max_duration_sec=3)
    video_chunks = chunker.split_video(temp_video)
    
    audio_chunks = chunker.split_audio(temp_audio, video_chunks)
    
    assert len(audio_chunks) == len(video_chunks)
    for chunk in audio_chunks:
        assert chunk.exists()
        assert chunk.suffix == ".wav"

def test_merge_videos(temp_video, tmp_path):
    chunker = VideoChunker(max_duration_sec=3)
    chunks = chunker.split_video(temp_video)
    
    output = tmp_path / "merged_output.mp4"
    result = chunker.merge_videos(chunks, output)
    
    assert result.exists()
    assert result.stat().st_size > 0
    
    # Verify duration approx 10s
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(result)],
        capture_output=True, text=True
    )
    duration = float(probe.stdout.strip())
    assert 9.5 < duration < 10.5

def test_merge_audio(temp_audio, temp_video, tmp_path):
    chunker = VideoChunker(max_duration_sec=3)
    # Split audio based on video chunks logic
    video_chunks = chunker.split_video(temp_video)
    audio_chunks = chunker.split_audio(temp_audio, video_chunks)
    
    output = tmp_path / "merged_audio.wav"
    result = chunker.merge_audio(audio_chunks, output)
    
    assert result.exists()
    assert result.stat().st_size > 0
