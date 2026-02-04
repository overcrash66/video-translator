import pytest
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.utils.chunker import VideoChunker
from src.utils import config as app_config

# Mock config.TEMP_DIR for tests to avoid polluting real temp
@pytest.fixture(autouse=True)
def mock_temp_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(app_config, "TEMP_DIR", tmp_path)
    return tmp_path

@pytest.fixture
def temp_video(tmp_path):
    """Generates a dummy video path"""
    video_path = tmp_path / "test_video.mp4"
    video_path.touch()
    return video_path

@pytest.fixture
def temp_audio(tmp_path):
    """Generates a dummy audio path"""
    audio_path = tmp_path / "test_audio.wav"
    audio_path.touch()
    return audio_path

def test_should_chunk(temp_video):
    with patch("src.utils.chunker.ffmpeg.probe") as mock_probe:
        # Case 1: Max duration 2s, Video 10s (Threshold 3s) -> Should Chunk
        mock_probe.return_value = {'format': {'duration': '10.0'}}
        chunker = VideoChunker(max_duration_sec=2)
        assert chunker.should_chunk(temp_video) == True
        
        # Case 2: Max duration 10s (Threshold 15s) -> Should NOT Chunk
        chunker = VideoChunker(max_duration_sec=10)
        assert chunker.should_chunk(temp_video) == False

def test_split_video(temp_video, tmp_path):
    with patch("src.utils.chunker.ffmpeg") as mock_ffmpeg:
        # Mock probe for should_chunk if called, but split_video doesn't call it? 
        # Actually split_video runs the command.
        
        # We need to simulate the file creation side effect that VideoChunker expects
        chunker = VideoChunker(max_duration_sec=3)
        
        def side_effect_run(*args, **kwargs):
            # Create fake chunks
            (tmp_path / f"{temp_video.stem}_chunk_000.mp4").touch()
            (tmp_path / f"{temp_video.stem}_chunk_001.mp4").touch()
            (tmp_path / f"{temp_video.stem}_chunk_002.mp4").touch()
            (tmp_path / f"{temp_video.stem}_chunk_003.mp4").touch()
            return MagicMock() # return proc
            
        # The chain is ffmpeg.input().output().run()
        # mock_ffmpeg.input is a Mock
        # .output is a Mock
        # .run is a Mock
        mock_run = mock_ffmpeg.input.return_value.output.return_value.run
        mock_run.side_effect = side_effect_run

        chunks = chunker.split_video(temp_video)
        
        assert len(chunks) == 4
        for chunk in chunks:
            assert chunk.exists()
            assert chunk.suffix == ".mp4"

def test_split_audio(temp_audio, temp_video, tmp_path):
    with patch("src.utils.chunker.ffmpeg") as mock_ffmpeg:
        chunker = VideoChunker(max_duration_sec=3)
        
        # 1. Setup video split mock result
        video_chunks = [
            tmp_path / "v_0.mp4", 
            tmp_path / "v_1.mp4",
            tmp_path / "v_2.mp4"
        ]
        
        # 2. Mock probe to return duration for audio splitting loop
        mock_ffmpeg.probe.return_value = {'format': {'duration': '3.0'}}
        
        # 3. Mock audio split run side effect
        def side_effect_run(*args, **kwargs):
            # In loop, create one chunk at a time.
            # But here we just need to ensure the run call doesn't crash.
            # We must verify files are "created".
            # The code calculates out_name. We can just create them all or trust the logic?
            # Creating them is safer for downstream assertions.
            pass

        mock_run = mock_ffmpeg.input.return_value.output.return_value.overwrite_output.return_value.run
        mock_run.side_effect = side_effect_run

        # Pre-create "audio chunks" because mocking the loop side effect is hard 
        # since it uses dynamic filenames based on loop index.
        # Alternatively, we just check call count?
        # But split_audio appends to list only if run succeeds.
        # And it returns the list.
        # Oh, split_audio does audio_chunks.append(out_name) inside try/except.
        # So it assumes file is created by ffmpeg.
        
        # Let's mock the file existence check? No it doesn't check existence.
        # It just returns the path.
        
        audio_chunks = chunker.split_audio(temp_audio, video_chunks)
        
        assert len(audio_chunks) == len(video_chunks)
        # Verify paths format
        assert "chunk_000" in str(audio_chunks[0])

def test_merge_videos(temp_video, tmp_path):
    with patch("src.utils.chunker.ffmpeg") as mock_ffmpeg:
        chunker = VideoChunker(max_duration_sec=3)
        chunks = [tmp_path / "c1.mp4", tmp_path / "c2.mp4"]
        output = tmp_path / "merged_output.mp4"
        
        def side_effect_run(*args, **kwargs):
            output.touch()
            return MagicMock()
            
        mock_run = mock_ffmpeg.input.return_value.output.return_value.overwrite_output.return_value.run
        mock_run.side_effect = side_effect_run
        
        result = chunker.merge_videos(chunks, output)
        
        assert result.exists()
        assert result == output

def test_merge_audio(temp_audio, temp_video, tmp_path):
    with patch("src.utils.chunker.ffmpeg") as mock_ffmpeg:
        chunker = VideoChunker(max_duration_sec=3)
        chunks = [tmp_path / "c1.wav", tmp_path / "c2.wav"]
        output = tmp_path / "merged_audio.wav"
        
        def side_effect_run(*args, **kwargs):
            output.touch()
            return MagicMock()
            
        mock_run = mock_ffmpeg.input.return_value.output.return_value.overwrite_output.return_value.run
        mock_run.side_effect = side_effect_run
        
        result = chunker.merge_audio(chunks, output)
        
        assert result.exists()
