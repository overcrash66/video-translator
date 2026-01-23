"""Tests for pipeline_stages.py."""
import pytest
from unittest.mock import MagicMock
from pathlib import Path
from src.core.pipeline_stages import PipelineContext, ExtractionStage
from src.processing.video import VideoProcessor

def test_pipeline_context_init():
    ctx = PipelineContext(
        video_path=Path("video.mp4"),
        source_lang="en",
        target_lang="fr",
        audio_model_name="demucs",
        tts_model_name="edge",
        translation_model_name="google",
        context_model_name="gpt3",
        transcription_model_name="whisper",
        optimize_translation=False,
        enable_diarization=False,
        enable_time_stretch=False,
        enable_vad=True,
        enable_lipsync=False,
        enable_visual_translation=False,
        enable_audio_enhancement=False
    )
    assert ctx.source_lang == "en"
    assert ctx.diarization_segments == []

def test_extraction_stage():
    processor = MagicMock(spec=VideoProcessor)
    processor.extract_audio.return_value = "video_full.wav"
    
    stage = ExtractionStage(processor)
    ctx = PipelineContext(
        video_path=Path("video.mp4"),
        source_lang="en",
        target_lang="fr",
        audio_model_name="demucs",
        tts_model_name="edge",
        translation_model_name="google",
        context_model_name="gpt3",
        transcription_model_name="whisper",
        optimize_translation=False,
        enable_diarization=False,
        enable_time_stretch=False,
        enable_vad=True,
        enable_lipsync=False,
        enable_visual_translation=False,
        enable_audio_enhancement=False
    )
    
    gen = stage.execute(ctx)
    result = list(gen)
    
    assert len(result) > 0
    assert ctx.extracted_audio_path == Path("video_full.wav")
    processor.extract_audio.assert_called_once()
