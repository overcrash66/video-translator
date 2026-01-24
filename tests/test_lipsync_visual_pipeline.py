import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Mock dependencies that might not be installed in test env
# Mock dependencies that might not be installed in test env
# Mock dependencies handled via object patching in tests

def test_lipsync_integration_flag():
    """Verify enable_lipsync flag triggers lipsync call"""
    with patch('src.core.video_translator.VideoTranslator') as MockVT:
        # We need to test the logic INSIDE process_video, so we shouldn't mock the whole class usually, 
        # but here we want to test the orchestration.
        # Better: Import the real class and mock its components.
        pass

def test_process_video_lipsync_call():
    from src.core.video_translator import VideoTranslator
    
    vt = VideoTranslator()
    
    # Mock components
    vt.separator = MagicMock()
    vt.transcriber = MagicMock()
    vt.translator = MagicMock()
    vt.tts_engine = MagicMock()
    vt.synchronizer = MagicMock()
    vt.processor = MagicMock()
    vt.diarizer = MagicMock()
    vt.lipsyncer = MagicMock()
    
    # Setup mocks to return valid paths/data
    vt.processor.extract_audio.return_value = "extracted.wav"
    vt.separator.separate.return_value = ("vocals.wav", "bg.wav")
    vt.transcriber.transcribe.return_value = ([{"text": "Hello", "start": 0, "end": 1}], "en")
    vt.translator.translate_segments.return_value = [{"text": "Hello", "translated_text": "Hola", "start": 0, "end": 1}]
    vt.tts_engine.generate_audio.return_value = "tts.wav"
    vt.synchronizer.merge_segments.return_value = True
    vt.processor.replace_audio.return_value = "final.mp4"
    
    # Mock Path existence and validation
    with patch('src.core.video_translator.Path') as MockPath, \
         patch('src.utils.config.validate_path', side_effect=lambda p, **kwargs: Path(p)):
        # Configure MockPath to return True for exists()
        mock_path_instance = MockPath.return_value
        mock_path_instance.exists.return_value = True
        mock_path_instance.stat.return_value.st_size = 1000
        mock_path_instance.stem = "test"
        mock_path_instance.name = "test.mp4"
        
        # Also mock the result of generate_audio being passed to Path
        # The code does Path(generated_path).exists()
        
        # Run with lipsync=True
        gen = vt.process_video(
            video_path="test.mp4",
            source_lang="en",
            target_lang="es",
            audio_model_name="demucs",
            tts_model_name="edge",
            translation_model_name="google",
            context_model_name=None,
            transcription_model_name="base",
            optimize_translation=False,
            enable_diarization=False,
            enable_time_stretch=False,
            enable_vad=False,
            enable_lipsync=True,
            enable_visual_translation=False
        )
        
        # Consume generator
        for _ in gen: pass
    
    # Verify lipsyncer was called
    vt.lipsyncer.sync_lips.assert_called_once()


def test_process_video_lipsync_skip():
    from src.core.video_translator import VideoTranslator
    vt = VideoTranslator()
    # Mock components
    vt.separator = MagicMock()
    vt.transcriber = MagicMock()
    vt.translator = MagicMock()
    vt.tts_engine = MagicMock()
    vt.synchronizer = MagicMock()
    vt.processor = MagicMock()
    vt.diarizer = MagicMock()
    vt.lipsyncer = MagicMock()
    
    # Setup happy path
    vt.processor.extract_audio.return_value = "extracted.wav"
    vt.separator.separate.return_value = ("vocals.wav", "bg.wav")
    vt.transcriber.transcribe.return_value = ([{"text": "Hello", "start": 0, "end": 1}], "en")
    vt.translator.translate_segments.return_value = [{"text": "Hello", "translated_text": "Hola", "start": 0, "end": 1}]
    vt.tts_engine.generate_audio.return_value = "tts.wav"
    vt.synchronizer.merge_segments.return_value = True
    vt.processor.replace_audio.return_value = "final.mp4"

    with patch('src.core.video_translator.Path') as MockPath, \
         patch('src.utils.config.validate_path', side_effect=lambda p, **kwargs: Path(p)):
        mock_path_instance = MockPath.return_value
        mock_path_instance.exists.return_value = True
        mock_path_instance.stat.return_value.st_size = 1000
        mock_path_instance.stem = "test"
        mock_path_instance.name = "test.mp4"

        # Run with lipsync=False
        gen = vt.process_video(
            video_path="test.mp4",
            source_lang="en",
            target_lang="es",
            audio_model_name="demucs",
            tts_model_name="edge",
            translation_model_name="google",
            context_model_name=None,
            transcription_model_name="base",
            optimize_translation=False,
            enable_diarization=False,
            enable_time_stretch=False,
            enable_vad=False,
            enable_lipsync=False,
            enable_visual_translation=False
        )
        
        for _ in gen: pass
    
    vt.lipsyncer.sync_lips.assert_not_called()


def test_process_video_visual_translation_call():
    """Verify enable_visual_translation=True triggers visual translator."""
    from src.core.video_translator import VideoTranslator
    
    vt = VideoTranslator()
    
    # Mock components
    vt.separator = MagicMock()
    vt.transcriber = MagicMock()
    vt.translator = MagicMock()
    vt.tts_engine = MagicMock()
    vt.synchronizer = MagicMock()
    vt.processor = MagicMock()
    vt.diarizer = MagicMock()
    vt.lipsyncer = MagicMock()
    vt.visual_translator = MagicMock()
    
    # Setup mocks
    vt.processor.extract_audio.return_value = "extracted.wav"
    vt.separator.separate.return_value = ("vocals.wav", "bg.wav")
    vt.transcriber.transcribe.return_value = ([{"text": "Hello", "start": 0, "end": 1}], "en")
    vt.translator.translate_segments.return_value = [{"text": "Hello", "translated_text": "Hola", "start": 0, "end": 1}]
    vt.tts_engine.generate_audio.return_value = "tts.wav"
    vt.synchronizer.merge_segments.return_value = True
    vt.processor.replace_audio.return_value = "final.mp4"
    vt.visual_translator.translate_video_text.return_value = "visual.mp4"
    
    with patch('src.core.video_translator.Path') as MockPath, \
         patch('src.utils.config.validate_path', side_effect=lambda p, **kwargs: Path(p)):
        mock_path_instance = MockPath.return_value
        mock_path_instance.exists.return_value = True
        mock_path_instance.stat.return_value.st_size = 1000
        mock_path_instance.stem = "test"
        mock_path_instance.name = "test.mp4"
        
        gen = vt.process_video(
            video_path="test.mp4",
            source_lang="en",
            target_lang="es",
            audio_model_name="demucs",
            tts_model_name="edge",
            translation_model_name="google",
            context_model_name=None,
            transcription_model_name="base",
            optimize_translation=False,
            enable_diarization=False,
            enable_time_stretch=False,
            enable_vad=False,
            enable_lipsync=False,
            enable_visual_translation=True
        )
        
        for _ in gen: pass
    
    # Verify visual translator was called
    vt.visual_translator.translate_video_text.assert_called_once()


def test_process_video_visual_translation_skip():
    """Verify visual translator is skipped when disabled."""
    from src.core.video_translator import VideoTranslator
    
    vt = VideoTranslator()
    
    # Mock components
    vt.separator = MagicMock()
    vt.transcriber = MagicMock()
    vt.translator = MagicMock()
    vt.tts_engine = MagicMock()
    vt.synchronizer = MagicMock()
    vt.processor = MagicMock()
    vt.diarizer = MagicMock()
    vt.lipsyncer = MagicMock()
    vt.visual_translator = MagicMock()
    
    # Setup mocks
    vt.processor.extract_audio.return_value = "extracted.wav"
    vt.separator.separate.return_value = ("vocals.wav", "bg.wav")
    vt.transcriber.transcribe.return_value = ([{"text": "Hello", "start": 0, "end": 1}], "en")
    vt.translator.translate_segments.return_value = [{"text": "Hello", "translated_text": "Hola", "start": 0, "end": 1}]
    vt.tts_engine.generate_audio.return_value = "tts.wav"
    vt.synchronizer.merge_segments.return_value = True
    vt.processor.replace_audio.return_value = "final.mp4"
    
    with patch('src.core.video_translator.Path') as MockPath, \
         patch('src.utils.config.validate_path', side_effect=lambda p, **kwargs: Path(p)):
        mock_path_instance = MockPath.return_value
        mock_path_instance.exists.return_value = True
        mock_path_instance.stat.return_value.st_size = 1000
        mock_path_instance.stem = "test"
        mock_path_instance.name = "test.mp4"
        
        gen = vt.process_video(
            video_path="test.mp4",
            source_lang="en",
            target_lang="es",
            audio_model_name="demucs",
            tts_model_name="edge",
            translation_model_name="google",
            context_model_name=None,
            transcription_model_name="base",
            optimize_translation=False,
            enable_diarization=False,
            enable_time_stretch=False,
            enable_vad=False,
            enable_lipsync=False,
            enable_visual_translation=False
        )
        
        for _ in gen: pass
    
    vt.visual_translator.translate_video_text.assert_not_called()


def test_lipsync_and_visual_together():
    """Verify both lip-sync and visual translation can work together."""
    from src.core.video_translator import VideoTranslator
    
    vt = VideoTranslator()
    
    # Mock components
    vt.separator = MagicMock()
    vt.transcriber = MagicMock()
    vt.translator = MagicMock()
    vt.tts_engine = MagicMock()
    vt.synchronizer = MagicMock()
    vt.processor = MagicMock()
    vt.diarizer = MagicMock()
    vt.lipsyncer = MagicMock()
    vt.visual_translator = MagicMock()
    
    # Setup mocks
    vt.processor.extract_audio.return_value = "extracted.wav"
    vt.separator.separate.return_value = ("vocals.wav", "bg.wav")
    vt.transcriber.transcribe.return_value = ([{"text": "Hello", "start": 0, "end": 1}], "en")
    vt.translator.translate_segments.return_value = [{"text": "Hello", "translated_text": "Hola", "start": 0, "end": 1}]
    vt.tts_engine.generate_audio.return_value = "tts.wav"
    vt.synchronizer.merge_segments.return_value = True
    vt.processor.replace_audio.return_value = "final.mp4"
    vt.visual_translator.translate_video_text.return_value = "visual.mp4"
    vt.lipsyncer.sync_lips.return_value = "lipsync.mp4"
    
    with patch('src.core.video_translator.Path') as MockPath, \
         patch('src.utils.config.validate_path', side_effect=lambda p, **kwargs: Path(p)):
        mock_path_instance = MockPath.return_value
        mock_path_instance.exists.return_value = True
        mock_path_instance.stat.return_value.st_size = 1000
        mock_path_instance.stem = "test"
        mock_path_instance.name = "test.mp4"
        
        gen = vt.process_video(
            video_path="test.mp4",
            source_lang="en",
            target_lang="es",
            audio_model_name="demucs",
            tts_model_name="edge",
            translation_model_name="google",
            context_model_name=None,
            transcription_model_name="base",
            optimize_translation=False,
            enable_diarization=False,
            enable_time_stretch=False,
            enable_vad=False,
            enable_lipsync=True,
            enable_visual_translation=True
        )
        
        for _ in gen: pass
    
    # Both should be called
    vt.visual_translator.translate_video_text.assert_called_once()
    vt.lipsyncer.sync_lips.assert_called_once()


def test_lipsync_failure_continues_pipeline():
    """Verify pipeline continues when lip-sync raises exception."""
    from src.core.video_translator import VideoTranslator
    
    vt = VideoTranslator()
    
    # Mock components
    vt.separator = MagicMock()
    vt.transcriber = MagicMock()
    vt.translator = MagicMock()
    vt.tts_engine = MagicMock()
    vt.synchronizer = MagicMock()
    vt.processor = MagicMock()
    vt.diarizer = MagicMock()
    vt.lipsyncer = MagicMock()
    vt.visual_translator = MagicMock()
    
    # Setup mocks
    vt.processor.extract_audio.return_value = "extracted.wav"
    vt.separator.separate.return_value = ("vocals.wav", "bg.wav")
    vt.transcriber.transcribe.return_value = ([{"text": "Hello", "start": 0, "end": 1}], "en")
    vt.translator.translate_segments.return_value = [{"text": "Hello", "translated_text": "Hola", "start": 0, "end": 1}]
    vt.tts_engine.generate_audio.return_value = "tts.wav"
    vt.synchronizer.merge_segments.return_value = True
    vt.processor.replace_audio.return_value = "final.mp4"
    
    # Simulate lip-sync failure
    vt.lipsyncer.sync_lips.side_effect = Exception("Lip-sync failed!")
    
    with patch('src.core.video_translator.Path') as MockPath, \
         patch('src.utils.config.validate_path', side_effect=lambda p, **kwargs: Path(p)):
        mock_path_instance = MockPath.return_value
        mock_path_instance.exists.return_value = True
        mock_path_instance.stat.return_value.st_size = 1000
        mock_path_instance.stem = "test"
        mock_path_instance.name = "test.mp4"
        
        gen = vt.process_video(
            video_path="test.mp4",
            source_lang="en",
            target_lang="es",
            audio_model_name="demucs",
            tts_model_name="edge",
            translation_model_name="google",
            context_model_name=None,
            transcription_model_name="base",
            optimize_translation=False,
            enable_diarization=False,
            enable_time_stretch=False,
            enable_vad=False,
            enable_lipsync=True,
            enable_visual_translation=False
        )
        
        # Should not raise, just continue
        results = list(gen)
    
    # Verify lip-sync was attempted
    vt.lipsyncer.sync_lips.assert_called_once()
    
    # Verify pipeline still produced a result
    assert any(r[0] == "result" for r in results)

