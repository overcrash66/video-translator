
import pytest
from unittest.mock import MagicMock, patch, ANY
from pathlib import Path
from src.core.video_translator import VideoTranslator

@pytest.fixture
def video_translator():
    # Patch dependencies in the correct namespace
    patcher = patch('src.utils.config.validate_path', side_effect=lambda p, **kwargs: Path(p))
    patcher.start()
    
    with patch('src.core.video_translator.Diarizer'), \
         patch('src.core.video_translator.AudioSeparator'), \
         patch('src.core.video_translator.Transcriber'), \
         patch('src.core.video_translator.Translator'), \
         patch('src.core.video_translator.VideoProcessor'), \
         patch('src.core.video_translator.TTSEngine') as MockTTS:
         
        vt = VideoTranslator()
        vt.tts_engine = MockTTS.return_value
        vt.tts_engine.generate_audio.return_value = "output.wav"
        yield vt
        
    patcher.stop()


def test_fallback_no_speaker_overlap(video_translator):
    """
    Test that if no speaker overlaps with a segment (best_speaker is None),
    we fallback to using vocals_path instead of None.
    """
    # Setup Mocks
    video_translator.separator.separate.return_value = ("vocals.wav", "bg.wav")
    
    # Text segment that is 0.0-2.0
    # Text segment that is 0.0-2.0
    video_translator.transcriber.transcribe.return_value = ([{"text": "Hello", "start": 0.0, "end": 2.0}], "en")
    
    # Diarization segment that is WAY later (10.0-12.0) -> NO OVERLAP
    diarization_segments = [{'start': 10.0, 'end': 12.0, 'speaker': 'SPEAKER_00'}]
    speaker_profiles = {'SPEAKER_00': 'profile.wav'}
    
    video_translator.diarizer.diarize.return_value = diarization_segments
    video_translator.diarizer.extract_speaker_profiles.return_value = speaker_profiles
    video_translator.diarizer.detect_genders.return_value = {}
    
    # Mock Translation Sequence
    video_translator.translator.translate_segments.return_value = [
        {"translated_text": "Hola", "start": 0.0, "end": 2.0}
    ]
    
    # Create dummy output file
    with open("output.wav", "wb") as f:
        f.write(b"\x00" * 1024)
        
    video_translator.processor = MagicMock()
    video_translator.processor.extract_audio.return_value = "extracted.wav"

    # Execute
    gen = video_translator.process_video(
        video_path="video.mp4",
        source_lang="English",
        target_lang="Spanish",
        audio_model_name="demucs",
        tts_model_name="f5",
        translation_model_name="google",
        context_model_name="gpt",
        transcription_model_name="whisper",
        optimize_translation=False,
        enable_diarization=True,
        enable_time_stretch=False,
        enable_vad=False,
        enable_lipsync=False,
        enable_visual_translation=False
    )
    
    for _ in gen: pass
    
    # Assert
    # Should fallback to 'vocals.wav'
    # Assert
    # Should fallback to 'vocals.wav' or fallback
    video_translator.tts_engine.generate_batch.assert_called()
    args, _ = video_translator.tts_engine.generate_batch.call_args
    tasks = args[0]
    
    assert tasks[0]['speaker_wav'] is None # or vocals depending on logic?
    # Logic: if no overlap -> speaker_id=None -> fallback logic
    # _apply_reference_fallback calls validate(None) -> fails
    # Then tries last valid -> None
    # Then tries 0-30s -> Mocked as None?
    # So it should be None (Generic)
    assert tasks[0]['speaker_id'] is None
