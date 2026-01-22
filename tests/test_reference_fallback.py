import pytest
from unittest.mock import MagicMock, patch, ANY
from pathlib import Path
from src.core.video_translator import VideoTranslator
import shutil

@pytest.fixture
def video_translator():
    # Patch dependencies in the correct namespace
    with patch('src.core.video_translator.Diarizer'), \
         patch('src.core.video_translator.AudioSeparator'), \
         patch('src.core.video_translator.Transcriber'), \
         patch('src.core.video_translator.Translator'), \
         patch('src.core.video_translator.VideoProcessor'), \
         patch('src.core.video_translator.TTSEngine') as MockTTS:
         
        vt = VideoTranslator()
        vt.tts_engine = MockTTS.return_value
        vt.tts_engine.generate_audio.return_value = "output.wav"
        vt.tts_engine.validate_reference.return_value = True # Default true
        
        # Create dummy output file for validation checks
        with open("output.wav", "wb") as f:
            f.write(b"\x00" * 1024)
            
        yield vt
        
        # Cleanup
        if Path("output.wav").exists():
            Path("output.wav").unlink()

def test_fallback_last_valid_reference(video_translator):
    """
    Test that we use the last valid reference if the current one is missing.
    Sequence:
    Seg 1: Valid Profile -> Uses Profile, Sets Last Valid
    Seg 2: No Profile -> Uses Last Valid
    """
    # Setup Data
    video_translator.separator.separate.return_value = ("vocals.wav", "bg.wav")
    
    # 2 Segments
    video_translator.translator.translate_segments.return_value = [
        {"translated_text": "Seg1", "start": 0.0, "end": 2.0},
        {"translated_text": "Seg2", "start": 2.0, "end": 4.0}
    ]
    
    # Diarization: Only Seg 1 has a speaker
    diarization_segments = [{'start': 0.0, 'end': 2.0, 'speaker': 'SPEAKER_01'}]
    speaker_profiles = {'SPEAKER_01': 'profile_01.wav'}
    
    video_translator.diarizer.diarize.return_value = diarization_segments
    video_translator.diarizer.extract_speaker_profiles.return_value = speaker_profiles
    video_translator.diarizer.detect_genders.return_value = {}
    video_translator.transcriber.transcribe.return_value = ([], "en")

    # Mock Validation
    video_translator.tts_engine.validate_reference.side_effect = lambda path, model_name: "profile" in str(path)

    # Execute
    gen = video_translator.process_video(
        video_path="video.mp4",
        source_lang="en", target_lang="es",
        audio_model_name="demucs", tts_model_name="f5",
        translation_model_name="google", context_model_name="gpt",
        transcription_model_name="whisper",
        optimize_translation=False, enable_diarization=True,
        enable_time_stretch=False, enable_vad=False,
        enable_lipsync=False, enable_visual_translation=False
    )
    
    # Mock extract
    video_translator.processor.extract_audio.return_value = "video.wav"
    
    for _ in gen: pass
    
    # Verify Calls
    calls = video_translator.tts_engine.generate_audio.call_args_list
    assert len(calls) == 2
    
    # Call 1: Should use profile_01.wav
    args1, _ = calls[0]
    assert args1[1] == 'profile_01.wav'
    
    # Call 2: Should fallback to profile_01.wav (Last Valid)
    args2, _ = calls[1]
    assert args2[1] == 'profile_01.wav' 

def test_fallback_0_30s_extraction(video_translator):
    """
    Test that if NO reference exists at all, we extract from 0-30s.
    """
    video_translator.separator.separate.return_value = ("vocals.wav", "bg.wav")
    video_translator.translator.translate_segments.return_value = [
        {"translated_text": "Seg1", "start": 0.0, "end": 2.0}
    ]
    
    
    # Mock diarization segments to EXIST but NOT OVERLAP with translation
    # Trans segment: 0.0 - 2.0
    # Diar segment: 10.0 - 12.0 (No overlap)
    video_translator.diarizer.diarize.return_value = [
        {'start': 10.0, 'end': 12.0, 'speaker': 'SPEAKER_00'}
    ]
    video_translator.diarizer.extract_speaker_profiles.return_value = {}
    video_translator.transcriber.transcribe.return_value = ([], "en")

    # Mock _extract_fallback_reference directly to avoid file IO
    with patch.object(video_translator, '_extract_fallback_reference', return_value='fallback_30.wav') as mock_extract:
    
        gen = video_translator.process_video(
            video_path="video.mp4",
            source_lang="en", target_lang="es",
            audio_model_name="demucs", tts_model_name="f5",
            translation_model_name="google", context_model_name="gpt",
            transcription_model_name="whisper",
            optimize_translation=False, enable_diarization=True,
            enable_time_stretch=False, enable_vad=False,
            enable_lipsync=False, enable_visual_translation=False
        )
        
        video_translator.processor.extract_audio.return_value = "video.wav"
        for _ in gen: pass
        
        # Verify
        mock_extract.assert_called_once()
        
        # Check generate audio call
        video_translator.tts_engine.generate_audio.assert_called_with(
            ANY, 
            'fallback_30.wav', # USED FALLBACK
            language=ANY, output_path=ANY, model='f5', 
            gender=ANY, speaker_id=ANY, guidance_scale=ANY, 
            force_cloning=ANY, voice_selector=ANY, source_lang=ANY, preferred_voice=ANY
        )
