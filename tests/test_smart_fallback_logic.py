
import pytest
from unittest.mock import MagicMock, patch, ANY
from pathlib import Path
from src.core.video_translator import VideoTranslator

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
        # Default behavior: generate_audio returns a valid path
        vt.tts_engine.generate_audio.return_value = "output.wav"
        return vt

def test_smart_fallback_profile_invalid(video_translator):
    """
    Test that if a speaker profile exists but fails validation, 
    we fallback to using vocals_path.
    """
    # Setup Mocks to inject data INTO process_video
    video_translator.separator.separate.return_value = ("vocals.wav", "bg.wav")
    video_translator.separator.separate.return_value = ("vocals.wav", "bg.wav")
    video_translator.transcriber.transcribe.return_value = ([{"text": "Hello", "start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}], "en")
    
    # Inject Diarization Data
    diarization_segments = [{'start': 0.0, 'end': 2.0, 'speaker': 'SPEAKER_00'}]
    speaker_profiles = {'SPEAKER_00': 'profile.wav'}
    
    video_translator.diarizer.diarize.return_value = diarization_segments
    video_translator.diarizer.extract_speaker_profiles.return_value = speaker_profiles
    video_translator.diarizer.detect_genders.return_value = {}
    
    # Mock TTS validation failure
    video_translator.tts_engine._check_reference_audio.return_value = False # INVALID PROFILE
    
    # Create dummy output file for validation checks
    with open("output.wav", "wb") as f:
        f.write(b"\x00" * 1024)

    # Mock Translation Sequence
    video_translator.translator.translate_segments.return_value = [
        {"translated_text": "Hola", "start": 0.0, "end": 2.0}
    ]

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
    
    # Consume generator
    video_translator.processor = MagicMock()
    video_translator.processor.extract_audio.return_value = "extracted.wav"
    
    # Run
    for _ in gen: pass
    
    # Assert
    # We expect 'vocals.wav' because validation failed
    video_translator.tts_engine.generate_audio.assert_called_with(
        ANY, 
        None, # EXPECTED FALLBACK to GENERIC (None)
        language='es', 
        output_path=ANY, 
        model='f5', 
        gender=ANY, 
        speaker_id='SPEAKER_00', 
        guidance_scale=ANY,
        force_cloning=ANY,
        voice_selector=ANY # Added voice_selector
    )

def test_smart_fallback_profile_valid(video_translator):
    """
    Test that if a speaker profile exists and PASSES validation,
    we use the profile.
    """
    # Setup Mocks
    video_translator.separator.separate.return_value = ("vocals.wav", "bg.wav")
    video_translator.separator.separate.return_value = ("vocals.wav", "bg.wav")
    video_translator.transcriber.transcribe.return_value = ([{"text": "Hello", "start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}], "en")
    
    diarization_segments = [{'start': 0.0, 'end': 2.0, 'speaker': 'SPEAKER_00'}]
    speaker_profiles = {'SPEAKER_00': 'profile.wav'}
    
    video_translator.diarizer.diarize.return_value = diarization_segments
    video_translator.diarizer.extract_speaker_profiles.return_value = speaker_profiles
    video_translator.diarizer.detect_genders.return_value = {}
    
    # Mock TTS validation SUCCESS
    video_translator.tts_engine._check_reference_audio.return_value = True # VALID PROFILE
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
    video_translator.tts_engine.generate_audio.assert_called_with(
        ANY, 
        'profile.wav', # EXPECTED NORMAL
        language='es', 
        output_path=ANY, 
        model='f5', 
        gender=ANY, 
        speaker_id='SPEAKER_00', 
        guidance_scale=ANY,
        force_cloning=ANY,
        voice_selector=ANY # Added voice_selector
    )
