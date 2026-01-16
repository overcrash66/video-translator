
import pytest
from unittest.mock import MagicMock, patch, ANY
from src.core.video_translator import VideoTranslator

@pytest.fixture
def video_translator():
    vt = VideoTranslator()
    vt.tts_engine = MagicMock()
    return vt

def test_tts_fallback_strategy(video_translator):
    """
    Test that VideoTranslator falls back to vocals_path when speaker profile is missing
    but diarization is enabled.
    """
    # Setup mocks
    video_translator.diarizer = MagicMock()
    
    # Mock inputs
    video_path = "test_video.mp4"
    vocals_path = "vocab_path.wav" # Acts as the "full audio" fallback
    
    # Simulate a scenario where diarization found a speaker, but NO profile was extracted
    diarization_segments = [{'start': 0.0, 'end': 1.0, 'speaker': 'SPEAKER_00'}]
    speaker_map = {'SPEAKER_00': 'Female'}
    speaker_profiles = {} # EMPTY profiles
    
    # Mock the generator execution flow manually since process_video is complex
    # We want to test the logic INSIDE step 6 (TTS)
    # So we'll inspect the logic trace or refactor? 
    # Actually, it's easier to verify implementation by checking the logic flow via a smaller mock
    # or by running the generator up to the point of TTS? No, too many dependencies.
    
    # Instead, let's look at the logic I just added. 
    # `if best_speaker in speaker_profiles: ... else: speaker_wav = vocals_path`
    
    pass 
    # Since I cannot easily unit test the huge process_video method without mocking 10 subsystems,
    # I will rely on the code review and the fact that the logic is straightforward:
    # IF enable_diarization AND best_speaker AND best_speaker NOT IN profiles:
    # 	speaker_wav SHOULD be vocals_path (not None)
    
    # Let's create a dummy segment processing verification using a stripped down logic test?
    # No, let's just create a test that mocks `generate_audio` and asserts it didn't receive None.
    
    # Mock everything needed to reach step 6
    video_translator.processor.extract_audio = MagicMock(return_value="extracted.wav")
    video_translator.separator.separate = MagicMock(return_value=(vocals_path, "bg.wav"))
    video_translator.diarizer.diarize = MagicMock(return_value=diarization_segments)
    video_translator.diarizer.detect_genders = MagicMock(return_value=speaker_map)
    video_translator.diarizer.extract_speaker_profiles = MagicMock(return_value=speaker_profiles) # Empty profiles
    video_translator.transcriber.transcribe = MagicMock(return_value=([{'start': 0.0, 'end': 1.0, 'text': 'Hello'}], "en"))
    video_translator.translator.translate_segments = MagicMock(return_value=[{'start': 0.0, 'end': 1.0, 'translated_text': 'Hola'}])
    
    # Mock synchronizer to avoid Step 7 crash
    video_translator.synchronizer.merge_segments = MagicMock(return_value=True)
    video_translator.processor.mix_tracks = MagicMock()
    video_translator.processor.replace_audio = MagicMock(return_value="output.mp4")
    
    # [Fix] Configure TTS mock to return a valid path that passes validation checks
    from pathlib import Path
    dummy_wav = Path("dummy_output.wav")
    dummy_wav.write_bytes(b'\x00' * 1024) # Write 1KB so st_size > 100 check passes
    video_translator.tts_engine.generate_audio = MagicMock(return_value=str(dummy_wav))

    # Run the generator
    gen = video_translator.process_video(
        video_path=video_path,
        source_lang="en",
        target_lang="es",
        audio_model_name="demucs",
        tts_model_name="f5", # F5 requires speaker_wav
        translation_model_name="google",
        context_model_name=None,
        transcription_model_name="tiny",
        optimize_translation=False,
        enable_diarization=True, # ENABLED
        enable_time_stretch=False,
        enable_vad=False,
        enable_lipsync=False,
        enable_visual_translation=False
    )
    
    # Consume generator
    for item in gen:
        pass
        
    # VERIFICATION
    # Check what generate_audio was called with
    # It should be called with speaker_wav = vocals_path, NOT None
    
    # args: (text, speaker_wav, ...)
    call_args = video_translator.tts_engine.generate_audio.call_args
    if call_args:
        args, kwargs = call_args
        speaker_wav_arg = args[1] if len(args) > 1 else kwargs.get('speaker_wav_path') # logic uses arg 1
        
        print(f"Called with speaker_wav: {speaker_wav_arg}")
        
        # We expect None (Generic Voice) because "vocals_path" fallback was removed to avoid demon voices
        assert speaker_wav_arg is None, f"Expected fallback to Generic (None), but got {speaker_wav_arg}"
    else:
        pytest.fail("generate_audio was never called!")
