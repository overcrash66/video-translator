import pytest
from unittest.mock import MagicMock, patch, ANY
from pathlib import Path
from src.core.video_translator import VideoTranslator
from src.core.session import SessionContext

@pytest.fixture
def mock_components():
    return {
        'separator': MagicMock(),
        'transcriber': MagicMock(),
        'translator': MagicMock(),
        'tts_engine': MagicMock(),
        'synchronizer': MagicMock(),
        'processor': MagicMock(),
        'diarizer': MagicMock(),
        'lipsyncer': MagicMock(),
        'visual_translator': MagicMock(),
        'voice_enhancer': MagicMock()
    }

def test_full_pipeline_integration(mock_components):
    """
    End-to-end integration test mocking all subsystems.
    Verifies that VideoTranslator orchestrates data flow correctly through all 9 steps.
    """
    # 1. Setup Data Mocks
    mock_components['processor'].extract_audio.return_value = "full_audio.wav"
    mock_components['separator'].separate.return_value = ("vocals.wav", "bg.wav")
    
    # Diarization
    mock_components['diarizer'].diarize.return_value = [{'start': 0, 'end': 5, 'speaker': 'SPK_01'}]
    mock_components['diarizer'].detect_genders.return_value = {'SPK_01': 'Male'}
    mock_components['diarizer'].extract_speaker_profiles.return_value = {'SPK_01': 'profile.wav'}
    
    # Transcription
    mock_components['transcriber'].transcribe.return_value = ([{'start': 0.0, 'end': 5.0, 'text': "Hello World"}], "en")
    
    # Translation
    # Note: Translator usually returns new list with 'translated_text'
    mock_components['translator'].translate_segments.return_value = [{'start': 0.0, 'end': 5.0, 'text': "Hello World", 'translated_text': "Hola Mundo"}]
    
    # TTS
    # Mock Batch API
    mock_components['tts_engine'].generate_batch.return_value = ["seg_0.mp3"]
    # Mock file validation for TTS output check (via Path)
    with patch('pathlib.Path.exists', return_value=True), \
         patch('pathlib.Path.stat') as mock_stat:
        mock_stat.return_value.st_size = 500 # Valid size
        mock_stat.return_value.st_mode = 16877 # Dir mode (0o40755)
        
        # Audio Sync
        mock_components['synchronizer'].merge_segments.return_value = "merged_tts.wav"
        
        # Final Mix
        mock_components['processor'].replace_audio.return_value = "final_output.mp4"

        # 2. Instantiate with DI
        vt = VideoTranslator(**mock_components)
        
        # 3. Run Pipeline
        gen = vt.process_video(
            video_path="input.mp4",
            source_lang="en", 
            target_lang="es",
            audio_model_name="demucs",
            tts_model_name="edge",
            translation_model_name="google",
            context_model_name=None,
            transcription_model_name="small",
            optimize_translation=False,
            enable_diarization=True,
            enable_time_stretch=False,
            enable_vad=False,
            enable_lipsync=False,
            enable_visual_translation=False
        )
        
        results = list(gen)
        
        # 4. Assertions
        
        # Step 1: Extraction
        mock_components['processor'].extract_audio.assert_called()
        
        # Step 2: Separation
        mock_components['separator'].separate.assert_called()
        
        # Step 3: Diarization
        mock_components['diarizer'].diarize.assert_called()
        
        # Step 4: Transcription
        mock_components['transcriber'].transcribe.assert_called()
        
        # Step 5: Translation
        mock_components['translator'].translate_segments.assert_called()
        
        # Step 6: TTS (Batch)
        mock_components['tts_engine'].generate_batch.assert_called()
        # Verify tasks contain translated text
        args, _ = mock_components['tts_engine'].generate_batch.call_args
        tasks = args[0]
        assert tasks[0]['text'] == "Hola Mundo"
        assert tasks[0]['language'] == "es"
        
        # Step 7: Sync
        mock_components['synchronizer'].merge_segments.assert_called()
        
        # Step 9: Final Mix
        mock_components['processor'].replace_audio.assert_called()
        
        # Check result
        assert results[-1] == ("result", "final_output.mp4")
