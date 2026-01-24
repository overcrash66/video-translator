"""
Unit tests for TTS Reference Voice Fallback Logic.

Tests the cascading fallback strategy in VideoTranslator for selecting
reference audio when generating cloned speech. Fallback hierarchy:
1. Speaker profile from diarization
2. Last valid reference used in session  
3. First 30 seconds of vocals track
4. None (use generic TTS voice)

Test Scenarios:
- Valid speaker profile available → Use profile
- No overlapping speaker → Fallback to 0-30s extraction or None
- Profile exists but fails validation → Fallback to 0-30s or None
- Multiple segments with partial profiles → Last valid reference persists
- All fallbacks fail → Use generic voice (None)
- Empty speaker profiles → Use extracted fallback reference
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.core.video_translator import VideoTranslator


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def video_translator_with_mocks():
    """
    Create VideoTranslator with all dependencies mocked.
    Includes cleanup of temp files created during tests.
    """
    validate_path_patcher = patch(
        'src.utils.config.validate_path', 
        side_effect=lambda p, **kwargs: Path(p)
    )
    validate_path_patcher.start()

    with patch('src.core.video_translator.Diarizer'), \
         patch('src.core.video_translator.AudioSeparator'), \
         patch('src.core.video_translator.Transcriber'), \
         patch('src.core.video_translator.Translator'), \
         patch('src.core.video_translator.VideoProcessor'), \
         patch('src.core.video_translator.TTSEngine') as MockTTS:
         
        vt = VideoTranslator()
        vt.tts_engine = MockTTS.return_value
        vt.tts_engine.generate_audio.return_value = "output.wav"
        vt.tts_engine.validate_reference.return_value = True
        vt.tts_engine._check_reference_audio.return_value = False
        
        # Create dummy output file for validation checks
        output_path = Path("output.wav")
        output_path.write_bytes(b"\x00" * 1024)
        
        yield vt
        
        # Cleanup
        if output_path.exists():
            output_path.unlink()
        
        dummy_wav = Path("dummy_output.wav")
        if dummy_wav.exists():
            dummy_wav.unlink()
            
    validate_path_patcher.stop()


# =============================================================================
# Test Class: Valid Speaker Profile Scenarios
# =============================================================================

class TestTTSFallbackWithValidProfile:
    """Tests when speaker profiles are available and valid."""

    def test_tts_uses_valid_speaker_profile(self, video_translator_with_mocks):
        """
        When a speaker profile exists and passes validation,
        the TTS engine should use that profile for voice cloning.
        """
        vt = video_translator_with_mocks
        
        # Setup mocks
        vt.separator.separate.return_value = ("vocals.wav", "bg.wav")
        vt.transcriber.transcribe.return_value = (
            [{"text": "Hello", "start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}], 
            "en"
        )
        
        diarization_segments = [{'start': 0.0, 'end': 2.0, 'speaker': 'SPEAKER_00'}]
        speaker_profiles = {'SPEAKER_00': 'profile.wav'}
        
        vt.diarizer.diarize.return_value = diarization_segments
        vt.diarizer.extract_speaker_profiles.return_value = speaker_profiles
        vt.diarizer.detect_genders.return_value = {}
        
        vt.tts_engine.validate_reference.return_value = True
        vt.translator.translate_segments.return_value = [
            {"translated_text": "Hola", "start": 0.0, "end": 2.0}
        ]
        
        vt.processor = MagicMock()
        vt.processor.extract_audio.return_value = "extracted.wav"

        # Execute
        gen = vt.process_video(
            video_path="video.mp4",
            source_lang="English", target_lang="Spanish",
            audio_model_name="demucs", tts_model_name="f5",
            translation_model_name="google", context_model_name="gpt",
            transcription_model_name="whisper",
            optimize_translation=False, enable_diarization=True,
            enable_time_stretch=False, enable_vad=False,
            enable_lipsync=False, enable_visual_translation=False
        )
        
        for _ in gen:
            pass
        
        # Assert: Profile was used
        vt.tts_engine.generate_batch.assert_called()
        args, _ = vt.tts_engine.generate_batch.call_args
        tasks = args[0]
        
        assert tasks[0]['speaker_wav'] == 'profile.wav'

    def test_tts_persists_last_valid_reference_across_segments(self, video_translator_with_mocks):
        """
        When processing multiple segments, if a later segment has no speaker
        profile, we should reuse the last valid reference from a previous segment.
        
        Segment 1: Valid profile → Uses profile, becomes "last valid"
        Segment 2: No matching profile → Falls back to last valid
        """
        vt = video_translator_with_mocks
        
        vt.separator.separate.return_value = ("vocals.wav", "bg.wav")
        
        # Two translated segments
        vt.translator.translate_segments.return_value = [
            {"translated_text": "Seg1", "start": 0.0, "end": 2.0},
            {"translated_text": "Seg2", "start": 2.0, "end": 4.0}
        ]
        
        # Diarization: Only Segment 1 has a speaker
        diarization_segments = [{'start': 0.0, 'end': 2.0, 'speaker': 'SPEAKER_01'}]
        speaker_profiles = {'SPEAKER_01': 'profile_01.wav'}
        
        vt.diarizer.diarize.return_value = diarization_segments
        vt.diarizer.extract_speaker_profiles.return_value = speaker_profiles
        vt.diarizer.detect_genders.return_value = {}
        vt.transcriber.transcribe.return_value = ([], "en")

        # Mock validation: profiles are valid
        vt.tts_engine.validate_reference.side_effect = lambda path, model_name: "profile" in str(path)

        # Execute
        gen = vt.process_video(
            video_path="video.mp4",
            source_lang="en", target_lang="es",
            audio_model_name="demucs", tts_model_name="f5",
            translation_model_name="google", context_model_name="gpt",
            transcription_model_name="whisper",
            optimize_translation=False, enable_diarization=True,
            enable_time_stretch=False, enable_vad=False,
            enable_lipsync=False, enable_visual_translation=False
        )
        
        vt.processor.extract_audio.return_value = "video.wav"
        for _ in gen:
            pass
        
        # Assert
        vt.tts_engine.generate_batch.assert_called()
        args, _ = vt.tts_engine.generate_batch.call_args
        tasks = args[0]
        
        assert len(tasks) == 2
        assert tasks[0]['speaker_wav'] == 'profile_01.wav'  # Direct profile
        assert tasks[1]['speaker_wav'] == 'profile_01.wav'  # Last valid fallback


# =============================================================================
# Test Class: Fallback to 0-30s Extraction
# =============================================================================

class TestTTSFallbackTo30sExtraction:
    """Tests when fallback to first 30 seconds of audio is needed."""

    def test_tts_extracts_30s_fallback_when_no_profile_overlap(self, video_translator_with_mocks):
        """
        When diarization segments don't overlap with translation segments,
        we should extract from the first 30 seconds as a fallback.
        """
        vt = video_translator_with_mocks
        
        vt.separator.separate.return_value = ("vocals.wav", "bg.wav")
        vt.translator.translate_segments.return_value = [
            {"translated_text": "Seg1", "start": 0.0, "end": 2.0}
        ]
        
        # Diarization segments exist but DON'T OVERLAP with translation
        vt.diarizer.diarize.return_value = [
            {'start': 10.0, 'end': 12.0, 'speaker': 'SPEAKER_00'}
        ]
        vt.diarizer.extract_speaker_profiles.return_value = {}
        vt.transcriber.transcribe.return_value = ([], "en")

        # Mock _extract_fallback_reference to return a fallback path
        with patch.object(vt, '_extract_fallback_reference', return_value='fallback_30.wav') as mock_extract:
            gen = vt.process_video(
                video_path="video.mp4",
                source_lang="en", target_lang="es",
                audio_model_name="demucs", tts_model_name="f5",
                translation_model_name="google", context_model_name="gpt",
                transcription_model_name="whisper",
                optimize_translation=False, enable_diarization=True,
                enable_time_stretch=False, enable_vad=False,
                enable_lipsync=False, enable_visual_translation=False
            )
            
            vt.processor.extract_audio.return_value = "video.wav"
            for _ in gen:
                pass
            
            # Verify fallback extraction was called
            mock_extract.assert_called_once()
            
            # Check that fallback was used in TTS
            args, kwargs = vt.tts_engine.generate_batch.call_args
            tasks = args[0]
            
            assert len(tasks) == 1
            assert tasks[0]['speaker_wav'] == 'fallback_30.wav'
            assert kwargs.get('model') == 'f5'

    def test_tts_uses_fallback_when_empty_speaker_profiles(self, video_translator_with_mocks):
        """
        When diarization finds speakers but no profiles are extracted,
        we should use the extracted fallback reference.
        """
        vt = video_translator_with_mocks
        
        vocals_path = "vocals.wav"
        diarization_segments = [{'start': 0.0, 'end': 1.0, 'speaker': 'SPEAKER_00'}]
        speaker_profiles = {}  # EMPTY profiles
        
        vt.processor.extract_audio = MagicMock(return_value="extracted.wav")
        vt.separator.separate = MagicMock(return_value=(vocals_path, "bg.wav"))
        vt.diarizer.diarize = MagicMock(return_value=diarization_segments)
        vt.diarizer.detect_genders = MagicMock(return_value={'SPEAKER_00': 'Female'})
        vt.diarizer.extract_speaker_profiles = MagicMock(return_value=speaker_profiles)
        vt.transcriber.transcribe = MagicMock(return_value=([{'start': 0.0, 'end': 1.0, 'text': 'Hello'}], "en"))
        vt.translator.translate_segments = MagicMock(return_value=[{'start': 0.0, 'end': 1.0, 'translated_text': 'Hola'}])
        
        vt.synchronizer.merge_segments = MagicMock(return_value=True)
        vt.processor.mix_tracks = MagicMock()
        vt.processor.replace_audio = MagicMock(return_value="output.mp4")

        dummy_wav = Path("dummy_output.wav")
        dummy_wav.write_bytes(b'\x00' * 1024)
        vt.tts_engine.generate_batch = MagicMock(return_value=[str(dummy_wav)])
        vt._extract_fallback_reference = MagicMock(return_value="fallback_ref.wav")

        # Execute
        gen = vt.process_video(
            video_path="test_video.mp4",
            source_lang="en", target_lang="es",
            audio_model_name="demucs", tts_model_name="f5",
            translation_model_name="google", context_model_name=None,
            transcription_model_name="tiny",
            optimize_translation=False, enable_diarization=True,
            enable_time_stretch=False, enable_vad=False,
            enable_lipsync=False, enable_visual_translation=False
        )

        for _ in gen:
            pass

        # Verify
        call_args = vt.tts_engine.generate_batch.call_args
        assert call_args is not None, "generate_batch was never called!"
        
        args, _ = call_args
        tasks = args[0]
        assert len(tasks) > 0
        assert tasks[0]['speaker_wav'] == "fallback_ref.wav"


# =============================================================================
# Test Class: Fallback to Generic Voice (None)
# =============================================================================

class TestTTSFallbackToGenericVoice:
    """Tests when all fallbacks fail and we must use generic TTS voice."""

    def test_tts_uses_none_when_profile_validation_fails_and_no_fallback(self, video_translator_with_mocks):
        """
        When a speaker profile exists but fails validation,
        and 0-30s extraction also fails, we should use None (generic voice).
        """
        vt = video_translator_with_mocks
        
        # Mock 0-30s extraction to fail
        with patch.object(vt, '_extract_fallback_reference', return_value=None):
            vt.separator.separate.return_value = ("vocals.wav", "bg.wav")
            vt.transcriber.transcribe.return_value = (
                [{"text": "Hello", "start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}], 
                "en"
            )
            
            diarization_segments = [{'start': 0.0, 'end': 2.0, 'speaker': 'SPEAKER_00'}]
            speaker_profiles = {'SPEAKER_00': 'profile.wav'}
            
            vt.diarizer.diarize.return_value = diarization_segments
            vt.diarizer.extract_speaker_profiles.return_value = speaker_profiles
            vt.diarizer.detect_genders.return_value = {}
            
            # Profile validation FAILS
            vt.tts_engine.validate_reference.return_value = False
            
            vt.translator.translate_segments.return_value = [
                {"translated_text": "Hola", "start": 0.0, "end": 2.0}
            ]

            vt.processor = MagicMock()
            vt.processor.extract_audio.return_value = "extracted.wav"
            
            # Execute
            gen = vt.process_video(
                video_path="video.mp4",
                source_lang="English", target_lang="Spanish",
                audio_model_name="demucs", tts_model_name="f5",
                translation_model_name="google", context_model_name="gpt",
                transcription_model_name="whisper",
                optimize_translation=False, enable_diarization=True,
                enable_time_stretch=False, enable_vad=False,
                enable_lipsync=False, enable_visual_translation=False
            )
            
            for _ in gen:
                pass
            
            # Assert: speaker_wav is None (generic voice)
            vt.tts_engine.generate_batch.assert_called()
            args, _ = vt.tts_engine.generate_batch.call_args
            tasks = args[0]
            
            assert tasks[0]['speaker_wav'] is None

    def test_tts_uses_none_when_no_speaker_overlap_and_extraction_fails(self, video_translator_with_mocks):
        """
        When no speaker overlaps with a segment (best_speaker is None)
        and 0-30s fallback extraction also fails, use None (generic voice).
        """
        vt = video_translator_with_mocks
        
        # Mock 0-30s extraction to fail
        with patch.object(vt, '_extract_fallback_reference', return_value=None):
            vt.separator.separate.return_value = ("vocals.wav", "bg.wav")
            
            # Text segment is 0.0-2.0
            vt.transcriber.transcribe.return_value = (
                [{"text": "Hello", "start": 0.0, "end": 2.0}], 
                "en"
            )
            
            # Diarization segment is WAY later (10.0-12.0) → NO OVERLAP
            diarization_segments = [{'start': 10.0, 'end': 12.0, 'speaker': 'SPEAKER_00'}]
            speaker_profiles = {'SPEAKER_00': 'profile.wav'}
            
            vt.diarizer.diarize.return_value = diarization_segments
            vt.diarizer.extract_speaker_profiles.return_value = speaker_profiles
            vt.diarizer.detect_genders.return_value = {}
            
            vt.translator.translate_segments.return_value = [
                {"translated_text": "Hola", "start": 0.0, "end": 2.0}
            ]
            
            vt.processor = MagicMock()
            vt.processor.extract_audio.return_value = "extracted.wav"

            # Execute
            gen = vt.process_video(
                video_path="video.mp4",
                source_lang="English", target_lang="Spanish",
                audio_model_name="demucs", tts_model_name="f5",
                translation_model_name="google", context_model_name="gpt",
                transcription_model_name="whisper",
                optimize_translation=False, enable_diarization=True,
                enable_time_stretch=False, enable_vad=False,
                enable_lipsync=False, enable_visual_translation=False
            )
            
            for _ in gen:
                pass
            
            # Assert: Both speaker_wav and speaker_id are None
            vt.tts_engine.generate_batch.assert_called()
            args, _ = vt.tts_engine.generate_batch.call_args
            tasks = args[0]
            
            assert tasks[0]['speaker_wav'] is None
            assert tasks[0]['speaker_id'] is None
