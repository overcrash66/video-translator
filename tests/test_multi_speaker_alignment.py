import os
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import pytest
from src.core.video_translator import VideoTranslator
from src.audio.transcription import Transcriber
from src.processing.synchronization import AudioSynchronizer

def test_split_segment_by_speaker():
    # Instantiate components
    vt = VideoTranslator()
    
    # Mock segment
    seg = {
        'start': 1.0,
        'end': 4.0,
        'text': "Hello how are you? I am fine.",
        'words': [
            {'word': "Hello", 'start': 1.0, 'end': 1.5, 'probability': 0.9},
            {'word': " how", 'start': 1.5, 'end': 1.8, 'probability': 0.9},
            {'word': " are", 'start': 1.8, 'end': 2.0, 'probability': 0.9},
            {'word': " you?", 'start': 2.0, 'end': 2.5, 'probability': 0.9},
            {'word': " I", 'start': 2.5, 'end': 2.8, 'probability': 0.9},
            {'word': " am", 'start': 2.8, 'end': 3.0, 'probability': 0.9},
            {'word': " fine.", 'start': 3.0, 'end': 3.5, 'probability': 0.9},
        ]
    }
    
    # Mock diarization segments
    diar_segments = [
        {'start': 0.8, 'end': 2.4, 'speaker': 'SPEAKER_01'},
        {'start': 2.4, 'end': 4.2, 'speaker': 'SPEAKER_02'}
    ]
    
    res = vt._split_segment_by_speaker(seg, diar_segments)
    
    # Verify split count
    assert len(res) == 2
    
    # Verify first speaker sub-segment
    assert res[0]['speaker_id'] == 'SPEAKER_01'
    assert res[0]['text'] == "Hello how are you?"
    assert res[0]['start'] == 1.0
    assert res[0]['end'] == 2.5
    
    # Verify second speaker sub-segment
    assert res[1]['speaker_id'] == 'SPEAKER_02'
    assert res[1]['text'] == "I am fine."
    assert res[1]['start'] == 2.5
    assert res[1]['end'] == 3.5

def test_merge_short_segments_respects_speakers():
    transcriber = Transcriber()
    
    # Scenario A: Same speaker -> Should merge
    segs_same = [
        {'start': 1.0, 'end': 1.8, 'text': "Hello", 'speaker_id': 'SPEAKER_01'},
        {'start': 2.0, 'end': 2.8, 'text': "world", 'speaker_id': 'SPEAKER_01'}
    ]
    res_same = transcriber.merge_short_segments(segs_same, min_duration=2.0, max_gap=0.5)
    assert len(res_same) == 1
    assert res_same[0]['text'] == "Hello world"
    
    # Scenario B: Different speakers -> Should NOT merge
    segs_diff = [
        {'start': 1.0, 'end': 1.8, 'text': "Hello", 'speaker_id': 'SPEAKER_01'},
        {'start': 2.0, 'end': 2.8, 'text': "world", 'speaker_id': 'SPEAKER_02'}
    ]
    res_diff = transcriber.merge_short_segments(segs_diff, min_duration=2.0, max_gap=0.5)
    assert len(res_diff) == 2
    assert res_diff[0]['text'] == "Hello"
    assert res_diff[1]['text'] == "world"

def test_synchronizer_merge_segments_smart_mixing():
    sync = AudioSynchronizer()
    
    # Create temp files for test audio segments
    temp_files = []
    try:
        for i in range(3):
            f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            # 1 second of audio at 24kHz
            data = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, 24000)) * 0.1
            sf.write(f.name, data, 24000)
            temp_files.append(f.name)
            
        segments = [
            {'audio_path': temp_files[0], 'start': 1.0, 'end': 2.0},
            {'audio_path': temp_files[1], 'start': 1.8, 'end': 2.8},  # Overlaps by 0.2s
            {'audio_path': temp_files[2], 'start': 3.0, 'end': 4.0}
        ]
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_f:
            output_path = out_f.name
            
        res = sync.merge_segments(segments, total_duration=5.0, output_path=output_path, enable_time_stretch=True)
        assert res is not None
        assert os.path.exists(output_path)
        
        # Verify file properties
        info = sf.info(output_path)
        assert info.samplerate == 24000
        # Expected duration should be at least 4.0s (last segment end) plus dynamic shift
        assert info.duration >= 4.0
        
    finally:
        for p in temp_files:
            try:
                os.unlink(p)
            except:
                pass
        try:
            os.unlink(output_path)
        except:
            pass
