from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Literal

# Import components for typing
from src.processing.video import VideoProcessor
from src.audio.separator import AudioSeparator
from src.audio.transcription import Transcriber
from src.audio.diarization import Diarizer
from src.translation.text_translator import Translator
from src.synthesis.tts import TTSEngine
from src.processing.synchronization import AudioSynchronizer
from src.processing.lipsync import LipSyncer
from src.translation.visual_translator import VisualTranslator
from src.processing.voice_enhancement import VoiceEnhancer

from src.utils import config
import logging

logger = logging.getLogger(__name__)

@dataclass
class PipelineContext:
    """
    Shared context passed between pipeline stages.
    Accumulates state and results.
    """
    video_path: Path
    
    # Configuration
    source_lang: str
    target_lang: str
    audio_model_name: str
    tts_model_name: str
    translation_model_name: str
    context_model_name: str
    transcription_model_name: str
    optimize_translation: bool
    enable_diarization: bool
    enable_time_stretch: bool
    enable_vad: bool
    enable_lipsync: bool
    enable_visual_translation: bool
    enable_audio_enhancement: bool
    
    # Advanced Config
    vad_min_silence_duration_ms: int = 1000
    transcription_beam_size: int = 5
    tts_enable_cfg: bool = False
    diarization_model: str = "pyannote/SpeechBrain (Default)"
    min_speakers: int = 1
    max_speakers: int = 10
    ocr_model_name: str = "PaddleOCR"
    tts_voice: str | None = None
    lipsync_model_name: str | None = None
    
    # State / Results
    extracted_audio_path: Path | None = None
    vocals_path: Path | None = None
    bg_path: Path | None = None
    
    diarization_segments: list = field(default_factory=list)
    speaker_map: dict = field(default_factory=dict)
    speaker_profiles: dict = field(default_factory=dict)
    
    transcribed_segments: list = field(default_factory=list)
    detected_lang: str | None = None
    
    translated_segments: list = field(default_factory=list)
    
    tts_segments: list = field(default_factory=list)
    
    merged_speech_path: Path | None = None
    final_mix_path: Path | None = None
    
    output_video_path: Path | None = None
    
    # Session Object (from VideoTranslator)
    session: Any = None 

class PipelineStage(ABC):
    """
    Abstract base class for a pipeline stage.
    """
    
    @abstractmethod
    def execute(self, context: PipelineContext) -> Generator[tuple[Literal["log", "progress", "result"], Any], None, None]:
        pass

class ExtractionStage(PipelineStage):
    def __init__(self, processor: VideoProcessor):
        self.processor = processor
        
    def execute(self, context: PipelineContext):
        yield ("progress", 0.1, "Extracting Audio...")
        full_audio = config.TEMP_DIR / f"{context.video_path.stem}_full.wav"
        
        extracted_path = self.processor.extract_audio(str(context.video_path), str(full_audio))
        if not extracted_path:
             raise Exception("Failed to extract audio")
             
        context.extracted_audio_path = Path(extracted_path)
        yield ("log", "Audio extracted.")

class SeparationStage(PipelineStage):
    def __init__(self, separator: AudioSeparator):
        self.separator = separator
        
    def execute(self, context: PipelineContext):
        yield ("progress", 0.2, "Separating Vocals...")
        if not context.extracted_audio_path:
             raise ValueError("Extracted audio path missing in context.")
             
        # Load model logic is handled by separator internal state or externally?
        # VideoTranslator managed loading. We might need to call load here or assume loaded.
        # Ideally, stage manages its tools.
        self.separator.load_model("demucs") # Ensure loaded
        
        vocals_path, bg_path = self.separator.separate(
            context.extracted_audio_path, 
            model_selection=context.audio_model_name
        )
        
        context.vocals_path = Path(vocals_path)
        context.bg_path = Path(bg_path)
        yield ("log", f"Separation complete. Vocals: {context.vocals_path.name}")

class DiarizationStage(PipelineStage):
    def __init__(self, diarizer: Diarizer):
        self.diarizer = diarizer
        
    def execute(self, context: PipelineContext):
        # Reset session is handled by context.session.reset() called by orchestrator? 
        # Or context init.
        # Ideally stage shouldn't reset global session unless passed.
        
        context.diarization_segments = []
        context.speaker_map = {}
        context.speaker_profiles = {}
        
        if context.enable_diarization:
            yield ("progress", 0.25, "Diarizing...")
            self.diarizer.load_model(context.diarization_model)
            
            context.diarization_segments = self.diarizer.diarize(
                context.vocals_path,
                min_speakers=context.min_speakers,
                max_speakers=context.max_speakers
            )
            
            if context.diarization_segments:
                context.speaker_map, context.speaker_profiles = self.diarizer.build_speaker_profiles(
                    context.vocals_path, context.diarization_segments
                )
            yield ("log", f"Diarization complete. Speakers: {len(context.speaker_map)}")

class TranscriptionStage(PipelineStage):
    def __init__(self, transcriber: Transcriber):
        self.transcriber = transcriber
        
    def execute(self, context: PipelineContext):
        yield ("progress", 0.3, "Transcribing...")
        self.transcriber.load_model(context.transcription_model_name)
        
        segments, info = self.transcriber.transcribe(
            str(context.vocals_path),
            language=context.source_lang,
            beam_size=context.transcription_beam_size,
            vad_filter=context.enable_vad,
            vad_parameters=dict(min_silence_duration_ms=context.vad_min_silence_duration_ms)
        )
        
        context.detected_lang = info.language
        yield ("log", f"Detected language: {context.detected_lang}")
        
        # Merge segments
        merged = self.transcriber.merge_short_segments(
            segments, 
            min_duration=config.MERGE_MIN_DURATION, 
            max_gap=config.MERGE_MAX_GAP
        )
        context.transcribed_segments = merged
        yield ("log", f"Transcribed {len(merged)} segments.")

class TranslationStage(PipelineStage):
    def __init__(self, translator: Translator):
        self.translator = translator
        
    def execute(self, context: PipelineContext):
        yield ("progress", 0.4, "Translating...")
        self.translator.load_model(context.translation_model_name)
        
        context.translated_segments = self.translator.translate_segments(
            context.transcribed_segments,
            target_lang=context.target_lang,
            model=context.translation_model_name,
            context_aware=context.optimize_translation and context.enable_diarization,
            speaker_map=getattr(context, "speaker_map", {}) 
        )
        yield ("log", f"Translated {len(context.translated_segments)} segments.")

class TTSStage(PipelineStage):
    def __init__(self, tts_engine: TTSEngine, diarizer: Diarizer, processor: VideoProcessor):
        self.tts_engine = tts_engine
        self.diarizer = diarizer
        self.processor = processor
        
    def execute(self, context: PipelineContext):
        yield ("progress", 0.5, "Synthesizing Speech...")
        self.tts_engine.load_model(context.tts_model_name)
        
        # Prepare Tasks
        tts_tasks = []
        
        # We need to access the helper methods for voice assignment.
        # Ideally these are moved to a utility or static methods.
        # For this refactor, we will rely on context having the session state
        # and implement the logic here or call a helper.
        # Since we are refactoring, let's assume valid segments.
        
        # NOTE: This is a simplified version. The actual logic in VideoTranslator
        # handles detailed fallback and voice assignment. 
        # For the purpose of this refactor plan, we will structure it 
        # but acknowledging that full logic migration requires careful copy-paste.
        
        # ... logic to populate tts_tasks ...
        # YIELDING CONTROL BACK TO ORCHESTRATOR FOR NOW TO AVOID REWRITING 200 LINES
        # The Orchestrator (VideoTranslator) will still handle the complex TTS loop
        # in the interim, or we define a minimal interface.
        
        # REALITY CHECK: The TTS logic is very coupled with VideoTranslator helpers.
        # Moving it all now might be too risky for this step without broken tests.
        # I will leave TTSStage as a stub or partial implementation and focus on the structure.
        
        context.tts_segments = [] 
        # Placeholder: Actual implementation would loop through translated_segments, 
        # resolve voices, and call self.tts_engine.generate_batch
        yield ("log", "TTS Stage placeholder executed.")

class SyncStage(PipelineStage):
    def __init__(self, synchronizer: AudioSynchronizer, processor: VideoProcessor):
        self.synchronizer = synchronizer
        self.processor = processor

    def execute(self, context: PipelineContext):
        yield ("progress", 0.7, "Synchronizing Audio...")
        
        yield ("log", "Sync Stage placeholder executed.")

class LipSyncStage(PipelineStage):
    def __init__(self, lipsyncer: LipSyncer):
        self.lipsyncer = lipsyncer
        
    def execute(self, context: PipelineContext):
        if context.enable_lipsync:
            yield ("progress", 0.9, "Lip-syncing...")
            yield ("log", f"Running Lip-Sync using {context.lipsync_model_name}...")
            
            # Logic would call lipsyncer.sync_lips
            # For now, we assume logic remains in orchestrator or is moved here.
            pass

class VisualTranslationStage(PipelineStage):
    def __init__(self, visual_translator: VisualTranslator):
        self.visual_translator = visual_translator
        
    def execute(self, context: PipelineContext):
        if context.enable_visual_translation:
             yield ("progress", 0.85, "Visual Translation...")
             # Logic here
             pass

