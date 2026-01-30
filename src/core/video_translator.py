from src.utils import config
import torch
import gc
import logging
import os
from pathlib import Path
from src.core.session import SessionContext

# Component imports
from src.audio.separator import AudioSeparator
from src.audio.transcription import Transcriber
from src.translation.text_translator import Translator
from src.synthesis.tts import TTSEngine
from src.processing.synchronization import AudioSynchronizer
from src.processing.video import VideoProcessor
from src.audio.diarization import Diarizer

from src.processing.lipsync import LipSyncer
from src.translation.visual_translator import VisualTranslator
from src.processing.voice_enhancement import VoiceEnhancer
from typing import Generator, Literal, Any, Optional

from src.utils import patches
from src.utils.chunker import VideoChunker

logger = logging.getLogger(__name__)

# Apply global patches
patches.apply_patches()

class VideoTranslator:
    """
    Central controller for the video translation pipeline.
    Enforces strict VRAM management by ensuring only one heavy model is loaded at a time.
    """
    def __init__(self,
                 separator: AudioSeparator | None = None,
                 transcriber: Transcriber | None = None,
                 translator: Translator | None = None,
                 tts_engine: TTSEngine | None = None,
                 synchronizer: AudioSynchronizer | None = None,
                 processor: VideoProcessor | None = None,
                 diarizer: Diarizer | None = None,
                 lipsyncer: LipSyncer | None = None,
                 visual_translator: VisualTranslator | None = None,
                 voice_enhancer: VoiceEnhancer | None = None,
                 live_portrait_acceleration: str = "ort") -> None:
        
        self.separator = separator or AudioSeparator()
        self.transcriber = transcriber or Transcriber()
        self.translator = translator or Translator()
        self.tts_engine = tts_engine or TTSEngine()
        self.synchronizer = synchronizer or AudioSynchronizer()
        self.processor = processor or VideoProcessor()
        self.diarizer = diarizer or Diarizer()
        self.lipsyncer = lipsyncer or LipSyncer(acceleration=live_portrait_acceleration)
        self.visual_translator = visual_translator or VisualTranslator()
        self.voice_enhancer = voice_enhancer or VoiceEnhancer()
        
        # Track currently loaded model to avoid redundant unloads/loads
        self.current_model = None
        
        # Session state
        self.session = SessionContext()

    def _get_assigned_voice(self, speaker_id, available_voices):
        """
        Deterministically assigns an unused voice to a speaker ID for the session.
        """
        # Check existing assignment
        existing = self.session.get_voice(speaker_id)
        if existing:
            return existing
            
        # Find first unused voice
        for voice in available_voices:
            if not self.session.is_voice_used(voice):
                self.session.assign_voice(speaker_id, voice)
                return voice
                
        # If all used, cycle using modulo (fallback)
        idx = len(self.session.speaker_voice_map) % len(available_voices)
        return available_voices[idx]

    def _resolve_speaker_reference(self, speaker_id: Optional[str], speaker_profiles: dict, vocals_path: str | Path, tts_model_name: str) -> tuple[Optional[str], bool]:
        """
        Determines the best reference audio for a speaker.
        Returns: (path_to_wav, is_fallback)
        """
        # Case 1: No speaker ID (Single speaker mode)
        if not speaker_id:
             return vocals_path, False

        # Case 2: Speaker has a clean profile
        if speaker_id in speaker_profiles:
            profile_path = speaker_profiles[speaker_id]
            # Use public validation API
            if self.tts_engine.validate_reference(profile_path, model_name=tts_model_name):
                return profile_path, False
            else:
                logger.warning(f"Profile validation failed for {speaker_id}. Fallback to generic voice.")
                return None, False

        # Case 3: Speaker identified but no profile extracted
        logger.warning(f"No clean profile for {speaker_id}. Fallback to generic voice.")
        return None, False  # Return None to signal use of generic voice, NOT full vocals

    def _extract_fallback_reference(self, vocals_path: str | Path) -> Optional[str]:
        """
        Extracts a valid speech clip from the first 30 seconds of the audio
        to use as a desperate fallback when no other reference is found.
        """
        fallback_path = config.TEMP_DIR / f"fallback_ref_0_30_{Path(vocals_path).stem}.wav"
        if fallback_path.exists():
            return str(fallback_path)
            
        try:
            logger.info("Comparing audio... Attempting to extract fallback reference from first 30s...")
            # We can use VAD or just simple slicing if VAD isn't desired/loaded.
            # But we have 'self.transcriber.vad' usually if initialized.
            
            # Simple approach: Slice 0-30s
            import soundfile as sf
            data, sr = sf.read(str(vocals_path))
            
            # Limit to 30s
            max_samples = 30 * sr
            if len(data) > max_samples:
                data = data[:max_samples]
                
            # If we have VAD loaded, try to find a speech segment
            # But 'self.transcriber' might not be loaded if we are in a different stage?
            # Actually process_video controls the flow, so Transcriber is initialized but maybe 'model' unloaded.
            # VAD is lightweight.
            
            # For robustness, let's just use the first 5-10 seconds of NON-SILENCE if possible.
            # Or just save the cropped 30s if it's long enough.
            
            if len(data) < 2 * sr: # Less than 2s
                logger.warning("First 30s too short for fallback.")
                return None
                
            sf.write(str(fallback_path), data, sr)
            
            # Check if valid (using TTS engine's validator)
            if self.tts_engine._check_reference_audio(str(fallback_path), min_duration=2.0):
                 logger.info(f"Created fallback reference: {fallback_path.name}")
                 return str(fallback_path)
            else:
                 logger.warning("Extracted fallback 30s failed validation.")
                 return None
                 
        except Exception as e:
            logger.error(f"Failed to extract fallback reference: {e}")
            return None

    def unload_all_models(self):
        """
        Force unloads all known models and clears CUDA cache.
        """
        logger.info("Unloading all models and clearing CUDA cache...")
        
        # Call unload methods on components if they exist
        if hasattr(self.separator, 'unload_model'):
            self.separator.unload_model()
            
        if hasattr(self.transcriber, 'unload_model'):
            self.transcriber.unload_model()

        if hasattr(self.translator, 'hymt') and self.translator.hymt:
             self.translator.hymt.unload_model()
             
        if hasattr(self.tts_engine, 'unload_model'):
            self.tts_engine.unload_model()
            
        if hasattr(self.diarizer, 'unload_model'):
            self.diarizer.unload_model()
            
        if hasattr(self.lipsyncer, 'unload_model'):
            self.lipsyncer.unload_model()
            
        if hasattr(self.visual_translator, 'unload_model'):
            self.visual_translator.unload_model()
            
        if hasattr(self.voice_enhancer, 'unload_model'):
            self.voice_enhancer.unload_model()
        
        self.current_model = None
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
        logger.info("VRAM cleared.")

    def load_model(self, model_type, **kwargs):
        """
        Generic loader that enforces the 'one heavy model' rule.
        """
        if self.current_model == model_type:
            return # Already loaded
            
        # Unload everything before loading a new heavy model
        self.unload_all_models()
        
        logger.info(f"Loading model: {model_type}")
        self.current_model = model_type
        
        # In the future, we can call specific load methods here if needed
        pass



    def get_available_tts_voices(self, model_name, language_code):
        """Wrapper to get available voices from TTSEngine."""
        return self.tts_engine.get_available_voices(model_name, language_code)

    def process_video(self, *args, **kwargs) -> Generator[tuple[Literal["log", "progress", "result"], Any] | list, None, None]:
        """
        Orchestrates the full pipeline, handling chunking if necessary.
        """
        video_path_arg = kwargs.get('video_path') or (args[0] if args else None)
        if not video_path_arg:
            raise ValueError("video_path is required")
            
        video_path = config.validate_path(video_path_arg, must_exist=False)
        
        # [FIX] Pop chunk_duration so it isn't passed to _process_pipeline
        chunk_val = kwargs.pop('chunk_duration', config.CHUNK_DURATION)
        
        # Only attempt chunking if the file actually exists (prevents failures in tests with mocked paths)
        if hasattr(video_path, 'exists') and video_path.exists() and video_path.stat().st_size > 0:
            chunker = VideoChunker(max_duration_sec=int(chunk_val))
            if chunker.should_chunk(video_path):
                yield ("log", f"Video is long (> {chunker.max_duration}s). Switching to Chunked Processing...")
                chunk_kwargs = kwargs.copy()
                chunk_kwargs.pop('video_path', None)
                yield from self._process_chunked(video_path, chunker, *args, **chunk_kwargs)
                return

        # Normal pipeline
        yield from self._process_pipeline(*args, **kwargs)

    def _process_chunked(self, video_path: Path, chunker: VideoChunker, *args, **kwargs):
        """
        Handles splitting, processing, and merging for long videos.
        """
        # 1. Global Diarization (if enabled)
        # We run this on the Full Audio to ensure speaker IDs are consistent across chunks.
        precomputed_diarization = None
        if kwargs.get('enable_diarization', False):
             yield ("progress", 0.05, "Global Diarization (Full Audio)...")
             yield ("log", "Running global diarization on full audio for consistency...")
             
             # Extract full audio temporarily
             full_audio = self._step_extraction(video_path)
             
             # Run Diarization Step
             self.load_model("diarization")
             # Use same params as pipeline
             precomputed_diarization = self._step_diarization(
                 full_audio, video_path, 
                 kwargs.get('diarization_model', "pyannote/SpeechBrain (Default)"),
                 kwargs.get('min_speakers', 1),
                 kwargs.get('max_speakers', 10)
             )
             self.unload_all_models()
             yield ("log", "Global diarization complete.")

        # 2. Split Video & Audio
        yield ("log", "Splitting video into chunks...")
        video_chunks = chunker.split_video(video_path)
        
        # We need audio chunks too? 
        # Actually _process_pipeline extracts audio from the video chunk passed to it.
        # So we just need video chunks.
        
        processed_chunks = []
        
        for i, chunk in enumerate(video_chunks):
            yield ("log", f"Refining Chunk {i+1}/{len(video_chunks)}...")
            yield ("progress", (i / len(video_chunks)), f"Processing Chunk {i+1}/{len(video_chunks)}")
            
            # Prepare kwargs for chunk
            chunk_kwargs = kwargs.copy()
            chunk_kwargs['video_path'] = chunk
            # Inject precomputed diarization
            chunk_kwargs['precomputed_diarization'] = precomputed_diarization
            
            # Run pipeline for chunk
            chunk_result = None
            
            # We strictly yield from _process_pipeline but we filter 'result'
            # We also might want to silence some logs or progress to avoid spam?
            # For now let's pass everything but capture the result.
            
            iterator = self._process_pipeline(**chunk_kwargs)
            for item in iterator:
                if isinstance(item, tuple):
                    if item[0] == "result":
                        chunk_result = item[1]
                    elif item[0] == "log":
                        # Indent logs for clarity
                        yield ("log", f"  [Chunk {i+1}] {item[1]}")
                    # Skip progress updates from sub-pipeline to avoid jumping 0-100
                elif isinstance(item, list):
                    # TTS segments list, ignore
                    pass
            
            if chunk_result:
                processed_chunks.append(Path(chunk_result))
            else:
                yield ("log", f"Error: Chunk {i+1} failed to produce result.")
                # Proceed? Or fail? failing 1 chunk ruins the video.
                raise Exception(f"Chunk {i+1} processing failed.")
                
            # Critical: Clear VRAM between chunks
            self.unload_all_models()
            
        # 3. Merge Results
        yield ("progress", 0.95, "Merging Chunks...")
        yield ("log", "Merging processed chunks...")
        
        final_output = config.OUTPUT_DIR / f"translated_{video_path.name}"
        chunker.merge_videos(processed_chunks, final_output)
        
        yield ("result", str(final_output))


    def _process_pipeline(self, 
                      video_path: str | Path, 
                      source_lang: str, 
                      target_lang: str, 
                      audio_model_name: str, 
                      tts_model_name: str, 
                      translation_model_name: str, 
                      context_model_name: str,
                      transcription_model_name: str, 
                      optimize_translation: bool, 
                      enable_diarization: bool, 
                      enable_time_stretch: bool, 
                      enable_vad: bool,
                      enable_lipsync: bool,
                      enable_visual_translation: bool,
                      enable_audio_enhancement: bool = False,
                      vad_min_silence_duration_ms: int = 1000,
                      transcription_beam_size: int = 5,
                      tts_enable_cfg: bool = False,
                      diarization_model: str = "pyannote/SpeechBrain (Default)",
                      min_speakers: int = 1,
                      max_speakers: int = 10,
                      ocr_model_name: str = "PaddleOCR",
                      tts_voice: str | None = None,
                      lipsync_model_name: str | None = "wav2lip",
                      live_portrait_acceleration: str = "ort",
                      precomputed_diarization: tuple | None = None) -> Generator[tuple[Literal["log", "progress", "result"], Any] | list, None, None]:
        """
        Internal pipeline logic (renamed from process_video).
        Supports precomputed_diarization for chunked execution.
        """
        
        # 0. Setup and Validation
        logger.info(f"DEBUG: VideoTranslator received live_portrait_acceleration='{live_portrait_acceleration}'")
        video_path = config.validate_path(video_path, must_exist=True)
        
        # ---------------------------------------------------------------------
        # 1. Audio Extraction Stage
        #    - Uses FFmpeg to extract audio track from video
        # ---------------------------------------------------------------------
        yield ("progress", 0.1, "Extracting Audio...")
        extracted_path = self._step_extraction(video_path)
        yield ("log", "Audio extracted.")
             
        # ---------------------------------------------------------------------
        # 2. Vocal Separation Stage
        #    - Uses Demucs to separate Vocals vs Background (Accompaniment)
        # ---------------------------------------------------------------------
        self.load_model("demucs") 
        yield ("progress", 0.2, "Separating Vocals...")
        vocals_path, bg_path = self._step_separation(extracted_path, audio_model_name)
        yield ("log", f"Separation complete. Vocals: {Path(vocals_path).name}")
        
        # ---------------------------------------------------------------------
        # 3. Speaker Diarization Stage
        #    - Identifies distinct speakers and extracts voice profiles
        # ---------------------------------------------------------------------
        # Reset session state for new video
        self.session.reset()
        
        diarization_segments, speaker_map, speaker_profiles = [], {}, {}
        if enable_diarization:
            if precomputed_diarization:
                 yield ("log", "Using precomputed global diarization...")
                 diarization_segments, speaker_map, speaker_profiles = precomputed_diarization
            else:
                self.load_model("diarization")
                yield ("progress", 0.25, "Diarizing...")
                diarization_segments, speaker_map, speaker_profiles = self._step_diarization(
                    vocals_path, video_path, diarization_model, min_speakers, max_speakers
                )
                yield ("log", f"Diarization complete. Speakers: {len(speaker_map)}")
            
        # ---------------------------------------------------------------------
        # 4. Transcription Stage
        #    - Converts speech to text (ASR) using Whisper
        #    - Incorporates VAD for clean segmentation
        # ---------------------------------------------------------------------
        self.load_model("whisper")
        yield ("progress", 0.3, "Transcribing...")
        # Resolve source/target codes
        source_code = config.get_language_code(source_lang)
        target_code = config.get_language_code(target_lang)
        
        segments, detected_lang = self._step_transcription(
             vocals_path, source_code, transcription_model_name, 
             enable_vad, transcription_beam_size, vad_min_silence_duration_ms
        )
        yield ("log", f"Transcription complete. {len(segments)} segments (Lang: {detected_lang}).")
        
        # Update source lang if auto
        if source_code == "auto" or source_code is None:
            logger.info(f"Updating source language from 'auto' to '{detected_lang}'")
            source_code = detected_lang
            if target_code == "auto": target_code = "en"
            
        # Merge short segments
        logger.info(f"Merging short segments (min_dur={config.MERGE_MIN_DURATION}s, max_gap={config.MERGE_MAX_GAP}s)...")
        before_count = len(segments)
        segments = self.transcriber.merge_short_segments(segments, min_duration=config.MERGE_MIN_DURATION, max_gap=config.MERGE_MAX_GAP)
        logger.info(f"Merged segments: {before_count} -> {len(segments)}")

        # ---------------------------------------------------------------------
        # 5. Translation Stage
        #    - Translates text segments to target language
        #    - Supports Context-Aware LLM translation
        # ---------------------------------------------------------------------
        self.load_model("translation_llm") 
        yield ("progress", 0.4, "Translating...")
        
        translated_segments = self._step_translation(
            segments, source_code, target_code, target_lang,
            translation_model_name, context_model_name, optimize_translation,
            video_path
        )
        yield ("log", f"Translation complete for {target_lang}.")
        
        # 6. TTS
        self.load_model("tts")
        yield ("progress", 0.5, "Generating Speech...")
        
        # Determine iterator for progress reporting
        gen_iterator = self._step_tts_generator(
            translated_segments, video_path, target_code, source_code,
            tts_model_name, tts_voice, tts_enable_cfg,
            enable_diarization, diarization_segments, speaker_map, speaker_profiles, vocals_path
        )
        
        tts_segments = []
        for item in gen_iterator:
            if isinstance(item, tuple) and item[0] == "progress":
                 yield item
            elif isinstance(item, list):
                 tts_segments = item
        
        # EQ Matching
        if enable_audio_enhancement and vocals_path and Path(vocals_path).exists():
             yield ("log", "Applying EQ matching...")
             count = self._apply_eq_matching_batch(vocals_path, tts_segments)
             yield ("log", f"Applied EQ matching to {count} segments.")

        yield ("log", "TTS Generation complete.")

        # 7. Sync & Mix
        self.unload_all_models() 
        yield ("progress", 0.7, "Synchronizing...")
        
        # Calculate duration
        import soundfile as sf
        try:
             duration_sec = sf.info(str(extracted_path)).duration
        except:
             duration_sec = tts_segments[-1]['end'] + 2.0 if tts_segments else 10.0
             
        merged_speech, final_mix = self._step_merge_mix(
            tts_segments, duration_sec, video_path, bg_path,
            enable_time_stretch, enable_audio_enhancement
        )
        
        # 8. Visual & LipSync
        if enable_visual_translation:
             yield ("progress", 0.8, "Translating Video Text...")
             self.load_model("visual")
             visual_out = config.TEMP_DIR / f"{video_path.stem}_visual.mp4"
             try:
                 self.visual_translator.translate_video_text(
                     str(video_path), str(visual_out),
                     target_lang=target_code, source_lang=source_code,
                     ocr_engine=ocr_model_name, ocr_interval_sec=2.0
                 )
                 if visual_out.exists():
                     video_path = visual_out
                     yield ("log", "Visual translation complete.")
             except Exception as e:
                logger.error(f"Visual translation failed: {e}")
        if enable_lipsync and lipsync_model_name:
            self.load_model("lipsync")
            
            # [Update] Propagate configuration to existing lipsyncer
            if hasattr(self.lipsyncer, 'acceleration') and live_portrait_acceleration:
                 if self.lipsyncer.acceleration != live_portrait_acceleration:
                     logger.info(f"Switching LivePortrait acceleration from {self.lipsyncer.acceleration} to {live_portrait_acceleration}")
                     self.lipsyncer.acceleration = live_portrait_acceleration
                     # Might typically need to reload if model was already loaded, but load_models handles it
            
            yield ("progress", 0.9, f"Lip-Syncing ({lipsync_model_name})...")
            out_path = self._step_lipsync(
                video_path,
                merged_speech, 
                lipsync_model_name
            )
            if out_path:
                # Use result as video source (silent)
                video_path = Path(out_path)
                yield ("log", "Lip-Sync complete.")
            else:
                yield ("log", "Lip-Sync failed or skipped.")
        
        # 9. Final Output
        output_video = config.OUTPUT_DIR / f"translated_{video_path.name}"
        result = self.processor.replace_audio(str(video_path), str(final_mix), str(output_video))
        
        yield ("result", str(result))

    # --- Orchestration Steps ---

    def _step_extraction(self, video_path: Path) -> str:
        """
        Extracts full audio track from the video file using ffmpeg.
        
        :param video_path: Path to the input video.
        :return: Path to the extracted .wav audio file.
        :raises Exception: If extraction fails.
        """
        full_audio = config.TEMP_DIR / f"{video_path.stem}_full.wav"
        extracted_path = self.processor.extract_audio(str(video_path), str(full_audio))
        if not extracted_path:
             raise Exception("Failed to extract audio")
        return extracted_path

    def _step_separation(self, extracted_path, model_name):
        """
        Separates audio into vocals and background (accompaniment).
        
        :param extracted_path: Path to the full audio file.
        :param model_name: Name/Type of the separation model (e.g., 'demucs', 'mdx').
        :return: Tuple (vocals_path, accompaniment_path).
        """
        return self.separator.separate(extracted_path, model_selection=model_name)

    def _step_diarization(self, vocals_path, video_path, model_name, min_spk, max_spk):
        """
        Performs speaker diarization to identify speakers and extract their profiles.
        
        :param vocals_path: Path to the isolated vocals audio.
        :param video_path: Path to the original video (used for naming).
        :param model_name: Name of the diarization model/backend.
        :param min_spk: Minimum number of speakers.
        :param max_spk: Maximum number of speakers.
        :return: Tuple (segments, speaker_map, speaker_profiles).
        """
        # Map UI name to backend key
        diar_backend = "speechbrain"
        if "NeMo" in model_name: diar_backend = "nemo"
        elif "Community" in model_name: diar_backend = "pyannote_community"
            
        segs = self.diarizer.diarize(vocals_path, backend=diar_backend, min_speakers=min_spk, max_speakers=max_spk)
        spk_map = self.diarizer.detect_genders(vocals_path, segs)
        
        profiles_dir = config.TEMP_DIR / f"{video_path.stem}_profiles"
        profiles = self.diarizer.extract_speaker_profiles(vocals_path, segs, profiles_dir)
        return segs, spk_map, profiles

    def _step_transcription(self, vocals_path, source_code, model_name, use_vad, beam_size, min_silence):
        """
        Transcribes the vocals track into text segments with timestamps.
        
        :param vocals_path: Path to isolated vocals.
        :param source_code: Language code of the source audio.
        :param model_name: Size of the Whisper model (e.g., 'tiny', 'large-v3').
        :return: List of transcription segments (dict with 'start', 'end', 'text').
        """
        return self.transcriber.transcribe(
            str(vocals_path), 
            language=source_code,
            beam_size=beam_size,
            use_vad=use_vad,
            model_size=model_name,
            min_silence_duration_ms=min_silence
        )

    def _step_translation(self, segments, source_code, target_code, target_lang, 
                          model_name, context_model, optimize, video_path):
        """
        Translates transcribed segments to the target language.
        Handles caching, optimization (context-aware), and SRT export.
        
        :param segments: List of source segments.
        :param source_code: Source language code.
        :param target_code: Target language code.
        :param model_name: Translation model key (e.g., 'google', 'gpt4').
        :param optimize: Whether to use context-aware translation.
        :return: List of segments with added 'translated_text' key.
        """
        # [Early Exit] Optimization
        if source_code == target_code:
            logger.info(f"Source matches Target ({source_code}). Skipping translation API.")
            translated = []
            for seg in segments:
                new_seg = seg.copy()
                new_seg["translated_text"] = seg["text"]
                translated.append(new_seg)
        else:
            effective_model = context_model if (optimize and context_model) else model_name
            
            trans_key = "google"
            if "HY-MT" in effective_model: trans_key = "hymt"
            elif "Llama" in effective_model: trans_key = "llama"
            elif "ALMA" in effective_model: trans_key = "alma"
            
            translated = self.translator.translate_segments(
                segments, target_code, model=trans_key, 
                source_lang=source_code, optimize=optimize
            )
        
        # Export SRT
        try:
            from src.utils.srt_generator import generate_srt
            srt_path = config.OUTPUT_DIR / f"{video_path.stem}.srt"
            config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            generate_srt(translated, str(srt_path))
        except Exception as e:
            logger.error(f"Failed to export subtitles: {e}")
            
        return translated

    def _step_tts_generator(self, translated_segments, video_path, target_code, source_code,
                            model_name, tts_voice, enable_cfg, enable_diarization, 
                            diarization_segments, speaker_map, speaker_profiles, vocals_path):
        """
        Generator that handles the loop for TTS segments.
        Now uses BATCH generation for performance.
        
        :param translated_segments: List of text segments to synthesize.
        :param video_path: Source video path.
        :param target_code: Target language code.
        :param source_code: Source language code.
        :param model_name: TTS model to use.
        :param tts_voice: Preferred generic voice.
        :param enable_cfg: Enable Classifier-Free Guidance.
        :param enable_diarization: Whether to use speaker profiles.
        :param diarization_segments: Diarization data.
        :param speaker_map: Map of speaker IDs to genders.
        :param speaker_profiles: Map of speaker IDs to reference wavs.
        :param vocals_path: Original vocals for fallback.
        :yields: Progress updates or log messages.
        :return: List of generated TTS segments (yielded as final item).
        """
        seg_dir = config.TEMP_DIR / video_path.stem
        seg_dir.mkdir(exist_ok=True)
        tts_segments = []
        
        # Prepare Batch Tasks
        tasks = []
        original_indices = []
        
        total_segs = len(translated_segments)
        
        for i, seg in enumerate(translated_segments):
             text = seg["translated_text"]
             if not text: continue
             
             ext = ".wav" if model_name in ["piper", "xtts"] else ".mp3"
             seg_out = seg_dir / f"seg_{i}_tts{ext}"
             
             gender, best_speaker, speaker_wav = "Female", None, None
             use_force_cloning = False
             
             # Resolve Speaker
             if enable_diarization:
                 best_speaker = self._find_overlapping_speaker(seg, diarization_segments) if diarization_segments else None
                 if best_speaker:
                     gender = speaker_map.get(best_speaker, "Female")
                     speaker_wav, use_force_cloning = self._resolve_speaker_reference(
                         best_speaker, speaker_profiles, vocals_path, model_name
                     )
                 else:
                     # logger.warning(f"No specific speaker detected for segment {i}.")
                     pass
             else:
                  # If no diarization, generic or fallback
                  speaker_wav = vocals_path 
                  
             # Fallback Logic
             speaker_wav, use_force_cloning = self._apply_reference_fallback(
                 speaker_wav, vocals_path, model_name, i
             )

             # Create Task
             task = {
                'text': text,
                'output_path': seg_out,
                'language': target_code,
                'speaker_wav': speaker_wav,
                'gender': gender,
                'speaker_id': best_speaker if enable_diarization else None,
                'guidance_scale': 1.3 if enable_cfg else None,
                'force_cloning': use_force_cloning,
                'voice_selector': self._get_assigned_voice,
                'source_lang': source_code,
                'preferred_voice': tts_voice
             }
             tasks.append(task)
             original_indices.append(i) # Needed to map back to segments

        yield ("log", f"Batch generating {len(tasks)} TTS segments...")
        
        # Execute Batch
        yield ("progress", 0.5, "Generating Speech (Batch)...")
        
        generated_paths = self.tts_engine.generate_batch(tasks, model=model_name)
        
        msg_count = 0
        for idx, (orig_i, path) in enumerate(zip(original_indices, generated_paths)):
             if path and Path(path).exists() and Path(path).stat().st_size > 100:
                seg = translated_segments[orig_i]
                tts_segments.append({
                    'audio_path': path,
                    'start': seg['start'],
                    'end': seg['end']
                })
             
        yield ("progress", 0.65, "TTS Generation Complete.")
        
        # Return the final list
        yield tts_segments

    def _find_overlapping_speaker(self, seg, diar_segments):
         """
         Identifies the speaker with the most overlap for a given segment.
         
         :param seg: The text segment {'start': float, 'end': float}.
         :param diar_segments: List of diarization segments with speaker info.
         :return: The speaker ID (e.g., 'SPEAKER_01') or None.
         """
         seg_start, seg_end = seg['start'], seg['end']
         max_overlap = 0
         best = None
         for d_seg in diar_segments:
              ov_start = max(seg_start, d_seg['start'])
              ov_end = min(seg_end, d_seg['end'])
              overlap = max(0, ov_end - ov_start)
              if overlap > max_overlap:
                  max_overlap = overlap
                  best = d_seg['speaker']
         return best

    def _apply_reference_fallback(self, speaker_wav, vocals_path, model_name, index):
         """
         Applies cascading fallback logic for speaker reference audio.
         1. Validates current reference.
         2. Tries 'Last Valid Reference'.
         3. Tries '0-30s Clip' from original vocals.
         
         :param speaker_wav: Current candidate for speaker reference.
         :param vocals_path: Path to full vocals (source for 30s clip).
         :param model_name: TTS model name (for validation rules).
         :param index: Segment index for logging.
         :return: Tuple (resolved_wav_path, force_cloning_flag).
         """
         # Logic from original code
         # 1. Update tracking
         if speaker_wav:
             if self.tts_engine.validate_reference(speaker_wav, model_name=model_name):
                 self.session.last_valid_reference_wav = speaker_wav
             else:
                 speaker_wav = None
         
         # 2. Last Valid
         use_force = False
         if not speaker_wav and self.session.last_valid_reference_wav:
             logger.info(f"Segment {index}: Using LAST VALID reference.")
             speaker_wav = self.session.last_valid_reference_wav
         
         # 3. 0-30s
         if not speaker_wav:
             fallback = self._extract_fallback_reference(vocals_path)
             if fallback:
                  logger.info(f"Segment {index}: Using 0-30s FALLBACK.")
                  speaker_wav = fallback
                  self.session.last_valid_reference_wav = fallback
         
         return speaker_wav, use_force

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _apply_eq_matching_batch(self, vocals_path, tts_segments):
         """
         Applies EQ matching to generated TTS segments to match the tone of the original vocals.
         
         :param vocals_path: Path to original vocals (reference).
         :param tts_segments: List of TTS segment dicts (modified in-place).
         :return: Count of successfully modified segments.
         """
         from src.audio.eq_matching import apply_eq_matching
         count = 0
         for tts_seg in tts_segments:
             out_path = str(tts_seg['audio_path']).replace('.wav', '_eq.wav').replace('.mp3', '_eq.wav')
             apply_eq_matching(str(vocals_path), str(tts_seg['audio_path']), out_path, strength=0.7)
             if Path(out_path).exists():
                 tts_seg['audio_path'] = out_path
                 count += 1
         return count

    def _step_merge_mix(self, tts_segments, duration, video_path, bg_path, time_stretch, enhance_audio):
        """
        Merges individual TTS segments into a single track and mixes with background audio.
        
        :param tts_segments: List of generated TTS segments.
        :param duration: Total duration of the detailed audio.
        :param video_path: Original video path.
        :param bg_path: Background music/noise path.
        :param time_stretch: Whether to stretch audio to fit time.
        :param enhance_audio: Whether to run VoiceFixer.
        :return: Tuple (merged_speech_path, final_mix_path).
        """
        merged_speech = config.TEMP_DIR / f"{video_path.stem}_merged_speech.wav"
        
        if not self.synchronizer.merge_segments(tts_segments, total_duration=duration, output_path=str(merged_speech), enable_time_stretch=time_stretch):
            raise Exception("Merging speech segments failed")
            
        if enhance_audio:
             self.load_model("voice_enhancer")
             enhanced = config.TEMP_DIR / f"{video_path.stem}_enhanced_speech.wav"
             try:
                 self.voice_enhancer.enhance_audio(merged_speech, enhanced)
                 if enhanced.exists(): merged_speech = enhanced
             except Exception as e:
                 logger.error(f"VoiceFixer failed: {e}")
                 
        final_mix = config.TEMP_DIR / f"{video_path.stem}_final_mix.wav"
        self.processor.mix_tracks(str(merged_speech), bg_path, str(final_mix))
        return merged_speech, final_mix
        
    def _step_lipsync(self, video_path, audio_path, model_name):
         """
         Runs lip-syncing on the video using the generated audio.
         
         :param video_path: Path to the video file.
         :param audio_path: Path to the new audio file.
         :param model_name: Name of the lipsync model from UI.
         :return: Path to the lipsynced video or None if failed.
         """
         lipsync_out = config.TEMP_DIR / f"{video_path.stem}_lipsync.mp4"
         try:
             # Resolve engine and enhancement
             engine_key = "wav2lip"
             enhance = "GFPGAN" in (model_name or "")
             
             if "LivePortrait" in (model_name or ""):
                 engine_key = "live_portrait"
             
             logger.info(f"Using lip-sync engine: {engine_key} (UI selected: {model_name})")
             
             self.lipsyncer.sync_lips(
                 str(video_path), 
                 str(audio_path), 
                 str(lipsync_out), 
                 model_name=engine_key,
                 enhance_face=enhance
             )
             return str(lipsync_out) if lipsync_out.exists() else None
         except Exception as e:
             logger.error(f"Lip-Sync failed: {e}")
             return None