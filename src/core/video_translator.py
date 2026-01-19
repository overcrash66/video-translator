import torch
import gc
import logging
import os
from pathlib import Path
from src.utils import config

# Component imports
from src.audio.separator import AudioSeparator
from src.audio.transcription import Transcriber
from src.translation.text_translator import Translator
from src.synthesis.tts import TTSEngine
from src.processing.synchronization import AudioSynchronizer
from src.processing.video import VideoProcessor
from src.audio.diarization import Diarizer

# Optional imports for new features (Placeholder for now until implemented)
from src.processing.lipsync import LipSyncer
from src.translation.visual_translator import VisualTranslator

logger = logging.getLogger(__name__)

class VideoTranslator:
    """
    Central controller for the video translation pipeline.
    Enforces strict VRAM management by ensuring only one heavy model is loaded at a time.
    """
    def __init__(self):
        self.separator = AudioSeparator()
        self.transcriber = Transcriber()
        self.translator = Translator()
        self.tts_engine = TTSEngine()
        self.synchronizer = AudioSynchronizer()
        self.processor = VideoProcessor()
        self.diarizer = Diarizer()
        self.lipsyncer = LipSyncer()
        self.visual_translator = VisualTranslator()
        
        # Track currently loaded model to avoid redundant unloads/loads
        self.current_model = None
        
        # Session state for deterministic voice mapping
        self.speaker_voice_map = {} 
        self.used_voices = set()

    def _get_assigned_voice(self, speaker_id, available_voices):
        """
        Deterministically assigns an unused voice to a speaker ID for the session.
        """
        if speaker_id in self.speaker_voice_map:
            return self.speaker_voice_map[speaker_id]
            
        # Find first unused voice
        for voice in available_voices:
            if voice not in self.used_voices:
                self.speaker_voice_map[speaker_id] = voice
                self.used_voices.add(voice)
                return voice
                
        # If all used, cycle using modulo (fallback)
        idx = len(self.speaker_voice_map) % len(available_voices)
        return available_voices[idx]

    def _resolve_speaker_reference(self, speaker_id, speaker_profiles, vocals_path, tts_model_name):
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

    def process_video(self, 
                      video_path, 
                      source_lang, 
                      target_lang, 
                      audio_model_name, 
                      tts_model_name, 
                      translation_model_name, 
                      context_model_name,
                      transcription_model_name, 
                      optimize_translation, 
                      enable_diarization, 
                      enable_time_stretch, 
                      enable_vad,
                      enable_lipsync,
                      enable_visual_translation,
                      vad_min_silence_duration_ms=1000,
                      transcription_beam_size=5,
                      tts_enable_cfg=False,
                      diarization_model="pyannote/SpeechBrain (Default)",
                      min_speakers=1,
                      max_speakers=10,
                      ocr_model_name="PaddleOCR",
                      tts_voice=None):
        """
        Orchestrates the full pipeline as a generator.
        Yields: ("log", message) or ("progress", value, desc) or ("result", path)
        """
        
        # 0. Setup
        video_path = Path(video_path)
        
        # 1. Extraction
        yield ("progress", 0.1, "Extracting Audio...")
        full_audio = config.TEMP_DIR / f"{video_path.stem}_full.wav"
        extracted_path = self.processor.extract_audio(str(video_path), str(full_audio))
        if not extracted_path:
             raise Exception("Failed to extract audio")
        yield ("log", "Audio extracted.")
             
        # 2. Separation
        self.load_model("demucs") 
        yield ("progress", 0.2, "Separating Vocals...")
        vocals_path, bg_path = self.separator.separate(extracted_path, model_selection=audio_model_name)
        yield ("log", f"Separation complete. Vocals: {Path(vocals_path).name}")
        
        # 3. Diarization
        speaker_map = {}
        diarization_segments = []
        speaker_profiles = {}
        
        # Reset voice map for new video
        self.speaker_voice_map = {}
        self.used_voices = set()
        
        if enable_diarization:
            self.load_model("diarization")
            yield ("progress", 0.25, "Diarizing...")
            
            # Map UI name to backend key
            diar_backend = "speechbrain"
            if "NeMo" in diarization_model:
                diar_backend = "nemo"
            elif "Community" in diarization_model:
                diar_backend = "pyannote_community"
                
            diarization_segments = self.diarizer.diarize(vocals_path, backend=diar_backend, min_speakers=min_speakers, max_speakers=max_speakers)
            speaker_map = self.diarizer.detect_genders(vocals_path, diarization_segments)
            
            # EXTRACT PROFILES FOR TTS CLONING
            profiles_dir = config.TEMP_DIR / f"{video_path.stem}_profiles"
            yield ("log", "Extracting speaker profiles...")
            speaker_profiles = self.diarizer.extract_speaker_profiles(vocals_path, diarization_segments, profiles_dir)
            
            yield ("log", f"Diarization complete. Speakers: {len(speaker_map)}")
            
        # 4. Transcription
        self.load_model("whisper")
        yield ("progress", 0.3, "Transcribing...")
        source_code = config.get_language_code(source_lang)
        segments = self.transcriber.transcribe(
            vocals_path, 
            language=source_code, 
            model_size=transcription_model_name, 
            use_vad=enable_vad, 
            beam_size=transcription_beam_size,
            min_silence_duration_ms=vad_min_silence_duration_ms
        )
        if not segments:
            raise Exception("No speech detected")
        yield ("log", f"Transcription complete. {len(segments)} segments.")
            
        # 5. Translation
        self.load_model("translation_llm") 
        yield ("progress", 0.4, "Translating...")
        target_code = config.get_language_code(target_lang)
        
        # Determine effective model
        effective_model_name = translation_model_name
        if optimize_translation and context_model_name:
             effective_model_name = context_model_name

        trans_model_key = "google"
        if "HY-MT" in effective_model_name:
            trans_model_key = "hymt"
        elif "Llama" in effective_model_name:
             trans_model_key = "llama"
        elif "ALMA" in effective_model_name:
             trans_model_key = "alma"
        
        # [Refactor] Transcriber now returns (segments, detected_lang_code)
        segments, detected_lang = self.transcriber.transcribe(
            str(vocals_path), 
            language=source_code,
            beam_size=5,
            use_vad=enable_vad
        )
        
        # [Fix] Update source_code if it was auto/unknown, so Translator uses correct source lang
        if source_code == "auto" or source_code is None:
            logger.info(f"Updating source language from 'auto' to '{detected_lang}'")
            source_code = detected_lang
            if target_code == "auto": # Corner case
                 target_code = "en"

        yield ("progress", 0.4, "Translating...")
        
        translated_segments = self.translator.translate_segments(
            segments, 
            target_code, 
            model=trans_model_key, 
            source_lang=source_code,
            optimize=optimize_translation
        )
        yield ("log", f"Translation complete for {target_lang}.")
        
        # 6. TTS
        self.load_model("tts")
        yield ("progress", 0.5, "Generating Speech...")
        
        seg_dir = config.TEMP_DIR / video_path.stem
        seg_dir.mkdir(exist_ok=True)
        tts_segments = []
        
        total_segs = len(translated_segments)
        for i, seg in enumerate(translated_segments):
             text = seg["translated_text"]
             if not text: continue
             
             ext = ".wav" if tts_model_name in ["piper", "xtts"] else ".mp3"
             seg_out = seg_dir / f"seg_{i}_tts{ext}"
             
             # Gender/Speaker logic
             gender = "Female"
             best_speaker = None
             
             # Speaker Logic Simplification
             speaker_wav = vocals_path
             use_force_cloning = False

             if enable_diarization:
                 if not diarization_segments:
                     use_force_cloning = True
                     logger.warning("Diarization enabled but no segments. Fallback to full vocals.")
                 else:
                     # Find overlapping speaker
                     seg_start = seg['start']
                     seg_end = seg['end']
                     max_overlap = 0
                     for d_seg in diarization_segments:
                          overlap_start = max(seg_start, d_seg['start'])
                          overlap_end = min(seg_end, d_seg['end'])
                          overlap = max(0, overlap_end - overlap_start)
                          if overlap > max_overlap:
                              max_overlap = overlap
                              best_speaker = d_seg['speaker']
                     
                     if best_speaker:
                         gender = speaker_map.get(best_speaker, "Female")
                         speaker_wav, use_force_cloning = self._resolve_speaker_reference(
                             best_speaker, speaker_profiles, vocals_path, tts_model_name
                         )
                     else:
                         # No overlap logic - likely just background noise or briefly spoken word
                         # We still try to find closest speaker... but for now, Generic.
                         use_force_cloning = False # Do NOT force clone full track
                         speaker_wav = None # Signal generic
                         logger.warning(f"No specific speaker detected for segment {i}. Using generic voice.")
                     
             generated_path = self.tts_engine.generate_audio(
                text, speaker_wav, 
                language=target_code, 
                output_path=seg_out, 
                model=tts_model_name, 
                gender=gender,

                speaker_id=best_speaker if enable_diarization else None,
                guidance_scale=1.3 if tts_enable_cfg else None,
                force_cloning=use_force_cloning,
                voice_selector=self._get_assigned_voice, # Pass function or we handle mapping here
                source_lang=source_code,  # For cross-lingual detection
                preferred_voice=tts_voice
            )
             
             if generated_path and Path(generated_path).exists() and Path(generated_path).stat().st_size > 100:
                tts_segments.append({
                    'audio_path': generated_path,
                    'start': seg['start'],
                    'end': seg['end']
                })
             
             # Report progress periodically
             if i % 5 == 0:
                 yield ("progress", 0.5 + (0.2 * (i / total_segs)), f"Processed segment {i+1}/{total_segs}")
        
        if not tts_segments:
            raise Exception("TTS Generation failed: No valid segments produced")
        yield ("log", "TTS Generation complete.")

        # 7. Sync & Mix
        self.unload_all_models() 
        yield ("progress", 0.7, "Synchronizing...")
        
        merged_speech = config.TEMP_DIR / f"{video_path.stem}_merged_speech.wav"
        
        import soundfile as sf
        try:
             duration_sec = sf.info(str(extracted_path)).duration
        except:
             if tts_segments:
                 duration_sec = tts_segments[-1]['end'] + 2.0
             else:
                 duration_sec = 10.0
             
        if not self.synchronizer.merge_segments(tts_segments, total_duration=duration_sec, output_path=str(merged_speech), enable_time_stretch=enable_time_stretch):
            raise Exception("Merging speech segments failed")
            
        final_mix = config.TEMP_DIR / f"{video_path.stem}_final_mix.wav"
        self.processor.mix_tracks(str(merged_speech), bg_path, str(final_mix))
        
        if enable_visual_translation:
             yield ("progress", 0.8, "Translating Video Text...")
             self.load_model("visual")
             visual_out = config.TEMP_DIR / f"{video_path.stem}_visual.mp4"
             try:
                 # Pass source and target language for proper text translation
                 self.visual_translator.translate_video_text(
                     str(video_path),
                     str(visual_out),
                     target_lang=target_code,
                     source_lang=source_code,
                     ocr_engine=ocr_model_name
                 )
                 if visual_out.exists():
                     video_path = visual_out
                     yield ("log", "Visual translation complete.")
             except Exception as e:
                logger.error(f"Visual translation failed: {e}")
                
        if enable_lipsync:
            yield ("progress", 0.9, "Lip-Syncing (Experimental)...")
            self.load_model("lipsync")
            
            # Lip-sync generated audio with original video (or re-dubbed video)
            # Actually we usually want to lip sync the translated audio onto the original video faces.
            # But we just replaced the audio. 
            # Flow: Original Video Frames + Final Mixed Audio -> Lip Synced Video
            
            # We use the video with the NEW Audio as input? No, typically MuseTalk takes:
            # - Input Video (Face Source)
            # - Input Audio (Driver)
            # - Output Video
            
            lipsync_out = config.TEMP_DIR / f"{video_path.stem}_lipsync.mp4"
            
            try:
                # We use the original video frames (video_path) and the merged speech (merged_speech) 
                # NOT the final mix with BG music, as that might confuse the model? 
                # Usually purely speech audio is best for driving lips.
                
                self.lipsyncer.sync_lips(str(video_path), str(merged_speech), str(lipsync_out))
                
                if lipsync_out.exists():
                    # Now assume this is the source for final mixing?
                    # The lipsynced video HAS no audio usually, or we mute it.
                    # We need to mix the final audio track back onto this NEW video.
                    video_path = lipsync_out 
                    yield ("log", "Lip-Sync complete.")
            except Exception as e:
                logger.error(f"Lip-Sync failed: {e}")
                yield ("log", f"Lip-Sync failed: {e}. Skipping.")
        
        output_video = config.OUTPUT_DIR / f"translated_{video_path.name}"
        result = self.processor.replace_audio(str(video_path), str(final_mix), str(output_video))
        
        yield ("result", str(result))