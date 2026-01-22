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
from src.processing.voice_enhancement import VoiceEnhancer

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
        self.voice_enhancer = VoiceEnhancer()
        
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

    def _extract_fallback_reference(self, vocals_path):
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
                      enable_audio_enhancement=False,
                      vad_min_silence_duration_ms=1000,
                      transcription_beam_size=5,
                      tts_enable_cfg=False,
                      diarization_model="pyannote/SpeechBrain (Default)",
                      min_speakers=1,
                      max_speakers=10,
                      ocr_model_name="PaddleOCR",
                      tts_voice=None,
                      lipsync_model_name=None):
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
        self._last_valid_reference_wav = None # Track last valid ref for fallback
        
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
        
        # [NEW] Merge short segments for smoother TTS
        logger.info(f"Merging short segments (min_dur=2.0s, max_gap=0.5s)...")
        before_count = len(segments)
        segments = self.transcriber.merge_short_segments(segments, min_duration=2.0, max_gap=0.5)
        logger.info(f"Merged segments: {before_count} -> {len(segments)}")

        yield ("progress", 0.4, "Translating...")
        
        translated_segments = self.translator.translate_segments(
            segments, 
            target_code, 
            model=trans_model_key, 
            source_lang=source_code,
            optimize=optimize_translation
        )
        yield ("log", f"Translation complete for {target_lang}.")
        
        # [NEW] Export Subtitles (SRT)
        try:
            from src.utils.srt_generator import generate_srt
            srt_path = config.OUTPUT_DIR / f"{video_path.stem}.srt"
            # Ensure output dir exists (it should, but safety first)
            config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            
            generate_srt(translated_segments, str(srt_path))
            yield ("log", f"Subtitles exported to {srt_path.name}")
        except Exception as e:
            logger.error(f"Failed to export subtitles: {e}")
            yield ("log", f"Subtitle export failed: {e}")
        
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
                         # No overlap logic
                         use_force_cloning = False
                         speaker_wav = None
                         logger.warning(f"No specific speaker detected for segment {i}.")
             
             # --- [NEW] Reference Voice Fallback Logic ---
             
             # 1. Update tracking if we have a valid speaker_wav
             if speaker_wav:
                 # We assume _resolve_speaker_reference returns None if invalid, 
                 # but double check validation if needed? 
                 # _resolve_speaker_reference already validates against profiles.
                 # If it returned something, it's likely good.
                 if self.tts_engine.validate_reference(speaker_wav, model_name=tts_model_name):
                     self._last_valid_reference_wav = speaker_wav
                 else:
                     speaker_wav = None # It was invalid
             
             # 2. If no valid speaker_wav, try Last Valid
             if not speaker_wav and getattr(self, '_last_valid_reference_wav', None):
                 logger.info(f"Segment {i}: Using LAST VALID reference: {Path(self._last_valid_reference_wav).name}")
                 speaker_wav = self._last_valid_reference_wav
                 # We don't force clone fallbacks, usually partial matches
                 use_force_cloning = False 

             # 3. If still no valid reference, try 0-30s Fallback
             if not speaker_wav:
                 # Try to get or create the 0-30s fallback
                 fallback_30s = self._extract_fallback_reference(vocals_path)
                 if fallback_30s:
                      logger.info(f"Segment {i}: Using 0-30s FALLBACK reference.")
                      speaker_wav = fallback_30s
                      self._last_valid_reference_wav = fallback_30s # Set as new valid
                 else:
                      logger.warning(f"Segment {i}: No reference found (Last or 30s). Will fallback to Edge-TTS.")
                      
             # --------------------------------------------
                     
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
            
        # [NEW] EQ Matching
        if enable_audio_enhancement:
             # We use the 'enable_audio_enhancement' flag for this, or should it be a separate flag?
             # The user request said "EQ Matching (Spectral Matching)". 
             # The plan didn't specify a new flag, but we should probably use one or just do it.
             # "The Fix: Capture the "EQ Curve"... and apply it".
             # Let's apply it by default if vocals exist, or maybe check a new arg?
             # For now, let's tie it to 'enable_audio_enhancement' OR just do it always if we want "Zero computing power" cost?
             # Actually, EQ matching changes the sound character significantly.
             # The prompt implied it's to fix the "studio clean" vs "hall" disconnect.
             # Let's apply it. 
             pass

        # Apply EQ Matching if we have reference vocals
        if vocals_path and Path(vocals_path).exists():
             yield ("log", "Applying EQ matching to match original vocal tone...")
             from src.audio.eq_matching import apply_eq_matching
             
             count = 0
             for tts_seg in tts_segments:
                 out_path = str(tts_seg['audio_path']).replace('.wav', '_eq.wav').replace('.mp3', '_eq.wav')
                 # Use 0.7 strength as a safe default
                 apply_eq_matching(str(vocals_path), str(tts_seg['audio_path']), out_path, strength=0.7)
                 
                 # Update path if success
                 if Path(out_path).exists():
                     tts_seg['audio_path'] = out_path
                     count += 1
             yield ("log", f"Applied EQ matching to {count} segments.")

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
            
        # 7.5. Audio Enhancement (VoiceFixer)
        if enable_audio_enhancement:
             yield ("progress", 0.75, "Enhancing Audio (VoiceFixer)...")
             self.load_model("voice_enhancer")
             enhanced_speech = config.TEMP_DIR / f"{video_path.stem}_enhanced_speech.wav"
             
             try:
                 yield ("log", "Enhancing audio with VoiceFixer...")
                 self.voice_enhancer.enhance_audio(merged_speech, enhanced_speech)
                 if enhanced_speech.exists():
                     merged_speech = enhanced_speech
                     yield ("log", "Audio enhancement complete.")
             except Exception as e:
                 logger.error(f"VoiceFixer failed: {e}. using non-enhanced audio.")
                 yield ("log", f"VoiceFixer failed: {e}. Skipping.")
        
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
                     ocr_engine=ocr_model_name,
                     ocr_interval_sec=10.0
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
                
                yield ("log", f"Starting Lip-Sync... (Model: {lipsync_model_name or 'Default'})")
                enhance_face = "GFPGAN" in (lipsync_model_name or "")
                
                self.lipsyncer.sync_lips(str(video_path), str(merged_speech), str(lipsync_out), enhance_face=enhance_face)
                
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