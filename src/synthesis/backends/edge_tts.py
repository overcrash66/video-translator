import logging
import asyncio
import edge_tts
import time
from pathlib import Path
from src.utils import config
from src.synthesis.backends.base import TTSBackend

logger = logging.getLogger(__name__)

class EdgeTTSBackend(TTSBackend):
    def __init__(self, voice_map):
        self.voice_map = voice_map

    def generate(self, text, output_path, language="en", speaker_wav=None, **kwargs):
        """
        Generates audio using Edge-TTS.
        kwargs can contain: gender, speaker_id, voice_selector, preferred_voice
        """
        gender = kwargs.get("gender", "Female")
        speaker_id = kwargs.get("speaker_id")
        voice_selector = kwargs.get("voice_selector")
        preferred_voice = kwargs.get("preferred_voice")

        # Get voice list for language and gender
        opts = self.voice_map.get(language, self.voice_map.get("en", {}))
        # Handle case where language entry exists but gender key is missing
        # Default to Female or first available if distinct structure
        # The map structure is {lang: {Gender: [voices]}}
        voice_list = opts.get(gender, opts.get("Female", ["en-US-AriaNeural"]))
        
        # Handle legacy single-voice format (backward compatibility)
        if isinstance(voice_list, str):
            voice_list = [voice_list]
        
        # Select voice
        voice = voice_list[0]
        
        if preferred_voice and preferred_voice != "Auto":
            voice = preferred_voice
            logger.info(f"Using preferred voice: {voice}")
        elif speaker_id:
            # deterministic voice selection via callback
            if voice_selector:
                voice = voice_selector(speaker_id, voice_list)
                logger.info(f"Selected voice via callback for {speaker_id}: {voice}")
            else:
                # Fallback to modulo
                try:
                    speaker_num = int(speaker_id.split("_")[-1])
                    voice_index = speaker_num % len(voice_list)
                    voice = voice_list[voice_index]
                except (ValueError, IndexError):
                    voice = voice_list[0]
        
        logger.info(f"Generating TTS (Edge): lang='{language}', voice='{voice}', text='{text[:50]}...'")
        
        
        # Retry logic
        max_retries = 3
        last_error = None
        voice_index = 0
        
        for attempt in range(max_retries):
            try:
                # Check for empty text again just in case
                if not text or not text.strip():
                     return None

                async def _gen():
                    communicate = edge_tts.Communicate(text, voice)
                    await communicate.save(str(output_path))
                
                asyncio.run(_gen())
                
                if Path(output_path).exists() and Path(output_path).stat().st_size > config.TTS_MIN_AUDIO_SIZE:
                    return output_path
                else:
                    raise RuntimeError("Edge-TTS produced invalid/empty file")
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Edge-TTS attempt {attempt + 1}/{max_retries} failed: {e}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying Edge-TTS in {wait_time}s...")
                    time.sleep(wait_time)
                    
                    # Try alternate voice on retry
                    if len(voice_list) > 1:
                        voice_index = (voice_index + 1) % len(voice_list)
                        voice = voice_list[voice_index]
                        logger.info(f"Switching to alternate voice: {voice}")
        
        raise RuntimeError(f"Edge-TTS failed after {max_retries} attempts: {last_error}")

    def generate_batch(self, tasks):
        """
        Batch generation using async parallelism.
        
        :param tasks: List of task dicts with keys: text, output_path, language, gender, speaker_id, preferred_voice, voice_selector
        :return: List of output paths (or None for failures)
        """
        MAX_RETRIES = 3
        
        async def _process_single(task):
            """Helper to process one task with retries."""
            text = task.get('text')
            out_path = task.get('output_path')
            
            # Extract task-specific overrides or use kwarg defaults from batch context?
            # Actually generate_batch receives full kwargs per task.
            # But we need voice logic too. 
            # We assume 'voice' is already resolved in task or we resolve it here.
            # The previous 'generate' resolved voice. We should likely reuse that logic 
            # or expect the caller (Facade) to have pre-resolved it?
            # Facade doesn't resolve voice, Backend does.
            # So we must resolve voice here.
            
            # This is complex because 'generate' does voice resolution.
            # We should probably refactor 'generate' to 'resolve_voice' + 'synthesize'.
            # For now, let's just duplicate the resolution logic or (better) let's wrap logic.
            
            # WAIT. 'generate' takes kwargs like 'gender', 'speaker_id'.
            # We can't easily reuse 'generate' because it's sync and calls run_until_complete.
            
            # Strategy: Resolve voice for each task first (fast, sync).
            # Then parallelize the IO.
            
            lang = task.get('language', 'en')
            gen = task.get('gender', 'Female')
            spk_id = task.get('speaker_id')
            pref_voice = task.get('preferred_voice')
            v_sel = task.get('voice_selector')
            
            # Resolve Voice (Sync)
            opts = self.voice_map.get(lang, self.voice_map.get("en", {}))
            v_list = opts.get(gen, opts.get("Female", ["en-US-AriaNeural"]))
            if isinstance(v_list, str): v_list = [v_list]
            
            voice = v_list[0]
            if pref_voice and pref_voice != "Auto":
                voice = pref_voice
            elif spk_id:
                if v_sel:
                     voice = v_sel(spk_id, v_list)
                else:
                     try:
                         s_num = int(spk_id.split("_")[-1])
                         idx = s_num % len(v_list)
                         voice = v_list[idx]
                     except: pass

            # Async Generation
            last_err = None
            for i in range(MAX_RETRIES):
                try:
                    comm = edge_tts.Communicate(text, voice)
                    await comm.save(str(out_path))
                    
                    if Path(out_path).exists() and Path(out_path).stat().st_size > config.TTS_MIN_AUDIO_SIZE:
                        return str(out_path)
                except Exception as e:
                    last_err = e
                    if i < MAX_RETRIES - 1:
                        await asyncio.sleep(2 ** i)
            
            logger.warning(f"Batch item failed: {last_err}")
            return None # Signal failure

        # Run Batch
        async def _run_batch():
            coroutines = [_process_single(t) for t in tasks]
            return await asyncio.gather(*coroutines)
            
        return asyncio.run(_run_batch())

