import gradio as gr
import os
import shutil
from pathlib import Path
from src.utils import config, languages
import logging
from src.core.video_translator import VideoTranslator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize central controller
video_translator = VideoTranslator()

def estimate_remaining_time(progress: float, elapsed_seconds: float) -> str:
    if progress <= 0:
        return "Calculating..."
    total_est = elapsed_seconds / progress
    remaining = total_est - elapsed_seconds
    if remaining < 60:
        return f"~{int(remaining)}s remaining"
    return f"~{int(remaining/60)}m remaining"

def process_video(video_path, source_language, target_language, audio_model, tts_model, translation_model, context_model, transcription_model, optimize_translation, enable_diarization, diarization_model, min_speakers, max_speakers, enable_time_stretch, enable_vad, vad_min_silence, enable_lipsync, lipsync_model, live_portrait_mode, enable_visual_translation, ocr_model, tts_voice, transcription_beam_size, tts_enable_cfg, enable_audio_enhancement, chunk_duration, progress=gr.Progress()):
    """
    Main pipeline entry point.
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"DEBUG: video_path={video_path}")
        logger.debug(f"DEBUG: source_language={source_language}")
        logger.debug(f"DEBUG: target_language={target_language}")
        logger.debug(f"DEBUG: audio_model={audio_model}")
        logger.debug(f"DEBUG: tts_model={tts_model}")
        logger.debug(f"DEBUG: translation_model={translation_model}")
        logger.debug(f"DEBUG: context_model={context_model}")
        logger.debug(f"DEBUG: transcribe_model={transcription_model}")
        logger.debug(f"DEBUG: optimize_translation={optimize_translation}")
        logger.debug(f"DEBUG: enable_diarization={enable_diarization}")
        logger.debug(f"DEBUG: diarization_model={diarization_model}")
        logger.debug(f"DEBUG: min_speakers={min_speakers}")
        logger.debug(f"DEBUG: max_speakers={max_speakers}")
        logger.debug(f"DEBUG: enable_time_stretch={enable_time_stretch}")
        logger.debug(f"DEBUG: enable_vad={enable_vad}")
        logger.debug(f"DEBUG: vad_min_silence={vad_min_silence}")
        logger.debug(f"DEBUG: enable_lipsync={enable_lipsync}")
        logger.debug(f"DEBUG: lipsync_model={lipsync_model}")
        logger.debug(f"DEBUG: live_portrait_mode={live_portrait_mode}")
        logger.debug(f"DEBUG: enable_visual_translation={enable_visual_translation}")
        logger.debug(f"DEBUG: ocr_model={ocr_model}")
        logger.debug(f"DEBUG: tts_voice={tts_voice}")
        logger.debug(f"DEBUG: transcription_beam_size={transcription_beam_size}")
        logger.debug(f"DEBUG: tts_enable_cfg={tts_enable_cfg}")
        logger.debug(f"DEBUG: enable_audio_enhancement={enable_audio_enhancement}")
        logger.debug(f"DEBUG: chunk_duration={chunk_duration}")
    
    if not video_path:
        return None, "Error: No video uploaded."
    
    logs = []
    def update_log(msg):
        logs.append(msg)
        logger.info(msg)
        return "\n".join(logs)

    try:
        # 1. Setup
        video_path = Path(video_path)
        log_msg = update_log(f"Starting pipeline for: {video_path.name}")
        yield None, log_msg
        
        # Ensure token is set from config (loaded from .env)
        if config.HF_TOKEN:
            os.environ["HF_TOKEN"] = config.HF_TOKEN
        
        # [Fix] Copy input video to local temp using shutil (safe for Windows)
        import uuid
        import time
        from src.utils import audio_utils
        
        time.sleep(0.5) # Brief delay
        local_video_path = config.TEMP_DIR / f"input_{uuid.uuid4().hex}.mp4"
        
        copied = False
        for attempt in range(5):
            try:
                shutil.copy2(str(video_path), str(local_video_path))
                if local_video_path.exists():
                    copied = True
                    break
            except Exception as e:
                update_log(f"Copy retry {attempt+1}... ({str(e)})")
                time.sleep(1)
        
        if not copied:
             return None, update_log("Error: Could not access uploaded file (Locked?).")

        video_path = local_video_path
        update_log(f"Created local copy: {local_video_path.name}")
        update_log(f"DEBUG: live_portrait_mode selected in UI: '{live_portrait_mode}'")
        
        # Delegate to VideoTranslator
        # Loop over the generator
        iterator = video_translator.process_video(
            video_path=video_path,
            source_lang=source_language,
            target_lang=target_language,
            audio_model_name=audio_model,
            tts_model_name=tts_model,
            translation_model_name=translation_model,
            context_model_name=context_model,
            transcription_model_name=transcription_model,
            optimize_translation=optimize_translation,
            enable_diarization=enable_diarization,
            diarization_model=diarization_model,
            enable_time_stretch=enable_time_stretch,
            enable_vad=enable_vad,
            vad_min_silence_duration_ms=vad_min_silence,
            enable_lipsync=enable_lipsync,

            enable_visual_translation=enable_visual_translation,
            transcription_beam_size=transcription_beam_size,
            tts_enable_cfg=tts_enable_cfg,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            ocr_model_name=ocr_model,
            tts_voice=tts_voice,
            lipsync_model_name=lipsync_model,
            enable_audio_enhancement=enable_audio_enhancement,
            live_portrait_acceleration=live_portrait_mode,
            chunk_duration=chunk_duration
        )
        
        final_video_path = None
        
        
        start_time = time.time()
        
        for item in iterator:
            msg_type = item[0]
            
            if msg_type == "log":
                log_msg = update_log(item[1])
                yield None, log_msg
                
            elif msg_type == "progress":
                # item = ("progress", val, desc)
                val, desc = item[1], item[2]
                elapsed = time.time() - start_time
                eta = estimate_remaining_time(val, elapsed)
                progress(val, desc=f"{desc} ({int(val*100)}% - {eta})")
                
            elif msg_type == "result":
                final_video_path = item[1]
                
        if final_video_path:
            total_time = time.time() - start_time
            update_log(f"Processing Complete in {int(total_time)}s! Saved to {final_video_path}")
            yield final_video_path, "\n".join(logs)
        else:
            yield None, update_log("Error: Pipeline finished but returned no video.")

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        logger.error(err)
        return None, update_log(f"Critical Error: {str(e)}\n{err}")

def create_ui():
    with gr.Blocks(title="Local AI Video Dubbing & Translation") as app:
        gr.Markdown("## Local AI Video Dubbing & Translation Tool")
        
        with gr.Row():
            with gr.Column():
                # extraction of the video takes place in the backend, but using gr.File prevents
                # the browser/gradio from trying to preview/stream the video which causes file locking errors on Windows
                video_input = gr.File(label="Input Video", file_types=[".mp4", ".avi", ".mov", ".mkv"])
                
                source_language = gr.Dropdown(
                    choices=["Auto Detect", "English", "Spanish", "French", "German", "Italian", "Portuguese", "Polish", "Turkish", "Russian", "Dutch", "Czech", "Arabic", "Chinese", "Japanese", "Korean", "Hindi"],
                    label="Source Language (Optional - forcing helps with noise)",
                    value="Auto Detect"
                )
                 
                target_language = gr.Dropdown(
                    choices=["English", "Spanish", "French", "German", "Italian", "Portuguese", "Polish", "Turkish", "Russian", "Dutch", "Czech", "Arabic", "Chinese (Simplified)", "Japanese", "Korean", "Hindi"],
                    label="Target Language",
                    value="Spanish"
                )
                
                translation_model = gr.Dropdown(
                    choices=[
                        "Google Translate (Online, Fast)", 
                        "Tencent HY-MT1.5 (Local, Better Context)",
                        "Llama 3.1 8B (Local, Instruct)",
                        "ALMA-R 7B (Local, Advanced)"
                    ],
                    label="Translation Model",
                    value="Google Translate (Online, Fast)",
                    visible=True
                )
                
                optimize_translation = gr.Checkbox(
                    label="Optimize Context (Experimental)", 
                    value=False,
                    info="Uses local AI to review and refine translations based on surrounding context. Slower but more accurate."
                )

                context_model = gr.Dropdown(
                    choices=[
                        "Tencent HY-MT1.5 (Local, Better Context)",
                        "Llama 3.1 8B (Local, Instruct)",
                        "ALMA-R 7B (Local, Advanced)"
                    ],
                    label="Context Translation Model",
                    value="Tencent HY-MT1.5 (Local, Better Context)",
                    visible=False,
                    info="Select the LLM to use for context-aware translation."
                )

                chunk_duration = gr.Slider(
                    minimum=60, maximum=900, value=300, step=30,
                    label="Chunk Duration (High RAM Safety)",
                    info="If video length exceeds this (seconds), it will be split processed. Default: 300s (5min)."
                )

                optimize_translation.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[optimize_translation],
                    outputs=[context_model]
                )
                
                audio_model = gr.Dropdown(
                    choices=["Torchaudio HDemucs (Recommended)"],
                    label="Audio Separator Model",
                    value="Torchaudio HDemucs (Recommended)"
                )
                
                enable_diarization = gr.Checkbox(
                    label="Enable Speaker Diarization (Multi-speaker)",
                    value=False,
                    info="Detects speakers and genders to assign appropriate TTS voices. Requires HF_TOKEN."
                )

                diarization_model = gr.Dropdown(
                    choices=["pyannote/SpeechBrain (Default)", "pyannote/Community-1 (Advanced)", "NVIDIA NeMo (Advanced)"],
                    label="Diarization Backend",
                    value="pyannote/SpeechBrain (Default)",
                    visible=True 
                )

                with gr.Row(visible=True) as diarization_params:
                    min_speakers = gr.Slider(1, 10, value=1, step=1, label="Min Speakers")
                    max_speakers = gr.Slider(1, 20, value=1, step=1, label="Max Speakers")

                def update_diarization_visibility(enabled):
                    return gr.update(visible=enabled), gr.update(visible=enabled)

                enable_diarization.change(
                    fn=update_diarization_visibility,
                    inputs=[enable_diarization],
                    outputs=[diarization_model, diarization_params]
                )
                
                transcription_model = gr.Dropdown(
                    choices=[
                        "Large v3 Turbo (Fast)",  # New recommended option
                        "Large v3",               # Best accuracy
                        "Medium", 
                        "Base",
                        "Small"
                    ],
                    label="Speech-to-Text Model",
                    value="Large v3 Turbo (Fast)"
                )

                transcription_beam_size = gr.Slider(
                    minimum=1, maximum=10, step=1, value=7, 
                    label="Beam Size (Accuracy vs Speed)",
                    info="Higher values improve accuracy but slow down transcription. (Default: 7)"
                )
                
                tts_model = gr.Dropdown(
                    choices=["edge", "piper", "xtts", "f5", "vibevoice", "vibevoice-7b"],
                    label="TTS Model (Edge=Online, Piper=Local, XTTS=Cloning, VibeVoice=Long-form)",
                    value="edge"
                )

                tts_enable_cfg = gr.Checkbox(
                    label="Enable CFG (Works with f5-TTS)-Better Quality, Slower)", 
                    value=False,
                    info="Applies Classifier-Free Guidance (scale 1.3) to f5-TTS for more natural speech."
                )

                enable_audio_enhancement = gr.Checkbox(
                    label="Enhance Audio (VoiceFixer)",
                    value=True,
                    info="Uses VoiceFixer to remove noise and restore speech quality in the finale output. (Recommended for Piper/Edge)"
                )
                
                tts_voice = gr.Dropdown(
                    choices=["Auto"],
                    label="TTS Voice (Specific Selection)",
                    value="Auto",
                    info="Select a specific voice to override default selection. (Updates based on Model/Language)"
                )
                
                # Update voices choices when model or language changes
                def update_tts_voices(model, lang_name):
                    try:
                        lang_code = languages.get_language_code(lang_name)
                        voices = video_translator.get_available_tts_voices(model, lang_code)
                        if not voices:
                            choices = ["Auto"]
                        else:
                            choices = ["Auto"] + voices
                        return gr.update(choices=choices, value="Auto")
                    except Exception as e:
                        logger.error(f"Failed to update voices: {e}")
                        return gr.update(choices=["Auto"], value="Auto")

                tts_model.change(update_tts_voices, inputs=[tts_model, target_language], outputs=[tts_voice])
                target_language.change(update_tts_voices, inputs=[tts_model, target_language], outputs=[tts_voice])


                
                enable_time_stretch = gr.Checkbox(
                    label="Enable Time-Stretch (Experimental)",
                    value=False,
                    info="Uses Rubberband/Librosa to stretch TTS audio to match original timing. May cause audio artifacts."
                )
                
                enable_vad = gr.Checkbox(
                    label="Enable VAD Filtering (Advanced)",
                    value=False,
                    info="Uses Voice Activity Detection to filter non-speech audio. May cut words if too aggressive."
                )

                vad_min_silence = gr.Number(
                    label="VAD Min Silence Duration (ms)",
                    value=120000,
                    minimum=0,
                    step=50,
                    visible=False,
                    info="Minimum duration of silence to separate speech segments. Lower values = more segments."
                )

                enable_vad.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[enable_vad],
                    outputs=[vad_min_silence]
                )

                enable_lipsync = gr.Checkbox(
                    label="Enable Lip-Sync (Generative - Experimental)",
                    value=False,
                    info="Synchronizes video lips to new audio using Wav2Lip-GAN."
                )
                
                lipsync_model = gr.Dropdown(
                    choices=["Wav2Lip-GAN (Low Quality - Fast)", "Wav2Lip + GFPGAN (Low Quality - Medium)", "LivePortrait (High Quality - Slow)"],
                    label="Lip-Sync Model",
                    value="Wav2Lip-GAN (Low Quality - Fast)",
                    visible=False,
                    info="Select 'LivePortrait' for the best visual quality."
                )

                enable_lipsync.change(
                     fn=lambda x: gr.update(visible=x),
                     inputs=[enable_lipsync],
                     outputs=[lipsync_model]
                )
                
                live_portrait_mode = gr.Dropdown(
                    choices=["ort", "tensorrt"],
                    label="LivePortrait Acceleration",
                    value="tensorrt",
                    visible=True,
                    info="Use 'tensorrt' for GPU acceleration (Requires Setup). 'ort' is standard."
                )
                
                def update_lp_mode_visibility(model_name):
                    visible = "LivePortrait" in (model_name or "")
                    return gr.update(visible=visible)

                lipsync_model.change(
                    fn=update_lp_mode_visibility,
                    inputs=[lipsync_model],
                    outputs=[live_portrait_mode]
                )
                


                # Lip-Sync Model hidden/removed (Defaulting to Wav2Lip-GAN internal)
                # enable_lipsync handles valid toggle

                enable_visual_translation = gr.Checkbox(
                    label="Enable Visual Text Translation (OCR + Inpainting)",
                    value=False,
                    info="Detects and translates text in the video (e.g. signs, slides)."
                )
                
                ocr_model = gr.Dropdown(
                    choices=["PaddleOCR", "EasyOCR"],
                    label="OCR Model",
                    value="EasyOCR",
                    visible=False,
                    info="Select OCR engine. EasyOCR is more robust on Windows. PaddleOCR is faster."
                )
                
                enable_visual_translation.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[enable_visual_translation],
                    outputs=[ocr_model]
                )

                with gr.Row():
                    process_btn = gr.Button("Process Video", variant="primary", scale=3)
                    cancel_btn = gr.Button("Cancel", variant="stop", scale=1)
            
            with gr.Column():
                video_output = gr.Video(label="Translated Video")
                logs_output = gr.Textbox(label="Processing Logs", lines=10)
        
        process_event = process_btn.click(
            fn=process_video,
            inputs=[video_input, source_language, target_language, audio_model, tts_model, translation_model, context_model, transcription_model, optimize_translation, enable_diarization, diarization_model, min_speakers, max_speakers, enable_time_stretch, enable_vad, vad_min_silence, enable_lipsync, lipsync_model, live_portrait_mode, enable_visual_translation, ocr_model, tts_voice, transcription_beam_size, tts_enable_cfg, enable_audio_enhancement, chunk_duration],
            outputs=[video_output, logs_output]
        )
        
        def abort_processing():
            """Signals the translator to stop."""
            logger.info("User requested cancellation.")
            video_translator.abort()
            return "Cancellation requested..."
        
        cancel_btn.click(
            fn=abort_processing, 
            inputs=None, 
            outputs=[logs_output], # Optional: update log to say "Cancelling..."
            cancels=[process_event]
        )
        
    return app

if __name__ == "__main__":
    # Config check passed
    pass

    demo = create_ui()
    demo.queue() # Enable queueing for progress bars
    demo.launch(server_name="127.0.0.1", share=False)
