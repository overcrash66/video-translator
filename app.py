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

def process_video(video_path, source_language, target_language, audio_model, tts_model, translation_model, context_model, transcription_model, optimize_translation, enable_diarization, diarization_model, min_speakers, max_speakers, enable_time_stretch, enable_vad, vad_min_silence, enable_lipsync, lipsync_model, enable_visual_translation, ocr_model, tts_voice, transcription_beam_size, tts_enable_cfg, enable_audio_enhancement, progress=gr.Progress()):
    """
    Main pipeline entry point.
    """
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
            enable_audio_enhancement=enable_audio_enhancement
        )
        
        final_video_path = None
        
        for item in iterator:
            msg_type = item[0]
            
            if msg_type == "log":
                log_msg = update_log(item[1])
                yield None, log_msg
                
            elif msg_type == "progress":
                # item = ("progress", val, desc)
                progress(item[1], desc=item[2])
                
            elif msg_type == "result":
                final_video_path = item[1]
                
        if final_video_path:
            update_log(f"Processing Complete! Saved to {final_video_path}")
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
                    max_speakers = gr.Slider(1, 20, value=5, step=1, label="Max Speakers")

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
                    minimum=1, maximum=10, step=1, value=5, 
                    label="Beam Size (Accuracy vs Speed)",
                    info="Higher values improve accuracy but slow down transcription. (Default: 5)"
                )
                
                tts_model = gr.Dropdown(
                    choices=["edge", "piper", "xtts", "f5"],
                    label="TTS Model (Edge=Online, Piper=Local, XTTS=Cloning)",
                    value="edge"
                )

                tts_enable_cfg = gr.Checkbox(
                    label="Enable CFG (Works with f5-TTS)-Better Quality, Slower)", 
                    value=False,
                    info="Applies Classifier-Free Guidance (scale 1.3) to f5-TTS for more natural speech."
                )

                enable_audio_enhancement = gr.Checkbox(
                    label="Enhance Audio (VoiceFixer)",
                    value=False,
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
                    value=1000,
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
                    choices=["Wav2Lip-GAN (Fast)", "Wav2Lip + GFPGAN (High Quality)"],
                    label="Lip-Sync Model",
                    value="Wav2Lip + GFPGAN (High Quality)",
                    visible=False,
                    info="Select 'Wav2Lip + GFPGAN' to drastically improve face quality (slower)."
                )

                enable_lipsync.change(
                     fn=lambda x: gr.update(visible=x),
                     inputs=[enable_lipsync],
                     outputs=[lipsync_model]
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
                    value="PaddleOCR",
                    visible=False,
                    info="Select OCR engine. EasyOCR is more robust on Windows. PaddleOCR is faster."
                )
                
                enable_visual_translation.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[enable_visual_translation],
                    outputs=[ocr_model]
                )

                process_btn = gr.Button("Process Video", variant="primary")
            
            with gr.Column():
                video_output = gr.Video(label="Translated Video")
                logs_output = gr.Textbox(label="Processing Logs", lines=10)
        
        process_btn.click(
            fn=process_video,
            inputs=[video_input, source_language, target_language, audio_model, tts_model, translation_model, context_model, transcription_model, optimize_translation, enable_diarization, diarization_model, min_speakers, max_speakers, enable_time_stretch, enable_vad, vad_min_silence, enable_lipsync, lipsync_model, enable_visual_translation, ocr_model, tts_voice, transcription_beam_size, tts_enable_cfg, enable_audio_enhancement],
            outputs=[video_output, logs_output]
        )
        
    return app

if __name__ == "__main__":
    # Config check passed
    pass

    demo = create_ui()
    demo.queue() # Enable queueing for progress bars
    demo.launch(server_name="127.0.0.1", share=False)
