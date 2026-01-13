import gradio as gr
import os
import shutil
from pathlib import Path
import config
import logging
from video_translator import VideoTranslator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize central controller
video_translator = VideoTranslator()

def process_video(video_path, source_language, target_language, audio_model, tts_model, translation_model, context_model, transcription_model, optimize_translation, enable_diarization, diarization_model, enable_time_stretch, enable_vad, enable_lipsync, enable_visual_translation, progress=gr.Progress()):
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
        
        # [Fix] Copy input video to local temp to avoid PermissionError on Windows
        import uuid
        import time
        import subprocess
        
        time.sleep(1) # Brief delay
        local_video_path = config.TEMP_DIR / f"input_{uuid.uuid4().hex}.mp4"
        
        copied = False
        for attempt in range(5):
            try:
                cmd = f'copy /Y "{str(video_path)}"\t"{str(local_video_path)}"'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0 and local_video_path.exists():
                    copied = True
                    break
            except Exception as e:
                update_log(f"Copy retry {attempt+1}...")
                time.sleep(1)
        
        if not copied:
            try:
                shutil.copyfile(str(video_path), str(local_video_path))
            except:
                return None, update_log("Error: Could not access uploaded file.")

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
            enable_lipsync=enable_lipsync,
            enable_visual_translation=enable_visual_translation
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
                    choices=["pyannote/SpeechBrain (Default)", "NVIDIA NeMo (Advanced)"],
                    label="Diarization Backend",
                    value="pyannote/SpeechBrain (Default)",
                    visible=True 
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
                
                tts_model = gr.Dropdown(
                    choices=["edge", "piper", "xtts", "f5"],
                    label="TTS Model (Edge=Online, Piper=Local, XTTS=Cloning)",
                    value="edge"
                )
                
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

                enable_lipsync = gr.Checkbox(
                    label="Enable Lip-Sync (Generative - Experimental)",
                    value=False,
                    info="Synchronizes video lips to new audio using MuseTalk. computationally expensive."
                )

                enable_visual_translation = gr.Checkbox(
                    label="Enable Visual Text Translation (OCR + Inpainting)",
                    value=False,
                    info="Detects and translates text in the video (e.g. signs, slides). Requires PaddleOCR."
                )

                process_btn = gr.Button("Process Video", variant="primary")
            
            with gr.Column():
                video_output = gr.Video(label="Translated Video")
                logs_output = gr.Textbox(label="Processing Logs", lines=10)
        
        process_btn.click(
            fn=process_video,
            inputs=[video_input, source_language, target_language, audio_model, tts_model, translation_model, context_model, transcription_model, optimize_translation, enable_diarization, diarization_model, enable_time_stretch, enable_vad, enable_lipsync, enable_visual_translation],
            outputs=[video_output, logs_output]
        )
        
    return app

if __name__ == "__main__":
    # Ensure config has language helper
    if not hasattr(config, 'get_language_code'):
        # Add basic mapper dynamically if missing from config
        def get_language_code(name):
             lang_map = {
                "English": "en", "Spanish": "es", "French": "fr", "German": "de",
                "Italian": "it", "Portuguese": "pt", "Polish": "pl", "Turkish": "tr",
                "Russian": "ru", "Dutch": "nl", "Czech": "cs", "Arabic": "ar",
                "Chinese (Simplified)": "zh-cn", "Japanese": "ja", "Korean": "ko",
                "Hindi": "hi"
            }
             return lang_map.get(name, "en")
        config.get_language_code = get_language_code

    demo = create_ui()
    demo.queue() # Enable queueing for progress bars
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
