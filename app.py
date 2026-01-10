import gradio as gr
import os
import shutil
from pathlib import Path
import config
import logging
from audio_separator import AudioSeparator
from transcriber import Transcriber
from translator import Translator
from tts_engine import TTSEngine
from synchronizer import AudioSynchronizer
from video_processor import VideoProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize components globally to avoid reloading models on every request?
# Or lazy load inside function. Lazy load is better for memory if single user.
# But for Gradio app, global is standard.
# However, we'll instantiate inside to keep state clean for this prototype, 
# relying on the classes' internal checks to avoid re-loading if they were global.
# Actually, the classes I wrote allow re-use. Let's make them global.

separator = AudioSeparator()
transcriber = Transcriber()
translator = Translator()
tts_engine = TTSEngine()
synchronizer = AudioSynchronizer()
processor = VideoProcessor()

def process_video(video_path, target_language, audio_model, progress=gr.Progress()):
    """
    Main pipeline entry point.
    """
    if not video_path:
        return None, "Error: No video uploaded."
    
    logs = []
    def log(msg):
        logs.append(msg)
        logger.info(msg)
        return "\n".join(logs)

    try:
        # 1. Setup
        video_path = Path(video_path)
        log_msg = log(f"Starting pipeline for: {video_path.name}")
        yield None, log_msg
        
        # Ensure token is set from config (loaded from .env)
        if config.HF_TOKEN:
            os.environ["HF_TOKEN"] = config.HF_TOKEN
        else:
             log("Warning: HF_TOKEN not found in environment. Model loading might fail if it requires authentication.")
        
        # [Fix] Copy input video to local temp to avoid PermissionError on Windows (file locking)
        # Use a simple ASCII name to avoid FFMPEG unicode issues.
        import uuid
        import time
        import shutil
        
        # Brief delay to let Gradio finish any initial file serving/locking (race condition mitigation)
        time.sleep(2)
        
        local_video_path = config.TEMP_DIR / f"input_{uuid.uuid4().hex}.mp4"
        
        # Retry loop for copying file
        copied = False
        import subprocess
        
        for attempt in range(5):
            try:
                # Use Windows system copy command which is often more robust with locks/sharing
                # strict=False allows fallback if copy fails but shell=True handles it
                # We use specific quoting for the paths
                cmd = f'copy /Y "{str(video_path)}"\t"{str(local_video_path)}"'
                # Note: f-string formatting handles the unicode string interpolation into the command
                
                # Execute copy command
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode == 0 and local_video_path.exists():
                    copied = True
                    break
                else:
                    log(f"System copy failed (code {result.returncode}): {result.stderr or result.stdout}")
                    raise PermissionError("Copy command failed")

            except Exception as e:
                log(f"File locked by Gradio/System, retrying copy ({attempt+1}/5)... Error: {str(e)[:100]}")
                time.sleep(2)
        
        if not copied:
            # Final fallback: Try Python generic copy as last resort
            try:
                 shutil.copyfile(str(video_path), str(local_video_path))
                 copied = True
            except:
                 return None, log("Error: Could not access uploaded file even after system retries. Please rename the file to something simple (A-Z) and try again.")

        video_path = local_video_path # Switch referencing variable to local copy
        


        video_path = local_video_path # Switch referencing variable to local copy
        log(f"Created local copy: {local_video_path.name}")

        # 2. Extract Audio
        progress(0.1, desc="Extracting Audio...")
        full_audio = config.TEMP_DIR / f"{video_path.stem}_full.wav"
        extracted_path = processor.extract_audio(str(video_path), str(full_audio))
        if not extracted_path:
            return None, log("Failed to extract audio.")
        log_msg = log("Audio extracted.")
        yield None, log_msg

        # 3. Separate Audio (Demucs)
        progress(0.2, desc=f"Separating Vocals with {audio_model}...")
        try:
            vocals_path, bg_path = separator.separate(extracted_path, model_selection=audio_model)
        except Exception as e:
            raise gr.Error(f"Separation failed: {e}")

        if not Path(vocals_path).exists() or not Path(bg_path).exists():
            raise gr.Error("Separation failed: Output files not found.")
            
        log_msg = log(f"Separation complete. \nVocals: {Path(vocals_path).name}\nBackground: {Path(bg_path).name}")
        yield None, log_msg

        # 4. Transcription
        progress(0.3, desc="Transcribing...")
        try:
            segments = transcriber.transcribe(vocals_path)
        except Exception as e:
            raise gr.Error(f"Transcription failed: {e}")

        # Validation: Transcription
        if not segments:
            raise gr.Error("Transcription returned no segments. Is the audio silent or unintelligible?")

        log_msg = log(f"Transcription complete. {len(segments)} segments found.")
        transcriber.unload_model() # Optimization
        yield None, log_msg

        # 5. Translate Segments
        progress(0.4, desc="Translating...")
        # Map friendly name to code
        target_code = config.get_language_code(target_language)
        try:
            translated_segments = translator.translate_segments(segments, target_code)
        except Exception as e:
             raise gr.Error(f"Translation failed: {e}")

        if not translated_segments:
             raise gr.Error("Translation return empty list.")

        log_msg = log(f"Translation complete for {target_language} ({target_code}).")
        yield None, log_msg

        # 6. Generate Translated Speech (TTS)
        progress(0.5, desc="Generating Speech...")
        
        # Create a temp dir for this run segments
        seg_dir = config.TEMP_DIR / video_path.stem
        seg_dir.mkdir(exist_ok=True)
        
        tts_segments = []
        
        for i, seg in enumerate(translated_segments):
            text = seg["translated_text"]
            if not text:
                 continue
                 
            seg_out = seg_dir / f"seg_{i}_tts.mp3" # edge-tts mp3
            generated_path = tts_engine.generate_audio(text, vocals_path, language=target_code, output_path=seg_out)
            
            # Validation: Individual TTS
            if not generated_path or not Path(generated_path).exists():
                log(f"Warning: TTS failed for segment {i}")
                continue
                
            tts_segments.append({
                'audio_path': generated_path,
                'start': seg['start'],
                'end': seg['end']
            })
            progress(0.5 + (0.2 * (i / len(translated_segments))), desc=f"Processed segment {i+1}/{len(translated_segments)}")
        
        if not tts_segments:
            raise gr.Error("TTS failed to generate any audio segments.")

        log_msg = log("TTS Generation complete.")
        yield None, log_msg

        # 7. Synchronize & Merge Speech
        progress(0.7, desc="Synchronizing...")
        merged_speech = config.TEMP_DIR / f"{video_path.stem}_merged_speech.wav"
        
        # Get total duration
        try:
             import soundfile as sf
             info = sf.info(str(extracted_path))
             duration_sec = info.duration
        except Exception as e:
             logger.warning(f"Could not read total duration: {e}")
             # Fallback to last segment end + small buffer
             if tts_segments:
                 duration_sec = tts_segments[-1]['end'] + 2.0
             else:
                 duration_sec = 10.0 # Should not happen given prev checks
             
        if not synchronizer.merge_segments(tts_segments, total_duration=duration_sec, output_path=str(merged_speech)):
             raise gr.Error("Merging speech segments failed.")
        yield None, log_msg

        # 8. Mixing
        progress(0.9, desc="Mixing Final Audio...")
        final_mix = config.TEMP_DIR / f"{video_path.stem}_final_mix.wav"
        # Determine background volume - if separate failed (dummy), background might be silent or full.
        # If dummy, vocals = full, background = silent.
        # If real, background is instrumental.
        processor.mix_tracks(str(merged_speech), bg_path, str(final_mix))
        
        # 9. Final Mux
        output_video = config.OUTPUT_DIR / f"translated_{video_path.name}"
        processor.replace_audio(str(video_path), str(final_mix), str(output_video))
        
        log_msg = log(f"Processing Complete! Saved to {output_video}")
        yield str(output_video), log_msg

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        logger.error(err)
        return None, log(f"Critical Error: {str(e)}\n{err}")

def create_ui():
    with gr.Blocks(title="Local AI Video Dubbing & Translation") as app:
        gr.Markdown("## Local AI Video Dubbing & Translation Tool")
        
        with gr.Row():
            with gr.Column():
                # extraction of the video takes place in the backend, but using gr.File prevents
                # the browser/gradio from trying to preview/stream the video which causes file locking errors on Windows
                video_input = gr.File(label="Input Video", file_types=[".mp4", ".avi", ".mov", ".mkv"]) 
                target_language = gr.Dropdown(
                    choices=["English", "Spanish", "French", "German", "Italian", "Portuguese", "Polish", "Turkish", "Russian", "Dutch", "Czech", "Arabic", "Chinese (Simplified)", "Japanese", "Korean", "Hindi"],
                    label="Target Language",
                    value="Spanish"
                )
                
                audio_model = gr.Dropdown(
                    choices=["Torchaudio HDemucs (Recommended)"],
                    label="Audio Separator Model",
                    value="Torchaudio HDemucs (Recommended)"
                )

                process_btn = gr.Button("Process Video", variant="primary")
            
            with gr.Column():
                video_output = gr.Video(label="Translated Video")
                logs_output = gr.Textbox(label="Processing Logs", lines=10)
        
        process_btn.click(
            fn=process_video,
            inputs=[video_input, target_language, audio_model],
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
