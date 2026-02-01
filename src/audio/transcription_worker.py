
import sys
import os
import json
import logging
import traceback
import argparse
from pathlib import Path

# Setup Path to find src modules
sys.path.append(os.getcwd())

# Configure minimal logging to stderr (so stdout is clean for JSON)
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='[Worker] %(message)s')
logger = logging.getLogger("TranscriptionWorker")

def run_transcription(audio_path, model_size, language, compute_type, device):
    """
    Runs transcription using faster-whisper in isolation.
    """
    try:
        from src.utils import config
        # Ensure DLLs are added
        config.setup_nvidia_dlls()
        
        from faster_whisper import WhisperModel
        
        logger.info(f"Loading Whisper {model_size} on {device} ({compute_type})...")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        
        logger.info(f"Transcribing {audio_path}...")
        segments_gen, info = model.transcribe(
            audio_path, 
            language=language if language != "auto" else None,
            beam_size=5,
            word_timestamps=True
        )
        
        segments = []
        for col in segments_gen:
            segments.append({
                "start": col.start,
                "end": col.end,
                "text": col.text,
                "words": [{"word": w.word, "start": w.start, "end": w.end, "probability": w.probability} for w in col.words] if col.words else []
            })
            
        result = {
            "segments": segments,
            "language": info.language,
            "language_probability": info.language_probability
        }
        
        # Print JSON result to STDOUT with delimiters
        json_str = json.dumps(result)
        print(f"<<<<JSON>>>>\n{json_str}\n<<<<ENDJSON>>>>")
        
    except Exception as e:
        logger.error(f"Worker Failed: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", required=True)
    parser.add_argument("--model_size", required=True)
    parser.add_argument("--language", default="auto")
    parser.add_argument("--compute_type", default="float16")
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    run_transcription(
        args.audio_path,
        args.model_size,
        args.language,
        args.compute_type,
        args.device
    )
