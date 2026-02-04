
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

# CUDA error patterns to detect GPU-specific failures
CUDA_ERROR_PATTERNS = [
    'cuda', 'gpu', 'out of memory', 'oom', 'cudnn', 'cublas',
    'illegal memory', 'device-side assert', 'cusparse', 'cufft',
    'nccl', 'nvrtc', 'curand'
]


def _is_cuda_error(error: Exception) -> bool:
    """Check if an exception is CUDA-related."""
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()
    return any(pattern in error_str or pattern in error_type for pattern in CUDA_ERROR_PATTERNS)


def run_transcription(audio_path, model_size, language, compute_type, device):
    """
    Runs transcription using faster-whisper in isolation.
    
    Exit codes:
        0 - Success
        1 - General error  
        2 - CUDA-specific error (triggers CPU fallback in parent)
    """
    try:
        from src.utils import config
        # Ensure DLLs are added
        config.setup_nvidia_dlls()
        
        from faster_whisper import WhisperModel
        
        # Log GPU memory if available (helps debug OOM)
        if device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated() / 1024**3
                    mem_reserved = torch.cuda.memory_reserved() / 1024**3
                    logger.info(f"GPU Memory before load: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
            except Exception:
                pass
        
        logger.info(f"Loading Whisper {model_size} on {device} ({compute_type})...")
        
        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception as load_error:
            if _is_cuda_error(load_error):
                logger.error(f"CUDA error loading model: {load_error}")
                sys.exit(2)  # Signal CUDA-specific failure
            raise
        
        logger.info(f"Transcribing {audio_path}...")
        
        try:
            segments_gen, info = model.transcribe(
                audio_path, 
                language=language if language != "auto" else None,
                beam_size=5,
                word_timestamps=True
            )
            
            # Process segments
            segments = []
            for col in segments_gen:
                segments.append({
                    "start": col.start,
                    "end": col.end,
                    "text": col.text,
                    "words": [{"word": w.word, "start": w.start, "end": w.end, "probability": w.probability} for w in col.words] if col.words else []
                })
                
        except Exception as transcribe_error:
            if _is_cuda_error(transcribe_error):
                logger.error(f"CUDA error during transcription: {transcribe_error}")
                sys.exit(2)  # Signal CUDA-specific failure
            raise
            
        result = {
            "segments": segments,
            "language": info.language,
            "language_probability": info.language_probability
        }
        
        # Print JSON result to STDOUT with delimiters
        json_str = json.dumps(result)
        print(f"<<<<JSON>>>>\n{json_str}\n<<<<ENDJSON>>>>")
        
    except SystemExit:
        # Re-raise SystemExit so exit codes are preserved
        raise
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
