import torch
import torchaudio
import soundfile as sf
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

def load_audio(file_path: str, target_sr: int = None, mono: bool = False):
    """
    Safely loads audio using soundfile backend to avoid Windows TorchCodec issues.
    Matches torchaudio.load return signature: (waveform, sample_rate)
    
    Args:
        file_path: Path to audio file
        target_sr: Optional, resample to this rate
        mono: Optional, convert to mono if True
        
    Returns:
        (waveform, sample_rate) where waveform is [Channels, Time] tensor
    """
    file_path = str(file_path)
    
    try:
        # Force float32 to match default PyTorch weights
        data, samplerate = sf.read(file_path, dtype='float32')
        
        # Handle empty files
        if len(data) == 0:
            raise ValueError(f"Audio file is empty: {file_path}")

        # specific fix for soundfile returning [Time, Channels] or [Time]
        # PyTorch expects [Channels, Time]
        tensor = torch.from_numpy(data)
        
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0) # (1, frames)
        else:
            tensor = tensor.transpose(0, 1) # (frames, channels)
            
        # Convert to mono if requested
        if mono and tensor.shape[0] > 1:
            tensor = tensor.mean(dim=0, keepdim=True)
            
        # Resample if requested
        if target_sr is not None and samplerate != target_sr:
            resampler = torchaudio.transforms.Resample(samplerate, target_sr)
            tensor = resampler(tensor)
            samplerate = target_sr
            
        return tensor, samplerate
        
    except Exception as e:
        logger.error(f"Failed to load audio {file_path}: {e}")
        raise e

def save_audio(file_path: str, waveform: torch.Tensor, sample_rate: int):
    """
    Safely saves audio using soundfile.
    waveform: [Channels, Time] tensor
    """
    try:
        # Convert to [Time, Channels] numpy array
        if waveform.ndim == 2:
            data = waveform.cpu().numpy().T
        elif waveform.ndim == 1:
            data = waveform.cpu().numpy()
        else:
            raise ValueError(f"Invalid waveform shape: {waveform.shape}")
            
        sf.write(str(file_path), data, sample_rate)
        
    except Exception as e:
        logger.error(f"Failed to save audio {file_path}: {e}")
        raise e
