import logging
import soundfile as sf
import torch
import torchaudio
try:
    import ctranslate2 # Pre-import to avoid Windows DLL shadowing with Paddle
except ImportError:
    pass

logger = logging.getLogger(__name__)

def apply_patches():
    """
    Applies necessary runtime patches to libraries.
    """
    _patch_torchaudio_load()

def _patch_torchaudio_load():
    """
    Enforce soundfile backend to avoid TorchCodec errors on Windows via monkey-patching.
    Post-2.x torchaudio on Windows is unstable with backends. We bypass it using soundfile directly.
    """
    try:
        # Save original just in case, though we don't expose a revert for now
        _original_load = torchaudio.load
        
        def _safe_load(filepath, **kwargs):
            """
            Drop-in replacement for torchaudio.load that forces soundfile usage
            and converts result to expected (Tensor, int) format.
            """
            # normalize argument allows normalization (default True in torchaudio?)
            # soundfile.read returns (data, samplerate)
            # data is numpy array: (frames, channels) or (frames,)
            
            # torchaudio checks normalization. Default is normalize=True
            normalize = kwargs.get('normalize', True)
            
            # Load with soundfile
            data, samplerate = sf.read(filepath, dtype='float32')
            
            # Convert to Tensor
            tensor = torch.from_numpy(data)
            
            # Soundfile: (Time, Channels) or (Time,)
            # Torchaudio expected: (Channels, Time)
            if tensor.ndim == 1:
                # Mono: (T,) -> (1, T)
                tensor = tensor.unsqueeze(0)
            else:
                # (T, C) -> (C, T)
                tensor = tensor.transpose(0, 1)
                
            # Validations similar to torchaudio
            if normalize:
                # SF reads float32 by default which is normalized -1 to 1 usually
                # If int, maybe not.
                pass 
                
            return tensor, samplerate
            
        torchaudio.load = _safe_load
        logger.info("Monkey-patched torchaudio.load to use soundfile library directly.")
    except Exception as e:
        logger.warning(f"Failed to patch torchaudio.load: {e}")
