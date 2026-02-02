import logging
import soundfile as sf
# Removed global torch/torchaudio imports to prevent early loading conflicts with CTranslate2
# Removed ctranslate2 import - handled properly in config.py now

logger = logging.getLogger(__name__)

def _patch_windows_encoding():
    """
    On Windows, open() defaults to cp1252. This breaks many ML libraries (like transformers) 
    that expect UTF-8. We monkey-patch open to default to UTF-8.
    """
    import sys
    if sys.platform != "win32":
        return

    import builtins
    _original_open = builtins.open

    def _utf8_open(*args, **kwargs):
        # Inspect mode to see if it's binary
        mode = 'r' # default
        if 'mode' in kwargs:
            mode = kwargs['mode']
        elif len(args) > 1:
            mode = args[1]
            
        # Only enforce UTF-8 if not in binary mode and encoding not specified
        if 'b' not in mode and 'encoding' not in kwargs:
            kwargs['encoding'] = 'utf-8'
            
        return _original_open(*args, **kwargs)

    builtins.open = _utf8_open
    logger.info("Monkey-patched builtins.open to force UTF-8 encoding on Windows.")

def _patch_torchaudio_load():
    """
    Enforce soundfile backend to avoid TorchCodec errors on Windows via monkey-patching.
    Post-2.x torchaudio on Windows is unstable with backends. We bypass it using soundfile directly.
    """
    # Import torch/torchaudio only when needed (LATE patch)
    import torch
    import torchaudio

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
                pass 
                
            return tensor, samplerate
            
        torchaudio.load = _safe_load
        logger.info("Monkey-patched torchaudio.load to use soundfile library directly.")
    except Exception as e:
        logger.warning(f"Failed to patch torchaudio.load: {e}")

def _patch_torchaudio_audiometadata():
    """
    Inject missing AudioMetaData class into torchaudio to satisfy pyannote.audio requirements.
    This class was removed in torchaudio 2.x but is still imported by pyannote.audio 3.x.
    """
    import torchaudio
    
    if hasattr(torchaudio, "AudioMetaData"):
        return

    # Define the missing class based on what pyannote expects
    class AudioMetaData:
        def __init__(self, sample_rate, num_frames, num_channels, bits_per_sample, encoding):
            self.sample_rate = sample_rate
            self.num_frames = num_frames
            self.num_channels = num_channels
            self.bits_per_sample = bits_per_sample
            self.encoding = encoding

    # Inject it into the module
    torchaudio.AudioMetaData = AudioMetaData
    logger.info("Monkey-patched torchaudio.AudioMetaData for pyannote compatibility.")


def _patch_transformers_qwen2_tokenizer():
    """
    Create compatibility shim for vibevoice's usage of internal transformers path.
    """
    import sys
    import types
    
    try:
        # Check if the problematic path exists
        from transformers.models.qwen2 import tokenization_qwen2_fast
        return  # Already exists, nothing to do
    except ImportError:
        pass
    
    try:
        # Import the working top-level class
        from transformers import Qwen2TokenizerFast
        
        # Create a fake module
        fake_module = types.ModuleType('transformers.models.qwen2.tokenization_qwen2_fast')
        fake_module.Qwen2TokenizerFast = Qwen2TokenizerFast
        
        # Register it in sys.modules so future imports work
        sys.modules['transformers.models.qwen2.tokenization_qwen2_fast'] = fake_module
        
        logger.info("Created compatibility shim for transformers.models.qwen2.tokenization_qwen2_fast")
    except ImportError as e:
        logger.warning(f"Could not create Qwen2 tokenizer shim: {e}")

def apply_encoding_patch():
    """
    Applies ONLY the Windows encoding fix.
    Must be run before any file I/O (e.g. dotenv loading in config).
    """
    _patch_windows_encoding()

def apply_transformers_patch():
    """
    Applies Transformers-related patches.
    Should be run AFTER config (and ctranslate2 setup) but BEFORE heavy usage.
    """
    _patch_transformers_qwen2_tokenizer()

def apply_early_patches():
    """
    Legacy method.
    """
    apply_encoding_patch()
    apply_transformers_patch()

def apply_late_patches():
    """
    Applies patches that require torch or other heavy libs.
    Should be called AFTER config setup.
    Includes: Torchaudio fix.
    """
    _patch_torchaudio_load()
    _patch_torchaudio_audiometadata()

def apply_patches():
    """
    Legacy entry point.
    """
    apply_early_patches()
    apply_late_patches()

