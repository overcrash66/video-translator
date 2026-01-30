import logging
import soundfile as sf
import torch
import torchaudio
try:
    import ctranslate2 # Pre-import to avoid Windows DLL shadowing with Paddle
except ImportError:
    pass

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

def _patch_transformers_qwen2_tokenizer():
    """
    Create compatibility shim for vibevoice's usage of internal transformers path.
    VibeVoice imports from 'transformers.models.qwen2.tokenization_qwen2_fast'
    which doesn't exist in transformers 5.x. We create a fake module with the right export.
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

def apply_patches():
    """
    Applies necessary runtime patches to libraries.
    """
    _patch_windows_encoding() # CRITICAL: Must be first
    _patch_torchaudio_load()
    _patch_transformers_qwen2_tokenizer()  # Fix VibeVoice import

