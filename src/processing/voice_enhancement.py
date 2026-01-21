import os
import logging
from pathlib import Path
import torch

try:
    from voicefixer import VoiceFixer
except ImportError:
    VoiceFixer = None

logger = logging.getLogger(__name__)

class VoiceEnhancer:
    """
    Wrapper for VoiceFixer to enhance audio quality.
    Restores degraded speech and removes noise.
    """
    def __init__(self):
        self.model = None

    def load_model(self):
        """Lazy loads the VoiceFixer model."""
        if self.model is None:
            if VoiceFixer is None:
                raise RuntimeError("voicefixer package not installed. Cannot enhance audio.")
            
            logger.info("Loading VoiceFixer model...")
            # Initialize VoiceFixer
            # It automatically downloads checkpoints to ~/.cache/voicefixer/
            # Initialize VoiceFixer
            # It automatically downloads checkpoints to ~/.cache/voicefixer/
            logger.info("Initializing VoiceFixer model...")
            self.model = VoiceFixer()

    def unload_model(self):
        """Unloads the model and clears GPU memory."""
        if self.model:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("VoiceFixer model unloaded.")

    def enhance_audio(self, input_path, output_path, mode=0, cuda=None):
        """
        Enhances the audio at input_path and saves to output_path.
        
        Args:
            input_path (str): Path to input audio.
            output_path (str): Path to save enhanced audio.
            mode (int): 0 for 'voicefixer' (restore), 1 for 'mel' (vocoder only), 2 for 'fbank' (vocoder only). 
                        Default 0 is typically best for general restoration.
            cuda (bool): Override cuda usage if needed.
        """
        self.load_model()
        
        input_path = str(input_path)
        output_path = str(output_path)
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input audio not found: {input_path}")
            
        logger.info(f"Enhancing audio: {input_path} -> {output_path}")
        
        # VoiceFixer restore method:
        # restore(input, output, cuda=True/False, mode=0/1/2) 
        # Note: self.model.restore signature might vary slightly by version, 
        # but typical usage is restore(input, output, ...)
        
        use_cuda = torch.cuda.is_available() if cuda is None else cuda
        
        try:
           self.model.restore(input=input_path, output=output_path, cuda=use_cuda, mode=mode)
           if not os.path.exists(output_path):
               raise RuntimeError("VoiceFixer finished but output file was not created.")
               
           logger.info("Audio enhancement complete.")
           
        except Exception as e:
            logger.error(f"VoiceFixer failed: {e}")
            raise
