import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
import config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioSeparator:
    def __init__(self):
        # Force soundfile backend if available, as default might be missing on Windows
        try:
             import torchaudio
             torchaudio.set_audio_backend("soundfile")
        except Exception:
             pass # Might already be soundfile or not available

        self.device = config.DEVICE
        self.output_dir = config.TEMP_DIR

        self.model = None
        self.processor = None
        self.loaded = False

    def unload_model(self):
        """Unloads the current model to free VRAM."""
        if self.loaded:
            logger.info("Unloading AudioSeparator model...")
            self.model = None
            self.processor = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
            self.loaded = False
            self.current_model_type = None

    def load_model(self, model_selection):
        """
        Loads the selected model.
        model_selection: "Torchaudio HDemucs (Recommended)"
        """
        # If we are already loaded with the SAME model, return
        if self.loaded and getattr(self, 'current_model_type', '') == model_selection:
            return
            
        # If switching models, unload previous
        if self.loaded:
             logger.info("Switching models, unloading previous...")
             self.model = None
             self.processor = None
             if self.device == "cuda":
                 torch.cuda.empty_cache()
             self.loaded = False

        self.current_model_type = model_selection
        
        self._load_demucs()

    def _load_demucs(self):
        logger.info(f"Loading HDemucs model on {self.device}...")
        try:
            self.bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
            self.model = self.bundle.get_model()
            self.model.to(self.device)
            self.sample_rate = self.bundle.sample_rate
            self.loaded = True
            logger.info("HDemucs loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load HDemucs: {e}")
            self.loaded = False



    def separate(self, audio_path, prompt="speech", model_selection="Torchaudio HDemucs (Recommended)"):
        """
        Separates audio.
        model_selection: Decides which backend to use.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self.load_model(model_selection)
        
        output_vocals = self.output_dir / "vocals.wav"
        output_bg = self.output_dir / "accompaniment.wav"

        if not self.loaded:
            logger.warning("Model not loaded. Performing dummy separation (copying).")
            import shutil
            shutil.copy(audio_path, output_vocals)
            self._create_silent_like(audio_path, output_bg)
            return str(output_vocals), str(output_bg)

        return self._separate_demucs(audio_path, output_vocals, output_bg)

    def _separate_demucs(self, audio_path, output_vocals, output_bg):
        logger.info(f"Separating with HDemucs: {audio_path}")
        try:
            import soundfile as sf
            
            # Use soundfile to read
            waveform_np, sr = sf.read(str(audio_path))
            
            # Soundfile returns [Time, Channels] or [Time] if mono
            # PyTorch expects [Channels, Time]
            if waveform_np.ndim == 1:
                waveform_np = waveform_np[np.newaxis, :]  # [1, Time]
            else:
                waveform_np = waveform_np.T # [Channels, Time]
                
            waveform = torch.from_numpy(waveform_np).float()
            
            # HDemucs requires Stereo (2 channels). If Mono, duplicate channel.
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1) # [2, Time]
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate).to(self.device)
                waveform = waveform.to(self.device)
                waveform = resampler(waveform)
            else:
                waveform = waveform.to(self.device)

            # Demucs inference
            ref = waveform.mean(0)
            waveform_in = waveform.unsqueeze(0)
            ref_mean = ref.mean()
            ref_std = ref.std()
            waveform_in = (waveform_in - ref_mean) / (ref_std + 1e-8)



            # Demucs inference with chunking and fallback
            try:
                sources = self._demucs_inference_chunked(waveform)
            except torch.cuda.OutOfMemoryError:
                logger.warning("CUDA Out of Memory during HDemucs inference. Falling back to CPU.")
                torch.cuda.empty_cache()
                self.model.to("cpu")
                waveform = waveform.to("cpu")
                sources = self._demucs_inference_chunked(waveform)
                self.model.to(self.device) # Move back for future? or keep on CPU? Move back.
            
            # 0=drums, 1=bass, 2=other, 3=vocals
            vocals = sources[3]
            background = sources[0] + sources[1] + sources[2]
            
            # Save using soundfile to avoid 'TorchCodec' errors
            import soundfile as sf
            
            # Torchaudio tensors are [Channel, Time], Soundfile expects [Time, Channel]
            vocals_np = vocals.cpu().numpy().T
            bg_np = background.cpu().numpy().T
            

            sf.write(str(output_vocals), vocals_np, self.sample_rate)
            sf.write(str(output_bg), bg_np, self.sample_rate)
            
            # Cleanup memory
            del waveform, sources, vocals, background
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return str(output_vocals), str(output_bg)

        except Exception as e:
            logger.error(f"HDemucs Inference failed: {e}")
            return self._fallback_dummy(audio_path, output_vocals, output_bg)

    def _demucs_inference_chunked(self, waveform):
        """
        Process audio in chunks to avoid OOM.
        waveform: [Channels, Time]
        """
        ref = waveform.mean(0)
        waveform_in = waveform
        ref_mean = ref.mean()
        ref_std = ref.std()
        
        # Normalize
        waveform_in = (waveform_in - ref_mean) / (ref_std + 1e-8)
        
        # Model expects [Batch, Channels, Time]
        # We process in chunks of X seconds
        chunk_seconds = 10 # Conservative chunk size for limited VRAM
        overlap_seconds = 1
        
        sr = self.sample_rate
        chunk_size = chunk_seconds * sr
        overlap = overlap_seconds * sr
        stride = chunk_size - overlap
        
        length = waveform_in.shape[-1]
        channels = waveform_in.shape[0]


        
        # Output container: Demucs outputs 4 sources [4, Channels, Time]
        # We need to accumulate.
        # We'll use a weight buffer for overlap-add
        final_sources = torch.zeros((4, channels, length), device=waveform_in.device)
        weights = torch.zeros((1, 1, length), device=waveform_in.device)
        
        # Create a linear window for cross-fading
        if overlap > 0:
            window = torch.ones(chunk_size, device=waveform_in.device)
            # Ramp up/down at edges
            ramp = torch.linspace(0, 1, overlap, device=waveform_in.device)
            window[:overlap] *= ramp
            window[-overlap:] *= ramp.flip(0)
        else:
            window = torch.ones(chunk_size, device=waveform_in.device)

        with torch.no_grad():
            from torch.nn import functional as F
            
            for start in range(0, length, stride):
                end = min(start + chunk_size, length)
                
                # Extract chunk
                chunk = waveform_in[:, start:end]
                
                # Pad if last chunk is smaller than expected? 
                # HDemucs is fully conv, should handle variable length, but padding is safer for edge effects.
                # Let's just pass what we have, but padding to valid stride might be needed.
                # Actually, simple passing usually works.
                
                # Add Batch Dim [1, Channels, Time]
                chunk_input = chunk.unsqueeze(0)
                
                # Inference
                # Output: [1, Sources, Channels, Time]
                chunk_out = self.model.forward(chunk_input)[0] 
                
                # Remove batch dim
                # [Sources, Channels, Time]
                
                # Add to final
                # Handle windowing if size matches
                current_window = window[:chunk.shape[-1]]
                
                # Expand window for broadcasting: [1, 1, Time]
                w_expanded = current_window.view(1, 1, -1)
                
                final_sources[:, :, start:end] += chunk_out * w_expanded
                weights[:, :, start:end] += w_expanded
                
        # Normalize by weights
        final_sources /= (weights + 1e-8)
        
        # Denormalize audio
        final_sources = final_sources * (ref_std + 1e-8) + ref_mean
        
        return final_sources



    def _fallback_dummy(self, audio_path, output_vocals, output_bg):
        logger.warning("Using fallback dummy separation.")
        import shutil
        shutil.copy(audio_path, output_vocals)
        self._create_silent_like(audio_path, output_bg)
        return str(output_vocals), str(output_bg)

    def _create_silent_like(self, ref_audio, output_path):
        """Creates a silent wav file with same duration/format as reference."""
        try:
            try:
                info = torchaudio.info(str(ref_audio))
            except AttributeError:
                # Fallback for older/different torchaudio versions or missing backend
                import soundfile as sf
                info = sf.info(str(ref_audio))
                # Create a specialized object or simple structure to match usage
                from collections import namedtuple
                AudioInfo = namedtuple('AudioInfo', ['sample_rate', 'num_frames', 'num_channels'])
                info = AudioInfo(sample_rate=info.samplerate, num_frames=info.frames, num_channels=info.channels)

            sr = info.sample_rate
            duration_frames = info.num_frames
            channels = info.num_channels
            
            # Create silent array (numpy) for soundfile
            import numpy as np
            import soundfile as sf
            
            # Channels, Frames -> Frames, Channels
            if channels == 1:
                silent_data = np.zeros((duration_frames,))
            else:
                silent_data = np.zeros((duration_frames, channels))
                
            sf.write(str(output_path), silent_data, sr)
        except Exception as e:
            logger.error(f"Error creating silent file: {e}")

if __name__ == "__main__":
    # Test stub
    sep = AudioSeparator()
    # Create dummy audio if not exists
    dummy_audio = config.TEMP_DIR / "test_input.wav"
    if not dummy_audio.exists():
        sr = 16000
        t = torch.linspace(0, 5, 5*sr)
        wav = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0) # 440Hz sine
        torchaudio.save(str(dummy_audio), wav, sr)
        
    v, b = sep.separate(dummy_audio)
    print(f"Vocals: {v}")
    print(f"Background: {b}")
