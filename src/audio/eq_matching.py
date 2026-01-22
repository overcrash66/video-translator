import numpy as np
import librosa
import soundfile as sf
import scipy.signal
import logging

logger = logging.getLogger(__name__)

def compute_spectral_envelope(prior, sr=22050):
    """
    Computes the spectral envelope (average log-magnitude spectrum) of a waveform.
    
    Args:
        prior: Audio time-series (waveform)
        sr: Sampling rate
        
    Returns:
        Frequency envelope
    """
    # Compute STFT
    # n_fft=2048 is standard for speech
    S = librosa.stft(prior, n_fft=2048)
    
    # Compute magnitude
    S_mag = np.abs(S)
    
    # Average over time to get global spectral profile
    # axis=1 is time
    S_mean = np.mean(S_mag, axis=1)
    
    # Optional: Smooth the envelope?
    # Raw average is usually fine for EQ matching
    
    return S_mean

def apply_eq_matching(source_path: str, target_path: str, output_path: str, strength: float = 0.5) -> str:
    """
    Applies the spectral envelope of the source audio to the target audio.
    
    Args:
        source_path: Path to reference audio (vocals)
        target_path: Path to target audio (TTS output)
        output_path: Path to save result
        strength: Blend factor (0.0 = no effect, 1.0 = full match). 
                  Default 0.5 is subtle but effective.
                  
    Returns:
        Path to output file
    """
    try:
        # Load source (Reference)
        y_ref, sr_ref = librosa.load(source_path, sr=None)
        
        # Load target (TTS)
        y_target, sr_target = librosa.load(target_path, sr=None)
        
        # Resample reference to target if needed (fft size depends on SR?)
        # Actually STFT bins depend on n_fft regardless of SR, but physical frequencies differ.
        # Ideally we resample one to match the other.
        if sr_ref != sr_target:
            y_ref = librosa.resample(y_ref, orig_sr=sr_ref, target_sr=sr_target)
            sr = sr_target
        else:
            sr = sr_target
            
        # Compute Envelopes
        # Use simple smoothing (window size 2048)
        n_fft = 2048
        hop_length = 512
        
        # 1. Target STFT
        S_target = librosa.stft(y_target, n_fft=n_fft, hop_length=hop_length)
        S_target_mag = np.abs(S_target)
        S_target_phase = np.angle(S_target)
        
        # 2. Compute average spectra
        # Add epsilon to avoid log(0)
        eps = 1e-8
        
        ref_spec = np.mean(np.abs(librosa.stft(y_ref, n_fft=n_fft)), axis=1) + eps
        target_spec = np.mean(S_target_mag, axis=1) + eps
        
        # 3. Calculate filter
        # The filter is the ratio of reference to target
        # filter = ref / target
        eq_filter = ref_spec / target_spec
        
        # 4. Limit the filter gain to avoid artifacts (e.g. boosting noise)
        # Clip max gain to e.g. 10dB (approx 3.0x amplitude)
        # Clip min attenuation to -10dB (approx 0.3x)
        eq_filter = np.clip(eq_filter, 0.3, 3.0)
        
        # 5. Apply strength (Interpolate between 1.0 (flat) and eq_filter)
        # New Filter = (Filter * Strength) + (1.0 * (1-Strength)) ? 
        # Actually power interpolation is better: width = filter ** strength
        # But linear blend is safer for simple EQ.
        final_filter = (eq_filter * strength) + (1.0 * (1.0 - strength))
        
        # 6. Apply to target
        # Reshape filter to (Frequency, 1) for broadcasting
        final_filter = final_filter[:, np.newaxis]
        
        S_target_matched = S_target_mag * final_filter
        
        # Reconstruct
        # Use original phase
        S_reconstructed = S_target_matched * np.exp(1j * S_target_phase)
        
        y_out = librosa.istft(S_reconstructed, hop_length=hop_length)
        
        # Fix length (ISTFT usually matches, but good to be safe)
        if len(y_out) > len(y_target):
            y_out = y_out[:len(y_target)]
        elif len(y_out) < len(y_target):
             y_out = np.pad(y_out, (0, len(y_target) - len(y_out)))
             
        # Save
        sf.write(output_path, y_out, sr)
        return output_path
        
    except Exception as e:
        logger.error(f"EQ Matching failed: {e}")
        # Just copy original if fail
        import shutil
        shutil.copy2(target_path, output_path)
        return output_path
