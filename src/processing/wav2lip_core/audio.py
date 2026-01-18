import librosa
import numpy as np

# Mel-spectrogram parameters
sample_rate = 16000
n_fft = 800
hop_size = 200
win_size = 800
num_mels = 80
fmin = 55
fmax = 7600
min_level_db = -100
ref_level_db = 20

def preemphasis(x):
    return np.append(x[0], x[1:] - 0.97 * x[:-1])

def inv_preemphasis(x):
    return np.append(x[0], x[1:] + 0.97 * x[:-1])

def normalize(S):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)

def denormalize(S):
    return (np.clip(S, 0, 1) * -min_level_db) + min_level_db

def wav2mel(file_path):
    wav = librosa.load(file_path, sr=sample_rate)[0]
    
    # Preemphasis
    wav = preemphasis(wav)
    
    # STFT
    D = librosa.stft(y=wav, n_fft=n_fft, hop_length=hop_size, win_length=win_size)
    
    # Mel basis convert
    S = librosa.feature.melspectrogram(S=np.abs(D), sr=sample_rate, n_fft=n_fft, n_mels=num_mels,
                                       fmin=fmin, fmax=fmax)
    
    # Log mel
    S = 20 * np.log10(np.maximum(1e-5, S)) - ref_level_db
    
    # Normalize
    S = normalize(S)
    
    return S.T # (T, 80)
