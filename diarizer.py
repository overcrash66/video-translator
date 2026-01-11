import logging
import os
import torch
import config
import numpy as np
from pathlib import Path
import torchaudio

# Fix for SpeechBrain compatibility with newer torchaudio versions
# SpeechBrain uses deprecated torchaudio.list_audio_backends which was removed
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']

logger = logging.getLogger(__name__)

class Diarizer:
    """
    Speaker diarization using SpeechBrain's ECAPA-TDNN embeddings + spectral clustering.
    This replaces pyannote to avoid PyTorch 2.6+ compatibility issues.
    """
    
    def __init__(self):
        self.embedding_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Diarization parameters
        self.segment_duration = 1.5  # seconds per segment for embedding extraction
        self.segment_overlap = 0.5   # overlap between segments
        self.min_speakers = 1
        self.max_speakers = 10
        
    def _load_model(self):
        """Load SpeechBrain ECAPA-TDNN speaker embedding model."""
        if self.embedding_model is not None:
            return
            
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            
            logger.info("Loading SpeechBrain ECAPA-TDNN model...")
            
            # Load pretrained ECAPA-TDNN from HuggingFace
            self.embedding_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(config.TEMP_DIR / "speechbrain_models"),
                run_opts={"device": str(self.device)}
            )
            
            logger.info(f"SpeechBrain model loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load SpeechBrain model: {e}")
            raise
    
    def _extract_embeddings(self, audio_path, segments):
        """Extract speaker embeddings for each audio segment."""
        # Load full audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to 16kHz if needed (ECAPA-TDNN expects 16kHz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        embeddings = []
        valid_segments = []
        
        for seg in segments:
            start_sample = int(seg['start'] * sample_rate)
            end_sample = int(seg['end'] * sample_rate)
            
            # Bounds check
            start_sample = max(0, start_sample)
            end_sample = min(waveform.shape[1], end_sample)
            
            if end_sample - start_sample < sample_rate * 0.3:  # Min 0.3s
                continue
                
            segment_audio = waveform[:, start_sample:end_sample]
            
            try:
                # Get embedding
                with torch.no_grad():
                    embedding = self.embedding_model.encode_batch(segment_audio.to(self.device))
                    embeddings.append(embedding.squeeze().cpu().numpy())
                    valid_segments.append(seg)
            except Exception as e:
                logger.warning(f"Failed to extract embedding for segment: {e}")
                continue
        
        return np.array(embeddings) if embeddings else None, valid_segments
    
    def _create_segments_from_vad(self, audio_path):
        """Create segments using simple energy-based VAD."""
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Simple energy-based VAD
        frame_size = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.010 * sample_rate)   # 10ms hop
        
        # Calculate energy per frame
        waveform_np = waveform.squeeze().numpy()
        num_frames = (len(waveform_np) - frame_size) // hop_length + 1
        
        energies = []
        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_size
            frame = waveform_np[start:end]
            energy = np.sum(frame ** 2)
            energies.append(energy)
        
        energies = np.array(energies)
        
        # Dynamic threshold (percentile-based)
        threshold = np.percentile(energies, 30)  # Bottom 30% is silence
        
        # Find speech regions
        is_speech = energies > threshold
        
        # Convert frames to segments
        segments = []
        in_speech = False
        seg_start = 0
        
        for i, speech in enumerate(is_speech):
            time = i * hop_length / sample_rate
            
            if speech and not in_speech:
                seg_start = time
                in_speech = True
            elif not speech and in_speech:
                if time - seg_start >= 0.5:  # Min segment length 0.5s
                    segments.append({'start': seg_start, 'end': time, 'speaker': 'UNKNOWN'})
                in_speech = False
        
        # Handle last segment
        if in_speech:
            end_time = len(waveform_np) / sample_rate
            if end_time - seg_start >= 0.5:
                segments.append({'start': seg_start, 'end': end_time, 'speaker': 'UNKNOWN'})
        
        # Merge close segments
        merged = []
        for seg in segments:
            if merged and seg['start'] - merged[-1]['end'] < 0.3:  # Gap < 0.3s
                merged[-1]['end'] = seg['end']
            else:
                merged.append(seg)
        
        return merged
    
    def _cluster_embeddings(self, embeddings, max_speakers=None):
        """Cluster embeddings using spectral clustering."""
        from sklearn.cluster import SpectralClustering
        from sklearn.metrics import silhouette_score
        
        if max_speakers is None:
            max_speakers = self.max_speakers
        
        n_samples = len(embeddings)
        
        if n_samples < 2:
            return np.zeros(n_samples, dtype=int)
        
        # Try different numbers of clusters
        best_score = -1
        best_labels = None
        best_n = 2
        
        max_clusters = min(max_speakers, n_samples)
        
        for n_clusters in range(2, max_clusters + 1):
            try:
                clustering = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity='nearest_neighbors',
                    n_neighbors=min(10, n_samples - 1),
                    random_state=42
                )
                labels = clustering.fit_predict(embeddings)
                
                # Silhouette score to evaluate clustering quality
                if len(set(labels)) > 1:
                    score = silhouette_score(embeddings, labels)
                    if score > best_score:
                        best_score = score
                        best_labels = labels
                        best_n = n_clusters
            except Exception as e:
                logger.debug(f"Clustering with {n_clusters} failed: {e}")
                continue
        
        if best_labels is None:
            return np.zeros(n_samples, dtype=int)
        
        logger.info(f"Best clustering: {best_n} speakers (score: {best_score:.3f})")
        return best_labels

    def diarize(self, audio_path):
        """
        Perform speaker diarization on audio file.
        Returns list of segments: [{'start': float, 'end': float, 'speaker': str}, ...]
        """
        self._load_model()
        
        logger.info(f"Diarizing {audio_path}...")
        
        # Step 1: Create segments using VAD
        logger.info("Running voice activity detection...")
        segments = self._create_segments_from_vad(audio_path)
        
        if not segments:
            logger.warning("No speech segments detected")
            return []
        
        logger.info(f"Found {len(segments)} speech segments")
        
        # Step 2: Extract embeddings for each segment
        logger.info("Extracting speaker embeddings...")
        embeddings, valid_segments = self._extract_embeddings(audio_path, segments)
        
        if embeddings is None or len(embeddings) == 0:
            logger.warning("No embeddings extracted, returning single speaker")
            for seg in segments:
                seg['speaker'] = 'SPEAKER_00'
            return segments
        
        # Step 3: Cluster embeddings
        logger.info("Clustering speakers...")
        labels = self._cluster_embeddings(embeddings)
        
        # Assign speaker labels
        for i, seg in enumerate(valid_segments):
            seg['speaker'] = f'SPEAKER_{labels[i]:02d}'
        
        logger.info(f"Diarization complete. Found {len(set(labels))} unique speakers.")
        return valid_segments

    def detect_genders(self, audio_path, segments):
        """
        Analyzes speaker segments to estimate gender (Male/Female) based on Pitch (F0).
        Returns dict {speaker_id: 'Male'|'Female'}
        """
        import librosa
        
        # Group segments by speaker
        speaker_segments = {}
        for seg in segments:
            sp = seg['speaker']
            if sp not in speaker_segments:
                speaker_segments[sp] = []
            speaker_segments[sp].append((seg['start'], seg['end']))
            
        genders = {}
        
        logger.info("Loading audio for gender detection...")
        try:
            y, sr = librosa.load(audio_path, sr=None, mono=True)
        except Exception as e:
            logger.error(f"Librosa load failed: {e}")
            return {sp: "Male" for sp in speaker_segments}
            
        logger.info("Analyzing pitch for speakers...")
        
        for sp, times in speaker_segments.items():
            # Collect audio for this speaker (max 20 seconds)
            speaker_audio = []
            total_dur = 0
            
            for start, end in times:
                s_sample = int(start * sr)
                e_sample = int(end * sr)
                s_sample = max(0, s_sample)
                e_sample = min(len(y), e_sample)
                
                if s_sample >= e_sample:
                    continue
                
                chunk = y[s_sample:e_sample]
                speaker_audio.append(chunk)
                total_dur += (end - start)
                
                if total_dur > 20.0:
                    break
            
            if not speaker_audio:
                genders[sp] = "Male"
                continue
                
            full_audio = np.concatenate(speaker_audio)
            
            # F0 estimation using PyIN
            try:
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    full_audio,
                    fmin=65,
                    fmax=300,
                    sr=sr
                )
                
                f0 = f0[~np.isnan(f0)]
                
                if len(f0) == 0:
                    genders[sp] = "Male"
                else:
                    mean_f0 = np.mean(f0)
                    # Threshold: Male ~85-155Hz, Female ~165-255Hz
                    if mean_f0 > 160:
                        genders[sp] = "Female"
                    else:
                        genders[sp] = "Male"
                        
                    logger.info(f"Speaker {sp}: Mean F0 = {mean_f0:.1f} Hz -> {genders[sp]}")
            except Exception as e:
                logger.warning(f"Pitch analysis failed for {sp}: {e}")
                genders[sp] = "Male"
        
        return genders


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    d = Diarizer()
    # d.diarize("path/to/test.wav")
