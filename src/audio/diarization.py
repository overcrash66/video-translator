import logging
import os
import torch
from src.utils import config
import numpy as np
from pathlib import Path
import torchaudio
import json
import shutil
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# [Fix] PyTorch 2.6+ changed torch.load to use weights_only=True by default
# PyAnnote models contain TorchVersion metadata that's not in the default safe globals
# Add to allowlist to prevent "Unsupported global: torch.torch_version.TorchVersion" error
try:
    from torch.torch_version import TorchVersion
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([TorchVersion])
except (ImportError, AttributeError):
    pass  # Older PyTorch version, not needed

# Fix for SpeechBrain compatibility with newer torchaudio versions
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']

# Fix for SpeechBrain 1.0.0 accessing torchaudio.io when it's not available
# This prevents AttributeError: module 'torchaudio' has no attribute 'io'
if not hasattr(torchaudio, 'io'):
    try:
        import torchaudio.io
    except ImportError:
        # If import fails, mock it to prevent AttributeError in SpeechBrain
        class MockIO:
            StreamReader = None
            AudioEffector = None
        torchaudio.io = MockIO()


logger = logging.getLogger(__name__)

class Diarizer:
    """
    Speaker diarization using SpeechBrain or NVIDIA NeMo.
    Also handles speaker profiling for TTS cloning.
    """
    
    def __init__(self):
        self.embedding_model = None
        self.nemo_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Diarization parameters
        self.segment_duration = 1.5
        self.segment_overlap = 0.5
        self.min_speakers = 1
        self.max_speakers = 10
        self.current_backend = None

    def unload_model(self):
        """Unload models to free VRAM."""
        if self.embedding_model:
            logger.info("Unloading SpeechBrain model...")
            del self.embedding_model
            self.embedding_model = None
            
        if self.nemo_model:
            logger.info("Unloading NeMo model...")
            del self.nemo_model
            self.nemo_model = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.current_backend = None

    def _load_speechbrain(self):
        """Load SpeechBrain ECAPA-TDNN speaker embedding model."""
        if self.embedding_model is not None:
            return
            
        try:
            from huggingface_hub import snapshot_download
            import platform
            
            # [CRITICAL] On Windows, monkey-patch SpeechBrain's fetch BEFORE importing EncoderClassifier
            # SpeechBrain imports fetch at module load time, so we must patch first
            if platform.system() == "Windows":
                self._patch_speechbrain_fetch_for_windows()
            
            # Now import EncoderClassifier AFTER the patch is applied
            from speechbrain.inference.speaker import EncoderClassifier
            
            logger.info("Loading SpeechBrain ECAPA-TDNN model...")
            
            # [Fix] Use local directory with no symlinks to avoid WinError 1314
            # We download to a persistent 'models' directory instead of temp
            model_dir = config.BASE_DIR / "models" / "speechbrain" / "spkrec-ecapa-voxceleb"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Force copy files instead of symlinking from cache
            local_path = snapshot_download(
                repo_id="speechbrain/spkrec-ecapa-voxceleb",
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            local_path = Path(local_path)
            
            # Override pretrained_path to point to local dir, inhibiting Hub fetch
            # ensure path string is safe for yaml
            local_path_str = str(local_path).replace("\\", "/")
            
            # [Fix] SpeechBrain 1.0.0 tries to symlink files which fails on Windows without admin
            # Pre-copy all expected model files to the savedir so SpeechBrain's fetch() 
            # sees them as already existing and skips symlink creation entirely
            expected_files = [
                "hyperparams.yaml",
                "embedding_model.ckpt", 
                "mean_var_norm_emb.ckpt",
                "classifier.ckpt",
                "label_encoder.ckpt",
                "custom.py",
            ]
            
            for fname in expected_files:
                src = local_path / fname
                # If source exists but is a symlink (broken or otherwise), resolve or copy
                if src.exists() or src.is_symlink():
                    # For Windows compatibility, ensure files are real copies not symlinks
                    if src.is_symlink() and platform.system() == "Windows":
                        try:
                            # Read symlink target and copy content
                            target = src.resolve()
                            if target.exists():
                                src.unlink()
                                shutil.copy2(target, src)
                        except Exception as e:
                            logger.warning(f"Could not resolve symlink {fname}: {e}")
            
            # Also handle label_encoder.txt -> label_encoder.ckpt copy
            lab_txt = local_path / "label_encoder.txt"
            lab_ckpt = local_path / "label_encoder.ckpt"
            if lab_txt.exists() and not lab_ckpt.exists():
                shutil.copy2(lab_txt, lab_ckpt)
            
            self.embedding_model = EncoderClassifier.from_hparams(
                source=str(local_path),
                savedir=str(local_path),
                run_opts={"device": str(self.device)},
                overrides={"pretrained_path": local_path_str}
            )
            self.current_backend = "speechbrain"
            logger.info(f"SpeechBrain model loaded on {self.device}")
        except Exception as e:
            if "CUDA" in str(e) and self.device.type == "cuda":
                logger.warning(f"CUDA Error loading SpeechBrain: {e}")
                logger.warning("Switching to CPU fallback for Diarization (SpeechBrain)...")
                self.device = torch.device("cpu")
                # Retry
                self._load_speechbrain()
                return

            logger.error(f"Failed to load SpeechBrain model: {e}")
            raise

    def _patch_speechbrain_fetch_for_windows(self):
        """
        Monkey-patch SpeechBrain's fetch function to use copy instead of symlink.
        
        This is needed for SpeechBrain 1.0.0 on Windows which doesn't have LocalStrategy
        and attempts to create symlinks which require admin privileges.
        """
        try:
            import speechbrain.utils.fetching as fetching_module
            import pathlib
            
            # Check if already patched
            if getattr(fetching_module, '_windows_patched', False):
                logger.debug("SpeechBrain fetch already patched.")
                return
            
            original_fetch = fetching_module.fetch
            
            def patched_fetch(
                filename,
                source,
                savedir="./pretrained_model_checkpoints",
                overwrite=False,
                save_filename=None,
                use_auth_token=False,
                revision=None,
                huggingface_cache_dir=None,
            ):
                """Patched fetch that uses copy instead of symlink on Windows."""
                if save_filename is None:
                    save_filename = filename
                savedir = pathlib.Path(savedir)
                savedir.mkdir(parents=True, exist_ok=True)
                destination = savedir / save_filename
                source_path = pathlib.Path(source)
                
                # [CRITICAL FIX] Handle edge case where source == savedir
                # This happens when from_hparams is called with source=savedir
                # SpeechBrain would try to symlink a file to itself, which fails
                if source_path.resolve() == savedir.resolve():
                    # Source and savedir are the same directory
                    if destination.exists() and not destination.is_symlink():
                        logger.debug(f"Fetch {filename}: Source==savedir, file already exists.")
                        return destination
                    elif destination.is_symlink():
                        # Convert symlink to real file
                        target = destination.resolve()
                        if target.exists() and target != destination:
                            destination.unlink()
                            shutil.copy2(target, destination)
                            logger.debug(f"Fetch {filename}: Converted symlink to real file.")
                        return destination
                    else:
                        # File doesn't exist - this is normal, SpeechBrain will create it
                        # For files like custom.py that may not exist, we just skip
                        logger.debug(f"Fetch {filename}: Source==savedir, file not found, skipping.")
                        # Return None or destination - caller will handle missing file
                        return destination
                
                # If destination already exists (as real file), skip
                if destination.exists() and not destination.is_symlink() and not overwrite:
                    logger.debug(f"Fetch {filename}: Using existing file in {str(destination)}.")
                    return destination
                
                # If source is a local directory, copy instead of symlink
                if source_path.is_dir():
                    sourcefile = source_path / filename
                    if sourcefile.exists():
                        # Skip if source and destination are the same file
                        if sourcefile.resolve() == destination.resolve():
                            logger.debug(f"Fetch {filename}: Source and destination are same file.")
                            return destination
                        # Remove any existing symlink or file if overwriting
                        if destination.exists() or destination.is_symlink():
                            destination.unlink()
                        # Copy instead of symlink
                        shutil.copy2(sourcefile, destination)
                        logger.debug(f"Fetch {filename}: Copied from {str(sourcefile)} to {str(destination)}.")
                        return destination
                
                # For other cases (HuggingFace, URL), call original but intercept symlinks
                # by checking if the result is a symlink and converting it
                result = original_fetch(
                    filename=filename,
                    source=source,
                    savedir=savedir,
                    overwrite=overwrite,
                    save_filename=save_filename,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    huggingface_cache_dir=huggingface_cache_dir,
                )
                
                # Convert symlink to real file
                result_path = pathlib.Path(result)
                if result_path.is_symlink():
                    target = result_path.resolve()
                    if target.exists():
                        result_path.unlink()
                        shutil.copy2(target, result_path)
                        logger.debug(f"Converted symlink {filename} to real file.")
                
                return result
            
            # Apply the patch to fetching module
            fetching_module.fetch = patched_fetch
            fetching_module._windows_patched = True
            
            # CRITICAL: Also patch the reference in speechbrain.inference.interfaces
            # because it does `from speechbrain.utils.fetching import fetch` at import time
            try:
                import speechbrain.inference.interfaces as interfaces_module
                interfaces_module.fetch = patched_fetch
                logger.debug("Also patched fetch in speechbrain.inference.interfaces.")
            except (ImportError, AttributeError) as e:
                logger.debug(f"Could not patch interfaces module: {e}")
            
            logger.info("Applied Windows symlink patch to SpeechBrain fetch.")
            
        except Exception as e:
            logger.warning(f"Could not patch SpeechBrain fetch: {e}")

    def _load_nemo(self):
        """Load NVIDIA NeMo for diarization."""
        if self.nemo_model is not None:
            return
            
        logger.info("Loading NVIDIA NeMo dependencies...")
        try:
            # NeMo is installed via pip install nemo_toolkit[asr]
            # It uses hydra/omegaconf for config.
            import nemo.collections.asr as nemo_asr
            # Just verifying import works
            self.current_backend = "nemo"
            self.nemo_model = "loaded" 
            logger.info("NeMo dependencies verified.")
            
        except ImportError:
            logger.error("NeMo Toolkit not installed. Please install `nemo_toolkit[asr]`.")
            raise
        except Exception as e:
            logger.error(f"Failed to load NeMo: {e}")
            raise

    def _extract_embeddings(self, audio_path, segments):
        """Extract speaker embeddings for each audio segment (SpeechBrain)."""
        from src.utils import audio_utils
        
        # Load full audio safely
        waveform, sample_rate = audio_utils.load_audio(audio_path, target_sr=16000, mono=True)
        
        embeddings = []
        valid_segments = []
        
        for seg in segments:
            start_sample = int(seg['start'] * sample_rate)
            end_sample = int(seg['end'] * sample_rate)
            
            start_sample = max(0, start_sample)
            end_sample = min(waveform.shape[1], end_sample)
            
            if end_sample - start_sample < sample_rate * 0.3:
                continue
                
            segment_audio = waveform[:, start_sample:end_sample]
            
            try:
                with torch.no_grad():
                    # SpeechBrain encode_batch expects [Batch, Time]
                    embedding = self.embedding_model.encode_batch(segment_audio.to(self.device))
                    embeddings.append(embedding.squeeze().cpu().numpy())
                    valid_segments.append(seg)
            except Exception as e:
                logger.warning(f"Failed to extract embedding: {e}")
                continue
        
        return np.array(embeddings) if embeddings else None, valid_segments

    def _run_nemo_diarization(self, audio_path):
        """
        Executes NeMo diarization pipeline via ClusteringDiarizer.
        """
        try:
            from nemo.collections.asr.models import ClusteringDiarizer
            from omegaconf import OmegaConf
        except ImportError:
            logger.error("NeMo not available.")
            return []
            
        # Prepare Manifest
        manifest_path = config.TEMP_DIR / "nemo_manifest.json"
        import soundfile as sf
        info = sf.info(str(audio_path))
        
        meta = {
            "audio_filepath": str(audio_path),
            "offset": 0,
            "duration": info.duration,
            "label": "infer",
            "text": "-",
            "num_speakers": None,
            "rttm_filepath": None,
            "uniq_id": ""
        }
        
        with open(manifest_path, "w") as f:
            json.dump(meta, f)
            f.write("\n")
            
        output_dir = config.TEMP_DIR / "nemo_output"
        output_dir.mkdir(exist_ok=True)
        
        # Config Construction
        cfg = OmegaConf.create({
            "name": "ClusterDiarizer",
            "num_workers": 0,
            "sample_rate": 16000,
            "batch_size": 16,
            "device": self.device.type, # 'cuda' or 'cpu'
            "diarizer": {
                "manifest_filepath": str(manifest_path),
                "out_dir": str(output_dir),
                "oracle_vad": False,
                "collar": 0.25,
                "ignore_overlap": True,
                "vad": {
                    "model_path": "vad_multilingual_marblenet",
                    "parameters": {"onset": 0.8, "offset": 0.6, "pad_onset": 0.05, "pad_offset": -0.1}
                },
                "speaker_embeddings": {
                    "model_path": "titanet_large",
                    "parameters": {"window_length_in_sec": 1.5, "shift_length_in_sec": 0.75, "multiscale_weights": [1,1,1,1,1]}
                },
                "clustering": {
                    "parameters": {"oracle_num_speakers": False, "max_num_speakers": 10, "enhanced_count_thresh": 80, "max_rp_threshold": 0.25, "sparse_search_volume": 30}
                }
            }
        })
        
        try:
            diarizer = ClusteringDiarizer(cfg=cfg)
            diarizer.diarize()
            
            # Parse RTTM output
            rttm_path = output_dir / "pred_rttm" / f"{Path(audio_path).stem}.rttm"
            
            # Helper to find any RTTM if name mismatch
            if not rttm_path.exists():
                 rttm_dir = output_dir / "pred_rttm"
                 if rttm_dir.exists():
                     rttms = list(rttm_dir.glob("*.rttm"))
                     if rttms: rttm_path = rttms[0]
            
            if not rttm_path.exists():
                 logger.error("NeMo RTTM output not found.")
                 return []
                     
            segments = []
            with open(rttm_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    # SPEAKER <NA> <channel> <start> <duration> <NA> <NA> <speaker> <NA> <NA>
                    if len(parts) >= 8 and parts[0] == "SPEAKER":
                        start = float(parts[3])
                        dur = float(parts[4])
                        speaker = parts[7]
                        segments.append({
                            "start": start,
                            "end": start + dur,
                            "speaker": speaker
                        })
            return segments
            
        except Exception as e:
            logger.error(f"NeMo diarization runtime error: {e}")
            return []

    def _run_pyannote(self, audio_path, model_name="pyannote/speaker-diarization-3.1", hf_token=None):
        """
        Run PyAnnote diarization pipeline.
        Supports mixed precision for speed and memory efficiency.
        """
        try:
            from pyannote.audio import Pipeline
            from src.utils import audio_utils
        except ImportError:
            logger.error("pyannote.audio not installed.")
            return []

        logger.info(f"Loading PyAnnote pipeline: {model_name}...")
        try:
            # Load pipeline
            # If explicit token is provided, use it. Otherwise rely on env var or cache.
            # [Fix] pyannote.audio 4.x uses 'token' parameter, while 3.x uses 'use_auth_token'
            # Try the newer API first, then fall back to the older API for compatibility
            auth_token = hf_token if hf_token else True
            
            try:
                # Try pyannote 4.x API first (token parameter)
                pipeline = Pipeline.from_pretrained(model_name, token=auth_token)
            except TypeError as e:
                if "unexpected keyword argument" in str(e) and "token" in str(e):
                    # Fall back to pyannote 3.x API (use_auth_token parameter)
                    logger.debug("Falling back to use_auth_token for older pyannote version")
                    pipeline = Pipeline.from_pretrained(model_name, use_auth_token=auth_token)
                else:
                    raise
            
            if pipeline is None:
                logger.error(f"Failed to load PyAnnote pipeline {model_name}. Ensure you have accepted the user agreement and have a valid token.")
                return []
                
            pipeline.to(self.device)
            logger.info(f"PyAnnote pipeline loaded on {self.device}")

            # Run inference
            # Use mixed precision for performance (approx 2x speedup on GPU)
            logger.info("Running diarization with mixed precision...")
            
            # Use autocast for mixed precision
            device_type = "cuda" if self.device.type == "cuda" else "cpu"
            
            # Load audio safely
            waveform, sample_rate = audio_utils.load_audio(audio_path)
            
            # PyAnnote expects dict for memory-based inference or path
            input_data = {"waveform": waveform, "sample_rate": sample_rate}
            
            with torch.amp.autocast(device_type=device_type, enabled=(device_type=="cuda")):
                diarization = pipeline(input_data)

            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                 segments.append({
                     "start": turn.start,
                     "end": turn.end,
                     "speaker": speaker
                 })
            
            logger.info(f"PyAnnote found {len(segments)} segments.")
            return segments

        except Exception as e:
            if "CUDA" in str(e) and self.device.type == "cuda":
                logger.warning(f"CUDA Error in PyAnnote: {e}")
                logger.warning("Switching to CPU fallback for Diarization (PyAnnote)...")
                self.device = torch.device("cpu")
                # Retry
                return self._run_pyannote(audio_path, model_name, hf_token)

            logger.error(f"PyAnnote diarization failed: {e}")
            return []

    def diarize(self, audio_path: str | Path, backend: str = "speechbrain", min_speakers: int = 1, max_speakers: int | None = None, hf_token: str | None = None) -> list[dict]:
        """
        Perform speaker diarization.
        backend: 'speechbrain' or 'nemo'
        """
        self.min_speakers = min_speakers
        if max_speakers: self.max_speakers = max_speakers

        logger.info(f"Diarizing {audio_path} using {backend} (min={self.min_speakers}, max={self.max_speakers})...")
        
        if backend == "nemo":
            try:
                self._load_nemo()
                segments = self._run_nemo_diarization(audio_path)
                if segments:
                    return segments
                logger.warning("NeMo backend returned 0 segments. Falling back to SpeechBrain.")
            except Exception as e:
                logger.warning(f"NeMo backend failed: {e}. Falling back to SpeechBrain.")
                # Fallthrough
        
        if backend == "pyannote" or backend == "pyannote_community":
            model = "pyannote/speaker-diarization-3.1"
            if backend == "pyannote_community":
                 # Use the community pipeline if specified
                 pass
            
            # Allow passing specific model ID via config if needed, but for now standardizing
            segments = self._run_pyannote(audio_path, model_name="pyannote/speaker-diarization-3.1", hf_token=hf_token)
            if segments:
                return segments
            logger.warning("PyAnnote failed. Falling back to SpeechBrain.")
                
        # SpeechBrain Fallback or Selection
        self._load_speechbrain()
        
        # 1. VAD
        segments = self._create_segments_from_vad(audio_path)
        if not segments: return []
        
        # 2. Embed
        embeddings, valid_segments = self._extract_embeddings(audio_path, segments)
        if embeddings is None: 
            for seg in segments: seg['speaker'] = 'SPEAKER_00'
            return segments
            
        # 3. Cluster
        labels = self._cluster_embeddings(embeddings)
        for i, seg in enumerate(valid_segments):
            seg['speaker'] = f'SPEAKER_{labels[i]:02d}'
            
        return valid_segments

    def _create_segments_from_vad(self, audio_path):
        """Create segments using simple energy-based VAD."""
        from src.utils import audio_utils
        
        waveform, sample_rate = audio_utils.load_audio(audio_path, mono=True)
        
        frame_size = int(0.025 * sample_rate)
        hop_length = int(0.010 * sample_rate)
        waveform_np = waveform.squeeze().cpu().numpy()
        num_frames = (len(waveform_np) - frame_size) // hop_length + 1
        
        energies = np.array([np.sum(waveform_np[i*hop_length : i*hop_length+frame_size]**2) for i in range(num_frames)])
        threshold = np.percentile(energies, 30)
        is_speech = energies > threshold
        
        segments = []
        in_speech = False
        seg_start = 0
        
        for i, speech in enumerate(is_speech):
            time = i * hop_length / sample_rate
            if speech and not in_speech:
                seg_start = time
                in_speech = True
            elif not speech and in_speech:
                if time - seg_start >= 0.5:
                    segments.append({'start': seg_start, 'end': time, 'speaker': 'UNKNOWN'})
                in_speech = False
        if in_speech:
            end_time = len(waveform_np) / sample_rate
            if end_time - seg_start >= 0.5:
                segments.append({'start': seg_start, 'end': end_time, 'speaker': 'UNKNOWN'})
                
        merged = []
        for seg in segments:
            if merged and seg['start'] - merged[-1]['end'] < 0.3:
                merged[-1]['end'] = seg['end']
            else:
                merged.append(seg)
        return merged

    def _cluster_embeddings(self, embeddings: np.ndarray, max_speakers: int | None = None) -> np.ndarray:
        from sklearn.cluster import SpectralClustering, AgglomerativeClustering
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import normalize
        
        if max_speakers is None: max_speakers = self.max_speakers
        n_samples = len(embeddings)
        embeddings_norm = normalize(embeddings)
        
        if n_samples < 2: return np.zeros(n_samples, dtype=int)
        
        best_score = -1
        best_labels = None
        max_clusters = min(max_speakers, n_samples)
        
        for n_clusters in range(2, max_clusters + 1):
             try:
                 c = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
                 l = c.fit_predict(embeddings_norm)
                 if len(set(l)) > 1:
                     s = silhouette_score(embeddings_norm, l)
                     if s > best_score:
                         best_score = s
                         best_labels = l
             except: continue
             
        if best_labels is None:
            from sklearn.cluster import KMeans
            # Fallback to KMeans if spectral failed, respecting min/max
            k = max(2, min(max_speakers, n_samples))
            best_labels = KMeans(n_clusters=k).fit_predict(embeddings_norm)
            
        return best_labels

    def detect_genders(self, audio_path, segments):
        import librosa
        speaker_segments = {}
        for seg in segments:
            sp = seg['speaker']
            if sp not in speaker_segments: speaker_segments[sp] = []
            speaker_segments[sp].append((seg['start'], seg['end']))
            
        genders = {}
        try:
            y, sr = librosa.load(audio_path, sr=None, mono=True)
        except: return {sp: "Male" for sp in speaker_segments}
        
        for sp, times in speaker_segments.items():
            speaker_audio = []
            total = 0
            for s, e in times:
                s_idx, e_idx = int(s*sr), int(e*sr)
                if s_idx < e_idx and e_idx <= len(y):
                    speaker_audio.append(y[s_idx:e_idx])
                    total += (e-s)
                if total > 20: break
            
            if not speaker_audio:
                genders[sp] = "Male"
                continue
                
            full = np.concatenate(speaker_audio)
            try:
                f0, _, _ = librosa.pyin(full, fmin=65, fmax=300, sr=sr)
                f0 = f0[~np.isnan(f0)]
                if len(f0) > 0 and np.mean(f0) > 160: genders[sp] = "Female"
                else: genders[sp] = "Male"
            except: genders[sp] = "Male"
            
        return genders

    def extract_speaker_profiles(self, audio_path: str | Path, segments: list[dict], output_dir: str | Path) -> dict[str, str]:
        """
        Extracts clean audio samples (10-15s) for each speaker.
        Calculates SNR to pick best segments.
        Returns: {speaker_id: path_to_wav}
        """
        import soundfile as sf
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        audio, sr = sf.read(str(audio_path))
        if len(audio.shape) > 1: audio = audio.mean(axis=1) # Mono
        
        profiles = {}
        grouped = {}
        for seg in segments:
            if seg['speaker'] not in grouped: grouped[seg['speaker']] = []
            grouped[seg['speaker']].append(seg)
            
        for sp, segs in grouped.items():
            # [Fix] Filter out very short segments to avoid "Frankenstein" audio
            valid_segs = [s for s in segs if (s['end'] - s['start']) >= 0.5]
            
            if not valid_segs:
                 logger.warning(f"No valid segments (>0.5s) found for {sp}. Skipping profile creation.")
                 continue

            # Sort by length, preferring longer continuous segments
            sorted_segs = sorted(valid_segs, key=lambda x: x['end']-x['start'], reverse=True)
            
            samples = []
            total_dur = 0
            
            for seg in sorted_segs:
                start = int(seg['start'] * sr)
                end = int(seg['end'] * sr)
                
                chunk = audio[start:end]
                
                # [Improvement] RMS/Energy Check to avoid silent chunks
                rms = np.sqrt(np.mean(chunk**2))
                if rms < 0.01: # Skip near-silent chunks
                    continue
                
                # [Improvement] Normalize Chunk (-3dB)
                max_val = np.max(np.abs(chunk))
                if max_val > 0:
                     target_amp = 10 ** (-3/20) # -3dB
                     chunk = chunk * (target_amp / max_val)
                
                # [Improvement] Add Micro-Fades (10ms) to prevent clicks
                fade_len = int(0.01 * sr)
                if len(chunk) > 2 * fade_len:
                    fade_in = np.linspace(0, 1, fade_len)
                    fade_out = np.linspace(1, 0, fade_len)
                    chunk[:fade_len] *= fade_in
                    chunk[-fade_len:] *= fade_out
                
                samples.append(chunk)
                total_dur += (len(chunk) / sr)
                if total_dur >= 15.0: break
            
            # [Fix] Ensure the final profile is long enough for stable cloning (> 1.0s)
            if samples and total_dur >= 1.0:
                full_sp_audio = np.concatenate(samples)
                if len(full_sp_audio) > 15 * sr:
                    full_sp_audio = full_sp_audio[:15*sr]
                    
                out_path = output_dir / f"{sp}_profile.wav"
                sf.write(str(out_path), full_sp_audio, sr)
                profiles[sp] = str(out_path)
                logger.info(f"Created profile for {sp}: {out_path} ({len(full_sp_audio)/sr:.1f}s)")
            else:
                logger.warning(f"Profile for {sp} too short ({total_dur:.1f}s < 1.0s). Skipping to avoid TTS crash.")
                
        return profiles

if __name__ == "__main__":
    d = Diarizer()
