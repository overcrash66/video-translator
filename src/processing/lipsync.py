"""
MuseTalk Lip-Sync Integration

This module provides lip synchronization for translated videos using the MuseTalk model.
MuseTalk is a real-time high-fidelity lip-sync model that modifies face regions to match audio.
"""
import logging
import torch
import gc
import os
import sys
import glob
import copy
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.utils import config
from src.processing.wav2lip import Wav2LipSyncer

logger = logging.getLogger(__name__)

# MuseTalk path configuration
MUSETALK_ROOT = Path(__file__).parent.parent.parent / "MuseTalk"


class LipSyncer:
    """
    Wrapper for MuseTalk (Real-time Audio-Driven Lip Syncing).
    Uses MuseTalk v1.5 for improved quality and lip-sync accuracy.
    """
    
    def __init__(self):
        self.model_loaded = False
        self.device = None
        self.vae = None
        self.unet = None
        self.pe = None
        self.whisper = None
        self.audio_processor = None
        self.face_parser = None
        self.use_float16 = True  # For VRAM efficiency
        
        # Model paths
        self.models_dir = MUSETALK_ROOT / "models"
        self.unet_model_path = self.models_dir / "musetalkV15" / "unet.pth"
        self.unet_config = self.models_dir / "musetalkV15" / "musetalk.json"
        self.whisper_dir = self.models_dir / "whisper"
        self.vae_type = "sd-vae"
        
        # Inference parameters
        self.batch_size = 8
        self.fps = 25
        self.extra_margin = 10
        self.version = "v15"
        self.parsing_mode = "jaw"

        self.wav2lip_engine = None
        
    def _get_wav2lip(self):
        if self.wav2lip_engine is None:
            self.wav2lip_engine = Wav2LipSyncer()
        return self.wav2lip_engine
        
    def _setup_paths(self):
        """Add MuseTalk directories to Python path."""
        musetalk_str = str(MUSETALK_ROOT)
        if musetalk_str not in sys.path:
            sys.path.insert(0, musetalk_str)
        logger.info(f"MuseTalk root: {MUSETALK_ROOT}")
        
    def _check_models_exist(self) -> bool:
        """Verify all required model files exist."""
        required_files = [
            self.unet_model_path,
            self.unet_config,
            self.whisper_dir / "pytorch_model.bin",
            self.models_dir / self.vae_type / "diffusion_pytorch_model.bin",
            self.models_dir / "dwpose" / "dw-ll_ucoco_384.pth",
        ]
        
        for f in required_files:
            if not f.exists():
                logger.error(f"Missing MuseTalk model file: {f}")
                return False
        return True
        
    def load_model(self):
        """Load all MuseTalk models for inference."""
        if self.model_loaded:
            return True
            
        logger.info("Loading MuseTalk models...")
        
        # Setup paths first
        self._setup_paths()
        
        # Check models exist
        if not self._check_models_exist():
            logger.error("MuseTalk models not found. Please run download_weights.bat in MuseTalk folder.")
            return False
            
        try:
            # Change working directory temporarily for MuseTalk imports
            original_cwd = os.getcwd()
            os.chdir(str(MUSETALK_ROOT))
            
            try:
                # Import MuseTalk modules
                from musetalk.utils.utils import load_all_model, get_file_type, get_video_fps, datagen
                from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
                from musetalk.utils.blending import get_image
                from musetalk.utils.face_parsing import FaceParsing
                from musetalk.utils.audio_processor import AudioProcessor
                from transformers import WhisperModel
                
                # Store imports for later use
                self._imports = {
                    'load_all_model': load_all_model,
                    'get_file_type': get_file_type,
                    'get_video_fps': get_video_fps,
                    'datagen': datagen,
                    'get_landmark_and_bbox': get_landmark_and_bbox,
                    'read_imgs': read_imgs,
                    'coord_placeholder': coord_placeholder,
                    'get_image': get_image,
                    'FaceParsing': FaceParsing,
                    'AudioProcessor': AudioProcessor,
                    'WhisperModel': WhisperModel,
                }
                
            finally:
                os.chdir(original_cwd)
            
            # Set device
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # Change to MuseTalk dir for model loading (paths are relative)
            os.chdir(str(MUSETALK_ROOT))
            
            try:
                # Load models
                self.vae, self.unet, self.pe = self._imports['load_all_model'](
                    unet_model_path=str(self.unet_model_path),
                    vae_type=self.vae_type,
                    unet_config=str(self.unet_config),
                    device=self.device
                )
                
                # Convert to half precision for VRAM efficiency
                if self.use_float16:
                    self.pe = self.pe.half()
                    self.vae.vae = self.vae.vae.half()
                    self.unet.model = self.unet.model.half()
                
                # Move to device
                self.pe = self.pe.to(self.device)
                self.vae.vae = self.vae.vae.to(self.device)
                self.unet.model = self.unet.model.to(self.device)
                
                # Load audio processor and Whisper
                self.audio_processor = self._imports['AudioProcessor'](
                    feature_extractor_path=str(self.whisper_dir)
                )
                
                weight_dtype = self.unet.model.dtype
                self.whisper = self._imports['WhisperModel'].from_pretrained(str(self.whisper_dir))
                self.whisper = self.whisper.to(device=self.device, dtype=weight_dtype).eval()
                self.whisper.requires_grad_(False)
                
                # Initialize face parser (v1.5 uses default params)
                self.face_parser = self._imports['FaceParsing']()
                
            finally:
                os.chdir(original_cwd)
            
            self.model_loaded = True
            logger.info("MuseTalk models loaded successfully.")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import MuseTalk modules: {e}")
            logger.error("Please ensure MuseTalk dependencies are installed (mmcv, mmpose, mmdet, etc.)")
            self.model_loaded = False
            return False
            
        except Exception as e:
            if "CUDA" in str(e) and self.device.type == "cuda":
                logger.warning(f"CUDA Error encountered during MuseTalk loading: {e}")
                logger.warning("Switching to CPU fallback for MuseTalk...")
                
                # Cleanup potential partial loads
                self.unload_model()
                
                # Switch to CPU
                self.device = torch.device("cpu")
                
                # Retry loading
                return self.load_model()
            
            logger.error(f"Failed to load MuseTalk models: {e}")
            self.model_loaded = False
            return False

    def unload_model(self):
        """Unload all models and free VRAM."""
        if self.vae:
            del self.vae
            self.vae = None
            
        if self.unet:
            del self.unet
            self.unet = None
            
        if self.pe:
            del self.pe
            self.pe = None
            
        if self.whisper:
            del self.whisper
            self.whisper = None
            
        if self.audio_processor:
            del self.audio_processor
            self.audio_processor = None
            
        if self.face_parser:
            del self.face_parser
            self.face_parser = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        gc.collect()
        self.model_loaded = False
        logger.info("MuseTalk models unloaded.")

    def sync_lips(self, video_path: str, audio_path: str, output_path: str, model_name: str = "musetalk") -> str:
        """
        Synchronizes lips in video_path to match audio_path.
        
        Args:
            video_path: Path to input video with face
            audio_path: Path to audio file to sync lips to
            output_path: Path for output video
            model_name: 'musetalk' or 'wav2lip'
            
        Returns:
            Path to output video
        """
        if model_name.lower() == "wav2lip":
            try:
                engine = self._get_wav2lip()
                return engine.sync_lips(video_path, audio_path, output_path)
            except Exception as e:
                logger.error(f"Wav2Lip failed: {e}")
                return video_path # Fallback or Raise?

        # Default to MuseTalk
        return self._sync_musetalk(video_path, audio_path, output_path)

    @torch.no_grad()
    def _sync_musetalk(self, video_path: str, audio_path: str, output_path: str) -> str:
        """Original MuseTalk pipeline"""
        # Ensure models are loaded
        if not self.model_loaded:
            if not self.load_model():
                logger.warning("MuseTalk not available. Returning original video.")
                shutil.copy(video_path, output_path)
                return output_path
                
        logger.info(f"Running Lip-Sync on {video_path} with {audio_path}...")
        
        # Create temp directory for processing
        temp_dir = config.TEMP_DIR / "lipsync_temp"
        temp_dir.mkdir(exist_ok=True)
        
        original_cwd = os.getcwd()
        
        try:
            # Change to MuseTalk directory for proper path resolution
            os.chdir(str(MUSETALK_ROOT))
            
            video_path = Path(video_path)
            audio_path = Path(audio_path)
            output_path = Path(output_path)
            
            # Get utilities from stored imports
            get_file_type = self._imports['get_file_type']
            get_video_fps = self._imports['get_video_fps']
            datagen = self._imports['datagen']
            get_landmark_and_bbox = self._imports['get_landmark_and_bbox']
            read_imgs = self._imports['read_imgs']
            coord_placeholder = self._imports['coord_placeholder']
            get_image = self._imports['get_image']
            
            # Setup paths
            input_basename = video_path.stem
            audio_basename = audio_path.stem
            output_basename = f"{input_basename}_{audio_basename}"
            
            frames_dir = temp_dir / f"{input_basename}_frames"
            result_img_dir = temp_dir / f"{output_basename}_results"
            frames_dir.mkdir(exist_ok=True)
            result_img_dir.mkdir(exist_ok=True)
            
            # Extract frames from video
            logger.info("Extracting frames from video...")
            if get_file_type(str(video_path)) == "video":
                cmd = f'ffmpeg -y -v warning -i "{video_path}" -start_number 0 "{frames_dir}/%08d.png"'
                os.system(cmd)
                input_img_list = sorted(glob.glob(str(frames_dir / "*.[pP][nN][gG]")))
                if not input_img_list:
                    input_img_list = sorted(glob.glob(str(frames_dir / "*.[jJ][pP][gG]")))
                fps = get_video_fps(str(video_path))
            else:
                raise ValueError(f"Input must be a video file: {video_path}")
                
            if not input_img_list:
                raise ValueError("Failed to extract frames from video")
                
            logger.info(f"Extracted {len(input_img_list)} frames at {fps} fps")
            
            # Extract audio features
            logger.info("Extracting audio features...")
            whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(str(audio_path))
            
            if whisper_input_features is None:
                raise ValueError(f"Failed to load audio: {audio_path}")
                
            weight_dtype = self.unet.model.dtype
            whisper_chunks = self.audio_processor.get_whisper_chunk(
                whisper_input_features,
                self.device,
                weight_dtype,
                self.whisper,
                librosa_length,
                fps=fps,
                audio_padding_length_left=2,
                audio_padding_length_right=2,
            )
            
            logger.info(f"Generated {len(whisper_chunks)} audio chunks")
            
            # Get landmarks and bounding boxes
            logger.info("Detecting faces and extracting landmarks...")
            coord_list, frame_list = get_landmark_and_bbox(input_img_list, upperbondrange=0)
            
            logger.info(f"Processed {len(frame_list)} frames for face detection")
            
            # Encode frames to latent space
            logger.info("Encoding frames to latent space...")
            input_latent_list = []
            for bbox, frame in tqdm(zip(coord_list, frame_list), total=len(frame_list), desc="Encoding"):
                if bbox == coord_placeholder:
                    continue
                x1, y1, x2, y2 = bbox
                if self.version == "v15":
                    y2 = y2 + self.extra_margin
                    y2 = min(y2, frame.shape[0])
                crop_frame = frame[y1:y2, x1:x2]
                crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                latents = self.vae.get_latents_for_unet(crop_frame)
                input_latent_list.append(latents)
                
            if not input_latent_list:
                raise ValueError("No faces detected in video frames")
                
            # Create cycled lists for smooth looping
            frame_list_cycle = frame_list + frame_list[::-1]
            coord_list_cycle = coord_list + coord_list[::-1]
            input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
            
            # Run inference
            logger.info("Running lip-sync inference...")
            timesteps = torch.tensor([0], device=self.device)
            video_num = len(whisper_chunks)
            
            gen = datagen(
                whisper_chunks=whisper_chunks,
                vae_encode_latents=input_latent_list_cycle,
                batch_size=self.batch_size,
                delay_frame=0,
                device=self.device,
            )
            
            res_frame_list = []
            total_batches = int(np.ceil(float(video_num) / self.batch_size))
            
            for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=total_batches, desc="Inference")):
                audio_feature_batch = self.pe(whisper_batch)
                latent_batch = latent_batch.to(dtype=self.unet.model.dtype)
                
                pred_latents = self.unet.model(
                    latent_batch, 
                    timesteps, 
                    encoder_hidden_states=audio_feature_batch
                ).sample
                
                recon = self.vae.decode_latents(pred_latents)
                for res_frame in recon:
                    res_frame_list.append(res_frame)
                    
            logger.info(f"Generated {len(res_frame_list)} lip-synced frames")
            
            # Blend results back into original frames
            logger.info("Blending results...")
            for i, res_frame in enumerate(tqdm(res_frame_list, desc="Blending")):
                bbox = coord_list_cycle[i % len(coord_list_cycle)]
                ori_frame = copy.deepcopy(frame_list_cycle[i % len(frame_list_cycle)])
                x1, y1, x2, y2 = bbox
                
                if self.version == "v15":
                    y2 = y2 + self.extra_margin
                    y2 = min(y2, ori_frame.shape[0])
                    
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                except Exception:
                    continue
                    
                # Merge with face parsing for natural blending
                if self.version == "v15":
                    combine_frame = get_image(
                        ori_frame, res_frame, [x1, y1, x2, y2], 
                        mode=self.parsing_mode, fp=self.face_parser
                    )
                else:
                    combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], fp=self.face_parser)
                    
                cv2.imwrite(str(result_img_dir / f"{str(i).zfill(8)}.png"), combine_frame)
                
            # Create output video
            logger.info("Creating output video...")
            temp_video = temp_dir / f"temp_{output_basename}.mp4"
            
            cmd_video = f'ffmpeg -y -v warning -r {fps} -f image2 -i "{result_img_dir}/%08d.png" -vcodec libx264 -vf format=yuv420p -crf 18 "{temp_video}"'
            os.system(cmd_video)
            
            # Combine with audio
            cmd_audio = f'ffmpeg -y -v warning -i "{audio_path}" -i "{temp_video}" -c:v copy -c:a aac "{output_path}"'
            os.system(cmd_audio)
            
            # Cleanup
            try:
                shutil.rmtree(frames_dir)
                shutil.rmtree(result_img_dir)
                if temp_video.exists():
                    temp_video.unlink()
            except Exception as e:
                logger.warning(f"Cleanup warning: {e}")
                
            if output_path.exists():
                logger.info(f"Lip-Sync complete. Saved to {output_path}")
                return str(output_path)
            else:
                raise ValueError("Output video was not created")
                
        except Exception as e:
            logger.error(f"Lip-Sync failed: {e}")
            # Fallback: copy original video
            try:
                shutil.copy(video_path, output_path)
                return str(output_path)
            except Exception:
                return str(video_path)
            
        finally:
            os.chdir(original_cwd)
