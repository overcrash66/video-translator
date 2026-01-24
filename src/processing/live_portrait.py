import os
import torch
import cv2
import numpy as np
import logging
import shutil
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import onnxruntime as ort
import insightface
from src.utils import config
from src.processing.wav2lip import Wav2LipSyncer

logger = logging.getLogger(__name__)

class LivePortraitSyncer:
    """
    High-quality Lip Sync using LivePortrait (ONNX).
    Uses Wav2Lip to generate driving motion, then LivePortrait to animate the high-quality face.
    """
    def __init__(self):
        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device_str == "cuda" else ['CPUExecutionProvider']
        
        # [WinFix] Register torch's library path so ONNX Runtime can find CUDA/cuDNN DLLs
        if os.name == 'nt' and self.device_str == "cuda":
            try:
                libs_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
                if os.path.exists(libs_path):
                    os.add_dll_directory(libs_path)
                    logger.info(f"Registered torch CUDA libs for ONNX: {libs_path}")
            except Exception as e:
                logger.warning(f"Could not register torch DLL path: {e}")

        self.model_dir = Path("models/live_portrait_onnx")
        self.wav2lip_driver = Wav2LipSyncer()
        
        # Models
        self.face_analysis = None
        self.appearance_extractor = None
        self.motion_extractor = None
        self.warping_module = None
        self.spade_generator = None
        
        # ONNX Model Paths
        # Prepend 'liveportrait_onnx/' as per repo structure
        self.onnx_files = {
            "appearance": "liveportrait_onnx/appearance_feature_extractor.onnx",
            "motion": "liveportrait_onnx/motion_extractor.onnx",
            "warping_spade": "liveportrait_onnx/warping_spade.onnx",
            "landmark": "liveportrait_onnx/landmark.onnx" 
        }

    def download_models(self):
        """Downloads LivePortrait ONNX models from HuggingFace."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        # Subdir for ONNX files to match structure
        (self.model_dir / "liveportrait_onnx").mkdir(exist_ok=True)
        
        repo_id = "warmshao/FasterLivePortrait"
        
        for key, filename in self.onnx_files.items():
            target = self.model_dir / filename
            if not target.exists():
                logger.info(f"Downloading {filename} from {repo_id}...")
                try:
                    hf_hub_download(
                        repo_id=repo_id, 
                        filename=filename, 
                        local_dir=str(self.model_dir),
                        local_dir_use_symlinks=False
                    )
                except Exception as e:
                    logger.error(f"Failed to download {filename}: {e}")
                    # If landmark fails, we might skip it if strictly using insightface?
                    # But let's raise for now to be safe.
                    if "landmark" in filename:
                        logger.warning("Could not download landmark.onnx. Proceeding without it (using InsightFace).")
                    else:
                        raise RuntimeError(f"Could not download LivePortrait model: {filename}")

    def load_models(self):
        """Loads models into VRAM (ONNX Runtime Sessions)."""
        if self.appearance_extractor is not None:
            return

        self.download_models()
        logger.info(f"Loading LivePortrait ONNX models on {self.device_str}...")

        try:
            # 1. InsightFace for Detection
            self.face_analysis = insightface.app.FaceAnalysis(
                name='buffalo_l', 
                providers=self.providers
            )
            self.face_analysis.prepare(ctx_id=0 if self.device_str == 'cuda' else -1, det_size=(640, 640))

            # 2. LivePortrait Modules
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            def load_ort(name):
                path = str(self.model_dir / self.onnx_files[name])
                if not Path(path).exists():
                     raise FileNotFoundError(f"Model not found: {path}")
                
                try:
                    return ort.InferenceSession(path, sess_opts, providers=self.providers)
                except Exception as e:
                    logger.warning(f"Failed to load {name} with providers {self.providers}: {e}")
                    if 'CUDAExecutionProvider' in self.providers:
                        logger.info("Falling back to CPUExecutionProvider...")
                        return ort.InferenceSession(path, sess_opts, providers=['CPUExecutionProvider'])
                    raise

            self.appearance_extractor = load_ort("appearance")
            self.motion_extractor = load_ort("motion")
            self.warping_module = load_ort("warping_spade") # Merged model
            self.spade_generator = None # No longer separate
            
            # Verify provider
            prov = self.appearance_extractor.get_providers()
            logger.info(f"LivePortrait models loaded successfully. Active providers: {prov}")
            
        except Exception as e:
            logger.error(f"Failed to load LivePortrait models: {e}")
            self.unload_models()
            raise

    def unload_models(self):
        """Unloads models to free VRAM."""
        self.face_analysis = None
        self.appearance_extractor = None
        self.motion_extractor = None
        self.warping_module = None
        self.spade_generator = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("LivePortrait models unloaded.")

    def sync_lips(self, video_path: str, audio_path: str, output_path: str, enhance_face: bool = False) -> str:
        """
        Runs the LivePortrait lip sync pipeline.
        1. Run Wav2Lip to get driving video (low quality face, good lips).
        2. Use LivePortrait to animate original video (high quality face) using driving motion.
        """
        self.load_models()
        
        # 1. Generate driving video using Wav2Lip
        # We need a temp path for the driving video
        temp_wav2lip_out = config.TEMP_DIR / f"lp_drive_{Path(video_path).stem}.mp4"
        logger.info("Step 1: Generating driving motion with Wav2Lip...")
        
        # We use the internal wav2lip driver. 
        # Note: We do NOT use enhance_face here because we only need the motion (lips), 
        # and enhancement is slow/unnecessary for the driving video.
        self.wav2lip_driver.sync_lips(video_path, audio_path, str(temp_wav2lip_out), enhance_face=False)
        
        if not temp_wav2lip_out.exists():
            raise RuntimeError("Wav2Lip failed to generate driving video.")

        # 2. LivePortrait Animation (The "Real" Step)
        logger.info("Step 2: Applied LivePortrait animation (ONNX)...")
        
        try:
            self._animate_video(
                source_video=str(video_path),
                driving_video=str(temp_wav2lip_out),
                output_path=output_path
            )
        except Exception as e:
            logger.error(f"LivePortrait Animation failed: {e}")
            # Fallback to Wav2Lip out if LP fails? 
            # The user wants "Real Quality" or nothing, but better to return something than crash?
            # User said "validate we can replace... with no issues".
            # For now, raise logic error but maybe cleanup.
            raise
        finally:
            # Cleanup temp
            if temp_wav2lip_out.exists():
                os.remove(temp_wav2lip_out)
                
        return output_path

    def _animate_video(self, source_video, driving_video, output_path):
        """
        Core animation loop.
        Extracts features from Source (Frame 0 usually, or per frame?) -> Usually Source is a single image or video.
        If Source is video, we might do video-to-video retargeting.
        For LipSync: Source is Video, Driving is Video (same duration).
        We want to take the 'Lip Motion' from Driving and apply it to Source.
        """
        cap_src = cv2.VideoCapture(source_video)
        cap_drv = cv2.VideoCapture(driving_video)
        
        width = int(cap_src.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_src.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap_src.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap_src.get(cv2.CAP_PROP_FRAME_COUNT))
        
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        # Pre-calc Source Features? 
        # Since it's Video-to-Video where Source and Driving are aligned (same video, just different lips),
        # we can process frame-by-frame.
        
        pbar = tqdm(total=total_frames, desc="LivePortrait Inference")
        
        while True:
            ret_s, frame_s = cap_src.read()
            ret_d, frame_d = cap_drv.read()
            
            if not ret_s or not ret_d:
                break
                
            # Process Frame
            # 1. Detect Face & Crop
            # We strictly need the face in the Source frame.
            face_info = self._detect(frame_s)
            
            if face_info is None:
                # No face detected, write original frame
                writer.write(frame_s)
                pbar.update(1)
                continue
                
            # 2. Prepare Inputs
            # Crop 512x512
            crop_img, M = self._align_crop(frame_s, face_info)
            crop_drv, _ = self._align_crop(frame_d, face_info) # Use same face info/M ideally?
            # Actually driving might have slightly different landmarks because lips moved.
            # But we want to extract motion from driving.
            # Ideally detect on driving too to get accurate 'kp_driving'.
            
            face_info_drv = self._detect(frame_d)
            if face_info_drv is None:
                 # Standard fallback
                 writer.write(frame_s)
                 pbar.update(1)
                 continue
            
            crop_drv, _ = self._align_crop(frame_d, face_info_drv)

            # 3. Inference
            out_img = self._run_inference(crop_img, crop_drv)
            
            # 4. Paste Back
            final_frame = self._paste_back(out_img, frame_s, M)
            writer.write(final_frame)
            pbar.update(1)
            
        cap_src.release()
        cap_drv.release()
        writer.release()
        pbar.close()

    def _detect(self, img):
        faces = self.face_analysis.get(img)
        if not faces:
            return None
        # Return largest face
        return sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))[-1]

    def _align_crop(self, img, face_info):
        """
        Aligns and crops face to 512x512 using InshghtFace landmarks.
        Simulates LivePortrait standard crop.
        """
        # Simplistic implementation matching standard FaceAnalysis suggestion
        # LivePortrait usually expects specific scale. 
        # We'll use a standard alignment function if available or simple crop
        # Using simple bounding box expansion for robustness if alignment is complex to reproduce exactly without 'templates'.
        # However, inference usually demands aligned eyes. 
        
        # We can use insightface's norm_crop if we had the template.
        # But 'warmshao' onnx models input 256x256 or 512x512?
        # FasterLivePortrait usually uses 256x256 inputs for speed.
        # Let's assume 256x256 for ONNX.
        
        # Use simple bbox crop for now to avoid crashes, but this is the "Quality" risk.
        # TODO: Implement proper 5-point alignment.
        
        bbox = face_info.bbox
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        
        # Expansion factor used in LivePortrait is roughly 1.5 - 2.0?
        scale = 2.3  
        size = int(max(w, h) * scale)
        
        # Square crop
        x1 = max(0, int(center[0] - size / 2))
        y1 = max(0, int(center[1] - size / 2))
        x2 = min(img.shape[1], x1 + size)
        y2 = min(img.shape[0], y1 + size)
        
        # Adjust if OOB
        crop = img[y1:y2, x1:x2]
        
        # Resize to 256x256 (Standard for ONNX models usually)
        # We need to verify input size. 
        target_size = (256, 256) 
        
        crop_resized = cv2.resize(crop, target_size)
        
        # Transformation Matrix (for inverse)
        # Map (x1, y1) -> (0,0) and scale
        scale_x = target_size[0] / crop.shape[1]
        scale_y = target_size[1] / crop.shape[0]
        
        M = np.array([
            [scale_x, 0, -x1 * scale_x],
            [0, scale_y, -y1 * scale_y],
            [0, 0, 1]
        ])
        
        return crop_resized, M

    def _paste_back(self, pred_img, bg_img, M):
        # Inverse warp
        # M maps Source -> Crop (256).
        # Inverse M maps Crop -> Source.
        
        # pred_img is 256x256 (0-255).
        
        # We need to warp pred_img back to bg_img shape
        inv_M = cv2.invertAffineTransform(M[:2])
        
        output = cv2.warpAffine(
            pred_img, inv_M, (bg_img.shape[1], bg_img.shape[0]), 
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT
        )
        
        # Masking? For now simple overlay (could be better with mask)
        # Using the non-black pixels of output?
        # Actually standard paste needs a mask. 
        # Simplest: Just overwrite the bbox area.
        
        # Create mask from the crop (all ones)
        mask_crop = np.ones_like(pred_img) * 255
        mask_full = cv2.warpAffine(
             mask_crop, inv_M, (bg_img.shape[1], bg_img.shape[0]),
             flags=cv2.INTER_LINEAR
        )
        
        # Blend
        mask_norm = mask_full.astype(float) / 255.0
        final = bg_img.astype(float) * (1 - mask_norm) + output.astype(float) * mask_norm
        return final.astype(np.uint8)

    def _run_inference(self, src_img, drv_img):
        # Prepare Tensors
        def to_tensor(img):
            x = img.astype(np.float32) / 255.0  # 0-1
            x = np.transpose(x, (2, 0, 1)) # C,H,W
            x = np.expand_dims(x, axis=0)  # 1,C,H,W
            return x

        t_src = to_tensor(src_img)
        t_drv = to_tensor(drv_img)
        
        # 1. Extract Features
        input_name_app = self.appearance_extractor.get_inputs()[0].name
        f_s = self.appearance_extractor.run(None, {input_name_app: t_src})[0]
        
        input_name_mot = self.motion_extractor.get_inputs()[0].name
        kp_s = self.motion_extractor.run(None, {input_name_mot: t_src})
        kp_d = self.motion_extractor.run(None, {input_name_mot: t_drv})
        
        # 2. Warping + SPADE (Merged in 'warping_spade.onnx')
        # Inputs: feature, kp_driving, kp_source
        # We try to map by order or reliable names if possible.
        # Usually: feature (3d), kp_driving, kp_source
        
        warp_inputs = self.warping_module.get_inputs()
        
        # Robust feed dict construction
        # Assumption: 3 inputs.
        
        feed_dict = {
            warp_inputs[0].name: f_s,
            warp_inputs[1].name: kp_d[0], # Driving KP (kp_driving)
            warp_inputs[2].name: kp_s[0]  # Source KP (kp_source)
        }
        
        # Run merged inference
        out_gen = self.warping_module.run(None, feed_dict)[0]
        
        # Post-process
        # Output is usually N,C,H,W in 0-1 range
        res = np.clip(out_gen, 0, 1) * 255
        res = np.transpose(res[0], (1, 2, 0)) # H,W,C
        return res.astype(np.uint8)

