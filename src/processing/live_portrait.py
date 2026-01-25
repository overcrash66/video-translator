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

# Constants for LivePortrait face processing
LIVE_PORTRAIT_FACE_SCALE = 1.6  # Reduced from 2.3 to prevent oversized face output (matches Wav2Lip)
LIVE_PORTRAIT_INPUT_SIZE = 256  # Standard input size for ONNX models

class LivePortraitSyncer:
    """
    High-quality Lip Sync using LivePortrait (ONNX).
    Uses Wav2Lip to generate driving motion, then LivePortrait to animate the high-quality face.
    """
    def __init__(self):
        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device_str == "cuda" else ['CPUExecutionProvider']
        
        # [WinFix] Register torch's library path so ONNX Runtime can find CUDA/cuDNN DLLs
        # IMPORTANT: We must keep the return values of add_dll_directory alive, otherwise 
        # the paths are removed when the objects are garbage collected.
        self.dll_handles = []
        
        if os.name == 'nt' and self.device_str == "cuda":
            # 1. Register Torch Libs
            try:
                libs_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
                if os.path.exists(libs_path):
                    self.dll_handles.append(os.add_dll_directory(libs_path))
                    logger.info(f"Registered torch CUDA libs for ONNX: {libs_path}")
            except Exception as e:
                logger.warning(f"Could not register torch DLL path: {e}")
            
            # 2. Register System CUDA Libs (detect v12.x, v11.x, v13.x)
            try:
                 import glob
                 cuda_bases = glob.glob("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*")
                 if cuda_bases:
                     # Register ALL found versions to handle mixed dependency requirements
                     # e.g. onnxruntime might want v12 libs while system has v13 installed
                     for cuda_dir in cuda_bases:
                         # Check standard bin
                         bin_path = os.path.join(cuda_dir, "bin")
                         if os.path.exists(bin_path):
                             self.dll_handles.append(os.add_dll_directory(bin_path))
                             os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"] # Legacy PATH support
                             logger.info(f"Registered system CUDA bin for ONNX: {bin_path}")
                             
                         # Check bin/x64 (sometimes used in newer versions)
                         bin_x64_path = os.path.join(bin_path, "x64")
                         if os.path.exists(bin_x64_path):
                             self.dll_handles.append(os.add_dll_directory(bin_x64_path))
                             os.environ["PATH"] = bin_x64_path + os.pathsep + os.environ["PATH"] # Legacy PATH support
                             logger.info(f"Registered system CUDA bin/x64 for ONNX: {bin_x64_path}")
            except Exception as e:
                logger.warning(f"Could not register system CUDA path: {e}")

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
            
            # [Fix] GridSample on CUDA only supports 4D tensors, but LivePortrait uses 5D (volumetric).
            # We must force the Warping/SPADE module to run on CPU to avoid the crash.
            # Extractors will still run on GPU for speed.
            try:
                path_warp = str(self.model_dir / self.onnx_files["warping_spade"])
                if not Path(path_warp).exists():
                     raise FileNotFoundError(f"Model not found: {path_warp}")
                logger.info(f"Loading warping_spade on CPU (Required for 5D GridSample support)...")
                self.warping_module = ort.InferenceSession(path_warp, sess_opts, providers=['CPUExecutionProvider'])
            except Exception as e:
                logger.error(f"Failed to load warping_spade on CPU: {e}")
                raise

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
        
        Uses Wav2Lip to generate driving motion (low quality face, good lips),
        then LivePortrait to animate the original video (high quality face) using driving motion.
        
        Args:
            video_path: Path to the input video file.
            audio_path: Path to the audio file to sync lips to.
            output_path: Path where the output video will be saved.
            enhance_face: Not used in LivePortrait (included for API compatibility).
            
        Returns:
            Path to the output video file.
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
        """
        cap_src = cv2.VideoCapture(source_video)
        cap_drv = cv2.VideoCapture(driving_video)
        writer = None
        
        try:
            width = int(cap_src.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap_src.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap_src.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap_src.get(cv2.CAP_PROP_FRAME_COUNT))
            
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            
            pbar = tqdm(total=total_frames, desc="LivePortrait Inference")
            
            while True:
                ret_s, frame_s = cap_src.read()
                ret_d, frame_d = cap_drv.read()
                
                if not ret_s or not ret_d:
                    break
                    
                # Process Frame
                # 1. Detect Face & Crop
                face_info = self._detect(frame_s)
                
                if face_info is None:
                    writer.write(frame_s)
                    pbar.update(1)
                    continue
                    
                # 2. Prepare Inputs
                # Use source face info for BOTH to ensure consistent cropping region
                # This fixes the issue where driving video face scale mismatch causes 
                # the pasted face to be the wrong size/position
                crop_img, M = self._align_crop(frame_s, face_info)
                crop_drv, _ = self._align_crop(frame_d, face_info)

                # 3. Inference
                out_img = self._run_inference(crop_img, crop_drv)
                
                # 4. Paste Back
                final_frame = self._paste_back(out_img, frame_s, M)
                writer.write(final_frame)
                pbar.update(1)
                
        except Exception as e:
            logger.error(f"Animation loop error: {e}")
            raise
        finally:
            if cap_src: cap_src.release()
            if cap_drv: cap_drv.release()
            if writer: writer.release()
            if 'pbar' in locals(): pbar.close()

    def _detect(self, img):
        faces = self.face_analysis.get(img)
        if not faces:
            return None
        # Return largest face
        return sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))[-1]

    def _align_crop(self, img, face_info):
        """
        Aligns and crops face to standard size using InsightFace landmarks.
        
        Args:
            img: Input image (H, W, C).
            face_info: Face detection result from InsightFace.
            
        Returns:
            Tuple of (cropped_resized_image, inverse_affine_matrix).
            The matrix maps from crop space (256x256) back to original image space.
        """
        bbox = face_info.bbox
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        
        # Expansion factor used in LivePortrait
        scale = LIVE_PORTRAIT_FACE_SCALE
        size = int(max(w, h) * scale)
        
        # Square crop coordinates
        x1 = max(0, int(center[0] - size / 2))
        y1 = max(0, int(center[1] - size / 2))
        x2 = min(img.shape[1], x1 + size)
        y2 = min(img.shape[0], y1 + size)
        
        # Actual crop dimensions (may differ from 'size' if clamped)
        crop_w = x2 - x1
        crop_h = y2 - y1
        
        # Crop the region
        crop = img[y1:y2, x1:x2]
        
        # Resize to standard input size for ONNX models
        target_size = (LIVE_PORTRAIT_INPUT_SIZE, LIVE_PORTRAIT_INPUT_SIZE)
        crop_resized = cv2.resize(crop, target_size)
        
        # Calculate scale factors from crop to target size
        scale_x = target_size[0] / crop_w
        scale_y = target_size[1] / crop_h
        
        # Build INVERSE affine matrix: maps from target (256x256) back to original image
        # Forward transform is: u = (x - x1) * scale_x, v = (y - y1) * scale_y
        # Inverse is: x = u / scale_x + x1, y = v / scale_y + y1
        M_inv = np.array([
            [1.0 / scale_x, 0, x1],
            [0, 1.0 / scale_y, y1]
        ], dtype=np.float32)
        
        return crop_resized, M_inv

    def _paste_back(self, pred_img, bg_img, M_inv):
        """
        Pastes the predicted face back onto the background image.
        
        Args:
            pred_img: Generated face image (256x256).
            bg_img: Original background frame.
            M_inv: Inverse affine matrix from _align_crop (maps crop → original).
            
        Returns:
            Blended output frame with same dimensions as bg_img.
        """
        # Warp the predicted face to original image coordinates
        # M_inv already maps from crop space to original space
        output = cv2.warpAffine(
            pred_img, M_inv, (bg_img.shape[1], bg_img.shape[0]), 
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT
        )
        
        # Create mask for blending
        # Use a feathered mask to avoid hard square edges
        h, w = pred_img.shape[:2]
        mask_crop = np.zeros((h, w), dtype=np.float32)
        
        # Define a center rectangle with margin to blur edges
        margin = int(min(h, w) * 0.05) # 5% margin
        cv2.rectangle(mask_crop, (margin, margin), (w-margin, h-margin), 1.0, -1)
        
        # Apply Gaussian blur to soften the transition
        mask_crop = cv2.GaussianBlur(mask_crop, (51, 51), 0)
        
        # Expand to 3 channels and scale to 255
        mask_crop = np.stack([mask_crop]*3, axis=-1) * 255

        mask_full = cv2.warpAffine(
             mask_crop, M_inv, (bg_img.shape[1], bg_img.shape[0]),
             flags=cv2.INTER_LINEAR
        )
        
        # Blend using the mask
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
        
        # Motion Extractor returns multiple outputs. 
        # Based on debug, Output 6 (index 6, or sometimes others depending on version) is KP.
        # Shape is (1, 63). We need (1, 21, 3).
        
        # Run and capture all outputs
        res_s = self.motion_extractor.run(None, {input_name_mot: t_src})
        res_d = self.motion_extractor.run(None, {input_name_mot: t_drv})
        
        # Heuristic: Find the output with size 63 (21*3)
        # Usually it is the last one or one of the middle ones.
        # In Warmshao v1.1 optimization, it seems to be index 6.
        
        def extract_kp(outputs):
            for out in outputs:
                if out.shape == (1, 63):
                    return out.reshape(1, 21, 3)
            # Fallback if specific shape not found (maybe (1, 21, 3) already?)
            for out in outputs:
                if out.shape == (1, 21, 3):
                    return out
            raise ValueError(f"Could not find Keypoint output in Motion Extractor. Shapes: {[o.shape for o in outputs]}")

        kp_s = extract_kp(res_s)
        kp_d = extract_kp(res_d)
        
        # 2. Warping + SPADE (Merged in 'warping_spade.onnx')
        warp_inputs = self.warping_module.get_inputs()
        
        feed_dict = {
            warp_inputs[0].name: f_s,
            warp_inputs[1].name: kp_d, # Driving KP (kp_driving)
            warp_inputs[2].name: kp_s  # Source KP (kp_source)
        }
        
        # Run merged inference
        out_gen = self.warping_module.run(None, feed_dict)[0]
        
        # Post-process
        # Output is usually N,C,H,W in 0-1 range
        res = np.clip(out_gen, 0, 1) * 255
        res = np.transpose(res[0], (1, 2, 0)) # H,W,C
        return res.astype(np.uint8)

