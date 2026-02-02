import os
import torch
import cv2
import numpy as np
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Any, Tuple, List, Union
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import onnxruntime as ort
import insightface
from src.utils import config
from src.processing.wav2lip import Wav2LipSyncer

# Optional TensorRT imports
# Optional TensorRT imports
try:
    import tensorrt as trt
    import ctypes
    TRT_AVAILABLE = True
except ImportError:
    trt = None
    ctypes = None
    TRT_AVAILABLE = False
    logging.getLogger(__name__).warning("TensorRT or PyCUDA not found. TensorRT acceleration will be unavailable.")

class TRTWrapper:
    """
    Wrapper for TensorRT engine to mimic ONNX Runtime InferenceSession interface.
    Handles buffer allocation, data transfer, and execution using PyTorch (no PyCUDA).
    """
    
    # Class-level flag to ensure TF32 override is set only once per process
    _tf32_override_set = False
    
    @staticmethod
    def _ensure_tf32_consistency():
        """
        Ensures consistent NVIDIA_TF32_OVERRIDE setting between engine build and runtime.
        
        TensorRT engines cache the TF32 mode setting at build time. If this environment
        variable differs at runtime, TensorRT will fail with:
        "Inconsistent setting of NVIDIA_TF32_OVERRIDE env var at build X and execution Y"
        
        The engines were typically built with TF32_OVERRIDE unset (-1), so we need to
        ensure it's also unset at runtime.
        """
        if not TRTWrapper._tf32_override_set:
            # Remove the environment variable if set, to match build-time state (unset = -1)
            if 'NVIDIA_TF32_OVERRIDE' in os.environ:
                logging.getLogger(__name__).info(
                    f"Removing NVIDIA_TF32_OVERRIDE env var (was: {os.environ['NVIDIA_TF32_OVERRIDE']}) "
                    "to match TensorRT engine build-time settings."
                )
                del os.environ['NVIDIA_TF32_OVERRIDE']
            TRTWrapper._tf32_override_set = True
    
    def __init__(self, engine_path: str):
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT dependencies not installed.")
        
        # Ensure TF32 override consistency BEFORE loading/deserializing any TRT engine
        TRTWrapper._ensure_tf32_consistency()
            
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
        if not self.engine:
            raise RuntimeError(f"Failed to load TRT engine: {engine_path}")
            
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        
        # Handle both legacy (< 10.0) and modern (10.0+) TRT APIs
        self.is_legacy = not hasattr(self.engine, "num_io_tensors")
        self.tensor_names = []
        
        if self.is_legacy:
            num_bindings = self.engine.num_bindings
            self.tensor_names = [self.engine.get_binding_name(i) for i in range(num_bindings)]
        else:
            num_tensors = self.engine.num_io_tensors
            self.tensor_names = [self.engine.get_tensor_name(i) for i in range(num_tensors)]
        
        # Allocate buffers using PyTorch
        self.inputs = []
        self.outputs = []
        self.input_names = []
        self.output_names = []
        
        # Mapping of name -> device tensor
        self.tensors = {}
        
        for name in self.tensor_names:
            if self.is_legacy:
                # Legacy indexing
                idx = self.engine.get_binding_index(name)
                shape = self.engine.get_binding_shape(idx)
                dtype_trt = self.engine.get_binding_dtype(idx)
                is_input = self.engine.binding_is_input(idx)
            else:
                # Modern API
                shape = self.engine.get_tensor_shape(name)
                dtype_trt = self.engine.get_tensor_dtype(name)
                is_input = (self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
            
            # Map TRT dtype to Torch dtype
            dtype_map = {
                trt.float32: torch.float32,
                trt.float16: torch.float16,
                trt.int32:   torch.int32,
                trt.int8:    torch.int8,
                trt.bool:    torch.bool,
            }
            torch_dtype = dtype_map.get(dtype_trt, torch.float32)
            
            # Create device buffer
            safe_shape = tuple([s if s > 0 else 1 for s in shape])
            device_tensor = torch.zeros(safe_shape, dtype=torch_dtype, device='cuda')
            
            self.tensors[name] = device_tensor
            
            if is_input:
                self.inputs.append({'tensor': device_tensor, 'name': name})
                self.input_names.append(name)
            else:
                self.outputs.append({'tensor': device_tensor, 'name': name})
                self.output_names.append(name)

    def get_inputs(self):
        """Mock method to match ORT session.get_inputs()[0].name usage."""
        class MockNode:
            def __init__(self, name): self.name = name
        return [MockNode(n) for n in self.input_names]

    def run(self, output_names, input_feed):
        """
        Execute inference.
        Args:
            output_names: List of output names to return (ignored, returns all ordered by binding).
            input_feed: Dict {input_name: numpy_array}.
        """
        # 1. Copy inputs to device tensors
        for inp in self.inputs:
            name = inp['name']
            if name in input_feed:
                data = input_feed[name]
                if not data.flags['C_CONTIGUOUS']:
                    data = np.ascontiguousarray(data)
                
                src_tensor = torch.from_numpy(data).to('cuda')
                
                # Cast if necessary
                if src_tensor.dtype != inp['tensor'].dtype:
                    src_tensor = src_tensor.to(inp['tensor'].dtype)
                
                inp['tensor'].copy_(src_tensor)
                
        # 2. Execute
        if self.is_legacy:
            bindings = [self.tensors[n].data_ptr() for n in self.tensor_names]
            self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.cuda_stream)
        else:
            for name, tensor in self.tensors.items():
                self.context.set_tensor_address(name, tensor.data_ptr())
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        
        # 3. Copy outputs back
        results = []
        for out in self.outputs:
            host_data = out['tensor'].cpu().numpy()
            results.append(host_data)
            
        self.stream.synchronize()
        return results

logger = logging.getLogger(__name__)

# Constants for LivePortrait face processing
LIVE_PORTRAIT_FACE_SCALE = 2.3  # Standard scale for LivePortrait to match training distribution
LIVE_PORTRAIT_INPUT_SIZE = 256  # Standard input size for ONNX models

class LivePortraitSyncer:
    """
    High-quality Lip Sync using LivePortrait (ONNX).
    Uses Wav2Lip to generate driving motion, then LivePortrait to animate the high-quality face.
    """
    def __init__(self, acceleration: str = 'ort'):
        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device_str == "cuda" else ['CPUExecutionProvider']
        
        self.acceleration = acceleration
        if self.acceleration == 'tensorrt' and not TRT_AVAILABLE:
            logger.warning("TensorRT acceleration requested but libraries missing. Falling back to ORT.")
            self.acceleration = 'ort'
        
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
        
        self.face_analysis = None
        self.appearance_extractor = None
        self.motion_extractor = None
        self.warping_module = None
        self.spade_generator = None
        self.stitching_module = None
        self.lip_retargeting = None
        
        # ONNX Model Paths
        # Prepend 'liveportrait_onnx/' as per repo structure
        self.onnx_files = {
            "appearance": "liveportrait_onnx/appearance_feature_extractor.onnx",
            "motion": "liveportrait_onnx/motion_extractor.onnx",
            "warping_spade": "liveportrait_onnx/warping_spade.onnx",
            "landmark": "liveportrait_onnx/landmark.onnx",
            "stitching": "liveportrait_onnx/stitching.onnx",
            "stitching_lip": "liveportrait_onnx/stitching_lip.onnx"
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
        """Loads models into VRAM (ONNX Runtime Sessions or TRT Engines)."""
        if self.appearance_extractor is not None:
            return

        self.download_models()
        logger.info(f"Loading LivePortrait models on {self.device_str} (Mode: {self.acceleration})...")

        if self.acceleration == 'tensorrt':
             try:
                 self._load_plugin()
                 try:
                    self._load_trt_models()
                    logger.info(f"Successfully loaded all TensorRT engines. (Models used: {list(self.onnx_files.keys())})")
                    # Load remaining small models (stitching) via ORT if needed, 
                    # but usually we can wrap everything or mix.
                    # For simplicity, let's load detection via InsightFace (standard)
                    # and stitching via ORT (fallback/standard) as they are fast enough.
                 except Exception as e:
                     import traceback
                     traceback.print_exc()
                     logger.error(f"Failed to load TensorRT engines: {e}. Falling back to ONNX Runtime.")
                     self.acceleration = 'ort' # Fallback
             except Exception as e:
                 logger.error(f"Failed to load TensorRT Plugin: {e}. Falling back to ONNX Runtime.")
                 self.acceleration = 'ort'

        if self.acceleration == 'ort':
             self._load_ort_models()

    def _load_plugin(self):
        """Loads the GridSample3D TensorRT plugin."""
        import platform
        is_windows = platform.system() == "Windows"
        ext = ".dll" if is_windows else ".so"
        
        plugin_path = self.model_dir / f"grid_sample_3d_plugin{ext}"
        if not plugin_path.exists():
            # Try alternative name or log warning
            # For Linux, it might be named differently or need compilation.
            # Assuming strictly same basename for now.
             raise FileNotFoundError(f"Plugin not found at {plugin_path}")
        
        logger.info(f"Loading TensorRT plugin: {plugin_path}")
        ctypes.CDLL(str(plugin_path))
        if hasattr(trt, "init_libnvinfer_plugins"):
            trt.init_libnvinfer_plugins(None, "")
        else:
            logger.info("DEBUG: trt.init_libnvinfer_plugins not found (ignoring for tests/compatibility)")

    def _load_trt_models(self):
        """Loads TensorRT engines."""
        # 1. InsightFace (still standard)
        self.face_analysis = insightface.app.FaceAnalysis(
            name='buffalo_l', 
            providers=self.providers
        )
        self.face_analysis.prepare(ctx_id=0 if self.device_str == 'cuda' else -1, det_size=(640, 640))

        # 2. Main Modules (TRT Wrappers)
        # Assumes .engine files exist next to .onnx
        def load_trt(name):
            onnx_path = self.model_dir / self.onnx_files[name]
            engine_path = onnx_path.with_suffix(".engine")
            if not engine_path.exists():
                raise FileNotFoundError(f"Engine not found: {engine_path}")
            return TRTWrapper(str(engine_path))

        self.appearance_extractor = load_trt("appearance")
        self.motion_extractor = load_trt("motion")
        # warping_spade is the key one needing plugin
        self.warping_module = load_trt("warping_spade")
        
        # 3. Stitching/Retargeting (Keep as ORT for now as they are small MLPs)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        try:
            path_stitch = str(self.model_dir / self.onnx_files["stitching"])
            self.stitching_module = ort.InferenceSession(path_stitch, sess_opts, providers=['CPUExecutionProvider'])
        except:
             self.stitching_module = None
             
        try:
             path_lip = str(self.model_dir / self.onnx_files["stitching_lip"])
             self.lip_retargeting = ort.InferenceSession(path_lip, sess_opts, providers=['CPUExecutionProvider'])
        except:
             self.lip_retargeting = None

    def _load_ort_models(self):
        """Original ONNX Runtime loading logic."""
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
            
            # [Optimization] CPU Threading for warping_spade (forced to CPU)
            sess_opts.intra_op_num_threads = 8 
            sess_opts.inter_op_num_threads = 2 
            
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
            
            # Warping/SPADE (CPU Force)
            try:
                path_warp = str(self.model_dir / self.onnx_files["warping_spade"])
                if not Path(path_warp).exists():
                     raise FileNotFoundError(f"Model not found: {path_warp}")
                
                logger.info("loading warping_spade with CPUExecutionProvider (Required for 5D GridSample support)...")
                self.warping_module = ort.InferenceSession(path_warp, sess_opts, providers=['CPUExecutionProvider'])
            except Exception as e:
                logger.error(f"Failed to load warping_spade on CPU: {e}")
                raise

            self.spade_generator = None 
            
            # Stitching/Retargeting
            try:
                self.stitching_module = load_ort("stitching")
            except:
                self.stitching_module = None
            try:
                self.lip_retargeting = load_ort("stitching_lip")
            except:
                self.lip_retargeting = None
            
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
        self.stitching_module = None
        self.lip_retargeting = None
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
        
        # 1. Generate driving video using Wav2Lip
        # We need a temp path for the driving video
        temp_wav2lip_out = config.TEMP_DIR / f"lp_drive_{Path(video_path).stem}.mp4"
        logger.info("Step 1: Generating driving motion with Wav2Lip...")
        
        # We use the internal wav2lip driver. 
        # Note: We do NOT use enhance_face here because we only need the motion (lips), 
        # and enhancement is slow/unnecessary for the driving video.
        self.wav2lip_driver.sync_lips(video_path, audio_path, str(temp_wav2lip_out), enhance_face=False)
        
        # Unload Wav2Lip to free VRAM for LivePortrait
        self.wav2lip_driver.unload_model()
        
        if not temp_wav2lip_out.exists():
            raise RuntimeError("Wav2Lip failed to generate driving video.")

        # 2. LivePortrait Animation (The "Real" Step)
        self.load_models() # Load LP models ONLY after Wav2Lip is done
        
        accel_label = "TENSORRT" if self.acceleration == 'tensorrt' else "ONNX"
        logger.info(f"Step 2: Applying LivePortrait animation ({accel_label})...")
        
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

    def _animate_video(self, source_video: str, driving_video: str, output_path: str) -> None:
        """
        Core animation loop.
        
        Args:
            source_video: Path to source video (high quality original).
            driving_video: Path to driving video (Wav2Lip output with lip motion).
            output_path: Path for output video file.
        """
        cap_src = cv2.VideoCapture(source_video)
        cap_drv = cv2.VideoCapture(driving_video)
        writer = None
        temp_raw_path = str(Path(output_path).with_suffix('.raw.mp4'))
        
        try:
            width = int(cap_src.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap_src.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap_src.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap_src.get(cv2.CAP_PROP_FRAME_COUNT))
            total_drv = int(cap_drv.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames != total_drv:
                logger.warning(f"Frame count mismatch: source={total_frames}, driving={total_drv}. Adjusting implementation.")

            # improved codec compatibility by writing to temp and remuxing later
            writer = cv2.VideoWriter(temp_raw_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            
            pbar = tqdm(total=total_frames, desc="LivePortrait Inference")
            
            # Cache for source face info to ensure consistent cropping region if detection jitters
            source_face_info_cache = None
            
            # Frame sync helpers
            last_drv_frame = None
            drv_frame_idx = 0
            
            while True:
                ret_s, frame_s = cap_src.read()
                
                if not ret_s:
                    break
                
                # Handle driving video logic (loop or hold last frame if shorter)
                ret_d, frame_d = cap_drv.read()
                if not ret_d:
                     if last_drv_frame is not None:
                         frame_d = last_drv_frame
                         logger.debug(f"Driving video exhausted, reusing last frame.")
                     else:
                         # Use source frame as fallback? Or just stop? 
                         # Stopping might shorten video. Let's use blank or break.
                         logger.warning("Driving video ended before source. Stopping.")
                         break
                else:
                    last_drv_frame = frame_d.copy()
                    drv_frame_idx += 1
                    
                # Process Frame
                # 1. Detect Face & Crop
                # Detect independently!
                face_info_s = self._detect(frame_s)
                face_info_d = self._detect(frame_d)
                
                if face_info_s is None:
                    # If we can't find face in source, we can't animate it.
                    writer.write(frame_s)
                    pbar.update(1)
                    continue
                
                # Update cache if we found a face
                if source_face_info_cache is None:
                    source_face_info_cache = face_info_s
                
                # 2. Prepare Inputs
                # Use cached source face info for source crop to keep it stable
                # Or use current frame's face info? Stable is better for background paste back.
                # But if head moves significantly, cache might be invalid.
                # LivePortrait is frame-by-frame. Let's use current frame's face info for alignment
                # to track head movement, but we need to ensure the crop region is valid.
                
                crop_img, M = self._align_crop(frame_s, face_info_s)
                
                # For driving video, use its OWN face detection for alignment
                if face_info_d is not None:
                    crop_drv, _ = self._align_crop(frame_d, face_info_d)
                else:
                    # Fallback: if driving face lost, use source alignment or previous?
                    # Using source alignment on driving frame might work if they are similar
                    crop_drv, _ = self._align_crop(frame_d, face_info_s)

                # 3. Inference
                out_img = self._run_inference(crop_img, crop_drv)
                
                # 4. Paste Back
                final_frame = self._paste_back(out_img, frame_s, M, face_info_s)
                writer.write(final_frame)
                pbar.update(1)
                
            writer.release()
            writer = None # Set to None to avoid double release in finally
            
            # Remux with FFmpeg for H.264
            # This ensures better browser compatibility than cv2's mp4v
            logger.info("Remuxing output to H.264...")
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-v", "warning",
                    "-i", temp_raw_path,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                    "-pix_fmt", "yuv420p", # Important for widespread compatibility
                    output_path
                ], check=True)
                # Remove temp file on success
                if os.path.exists(temp_raw_path):
                    os.remove(temp_raw_path)
            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg remux failed: {e}. Moving temp file to output as fallback.")
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(temp_raw_path, output_path)

        except Exception as e:
            logger.error(f"Animation loop error: {e}")
            raise
        finally:
            if cap_src: cap_src.release()
            if cap_drv: cap_drv.release()
            if writer: writer.release()
            if 'pbar' in locals(): pbar.close()
            # Cleanup temp if it exists and wasn't renamed
            if 'temp_raw_path' in locals() and os.path.exists(temp_raw_path):
                 os.remove(temp_raw_path)

    def _detect(self, img: np.ndarray) -> Optional[Any]:
        """
        Detect the largest face in an image.
        
        Args:
            img: BGR image as numpy array.
            
        Returns:
            InsightFace detection object or None if no face found.
        """
        faces = self.face_analysis.get(img)
        if not faces:
            return None
        # Return largest face
        return sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))[-1]

    def _align_crop(self, img, face_info):
        """
        Aligns and crops face using Naive Square Crop (Scale 2.3).
        This matches the original logic and prevents ghosting caused by incompatible Affine Transforms.
        """
        bbox = face_info.bbox
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        
        # Expansion factor used in LivePortrait (Standard 2.3)
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

    def _paste_back(self, pred_img, bg_img, M_inv, face_info=None):
        """
        Pastes the predicted face back onto the background image using Poisson Blending.
        
        Args:
            pred_img: Generated face image (256x256).
            bg_img: Original background frame.
            M_inv: Inverse affine matrix from _align_crop (maps crop → original).
            face_info: InsightFace detection object containing landmarks (kps).
            
        Returns:
            Blended output frame with same dimensions as bg_img.
        """
        try:
             # 1. Extract Original Background Crop (Target for Blending)
             # Invert M_inv to get M (Original -> Crop)
             M_fwd = cv2.invertAffineTransform(M_inv)
             
             crop_h, crop_w = pred_img.shape[:2]
             
             # Extract the original face crop from bg_img
             bg_crop = cv2.warpAffine(
                 bg_img, M_fwd, (crop_w, crop_h),
                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
             )
             
             # 2. Generate Mouth Mask for Seamless Cloning
             # The mask determines WHICH part of pred_img is cloned into bg_crop.
             # It must be a binary mask (255 where we want the update).
             mask_clone = np.zeros((crop_h, crop_w), dtype=np.uint8)
             center_clone = (crop_w // 2, crop_h // 2)
             
             use_seamless = False
             
             if face_info is not None and hasattr(face_info, 'kps'):
                try:
                    # Map landmarks to crop space
                    kps_orig = face_info.kps
                    ones = np.ones((kps_orig.shape[0], 1))
                    kps_homo = np.concatenate([kps_orig, ones], axis=1)
                    kps_crop = np.dot(kps_homo, M_fwd.T) # (5, 2)
                    
                    # Define mouth region (Mouth corners + Nose hint)
                    mouth_left = kps_crop[3]
                    mouth_right = kps_crop[4]
                    
                    # Calculate center and axes
                    mouth_center = (mouth_left + mouth_right) / 2
                    width = np.linalg.norm(mouth_left - mouth_right)
                    
                    center_x = int(mouth_center[0])
                    center_y = int(mouth_center[1] + width * 0.1) # Slightly lower
                    
                    # Ellipse size (Reduced to ensure it doesn't touch borders)
                    axis_x = int(width * 0.85) 
                    axis_y = int(width * 0.6)
                    
                    # Draw filled ellipse (white)
                    # cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness)
                    cv2.ellipse(mask_clone, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, 255, -1)
                    
                    # IMPORTANT: seamlessClone requires mask to NOT touch borders.
                    # Zero out a safe margin (4 pixels)
                    mask_clone[:4, :] = 0
                    mask_clone[-4:, :] = 0
                    mask_clone[:, :4] = 0
                    mask_clone[:, -4:] = 0
                    
                    # CORRECTION: The source and dest are already aligned. 
                    # We want to place the source exactly on top of the dest.
                    # seamlessClone 'center' is where the CENTER of 'src' should be placed in 'dst'.
                    center_clone = (crop_w // 2, crop_h // 2)
                    use_seamless = True
                except Exception as e:
                    logger.warning(f"Mask generation failed: {e}")
            
             if not use_seamless:
                 # Fallback: Center circle
                 cv2.circle(mask_clone, (crop_w//2, crop_h//2 + 20), 80, 255, -1)
                 center_clone = (crop_w//2, crop_h//2 + 20)
             
             # 3. Apply Poisson Blending (Seamless Clone)
             # src: pred_img (The new mouth)
             # dst: bg_crop (The original face)
             # mask: mask_clone (Where to paste)
             # center: Center of the mask in dst
             
             # Ensure types
             pred_img = pred_img.astype(np.uint8)
             bg_crop = bg_crop.astype(np.uint8)
             
             try:
                 # Check mask_clone validity for seamlessClone
                 if cv2.countNonZero(mask_clone) == 0:
                     raise ValueError("Mask is empty")

                 # Ensure mask is within bounds (should be guaranteed by construction, but safe to check)
                 if mask_clone.shape != bg_crop.shape[:2]:
                     raise ValueError(f"Mask shape mismatch: {mask_clone.shape} vs {bg_crop.shape}")

                 # NORMAL_CLONE is best for inserting objects with different lighting
                 try:
                    blended_crop = cv2.seamlessClone(pred_img, bg_crop, mask_clone, center_clone, cv2.NORMAL_CLONE)
                 except Exception:
                    # Retry with MIXED_CLONE (sometimes more robust to edge artifacts)
                    blended_crop = cv2.seamlessClone(pred_img, bg_crop, mask_clone, center_clone, cv2.MIXED_CLONE)
                    
             except Exception as e:
                 logger.warning(f"Seamless clone failed: {e}. Falling back to alpha blend.")
                 # Fallback manual blend
                 mask_f = cv2.GaussianBlur(mask_clone, (51, 51), 0).astype(float) / 255.0
                 mask_f = np.stack([mask_f]*3, axis=-1)
                 blended_crop = (bg_crop * (1 - mask_f) + pred_img * mask_f).astype(np.uint8)

             # 4. Warp the fully blended crop back to original space
             # Since 'blended_crop' now matches the original 'bg_crop' at the edges perfectly,
             # we can just paste it back. But to be safe vs alignment jitter, we use a soft mask for the whole crop.
             
             output_full = cv2.warpAffine(
                blended_crop, M_inv, (bg_img.shape[1], bg_img.shape[0]), 
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT
             )
             
             # Create a full-crop transparency mask (Soft box)
             mask_box = np.zeros((crop_h, crop_w), dtype=np.float32)
             margin = int(crop_w * 0.02)
             cv2.rectangle(mask_box, (margin, margin), (crop_w-margin, crop_h-margin), 1.0, -1)
             mask_box = cv2.GaussianBlur(mask_box, (21, 21), 0)
             mask_box = np.stack([mask_box]*3, axis=-1)
             
             mask_full_box = cv2.warpAffine(
                 mask_box, M_inv, (bg_img.shape[1], bg_img.shape[0]),
                 flags=cv2.INTER_LINEAR
             )
             
             # Final Blend
             final = bg_img.astype(float) * (1 - mask_full_box) + output_full.astype(float) * mask_full_box
             return final.astype(np.uint8)

        except Exception as e:
            logger.error(f"Paste back failed: {e}")
            return bg_img 

    def _run_inference(self, src_img, drv_img):
        """
        Run LivePortrait inference with lip-only motion transfer.
        
        Uses the lip retargeting module to extract ONLY lip motion from the driving
        video and apply it to source keypoints, preserving head pose and face shape.
        """
        # Prepare Tensors
        def to_tensor(img):
            x = img.astype(np.float32) / 255.0  # 0-1
            x = np.transpose(x, (2, 0, 1)) # C,H,W
            x = np.expand_dims(x, axis=0)  # 1,C,H,W
            return x

        t_src = to_tensor(src_img)
        t_drv = to_tensor(drv_img)
        
        # 1. Extract Appearance Features from SOURCE only
        input_name_app = self.appearance_extractor.get_inputs()[0].name
        f_s = self.appearance_extractor.run(None, {input_name_app: t_src})[0]
        
        # 2. Extract Motion (Keypoints) from both source and driving
        input_name_mot = self.motion_extractor.get_inputs()[0].name
        
        res_s = self.motion_extractor.run(None, {input_name_mot: t_src})
        res_d = self.motion_extractor.run(None, {input_name_mot: t_drv})
        
        def extract_kp(outputs):
            """Extract keypoints tensor from motion extractor outputs."""
            for out in outputs:
                if out.shape == (1, 63):
                    return out.reshape(1, 21, 3)
            for out in outputs:
                if out.shape == (1, 21, 3):
                    return out
            raise ValueError(f"Could not find Keypoint output. Shapes: {[o.shape for o in outputs]}")

        kp_s = extract_kp(res_s)
        kp_d = extract_kp(res_d)
        
        # 3. Apply Lip Retargeting (KEY FIX)
        # Instead of using kp_d directly (which transfers ALL motion including head pose),
        # we use the lip retargeting module to compute only the lip delta
        kp_driving = self._apply_lip_retargeting(kp_s, kp_d)
        
        # Apply stitching if available (blends keypoints for seamless transition)
        if self.stitching_module is not None:
            kp_driving = self._apply_stitching(kp_s, kp_driving)

        # 4. Warping + SPADE
        warp_inputs = self.warping_module.get_inputs()
        
        feed_dict = {
            warp_inputs[0].name: f_s,
            warp_inputs[1].name: kp_driving,  # Modified keypoints (lip motion only)
            warp_inputs[2].name: kp_s         # Source KP (unchanged)
        }
        
        out_gen = self.warping_module.run(None, feed_dict)[0]
        
        # Post-process
        res = np.clip(out_gen, 0, 1) * 255
        res = np.transpose(res[0], (1, 2, 0)) # H,W,C
        return res.astype(np.uint8)
    
    def _apply_lip_retargeting(self, kp_source: np.ndarray, kp_driving: np.ndarray) -> np.ndarray:
        """
        Apply lip-only retargeting using the stitching_lip module.
        
        This preserves the source head pose and face shape while transferring
        only the lip motion from the driving keypoints.
        
        The stitching_lip.onnx model expects:
        - Input: [1, 65] = 63 (source keypoints) + 1 (source lip-open) + 1 (driving lip-open)
        - Output: [1, 63] = delta to apply to source keypoints
        
        Args:
            kp_source: Source keypoints (1, 21, 3)
            kp_driving: Driving keypoints (1, 21, 3)
            
        Returns:
            Modified keypoints with lip motion applied to source (1, 21, 3)
        """
        if self.lip_retargeting is None:
            # Fallback: If lip retargeting not available, use simple lip interpolation
            return self._simple_lip_transfer(kp_source, kp_driving)
        
        try:
            # Flatten keypoints
            kp_s_flat = kp_source.reshape(1, -1).astype(np.float32)  # (1, 63)
            kp_d_flat = kp_driving.reshape(1, -1).astype(np.float32)  # (1, 63)
            
            # Calculate lip-open ratio from keypoints
            # Lip keypoints are typically at indices 17-20 in 21-point layout
            # Lip-open = vertical distance between upper and lower lip keypoints
            lip_open_src = self._compute_lip_open(kp_source)
            lip_open_drv = self._compute_lip_open(kp_driving)
            
            # Build input: [kp_source(63), lip_open_src(1), lip_open_drv(1)] = (1, 65)
            lip_input = np.concatenate([
                kp_s_flat,
                np.array([[lip_open_src]], dtype=np.float32),
                np.array([[lip_open_drv]], dtype=np.float32)
            ], axis=1)  # (1, 65)
            
            # Run lip retargeting
            lip_inputs = self.lip_retargeting.get_inputs()
            feed_dict = {lip_inputs[0].name: lip_input}
            
            delta_lip = self.lip_retargeting.run(None, feed_dict)[0]  # (1, 63)
            
            # Reshape delta and apply to source keypoints
            delta_lip = delta_lip.reshape(1, 21, 3)
            kp_result = kp_source + delta_lip
            
            return kp_result.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Lip retargeting failed: {e}. Using simple lip transfer.")
            return self._simple_lip_transfer(kp_source, kp_driving)
    
    def _compute_lip_open(self, kp: np.ndarray) -> float:
        """
        Compute lip-open ratio from keypoints.
        
        Uses the vertical distance between upper and lower lip keypoints.
        LivePortrait 21-point layout: indices 17-20 are around mouth area.
        
        Args:
            kp: Keypoints (1, 21, 3)
            
        Returns:
            Lip-open ratio (normalized value)
        """
        # For 21-point layout, estimate mouth opening from relevant indices
        # Upper lip ~ index 18, Lower lip ~ index 19 (approximate)
        # If indices don't exist or are wrong, return neutral value
        try:
            # Y-coordinate difference gives vertical opening
            upper_lip_y = kp[0, 18, 1]  # Y of upper lip point
            lower_lip_y = kp[0, 19, 1]  # Y of lower lip point
            lip_open = abs(lower_lip_y - upper_lip_y)
            return float(lip_open)
        except (IndexError, TypeError):
            return 0.0  # Neutral - lips closed
    
    def _simple_lip_transfer(self, kp_source: np.ndarray, kp_driving: np.ndarray) -> np.ndarray:
        """
        Fallback: Simple lip motion transfer by blending only lip-related keypoints.
        
        LivePortrait uses 21 keypoints. Based on standard face keypoint layouts,
        lower face keypoints (roughly indices 15-20) correspond to mouth/chin area.
        
        Args:
            kp_source: Source keypoints (1, 21, 3)
            kp_driving: Driving keypoints (1, 21, 3)
            
        Returns:
            Blended keypoints with lip motion from driving (1, 21, 3)
        """
        kp_result = kp_source.copy()
        
        # Lip keypoint indices (estimated for 21-point layout)
        # Typically: 0-4 = face contour, 5-9 = eyebrows, 10-14 = eyes, 15-20 = nose/mouth
        LIP_INDICES = [17, 18, 19, 20]  # Lower lip/mouth area
        
        # Calculate the delta for lip keypoints only
        for idx in LIP_INDICES:
            if idx < kp_source.shape[1]:
                # Transfer driving lip position relative to source
                delta = kp_driving[0, idx] - kp_source[0, idx]
                # Apply with dampening to avoid extreme movements
                kp_result[0, idx] = kp_source[0, idx] + delta * 0.8
        
        return kp_result.astype(np.float32)

    def _apply_stitching(self, kp_source: np.ndarray, kp_driving: np.ndarray) -> np.ndarray:
        """
        Apply stitching module to blend driving keypoints with source for smooth transitions.
        
        Args:
            kp_source: Source keypoints (1, 21, 3)
            kp_driving: Driving keypoints after lip retargeting (1, 21, 3)
            
        Returns:
            Stitched keypoints (1, 21, 3)
        """
        if self.stitching_module is None:
             return kp_driving

        try:
            # Flatten and concatenate for stitching input
            kp_s_flat = kp_source.reshape(1, -1).astype(np.float32)  # (1, 63)
            kp_d_flat = kp_driving.reshape(1, -1).astype(np.float32)  # (1, 63)
            
            stitch_input = np.concatenate([kp_s_flat, kp_d_flat], axis=1)  # (1, 126)
            
            stitch_inputs = self.stitching_module.get_inputs()
            feed_dict = {stitch_inputs[0].name: stitch_input}
            
            delta = self.stitching_module.run(None, feed_dict)[0]
            
            # [Fix] Handle output size 65 (63 params + 2 ratios?)
            if delta.size > 63:
                delta = delta[:, :63]
                
            delta = delta.reshape(1, 21, 3)
            
            return (kp_driving + delta * 0.5).astype(np.float32)  # Blend with dampening
            
        except Exception as e:
            logger.warning(f"Stitching failed: {e}. Using unstitched keypoints.")
            return kp_driving

