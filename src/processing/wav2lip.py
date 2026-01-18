import torch
import numpy as np
import cv2
import os
import logging
from tqdm import tqdm
from pathlib import Path
import face_alignment

from src.processing.wav2lip_core.models import Wav2Lip
from src.processing.wav2lip_core import audio as audio_utils
from src.utils import config

logger = logging.getLogger(__name__)

class Wav2LipSyncer:
    def __init__(self):
        self.model = None
        self.detector = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = 96
        self.batch_size = 32 # Default batch size
        self.model_path = Path("models/wav2lip/wav2lip.pth")

    def load_model(self):
        if self.model is not None:
            return

        logger.info(f"Loading Wav2Lip model from {self.model_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Wav2Lip model not found at {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model = Wav2Lip()
        
        # Handle state dict mismatch if any (usually 'module.' prefix)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
            
        self.model.load_state_dict(new_s)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info("Loading Face Detection model (face_alignment)...")
        self.detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, 
                                                     flip_input=False, device=str(self.device).split(":")[0])
                                                     
    def get_smoth_box(self, boxes, window_size=5):
        # Smoothing logic for simple boxes
        # Assume boxes is (N, 4)
        pass 
        # For now, simplistic approach: use detections frame-by-frame or interpolate
        # We'll stick to frame-by-frame for robustness in v1

    def detect_faces(self, frames):
        """
        Detect faces in frames using face_alignment.
        Returns list of [x1, y1, x2, y2]. Selects the largest face if multiple are found.
        Handles CUDA errors by falling back to CPU.
        """
        results = []
        
        logger.info("Detecting faces...")
        
        # We need to restart detection if fallback happens, or handle per-frame.
        # Since 'frames' is a list, we can just continue from current index if we are careful.
        # But 'enumerate' in loop makes it tricky to retry current frame cleanly without complex iterator logic.
        # Simpler approach: Iterate by index.
        
        i = 0
        while i < len(frames):
            frame = frames[i]
            
            # Progress bar update (manual)
            if i % 100 == 0:
                logger.debug(f"Detecting face {i}/{len(frames)}")

            try:
                # Face Alignment expects RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                preds = self.detector.get_landmarks(rgb_frame)
                
                if preds:
                    # Choose largest face
                    best_box = None
                    max_area = -1
                    
                    for lm in preds:
                        x_min, y_min = np.min(lm, axis=0)
                        x_max, y_max = np.max(lm, axis=0)
                        
                        w = x_max - x_min
                        h = y_max - y_min
                        area = w * h
                        
                        if area > max_area:
                            max_area = area
                            
                            # Expand box
                            scale = 1.6 # typical scaling for Wav2Lip crop
                            
                            cx = (x_min + x_max) / 2
                            cy = (y_min + y_max) / 2
                            
                            nw = w * scale
                            nh = h * scale
                            
                            x1 = int(cx - nw / 2)
                            y1 = int(cy - nh / 2)
                            x2 = int(cx + nw / 2)
                            y2 = int(cy + nh / 2)
                            
                            # Pad / Clamp
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(frame.shape[1], x2)
                            y2 = min(frame.shape[0], y2)
                            
                            best_box = [x1, y1, x2, y2]
                    
                    results.append(best_box)
                else:
                    results.append(None)
                
                # Move to next frame
                i += 1
                
            except Exception as e:
                # check for CUDA error
                if "CUDA" in str(e) and self.device.type == "cuda":
                    logger.warning(f"CUDA Error during face detection on frame {i}: {e}")
                    logger.warning("Switching Face Detector to CPU fallback...")
                    
                    # Switch to CPU globally for this instance
                    self.device = torch.device("cpu")
                    
                    # Re-init detector on CPU
                    del self.detector
                    torch.cuda.empty_cache()
                    
                    self.detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, 
                                                     flip_input=False, device='cpu')
                                                     
                    # Retry CURRENT frame (do not increment i)
                    logger.info("Retrying current frame on CPU...")
                    continue
                    
                else:
                    logger.warning(f"Face detection error on frame {i}: {e}")
                    results.append(None)
                    i += 1
                 
        return results

    def sync_lips(self, video_path: str, audio_path: str, output_path: str):
        self.load_model()
        
        video_path = Path(video_path)
        audio_path = Path(audio_path)
        
        # 1. Read Video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()
        
        if not frames:
            raise ValueError("No frames extracted from video")

        # 2. Process Audio (Mel)
        mel = audio_utils.wav2mel(str(audio_path)) # (T, 80)
        
        # Wav2Lip Params
        mel_step_size = 16
        mel_idx_multiplier = 80.0 / fps 
        
        # 3. Detect Faces
        face_boxes = self.detect_faces(frames)
        
        # Fill missing boxes (Interpolation / Forward-Backward Fill)
        # Avoid defaulting to center unless absolutely no faces are found anywhere
        
        # Find at least one valid box to start with
        curr_box = None
        for box in face_boxes:
            if box is not None:
                curr_box = box
                break
                
        if curr_box is None:
             logger.warning("No faces detected in ANY frame. Falling back to center crop (Quality will be poor).")
             h, w = frames[0].shape[:2]
             curr_box = [w//4, h//4, 3*w//4, 3*h//4]
             
        # Forward Fill
        for i in range(len(face_boxes)):
            if face_boxes[i] is None:
                face_boxes[i] = curr_box
            else:
                curr_box = face_boxes[i]
                
        # Backward Fill (for leading None frames)
        for i in range(len(face_boxes)-1, -1, -1):
            if face_boxes[i] is None:
                # Should not happen due to forward fill unless start was None and we used the 'first valid' logic
                # But if we did forward fill with initial curr_box, we are good.
                # Just in case:
                 pass

        # 4. Batch Inference
        mel_chunks = []
        mel_bs = [] # Batch indices corresponding to frames
        
        # Generate Audio Chunks aligned with Frames
        # Frame i corresponds to audio window centered around i?
        # Wav2Lip Logic: Input 5 frames (i-2, i-1, i, i+1, i+2)
        # Audio: Corresponding segment.
        
        full_frames = frames
        processed_frames = frames.copy()
        
        input_batches = []
        mel_batches = []
        coords_batches = []
        
        current_faces = []
        current_mels = []
        current_coords = []
        
        mel_chunk_idx = 0
        
        logger.info("Preparing batches...")
        for i in range(len(frames)):
            # Audio window
            # Frame i timestamp = i / fps
            # Mel index = (i / fps) * 80? No, mel is 80Hz resolution?
            # Standard Wav2Lip:
            # mel_idx_multiplier = 80. / fps
            start_idx = int(i * mel_idx_multiplier)
            end_idx = start_idx + mel_step_size # 16 steps = ? ms
            
            # Pad Mel if needed
            if end_idx > mel.shape[0]:
                 # Pad with zeros
                 diff = end_idx - mel.shape[0]
                 m = np.concatenate((mel, np.zeros((diff, 80))), axis=0) # (T+d, 80)
                 m_chunk = m[start_idx:end_idx]
            else:
                m_chunk = mel[start_idx:end_idx]
                
            m_chunk = np.transpose(m_chunk, (1, 0)) # (80, 16)
            
            # Video Window (5 frames)
            window_frames = []
            audio_window = []
            
            for j in range(i-2, i+3):
                idx = max(0, min(len(frames)-1, j))
                window_frames.append(frames[idx])
                
                # Audio for this specific frame
                s_idx = int(idx * mel_idx_multiplier)
                e_idx = s_idx + mel_step_size
                
                if e_idx > mel.shape[0]:
                    diff = e_idx - mel.shape[0]
                    m = np.concatenate((mel, np.zeros((diff, 80))), axis=0)
                    chunk = m[s_idx:e_idx]
                else:
                    chunk = mel[s_idx:e_idx]
                    
                chunk = np.transpose(chunk, (1, 0)) # (80, 16)
                audio_window.append(chunk)

            # Crop window frames
            x1, y1, x2, y2 = face_boxes[i]
            
            face_crops = []
            for wf in window_frames:
                f_crop = wf[y1:y2, x1:x2]
                f_crop = cv2.resize(f_crop, (self.img_size, self.img_size))
                face_crops.append(f_crop)
                
            processed_window = []
            for fc in face_crops:
                masked = fc.copy()
                masked[self.img_size//2:, :] = 0
                combined = np.concatenate((masked, fc), axis=2) # (96, 96, 6)
                processed_window.append(combined)

            window_np = np.stack(processed_window, axis=0) # (5, 96, 96, 6)
            window_np = np.transpose(window_np, (0, 3, 1, 2)) # (5, 6, 96, 96)
            
            audio_np = np.stack(audio_window, axis=0) # (5, 80, 16)
            
            current_faces.append(window_np)
            current_mels.append(audio_np) # (5, 80, 16)
            current_coords.append((x1, y1, x2, y2))
            
            if len(current_faces) >= self.batch_size:
                input_batches.append(np.array(current_faces))
                mel_batches.append(np.array(current_mels))
                coords_batches.append(current_coords)
                current_faces = []
                current_mels = []
                current_coords = []
                
        if current_faces:
             input_batches.append(np.array(current_faces))
             mel_batches.append(np.array(current_mels))
             coords_batches.append(current_coords)
             
        # Inference
        logger.info("Running inference...")
        
        gen_frames = []
        gen_coords = [] # corresponding coords

        try:
            with torch.no_grad():
                for faces, mels, coords in zip(tqdm(input_batches), mel_batches, coords_batches):
                    # faces: (B, 5, 6, 96, 96)
                    # mels: (B, 80, 16)
                    
                    img_batch = torch.FloatTensor(faces).to(self.device).float() / 255.0
                    mel_batch = torch.FloatTensor(mels).to(self.device).float().unsqueeze(2) # (B, 5, 1, 80, 16)
                    
                    pred = self.model(mel_batch, img_batch) 
                    
                    pred = pred.cpu().numpy().transpose(0, 2, 3, 4, 1) * 255. # (B, T, 96, 96, 3)
                    
                    for b_i in range(len(coords)):
                        # Take middle frame (index 2)
                        p_frame = pred[b_i, 2] # (96, 96, 3)
                        gen_frames.append(p_frame.astype(np.uint8))
                        
                    gen_coords.extend(coords)
                    
        except RuntimeError as e:
            if "CUDA" in str(e) and self.device.type == "cuda":
                logger.warning(f"CUDA Error encountered: {e}")
                logger.warning("Switching to CPU fallback for Wav2Lip...")
                
                # Cleanup GPU memory
                try:
                    del img_batch
                except UnboundLocalError: pass
                
                try:
                    del mel_batch
                except UnboundLocalError: pass
                
                try:
                    del pred
                except UnboundLocalError: pass
                
                torch.cuda.empty_cache()
                
                # Switch to CPU
                self.device = torch.device("cpu")
                self.model = self.model.to(self.device)
                
                # Retry inference on CPU
                gen_frames = []
                gen_coords = []
                
                with torch.no_grad():
                    for faces, mels, coords in zip(tqdm(input_batches, desc="Inference (CPU Fallback)"), mel_batches, coords_batches):
                        img_batch = torch.FloatTensor(faces).to(self.device).float() / 255.0
                        mel_batch = torch.FloatTensor(mels).to(self.device).float().unsqueeze(2)
                        
                        pred = self.model(mel_batch, img_batch)
                        pred = pred.numpy().transpose(0, 2, 3, 4, 1) * 255.
                        
                        for b_i in range(len(coords)):
                            p_frame = pred[b_i, 2]
                            gen_frames.append(p_frame.astype(np.uint8))
                            
                        gen_coords.extend(coords)
            else:
                raise e

        # Blend Back
        logger.info("Blending results...")
        
        final_video_path = str(config.TEMP_DIR / "wav2lip_out.mp4")
        out = cv2.VideoWriter(final_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frames[0].shape[1], frames[0].shape[0]))
        
        for i, (g_crop, box) in enumerate(zip(gen_frames, gen_coords)):
            frame = frames[i]
            x1, y1, x2, y2 = box
            
            # Simple Paste
            # Ideally use mask blending, but direct paste is standard Wav2Lip level
            try:
                g_crop_resized = cv2.resize(g_crop, (x2 - x1, y2 - y1))
                frame[y1:y2, x1:x2] = g_crop_resized
            except Exception:
                pass
                
            out.write(frame)
            
        out.release()
        
        # Merge Audio using FFMPEG
        cmd = f'ffmpeg -y -v warning -i "{final_video_path}" -i "{str(audio_path)}" -c:v copy -c:a aac "{output_path}"'
        os.system(cmd)
        
        return output_path
