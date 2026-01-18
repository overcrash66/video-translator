"""
Visual Text Translation module.

Detects text in video frames using PaddleOCR, translates it, 
and overlays the translated text using OpenCV inpainting.
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from src.utils import config
from deep_translator import GoogleTranslator

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

logger = logging.getLogger(__name__)


class VisualTranslator:
    """
    Handles visual text translation in video frames.
    Uses PaddleOCR for detection and OpenCV for simple inpainting.
    """
    
    def __init__(self):
        self.ocr_model = None
        self.model_loaded = False
        self.translator_cache = {}
        self.current_engine = None
        

    def load_model(self, source_lang: str = 'en', ocr_engine: str = "PaddleOCR"):
        """
        Load OCR model (PaddleOCR or EasyOCR).
        ocr_engine: "PaddleOCR" or "EasyOCR"
        """
        if self.model_loaded and self.current_engine == ocr_engine:
            return
            
        # Unload previous model if engine changed
        if self.model_loaded and self.current_engine != ocr_engine:
            self.unload_model()

        self.current_engine = ocr_engine
        
        if ocr_engine == "PaddleOCR":
            if not PADDLE_AVAILABLE:
                logger.warning("PaddleOCR not installed. Visual translation disabled.")
                return

            logger.info(f"Loading PaddleOCR model for language: {source_lang}...")
            try:
                # Map common language codes to PaddleOCR lang codes
                paddle_lang_map = {
                    'en': 'en', 'ar': 'ar', 'zh': 'ch', 'fr': 'fr', 'de': 'german',
                    'es': 'es', 'it': 'it', 'ja': 'japan', 'ko': 'korean', 'ru': 'ru', 'pt': 'pt',
                }
                ocr_lang = paddle_lang_map.get(source_lang, 'en')
                
                # use_angle_cls=True for robust detection of rotated text
                # enable_mkldnn=False to fix oneDNN crash on Windows
                self.ocr_model = PaddleOCR(use_angle_cls=True, lang=ocr_lang, enable_mkldnn=False)
                self.model_loaded = True
                logger.info("PaddleOCR loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load PaddleOCR: {e}")
                raise
                
        elif ocr_engine == "EasyOCR":
            logger.info(f"Loading EasyOCR model for language: {source_lang}...")
            try:
                import easyocr
                # Map common language codes to EasyOCR
                # EasyOCR uses standard codes mostly, but let's be safe
                if source_lang == 'zh': source_lang = 'ch_sim'
                
                self.ocr_model = easyocr.Reader([source_lang, 'en']) # Always include English for robustness
                self.model_loaded = True
                logger.info("EasyOCR loaded successfully.")
            except ImportError:
                 logger.error("EasyOCR not installed. Run `pip install easyocr`.")
                 raise
            except Exception as e:
                logger.error(f"Failed to load EasyOCR: {e}")
                raise

    def unload_model(self):
        """Unload OCR model to free memory."""
        if self.ocr_model:
            del self.ocr_model
            self.ocr_model = None
        self.model_loaded = False
        self.current_engine = None
        self.translator_cache = {}
        logger.info("VisualTranslator model unloaded.")

    def _get_translator(self, target_lang: str) -> GoogleTranslator:
        """Get or create a cached translator for the target language."""
        if target_lang not in self.translator_cache:
            self.translator_cache[target_lang] = GoogleTranslator(source='auto', target=target_lang)
        return self.translator_cache[target_lang]

    def _translate_text(self, text: str, target_lang: str) -> str:
        """Translate text to target language using Google Translate."""
        if not text or len(text.strip()) < 2:
            return text
        try:
            translator = self._get_translator(target_lang)
            translated = translator.translate(text)
            return translated if translated else text
        except Exception as e:
            logger.warning(f"Translation failed for '{text}': {e}")
            return text

    def _create_text_mask(self, frame: np.ndarray, boxes: list) -> np.ndarray:
        """Create a binary mask for detected text regions."""
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for box in boxes:
            pts = np.array(box, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        return mask

    def _inpaint_text_regions(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Use OpenCV inpainting to remove text from frame."""
        # Dilate mask slightly for better inpainting
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Use TELEA inpainting algorithm
        inpainted = cv2.inpaint(frame, dilated_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
        return inpainted

    def _overlay_translated_text(self, frame: np.ndarray, boxes: list, 
                                  original_texts: list, translated_texts: list) -> np.ndarray:
        """Overlay translated text on the frame at detected positions."""
        result = frame.copy()
        
        for box, orig_text, trans_text in zip(boxes, original_texts, translated_texts):
            if not trans_text or trans_text == orig_text:
                continue
                
            pts = np.array(box, dtype=np.int32)
            
            # Calculate text position (center of bounding box)
            x_coords = pts[:, 0]
            y_coords = pts[:, 1]
            x_center = int(np.mean(x_coords))
            y_center = int(np.mean(y_coords))
            
            # Estimate font scale based on box height
            box_height = max(y_coords) - min(y_coords)
            font_scale = max(0.4, box_height / 30)
            thickness = max(1, int(font_scale))
            
            # Get text size for centering
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), baseline = cv2.getTextSize(
                trans_text, font, font_scale, thickness
            )
            
            # Position text centered on the original location
            x_pos = x_center - text_width // 2
            y_pos = y_center + text_height // 2
            
            # Draw background rectangle for better visibility
            padding = 4
            cv2.rectangle(
                result,
                (x_pos - padding, y_pos - text_height - padding),
                (x_pos + text_width + padding, y_pos + padding),
                (255, 255, 255),  # White background
                -1
            )
            
            # Draw translated text
            cv2.putText(
                result, trans_text, (x_pos, y_pos),
                font, font_scale, (0, 0, 0),  # Black text
                thickness, cv2.LINE_AA
            )
            
        return result

    def translate_video_text(self, video_path: str, output_path: str, 
                              target_lang: str = 'fr', source_lang: str = 'en',
                              ocr_engine: str = "PaddleOCR",
                              process_interval: int = 1) -> str:
        """
        Process video to detect and translate text.
        """
        if not self.model_loaded or self.current_engine != ocr_engine:
            self.load_model(source_lang, ocr_engine)
            
        if not self.model_loaded:
            logger.warning(f"Skipping visual translation ({ocr_engine} not available).")
            import shutil
            shutil.copy(video_path, output_path)
            return output_path

        logger.info(f"Starting visual text translation: {video_path} using {ocr_engine}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            import shutil
            shutil.copy(video_path, output_path)
            return output_path
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        processed_count = 0
        
        # Cache for text detection results
        last_boxes = []
        last_mask = None
        last_translated_texts = []
        last_original_texts = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_count % process_interval == 0:
                try:
                    boxes = []
                    original_texts = []
                    
                    if ocr_engine == "PaddleOCR":
                        result = self.ocr_model.ocr(frame)
                        if result and result[0]:
                             for line in result[0]:
                                boxes.append(line[0])
                                original_texts.append(line[1][0])
                                
                    elif ocr_engine == "EasyOCR":
                        # EasyOCR returns (bbox, text, prob)
                        # bbox is [ [x1,y1], [x2,y2], [x3,y3], [x4,y4] ]
                        results = self.ocr_model.readtext(frame)
                        for (bbox, text, prob) in results:
                            if prob > 0.4:
                                boxes.append(bbox)
                                original_texts.append(text)
                    
                    if boxes:
                         translated_texts = []
                         for text in original_texts:
                             translated_texts.append(self._translate_text(text, target_lang))
                             
                         last_mask = self._create_text_mask(frame, boxes)
                         frame = self._inpaint_text_regions(frame, last_mask)
                         frame = self._overlay_translated_text(frame, boxes, original_texts, translated_texts)
                         
                         last_boxes = boxes
                         last_original_texts = original_texts
                         last_translated_texts = translated_texts
                    else:
                        last_boxes = [] # clear cache if no text
                        
                    processed_count += 1
                except Exception as e:
                     logger.warning(f"Error processing frame {frame_count}: {e}")
            else:
                 # Apply cached
                 if last_boxes and last_mask is not None:
                      try:
                          frame = self._inpaint_text_regions(frame, last_mask)
                          frame = self._overlay_translated_text(frame, last_boxes, last_original_texts, last_translated_texts)
                      except: pass
                      
            out.write(frame)
            frame_count += 1
            if frame_count % 100 == 0: logger.info(f"Processed {frame_count}/{total_frames}...")
            
        cap.release()
        out.release()
        return output_path
