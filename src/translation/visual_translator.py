"""
Visual Text Translation module.

Detects text in video frames using PaddleOCR, translates it, 
and overlays the translated text using OpenCV inpainting and PIL for text rendering.
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from src.utils import config
from deep_translator import GoogleTranslator
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFont

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

logger = logging.getLogger(__name__)


class VisualTranslator:
    """
    Handles visual text translation in video frames.
    Uses PaddleOCR for detection and OpenCV for simple inpainting.
    Uses PIL for high-quality text rendering (Unicode support).
    """
    
    def __init__(self):
        self.ocr_model = None
        self.model_loaded = False
        self.translator_cache = {}
        self.current_engine = None
        self.font_path = self._find_font()
        
    def _find_font(self):
        """Find a suitable Unicode font on Windows."""
        # Common Windows fonts with good Unicode support
        candidates = ["arial.ttf", "segoeui.ttf", "tahoma.ttf", "msgothic.ttc"]
        for font in candidates:
             try:
                 # Check if loadable (PIL searches system path on Windows)
                 ImageFont.truetype(font, 10)
                 return font
             except:
                 continue
        return "arial.ttf" # Fallback

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
                
                # Try HPI -> GPU -> CPU fallback chain for best performance
                # HPI: High Performance Inference with OpenVINO/ONNX Runtime
                # GPU: Basic CUDA acceleration
                # CPU: Slow fallback
                try:
                    self.ocr_model = PaddleOCR(
                        use_angle_cls=True,
                        lang=ocr_lang,
                        use_gpu=True,
                        enable_hpi=True,  # Auto-selects OpenVINO/ONNX Runtime
                    )
                    self.model_loaded = True
                    logger.info("PaddleOCR loaded with HPI acceleration.")
                except Exception as hpi_err:
                    logger.warning(f"HPI mode failed ({hpi_err}). Trying GPU-only...")
                    try:
                        self.ocr_model = PaddleOCR(
                            use_angle_cls=True,
                            lang=ocr_lang,
                            use_gpu=True,
                        )
                        self.model_loaded = True
                        logger.info("PaddleOCR loaded with GPU acceleration.")
                    except Exception as gpu_err:
                        logger.warning(f"GPU mode failed ({gpu_err}). Falling back to CPU...")
                        self.ocr_model = PaddleOCR(
                            use_angle_cls=True,
                            lang=ocr_lang,
                            enable_mkldnn=False,  # Disable oneDNN to avoid Windows crash
                        )
                        self.model_loaded = True
                        logger.info("PaddleOCR loaded with CPU mode (slower).")
            except Exception as e:
                logger.error(f"Failed to load PaddleOCR: {e}")
                raise
                
        elif ocr_engine == "EasyOCR":
            logger.info(f"Loading EasyOCR model for language: {source_lang}...")
            try:
                import easyocr
                # Map common language codes to EasyOCR
                if source_lang == 'zh': source_lang = 'ch_sim'
                
                self.ocr_model = easyocr.Reader([source_lang, 'en'], gpu=True) 
                self.model_loaded = True
                logger.info("EasyOCR loaded successfully.")
            except ImportError:
                 logger.error("EasyOCR not installed. Run `pip install easyocr`.")
                 raise
            except Exception as e:
                logger.warning(f"Failed to load EasyOCR with GPU: {e}. Retrying with CPU...")
                try:
                    self.ocr_model = easyocr.Reader([source_lang, 'en'], gpu=False)
                    self.model_loaded = True
                    logger.info("EasyOCR loaded successfully (CPU mode).")
                except Exception as e_cpu:
                    logger.error(f"Failed to load EasyOCR with CPU: {e_cpu}")
                    raise

    def unload_model(self):
        """Unload OCR model to free memory."""
        if self.ocr_model:
            del self.ocr_model
            self.ocr_model = None
        self.model_loaded = False
        self.current_engine = None
        self.translator_cache = {}
        # Clear LRU cache
        self._cached_translate.cache_clear()
        logger.info("VisualTranslator model unloaded.")

    def _get_translator(self, target_lang: str) -> GoogleTranslator:
        """Get or create a cached translator for the target language."""
        if target_lang not in self.translator_cache:
            self.translator_cache[target_lang] = GoogleTranslator(source='auto', target=target_lang)
        return self.translator_cache[target_lang]

    @lru_cache(maxsize=2000)
    def _cached_translate(self, text: str, target_lang: str) -> str:
        """Cached translation."""
        try:
            translator = self._get_translator(target_lang)
            return translator.translate(text)
        except Exception as e:
            logger.warning(f"Translation failed for '{text}': {e}")
            return text

    def _detect_language(self, text: str) -> str:
        """Detect language of text using langdetect."""
        if not LANGDETECT_AVAILABLE: 
            return "unknown"
        
        # Clean text for detection
        clean = ''.join([c for c in text if c.isalpha() or c.isspace()])
        if len(clean.strip()) < 3:
            return "unknown"
            
        try:
            return detect(clean)
        except:
            return "unknown"

    @lru_cache(maxsize=1000)
    def _translate_text(self, text: str, target_lang: str, source_lang: str = None) -> str:
        """Translate text if it matches source language or if source is unknown."""
        if not text or len(text.strip()) < 2:
            return text
            
        # Language filtering
        detected = self._detect_language(text)
        
        # If source_lang is specified, only translate if detected matches source
        # But 'unknown' should probably pass through or be skipped depending on strictness
        # Here we translate if detected matches source OR detected is 'unknown' 
        # (to avoid missing short text) OR if source_lang is not strictly enforced.
        
        # Simplification: If detected is explicitly the TARGET language, skip it.
        # This prevents translating French back to French (glitch).
        if detected == target_lang:
            return text
            
        # If source_lang provided, and detected is a DIFFERENT known language, skip
        if source_lang and detected != "unknown" and detected != source_lang and detected != 'en': 
            # Allow EN as universal fallback or if user said source=EN?
            # If user said source=EN, and we detect AR, we should skip? Yes.
            if detected != source_lang:
                 # Special case: 'en' often detects as other latins if short, but let's trust detector
                 return text
        
        translated = self._cached_translate(text, target_lang)
        return translated if translated else text

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
        inpainted = cv2.inpaint(frame, dilated_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        return inpainted

    def _overlay_translated_text_pil(self, frame: np.ndarray, boxes: list, 
                                  translated_texts: list) -> np.ndarray:
        """Overlay translated text using PIL for Unicode support."""
        
        # Convert to PIL Image
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil, 'RGBA')
        
        for box, text in zip(boxes, translated_texts):
             if not text: continue
             
             pts = np.array(box, dtype=np.int32)
             x, y, w, h = cv2.boundingRect(pts)
             
             # Calculate font size
             # Heuristic: Match original height plus a small margin (2px)
             target_height = h + 2
             font_size = int(target_height) 
             if font_size < 8: font_size = 8
             
             try:
                font = ImageFont.truetype(self.font_path, font_size)
             except:
                font = ImageFont.load_default()

             # Measure text
             # getbbox returns (left, top, right, bottom)
             left, top, right, bottom = font.getbbox(text)
             text_w = right - left
             text_h = bottom - top
             
             # Auto-scaling: Reduce font size if text is too wide
             while text_w > w and font_size > 8:
                 font_size -= 2
                 try:
                    font = ImageFont.truetype(self.font_path, font_size)
                 except: break # generic default font doesn't scale
                 left, top, right, bottom = font.getbbox(text)
                 text_w = right - left
                 text_h = bottom - top
                 
             # Center text
             center_x = x + w // 2
             center_y = y + h // 2
             
             text_x = center_x - text_w // 2
             text_y = center_y - text_h // 2 - top # Adjust for baseline
             
             # Draw text with outline for better visibility
             shadow_color = (0, 0, 0, 200)
             text_color = (255, 255, 255, 255)
             
             # Draw shadow/outline
             outline_width = max(1, font_size // 15)
             for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    draw.text((text_x + dx, text_y + dy), text, font=font, fill=shadow_color)
             
             # Main text
             draw.text((text_x, text_y), text, font=font, fill=text_color)
        
        # Convert back to OpenCV
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        

    def translate_video_text(self, video_path: str, output_path: str, 
                              target_lang: str = 'fr', source_lang: str = 'en',
                              ocr_engine: str = "PaddleOCR",
                              ocr_interval_sec: float = 1.0) -> str: # DEFAULT interval 1.0s or user defined
        """
        Process video to detect and translate text.
        ocr_interval_sec: Run OCR every N seconds. In between, reuse last detection.
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
        if fps <= 0: fps = 30.0
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Calculate frame interval
        interval_frames = max(1, int(fps * ocr_interval_sec))
        
        frame_count = 0
        
        # Cache for text detection results
        last_boxes = []
        last_mask = None
        last_translated_valid = [] # Subset of texts that were valid for translation
        last_boxes_valid = []      # Subset of boxes corresponding to valid texts
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Run OCR at interval
            if frame_count % interval_frames == 0:
                try:
                    boxes = []
                    original_texts = []
                    
                    if ocr_engine == "PaddleOCR":
                        result = self.ocr_model.ocr(frame)
                        if result and result[0]:
                             for line in result[0]:
                                boxes.append(line[0])
                                # line[1][0] is text, line[1][1] is confidence
                                original_texts.append(line[1][0])
                                
                    elif ocr_engine == "EasyOCR":
                        results = self.ocr_model.readtext(frame)
                        for (bbox, text, prob) in results:
                            if prob > 0.4:
                                boxes.append(bbox)
                                original_texts.append(text)
                    
                    # Filter and Translate
                    valid_translations = []
                    valid_boxes = []
                    
                    if boxes:
                         for box, text in zip(boxes, original_texts):
                             # Translate with language checking
                             translated = self._translate_text(text, target_lang, source_lang)
                             
                             # Only keep if translation occurred (and isn't same as original) 
                             # OR if we want to overwrite even identical text (to fix style)?
                             # Usually we overwrite to ensure consistent look.
                             
                             # Filter: If translated == original, it might be a name or untranslated.
                             # If we detected it was NOT source language, _translate_text returns original.
                             # We should check if we want to overlay original text? 
                             # Probably NOT. We only want to overlay TRANSLATED text.
                             # If text was skipped due to language mismatch, we shouldn't draw over it.
                             
                             if translated != text:
                                  valid_translations.append(translated)
                                  valid_boxes.append(box)
                    
                    if valid_boxes:
                         # Create mask from ALL boxes (to remove original text)
                         # Wait, if we only translate SOME text, we should only inpaint SOME text?
                         # Ideally yes. If we have English and French on screen, and we translate EN->FR.
                         # We should Inpaint EN, write FR. Leave original FR alone.
                         # So we should use 'valid_boxes' for mask too?
                         # Yes, otherwise we delete the FR text and don't write anything back.
                         
                         last_mask = self._create_text_mask(frame, valid_boxes)
                         last_boxes_valid = valid_boxes
                         last_translated_valid = valid_translations
                         
                         frame = self._inpaint_text_regions(frame, last_mask)
                         frame = self._overlay_translated_text_pil(frame, last_boxes_valid, last_translated_valid)
                    else:
                        last_mask = None # Clear mask if no text to translate
                        last_boxes_valid = []
                        last_translated_valid = []
                        
                except Exception as e:
                     logger.warning(f"Error processing frame {frame_count}: {e}")
            else:
                 # Apply cached (hold the translation)
                 if last_mask is not None and last_boxes_valid:
                      try:
                          frame = self._inpaint_text_regions(frame, last_mask)
                          frame = self._overlay_translated_text_pil(frame, last_boxes_valid, last_translated_valid)
                      except Exception as e:
                          pass # If painting fails, just output original frame
                       
            out.write(frame)
            frame_count += 1
            if frame_count % 100 == 0: logger.info(f"Processed {frame_count}/{total_frames}...")
            
            # Periodic VRAM cleanup
            if frame_count % 500 == 0:
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                except: pass
            
        cap.release()
        out.release()
        return output_path
