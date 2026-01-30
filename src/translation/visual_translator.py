"""
Visual Text Translation module.

Detects text in video frames using PaddleOCR, translates it, 
and overlays the translated text using OpenCV inpainting and PIL for text rendering.

Memory Optimization Notes:
- Frames are resized before OCR to reduce VRAM usage
- LRU caches are bounded to prevent RAM growth
- Periodic GC is performed during video processing
"""

import gc
import logging
import threading
import cv2
# Save cv2.error reference at import time (before any mocking in tests)
OpenCVError = cv2.error
import numpy as np
from pathlib import Path
from src.utils import config
from deep_translator import GoogleTranslator
from cachetools import TTLCache
from PIL import Image, ImageDraw, ImageFont
import sys
import os

# Note: PaddlePaddle is now CPU-only to avoid cuDNN conflicts with PyTorch
# Set PaddlePaddle flags BEFORE importing to suppress oneDNN/PIR warnings
# Version 3.3.0 has a regression on Windows; disabling PIR and New Executor helps.
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_enable_pir_in_executor'] = '0'
os.environ['FLAGS_enable_new_executor'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_cpu_deterministic'] = '1'
os.environ['GLOG_minloglevel'] = '2'  # Suppress verbose paddle logging
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'  # Skip connectivity check

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

# Memory optimization constants
MAX_OCR_WIDTH = 1280  # Max width for OCR processing (reduces VRAM significantly)
GC_INTERVAL_FRAMES = 100  # Run GC every N frames (was 500)
MIN_OCR_CONFIDENCE = 0.4  # Minimum confidence threshold for OCR detections
TRANSLATION_CACHE_TTL = 4 * 3600  # 4 hours TTL for translation cache (supports long videos)
TRANSLATION_CACHE_MAXSIZE = 500  # Max entries in translation cache


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
        self._ocr_scale_factor = 1.0  # Track scaling for coordinate mapping
        # Thread-safe translation cache with TTL
        self._translation_cache = TTLCache(maxsize=TRANSLATION_CACHE_MAXSIZE, ttl=TRANSLATION_CACHE_TTL)
        self._cache_lock = threading.Lock()
        # CJK font paths for Asian language support
        self._cjk_fonts = {
            'ja': ['msgothic.ttc', 'meiryo.ttc', 'yugothic.ttc'],  # Japanese
            'zh': ['simsun.ttc', 'msyh.ttc', 'simhei.ttf'],  # Chinese
            'ko': ['malgun.ttf', 'gulim.ttc'],  # Korean
        }
        
    def _find_font(self, candidates: list = None) -> str:
        """
        Find a suitable Unicode font on Windows.
        
        Args:
            candidates: Optional list of font names to try. Defaults to common Unicode fonts.
            
        Returns:
            The first loadable font name, or 'arial.ttf' as fallback.
        """
        if candidates is None:
            # Common Windows fonts with good Unicode support
            candidates = ["arial.ttf", "segoeui.ttf", "tahoma.ttf", "msgothic.ttc"]
        
        for font in candidates:
            try:
                # Check if loadable (PIL searches system path on Windows)
                ImageFont.truetype(font, 10)
                return font
            except OSError:
                continue
        return "arial.ttf"  # Fallback
    
    def _get_font_for_language(self, target_lang: str) -> str:
        """
        Get the best font for a specific target language.
        
        For CJK languages (Chinese, Japanese, Korean), tries language-specific fonts first.
        Falls back to the default font path for other languages.
        
        Args:
            target_lang: Target language code ('ja', 'zh', 'ko', etc.)
            
        Returns:
            Font path suitable for the target language.
        """
        # Check if CJK language
        if target_lang in self._cjk_fonts:
            cjk_candidates = self._cjk_fonts[target_lang]
            font = self._find_font(cjk_candidates)
            if font != "arial.ttf":  # Found a CJK font
                return font
            # CJK font not found, log warning
            logger.warning(f"No CJK font found for language '{target_lang}'. Using fallback: {self.font_path}")
        
        return self.font_path

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

            logger.info(f"Initializing PaddleOCR (CPU mode) for language: {source_lang}...")
            
            try:
                # Map common language codes
                paddle_lang_map = {
                    'en': 'en', 'ar': 'ar', 'zh': 'ch', 'fr': 'fr', 'de': 'german',
                    'es': 'es', 'it': 'it', 'ja': 'japan', 'ko': 'korean', 'ru': 'ru', 'pt': 'pt',
                }
                ocr_lang = paddle_lang_map.get(source_lang, 'en')
                
                # [FIX] Simplified loader: Force CPU immediately
                # This avoids the complex GPU/HPI detection which hangs on some Windows setups
                # with PaddlePaddle 3.x.
                self.ocr_model = PaddleOCR(
                    use_angle_cls=True,
                    lang=ocr_lang,
                    device='cpu',
                    enable_mkldnn=False # Stabilizes Windows CPU inference
                )
                self.model_loaded = True
                self.current_engine = ocr_engine
                logger.info(f"PaddleOCR (engine={ocr_engine}, lang={ocr_lang}) loaded successfully.")
            except Exception as e:
                logger.error(f"Critical failure loading PaddleOCR: {e}")
                self.model_loaded = False
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
        # Clear translation cache (thread-safe)
        with self._cache_lock:
            self._translation_cache.clear()
        logger.info("VisualTranslator model unloaded.")

    def _get_translator(self, target_lang: str) -> GoogleTranslator:
        """Get or create a cached translator for the target language."""
        if target_lang not in self.translator_cache:
            self.translator_cache[target_lang] = GoogleTranslator(source='auto', target=target_lang)
        return self.translator_cache[target_lang]

    def _cached_translate(self, text: str, target_lang: str) -> str:
        """
        Thread-safe cached translation using TTLCache.
        
        Args:
            text: Text to translate.
            target_lang: Target language code.
            
        Returns:
            Translated text, or original text on failure.
        """
        cache_key = (text, target_lang)
        
        # Check cache first (thread-safe read)
        with self._cache_lock:
            if cache_key in self._translation_cache:
                return self._translation_cache[cache_key]
        
        # Perform translation outside lock to avoid blocking
        try:
            translator = self._get_translator(target_lang)
            translated = translator.translate(text)
        except ConnectionError as e:
            logger.error(f"Network error during translation for '{text}': {type(e).__name__}: {e}")
            return text
        except Exception as e:
            logger.warning(f"Translation failed for '{text}': {type(e).__name__}: {e}")
            return text
        
        # Store in cache (thread-safe write)
        with self._cache_lock:
            self._translation_cache[cache_key] = translated
        
        return translated

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

    def _translate_text(self, text: str, target_lang: str, source_lang: str = None) -> str:
        """Translate text if it matches source language or if source is unknown."""
        if not text or len(text.strip()) < 2:
            return text
            
        # Language filtering
        detected = self._detect_language(text)
        
        # Relaxed filtering: Only skip if it's DEFINITELY the target language.
        # Short strings often lead to incorrect language detection.
        if detected == target_lang:
            logger.info(f"[VisualTranslator] Text='{text}' is already in target language ({target_lang}), skipping.")
            return text
            
        # [REMOVED] Strict source_lang check as it was too aggressive for short OCR snippets.
        
        translated = self._cached_translate(text, target_lang)
        if translated and translated != text:
            logger.info(f"[VisualTranslator] Translated '{text}' -> '{translated}'")
        return translated if translated else text

    def _resize_for_ocr(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Resize frame for OCR processing to reduce VRAM usage.
        Returns resized frame and scale factor for coordinate mapping.
        """
        h, w = frame.shape[:2]
        if w <= MAX_OCR_WIDTH:
            return frame, 1.0
        
        scale = MAX_OCR_WIDTH / w
        new_w = MAX_OCR_WIDTH
        new_h = int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale

    def _scale_boxes(self, boxes: list, scale: float) -> list:
        """
        Scale bounding boxes back to original frame coordinates.
        """
        if scale == 1.0:
            return boxes
        
        inv_scale = 1.0 / scale
        scaled_boxes = []
        for box in boxes:
            scaled_box = [[pt[0] * inv_scale, pt[1] * inv_scale] for pt in box]
            scaled_boxes.append(scaled_box)
        return scaled_boxes

    def _create_text_mask(self, frame: np.ndarray, boxes: list) -> np.ndarray:
        """Create a binary mask for detected text regions."""
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for box in boxes:
            pts = np.array(box, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        return mask

    def _inpaint_text_regions(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Use OpenCV inpainting to remove text from frame.
        
        Uses the TELEA algorithm (Fast Marching Method) which provides good results
        for small text regions. For higher quality results on larger regions,
        consider integrating GAN-based inpainting models like:
        - LaMa (https://github.com/advimman/lama) - Large Mask Inpainting
        - MAT (Mask-Aware Transformer)
        
        Note: GAN-based models require additional model downloads and GPU resources.
        The TELEA algorithm is used by default for speed and simplicity.
        
        Args:
            frame: Input BGR frame.
            mask: Binary mask where 255 indicates regions to inpaint.
            
        Returns:
            Inpainted frame with text regions filled.
        """
        # Dilate mask slightly for better inpainting
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Use TELEA inpainting algorithm
        inpainted = cv2.inpaint(frame, dilated_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        return inpainted

    def _overlay_translated_text_pil(self, frame: np.ndarray, boxes: list, 
                                  translated_texts: list, target_lang: str = 'en') -> np.ndarray:
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
             
             # Select font based on target language (CJK support)
             font_path = self._get_font_for_language(target_lang)
             try:
                font = ImageFont.truetype(font_path, font_size)
             except OSError:
                logger.warning(f"Failed to load font '{font_path}', using default.")
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
                    font = ImageFont.truetype(font_path, font_size)
                 except OSError:
                    break  # Font doesn't scale, use current size
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
        
        logger.info(f"Successfully opened video for OCR: {video_path}")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0: fps = 30.0
        
        # Log memory optimization info
        if width > MAX_OCR_WIDTH:
            logger.info(f"Frame size {width}x{height} exceeds MAX_OCR_WIDTH ({MAX_OCR_WIDTH}). OCR will use resized frames.")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
             logger.error(f"Failed to initialize VideoWriter for: {output_path}")
             cap.release()
             import shutil
             shutil.copy(video_path, output_path)
             return output_path
             
        logger.info(f"VideoWriter initialized: {width}x{height} at {fps} FPS")
        
        # Calculate frame interval
        interval_frames = max(1, int(fps * ocr_interval_sec))
        
        frame_count = 0
        
        # Cache for text detection results
        last_boxes = []
        last_mask = None
        last_translated_valid = [] # Subset of texts that were valid for translation
        last_boxes_valid = []      # Subset of boxes corresponding to valid texts
        
        # Pre-import torch for CUDA cleanup (avoid repeated imports in loop)
        try:
            import torch
            has_torch = torch.cuda.is_available()
        except ImportError:
            has_torch = False
        
        logger.info(f"Visual Translation Loop Starting: {total_frames} frames to process at 2s OCR interval...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Run OCR at interval
            if frame_count % interval_frames == 0:
                try:
                    boxes = []
                    original_texts = []
                    
                    # [MEMORY OPT] Resize frame for OCR to reduce VRAM usage
                    ocr_frame, scale_factor = self._resize_for_ocr(frame)
                    
                    if ocr_engine == "PaddleOCR":
                        result = self.ocr_model.ocr(ocr_frame)
                        
                        # Parse PaddleOCR result - handle both old list format and new HPI dict format
                        if result:
                            # Handle nested list format: [[[[box], (text, conf)], ...]]
                            if result[0] is not None:
                                for line in result[0]:
                                    try:
                                        # New HPI/dict format: {'text': str, 'confidence': float, 'text_region': list}
                                        if isinstance(line, dict):
                                            if 'text_region' in line and 'text' in line:
                                                confidence = line.get('confidence', 1.0)
                                                if confidence < MIN_OCR_CONFIDENCE:
                                                    logger.debug(f"[VisualTranslator] Skipping low-confidence detection: '{line['text']}' (conf={confidence:.2f})")
                                                    continue
                                                boxes.append(line['text_region'])
                                                original_texts.append(line['text'])
                                        # Old list format: [[box], (text, confidence)]
                                        elif isinstance(line, (list, tuple)) and len(line) >= 2:
                                            box_data = line[0]
                                            text_data = line[1]
                                            
                                            # text_data can be tuple (text, conf) or just text
                                            if isinstance(text_data, (list, tuple)):
                                                text = text_data[0]
                                                confidence = text_data[1] if len(text_data) > 1 else 1.0
                                            else:
                                                text = str(text_data)
                                                confidence = 1.0
                                            
                                            # Apply confidence filter
                                            if confidence < MIN_OCR_CONFIDENCE:
                                                logger.debug(f"[VisualTranslator] Skipping low-confidence detection: '{text}' (conf={confidence:.2f})")
                                                continue
                                            
                                            boxes.append(box_data)
                                            original_texts.append(text)
                                    except (IndexError, KeyError, TypeError) as parse_err:
                                        logger.warning(f"[VisualTranslator] Failed to parse OCR line: {type(parse_err).__name__}: {parse_err}")
                        
                        if boxes:
                            logger.info(f"[VisualTranslator] Frame {frame_count}: Detected {len(boxes)} text regions")
                                
                    elif ocr_engine == "EasyOCR":
                        results = self.ocr_model.readtext(ocr_frame)
                        for (bbox, text, prob) in results:
                            if prob > 0.4:
                                boxes.append(bbox)
                                original_texts.append(text)
                        logger.info(f"[VisualTranslator] Frame {frame_count}: EasyOCR detected {len(boxes)} text regions")
                    
                    # [MEMORY OPT] Scale boxes back to original frame coordinates
                    boxes = self._scale_boxes(boxes, scale_factor)
                    
                    # Cleanup resized frame immediately
                    del ocr_frame
                    
                    # Filter and Translate
                    valid_translations = []
                    valid_boxes = []
                    
                    if boxes:
                         for box, text in zip(boxes, original_texts):
                             # Translate with language checking
                             translated = self._translate_text(text, target_lang, source_lang)
                             
                             # [FIX] Always include detected text for overlay
                             # Even if translation == original (proper names, numbers), we still
                             # want to render it with consistent styling over the inpainted area
                             if translated and len(translated.strip()) > 0:
                                  valid_translations.append(translated)
                                  valid_boxes.append(box)
                    
                    if valid_boxes:
                         logger.info(f"[VisualTranslator] Frame {frame_count}: Overlaying {len(valid_boxes)} translated texts")
                         # Create mask only for boxes we're replacing
                         last_mask = self._create_text_mask(frame, valid_boxes)
                         last_boxes_valid = valid_boxes
                         last_translated_valid = valid_translations
                         
                         frame = self._inpaint_text_regions(frame, last_mask)
                         frame = self._overlay_translated_text_pil(frame, last_boxes_valid, last_translated_valid, target_lang)
                    else:
                        last_mask = None # Clear mask if no text to translate
                        last_boxes_valid = []
                        last_translated_valid = []
                        
                except OpenCVError as e:
                    # OpenCV errors are critical - log and continue
                    logger.error(f"[VisualTranslator] OpenCV error in frame {frame_count}: {type(e).__name__}: {e}")
                except Exception as e:
                    logger.warning(f"[VisualTranslator] Error processing frame {frame_count}: {type(e).__name__}: {e}")
            else:
                 # Apply cached (hold the translation)
                 if last_mask is not None and last_boxes_valid:
                      try:
                          frame = self._inpaint_text_regions(frame, last_mask)
                          frame = self._overlay_translated_text_pil(frame, last_boxes_valid, last_translated_valid, target_lang)
                      except OpenCVError as e:
                          logger.error(f"[VisualTranslator] OpenCV error applying cached mask: {type(e).__name__}: {e}")
                      except Exception as e:
                          pass # If painting fails, just output original frame
                       
            out.write(frame)
            frame_count += 1
            if frame_count % 50 == 0: logger.info(f"Visual Progress: {frame_count}/{total_frames} frames ({frame_count/total_frames:.1%})")
            
            # [MEMORY OPT] Periodic VRAM cleanup - now uses constant (every 100 frames vs 500)
            if frame_count % GC_INTERVAL_FRAMES == 0:
                gc.collect()
                if has_torch:
                    torch.cuda.empty_cache()
            
        cap.release()
        out.release()
        logger.info(f"Visual translation processing complete. Output saved to: {output_path}")
        return output_path
