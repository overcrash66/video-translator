from deep_translator import GoogleTranslator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Translator:
    def __init__(self):
        self.translator_cache = {}

    def get_translator(self, target_lang):
        # Allow input of full language name or code.
        # deep_translator expects generic names or codes.
        # We might need a mapper for "Spanish" -> "es".
        # For simplicity, we assume the UI passes something compatible or we map it.
        
        # Simple mapping for the UI dropdown values
        lang_map = {
            "English": "en", "Spanish": "es", "French": "fr", "German": "de",
            "Italian": "it", "Portuguese": "pt", "Polish": "pl", "Turkish": "tr",
            "Russian": "ru", "Dutch": "nl", "Czech": "cs", "Arabic": "ar",
            "Chinese (Simplified)": "zh-CN", "Japanese": "ja", "Korean": "ko",
            "Hindi": "hi"
        }
        
        code = lang_map.get(target_lang, target_lang.lower())
        
        if code not in self.translator_cache:
            self.translator_cache[code] = GoogleTranslator(source='auto', target=code)
        
        return self.translator_cache[code]

    def translate_text(self, text, target_lang):
        if not text:
            return ""
        
        try:
            translator = self.get_translator(target_lang)
            translated = translator.translate(text)
            return translated
        except Exception as e:
            logger.error(f"Translation failed for '{text}' to {target_lang}: {e}")
            return text # Fallback to original

    def translate_segments(self, segments, target_lang):
        """
        Translates a list of segments in-place or returns new list.
        """
        translated_segments = []
        logger.info(f"Translating {len(segments)} segments to {target_lang}...")
        
        for seg in segments:
            new_seg = seg.copy()
            new_seg["translated_text"] = self.translate_text(seg["text"], target_lang)
            translated_segments.append(new_seg)
            
        return translated_segments

if __name__ == "__main__":
    tr = Translator()
    print(tr.translate_text("Hello world", "Spanish"))
