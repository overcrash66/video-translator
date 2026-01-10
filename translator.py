import logging
import torch
from deep_translator import GoogleTranslator
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HYMTTranslator:
    def __init__(self, model_id="tencent/HY-MT1.5-1.8B"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        if self.model:
            return
        
        logger.info(f"Loading {self.model_id} on {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                device_map="auto", 
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            logger.info("HY-MT1.5 model loaded.")
        except Exception as e:
            logger.error(f"Failed to load HY-MT model: {e}")
            raise

    def unload_model(self):
        if self.model:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            self.model = None
            self.tokenizer = None
            logger.info("HY-MT1.5 model unloaded.")

    def translate(self, text, source_lang_code, target_lang_code):
        if not text.strip():
            return ""
        
        self.load_model()
        
        # HY-MT1.5 uses standard chat messages for translation prompt
        # User: Translate to [Target]: [Source Text]
        # Or specialized tokens if available. 
        # Based on research, we rely on chat template or standard prompt.
        
        # Let's use a explicit prompt structure that works for general LLMs
        # "Translate the following text from {src} to {tgt}:\n{text}"
        
        # However, HY-MT is fine-tuned. 
        # Standard format likely: "[src_lang] text [tgt_lang]" e.g. "en hello es"
        # Since I cannot verify exact custom format without docs, I will use a descriptive prompt
        # which usually works for instruction tuned models.
        
        # Better: use apply_chat_template if available
        messages = [
            {"role": "user", "content": f"Translate the following text from {source_lang_code} to {target_lang_code}:\n{text}"}
        ]
        
        try:
            if self.tokenizer.chat_template:
                text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                # Fallback manual formatting
                text_input = f"<|user|>\nTranslate from {source_lang_code} to {target_lang_code}:\n{text}\n<|assistant|>\n"

            inputs = self.tokenizer(text_input, return_tensors="pt", return_token_type_ids=False).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=256, 
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
            # Decode
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant response (post-processing)
            # Check if input is echoed
            # Usually transformers.decode includes input? No, generate output includes input tokens in 'outputs' usually.
            
            full_text = decoded
            # Naive parse: remove input prompt if present
            # Actually simplest is to decode only the new tokens:
            new_tokens = outputs[0][inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"HY-MT translation error: {e}")
            return text # Fallback


class Translator:
    def __init__(self):
        self.translator_cache = {}
        self.hymt = None

    def get_google_translator(self, target_lang):
        # ... logic as before ...
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
        
        return self.translator_cache[code], code

    def translate_text(self, text, target_lang, model="google", source_lang="auto"):
        if not text:
            return ""
            
        try:
            # Map target_lang to code first
            _, target_code = self.get_google_translator(target_lang)
            
            if model == "hymt":
                if not self.hymt:
                    self.hymt = HYMTTranslator()
                
                # We need source code for HYMT
                # If auto, we assume 'en' or try to reuse UI source selection
                source_code = source_lang if source_lang != "auto" else "en" # fallback
                
                return self.hymt.translate(text, source_code, target_code)
                
            else:
                translator, _ = self.get_google_translator(target_lang)
                return translator.translate(text)
                
        except Exception as e:
            logger.error(f"Translation failed ({model}): {e}")
            return text

    def translate_segments(self, segments, target_lang, model="google", source_lang="auto"):
        """
        Translates a list of segments.
        """
        # Prepare HYMT if needed (batched loading, but sequential generation for now)
        if model == "hymt" and not self.hymt:
            self.hymt = HYMTTranslator()
            self.hymt.load_model()
            
        translated_segments = []
        logger.info(f"Translating {len(segments)} segments to {target_lang} using {model}...")
        
        for seg in segments:
            new_seg = seg.copy()
            new_seg["translated_text"] = self.translate_text(
                seg["text"], target_lang, model, source_lang
            )
            translated_segments.append(new_seg)
            
        # Unload HYMT to save vram
        if model == "hymt" and self.hymt:
            self.hymt.unload_model()
            
        return translated_segments

if __name__ == "__main__":
    tr = Translator()
    # print(tr.translate_text("Hello world", "Spanish", model="hymt"))

