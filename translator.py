import logging
import torch
from deep_translator import GoogleTranslator
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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
            # Load with 4-bit quantization for VRAM efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                device_map="auto", 
                quantization_config=quantization_config,
                trust_remote_code=True
            )
            logger.info("HY-MT1.5 model loaded (4-bit quantized).")
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

    def translate_batch(self, texts, source_lang_code, target_lang_code):
        if not texts:
            return []
            
        self.load_model()
        
        # Prepare batch prompts
        prompts = []
        for text in texts:
             messages = [{"role": "user", "content": f"Translate the following text from {source_lang_code} to {target_lang_code}:\n{text}"}]
             if self.tokenizer.chat_template:
                 prompts.append(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
             else:
                 prompts.append(f"<|user|>\nTranslate from {source_lang_code} to {target_lang_code}:\n{text}\n<|assistant|>\n")

        try:
            # Batch tokenization
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, return_token_type_ids=False).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=256, 
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
            # Decode batch
            # Slice input tokens from output
            generated_tokens = outputs[:, inputs.input_ids.shape[1]:]
            responses = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            return [r.strip() for r in responses]
            
        except Exception as e:
            logger.error(f"HY-MT batch translation error: {e}")
            return texts # Fallback
            
    def translate(self, text, source_lang_code, target_lang_code):
         # Wrapper for single
         return self.translate_batch([text], source_lang_code, target_lang_code)[0]

    def refine_with_context(self, current_text, prev_text, next_text, source_text, source_lang, target_lang):
        """
        Refines the translation using context.
        """
        self.load_model()
        
        # Prompt Engineering for Improvement
        # We explicitly ask the model to review and improve.
        
        prompt = (
            f"Review and improve the translation from {source_lang} to {target_lang}.\n"
            f"Context (Previous): {prev_text}\n"
            f"Context (Next): {next_text}\n"
            f"Source Text: {source_text}\n"
            f"Current Translation: {current_text}\n\n"
            f"If the current translation is accurate and fits the context, repeat it exactly.\n"
            f"If it needs improvement (grammar, tone, context), provide ONLY the improved text.\n"
            f"Improved Translation:"
        )

        messages = [{"role": "user", "content": prompt}]
        
        try:
             # Similar generation logic as translate
             if self.tokenizer.chat_template:
                text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
             else:
                text_input = f"<|user|>\n{prompt}\n<|assistant|>\n"

             inputs = self.tokenizer(text_input, return_tensors="pt", return_token_type_ids=False).to(self.model.device)
             
             with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=256, 
                    temperature=0.2, # Lower temp for strict review
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
             
             new_tokens = outputs[0][inputs.input_ids.shape[1]:]
             response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
             
             # Basic sanity check: Don't accept if it hallucinated a huge text or empty
             if not response or len(response) > len(current_text) * 3:
                 return current_text
                 
             return response
             
        except Exception as e:
             logger.warning(f"Refinement failed: {e}")
             return current_text


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

            return text
    
    def translate_segments(self, segments, target_lang, model="google", source_lang="auto", optimize=False):
        """
        Translates a list of segments.
        optimize: If True, performs a second pass with local LLM to refine context.
        """
        # Prepare HYMT if needed (batched loading, but sequential generation for now)
        if model == "hymt" and not self.hymt:
            self.hymt = HYMTTranslator()
            self.hymt.load_model()
            
        translated_segments = []
        logger.info(f"Translating {len(segments)} segments to {target_lang} using {model}...")
        
        if model == "hymt":
            # Batch process for HYMT
            if not self.hymt:
                self.hymt = HYMTTranslator()
            
            # We assume single source language for the batch
            source_code = source_lang if source_lang != "auto" else "en"
            _, target_code = self.get_google_translator(target_lang)
            
            texts = [s["text"] for s in segments]
            
            # Process in chunks of 8 to prevent OOM even with 4-bit
            chunk_size = 8
            translated_texts = []
            
            for i in range(0, len(texts), chunk_size):
                chunk = texts[i:i+chunk_size]
                translated_texts.extend(self.hymt.translate_batch(chunk, source_code, target_code))
            
            for i, seg in enumerate(segments):
                new_seg = seg.copy()
                new_seg["translated_text"] = translated_texts[i]
                translated_segments.append(new_seg)

        else:
            # Google Translate (Sequential)
            for seg in segments:
                new_seg = seg.copy()
                new_seg["translated_text"] = self.translate_text(
                    seg["text"], target_lang, model, source_lang
                )
                translated_segments.append(new_seg)
            
        # Optimization Pass
        if optimize and translated_segments:
            logger.info("Starting Contextual Optimization Pass...")
            if not self.hymt:
                self.hymt = HYMTTranslator()
            
            # Ensure model is loaded
            self.hymt.load_model()
            
            count = 0
            total = len(translated_segments)
            
            for i in range(total):
                seg = translated_segments[i]
                curr_text = seg["translated_text"]
                source_text = seg["text"]
                
                # Get Context
                prev_seg = translated_segments[i-1] if i > 0 else None
                next_seg = translated_segments[i+1] if i < total - 1 else None
                
                prev_text = prev_seg["translated_text"] if prev_seg else "[START]"
                next_text = next_seg["translated_text"] if next_seg else "[END]"
                
                # Check for meaningful content to optimize (skip very short)
                if len(curr_text.split()) < 2:
                    continue
                    
                refined = self.hymt.refine_with_context(
                    curr_text, prev_text, next_text, source_text, source_lang, target_lang
                )
                
                if refined != curr_text:
                    logger.info(f"Refined seg {i}: '{curr_text}' -> '{refined}'")
                    translated_segments[i]["translated_text"] = refined
                    count += 1
            
            logger.info(f"Optimization complete. Refined {count}/{total} segments.")

            
        # Unload HYMT to save vram
        if model == "hymt" and self.hymt:
            self.hymt.unload_model()
            
        return translated_segments

if __name__ == "__main__":
    tr = Translator()
    # print(tr.translate_text("Hello world", "Spanish", model="hymt"))

