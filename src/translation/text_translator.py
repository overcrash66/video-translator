import logging
import torch
from deep_translator import GoogleTranslator
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMTranslator:
    def __init__(self, model_id="tencent/HY-MT1.5-1.8B"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Determine model capabilities/prompt style
        self.is_llama = "llama" in model_id.lower()
        self.is_alma = "alma" in model_id.lower()
        self.is_hymt = "hy-mt" in model_id.lower()

    def unload_model(self):
        """Unload model to free VRAM."""
        if self.model:
            logger.info(f"Unloading {self.model_id}...")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("LLM Translator unloaded.")

    def load_model(self):
        if self.model:
            return
        
        logger.info(f"Loading {self.model_id} on {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            
            # Use 4-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                device_map="auto", 
                quantization_config=quantization_config,
                trust_remote_code=True
            )
            
            # Ensure proper pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Critical for batched generation with LLaMA/decoder models
            self.tokenizer.padding_side = 'left'
                
            logger.info(f"{self.model_id} loaded successfully.")
            
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            raise

    def get_prompt(self, text, source_lang, target_lang, context_prev=None, context_next=None):
        """Generates prompt based on model type and global context."""
        from src.utils.languages import get_language_name
        
        # Convert codes to full names (e.g. "en" -> "English")
        source_name = get_language_name(source_lang)
        target_name = get_language_name(target_lang)
        
        # Standard Translation Prompt
        if self.is_llama:
            # Llama 3.1 Instruct
            sys_prompt = (
                "You are a professional video translator. Translate the user's text accurately, "
                "preserving tone and style. Do not add any explanations."
            )
            if context_prev:
                sys_prompt += f"\nContext (Previous Line): {context_prev}"
                
            user_prompt = f"Translate from {source_name} to {target_name}:\n\n{text}"
            
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            if self.tokenizer.chat_template:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                 return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        elif self.is_alma:
            # ALMA-R (Advanced Language Model-based Translator)
            # ALMA uses specific formatting: "Translate this from {src} to {tgt}:\n{src_text}\n{tgt_text}"
            # Reference: https://github.com/fe1ixxu/ALMA
            # NOTE: ALMA expects full English names (e.g. English, French)
            prompt = f"Translate this from {source_name} to {target_name}:\n{source_name}: {text}\n{target_name}:"
            logger.info(f"ALMA Prompt: {repr(prompt)}")
            return prompt
            
        else:
            # HY-MT (Default / Fallback)
            messages = [{"role": "user", "content": f"Translate the following text from {source_name} to {target_name}:\n{text}"}]
            if self.tokenizer.chat_template:
                 return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return f"<|user|>\nTranslate from {source_name} to {target_name}:\n{text}\n<|assistant|>\n"

    def translate_batch(self, texts, source_lang_code, target_lang_code):
        if not texts: return []
        self.load_model()
        
        prompts = [self.get_prompt(t, source_lang_code, target_lang_code) for t in texts]
        
        try:
            # [Fix] Enforce truncation and max length
            self.tokenizer.model_max_length = 2048 # Explicitly set
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(self.device)
            
            # [Fix] Safety check for Token ID OOB (Root cause of CUDA assert in Embedding)
            vocab_size = self.model.config.vocab_size
            
            # Check for negatives
            if (inputs.input_ids < 0).any():
                logger.warning("Found negative token IDs. Clamping to 0.")
                inputs.input_ids[inputs.input_ids < 0] = 0
                
            # Check for overflow
            if (inputs.input_ids >= vocab_size).any():
                logger.warning("Found token IDs exceeding vocab size. Clamping...")
                inputs.input_ids[inputs.input_ids >= vocab_size] = self.tokenizer.unk_token_id or 0
            
            # Verify pad_token_id
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None or pad_id >= vocab_size or pad_id < 0:
                logger.warning(f"Invalid pad_token_id {pad_id}. Resetting to eos_token_id or 0.")
                pad_id = self.tokenizer.eos_token_id
                if pad_id is None or pad_id >= vocab_size:
                    pad_id = 0
                self.tokenizer.pad_token_id = pad_id
            
            with torch.no_grad():
                # [Stability Fix] Disable autocast and use greedy decoding to prevent CUDA asserts
                # caused by sampling from unstable distributions in 4-bit models on Windows.
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=512, 
                    do_sample=False, # Robustness: greedy decoding
                    num_beams=1,
                    pad_token_id=pad_id
                )
                torch.cuda.synchronize()
                
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            results = []
            for i, raw in enumerate(decoded):
                # Clean up retrieval: remove prompt if chat template echoes it (model dependent)
                # Llama 3 often only outputs answer, but let's be safe.
                # ALMA outputs "English: Hello\nGerman: Hallo", we need to extract target.
                
                clean = raw
                if self.is_alma:
                    from src.utils.languages import get_language_name
                    target_name = get_language_name(target_lang_code)
                    # ALMA output structure: ...\n{target_lang}: {translation}
                    if f"{target_name}:" in raw:
                        clean = raw.split(f"{target_name}:")[-1].strip()
                    elif f"{target_lang_code}:" in raw: # Fallback for code
                         clean = raw.split(f"{target_lang_code}:")[-1].strip()
                    else:
                        # Fallback heuristic
                         clean = raw.replace(prompts[i], "").strip()
                elif self.is_llama:
                     # Llama typically just answers.
                     # If prompt is included, strip it.
                     pass # Transformers usually handles this with decode if not 'return_full_text=? '
                     # Actually generate() by default returns input+output.
                     pass
                     
                # Correct slicing for all models (transformers default behavior for causal LM is to return input+output)
                # We should slice based on input length
                results.append(clean)
                
            # Better slicing logic:
            final_results = []
            input_len = inputs.input_ids.shape[1]
            gen_tokens = outputs[:, input_len:]
            
            final_decoded = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            
            return [r.strip() for r in final_decoded]

        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            return texts
            
    def translate_context_aware(self, text, source_lang, target_lang, prev_text=None):
        """Translates a single segment with context."""
        self.load_model()
        prompt = self.get_prompt(text, source_lang, target_lang, context_prev=prev_text)
        
        try:
            # [Fix] Enforce truncation and max length
            self.tokenizer.model_max_length = 2048
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
            
            # [Fix] Safety check
            vocab_size = self.model.config.vocab_size
            if (inputs.input_ids < 0).any(): inputs.input_ids[inputs.input_ids < 0] = 0
            if (inputs.input_ids >= vocab_size).any():
                inputs.input_ids[inputs.input_ids >= vocab_size] = self.tokenizer.unk_token_id or 0
                
            # Verify pad_token_id
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None or pad_id >= vocab_size or pad_id < 0:
                pad_id = self.tokenizer.eos_token_id or 0
                self.tokenizer.pad_token_id = pad_id

            with torch.no_grad():
                # [Stability Fix] Disable autocast and use greedy decoding to prevent CUDA asserts
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=512, 
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=pad_id
                )
                torch.cuda.synchronize()
             
            # Slice input from output
            input_len = inputs.input_ids.shape[1]
            gen_tokens = outputs[0][input_len:]
            decoded = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            
            # Post-process ALMA
            if self.is_alma:
                 from src.utils.languages import get_language_name
                 target_name = get_language_name(target_lang)
                 if f"{target_name}:" in decoded:
                     decoded = decoded.split(f"{target_name}:")[-1].strip()
                 
            return decoded
            
        except Exception as e:
            logger.error(f"Context translation failed: {e}")
            return text

    def translate(self, text, source_lang, target_lang):
        return self.translate_batch([text], source_lang, target_lang)[0]


class Translator:
    def __init__(self):
        self.translator_cache = {}
        self.llm_translator = None 
        # Import centralized languages
        from src.utils import languages
        self.languages = languages

    def get_google_translator(self, target_lang):
        # Use centralized map
        code = self.languages.get_language_code(target_lang)
        
        if code not in self.translator_cache:
            self.translator_cache[code] = GoogleTranslator(source='auto', target=code)
        return self.translator_cache[code], code

    def translate_segments(self, segments, target_lang, model="google", source_lang="auto", optimize=False):
        """
        Translates segments using selected model.
        optimize: If True, uses sequential context-aware translation (slower but better).
        """
        if not segments: return []
        
        _, target_code = self.get_google_translator(target_lang)
        source_code = source_lang if source_lang != "auto" else "en"
        
        translated_segments = []
        
        if model in ["hymt", "llama", "alma"]:
            model_id_map = {
                "hymt": "tencent/HY-MT1.5-1.8B",
                "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "alma": "haoranxu/ALMA-7B-R"
            }
            mid = model_id_map.get(model, "tencent/HY-MT1.5-1.8B")
            
            if self.llm_translator and self.llm_translator.model_id != mid:
                self.llm_translator.unload_model()
                self.llm_translator = None
                
            if not self.llm_translator:
                self.llm_translator = LLMTranslator(mid)
                
            logger.info(f"Translating {len(segments)} items with {model} (Optimize={optimize})...")
            
            if optimize:
                # Sequential Context-Aware Mode
                prev_text = None
                for i, seg in enumerate(segments):
                    logger.info(f"Translating segment {i+1}/{len(segments)} (Context-Aware)...")
                    trans_text = self.llm_translator.translate_context_aware(
                        seg['text'], source_code, target_code, prev_text=prev_text
                    )
                    
                    new_s = seg.copy()
                    new_s['translated_text'] = trans_text
                    translated_segments.append(new_s)
                    
                    # Update context for next iteration
                    prev_text = trans_text
            else:
                # Fast Batch Mode (No inter-segment context)
                texts = [s['text'] for s in segments]
                # Smaller chunks for Llama
                chunk_size = 4 if "llama" in model or "alma" in model else 8 
                
                trans_texts = []
                for i in range(0, len(texts), chunk_size):
                    chunk = texts[i:i+chunk_size]
                    trans_texts.extend(self.llm_translator.translate_batch(chunk, source_code, target_code))
                    
                for i, seg in enumerate(segments):
                    new_s = seg.copy()
                    new_s['translated_text'] = trans_texts[i]
                    translated_segments.append(new_s)
            
            # Unload after use
            self.llm_translator.unload_model()
            
        else:
            # Google
            translator, _ = self.get_google_translator(target_lang)
            for seg in segments:
                s = seg.copy()
                s['translated_text'] = translator.translate(seg['text'])
                translated_segments.append(s)
                
        return translated_segments

if __name__ == "__main__":
    tr = Translator()
    # print(tr.translate_text("Hello world", "Spanish", model="hymt"))

