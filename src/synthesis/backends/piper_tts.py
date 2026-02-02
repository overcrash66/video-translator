import logging
import subprocess
import requests
import zipfile
import io
from pathlib import Path
from src.utils import config
from src.synthesis.backends.base import TTSBackend

logger = logging.getLogger(__name__)

class PiperTTSBackend(TTSBackend):
    def __init__(self, piper_map):
        self.piper_map = piper_map

    def generate(self, text, output_path, language="en", speaker_wav=None, **kwargs):
        try:
            # 1. Resolve Voice Model
            model_name = self.piper_map.get(language, "en_US-lessac-high")
            model_dir = config.TEMP_DIR / "piper_models"
            model_dir.mkdir(exist_ok=True)
            
            onnx_path = model_dir / f"{model_name}.onnx"
            # conf_path = model_dir / f"{model_name}.onnx.json" # Not explicitly used in CLI but needed by binary
            
            # 2. Download Model if missing
            if not onnx_path.exists():
                logger.info(f"Downloading Piper model: {model_name}...")
                self._download_piper_model(model_name, model_dir)
            
            # 3. Check for Piper Binary
            piper_bin_dir = config.TEMP_DIR / "piper_bin"
            
            # Detect platform
            import platform
            system = platform.system()
            if system == "Windows":
                 exe_name = "piper.exe"
            else:
                 exe_name = "piper"
                 
            piper_exe = piper_bin_dir / "piper" / exe_name
            
            if not piper_exe.exists():
                logger.info(f"Piper binary ({exe_name}) not found. Downloading...")
                self._download_piper_binary(piper_bin_dir, system)
                
            # Ensure Linux binary is executable
            if system != "Windows" and piper_exe.exists():
                import os
                st = os.stat(piper_exe)
                os.chmod(piper_exe, st.st_mode | 0o111)
                
            if not piper_exe.exists():
                raise RuntimeError("Piper binary missing after download attempt.")

            logger.info(f"Generating Piper TTS (Binary): '{text[:20]}...' ({model_name})")
            
            # 4. Execute Piper
            cmd = [
                str(piper_exe),
                "--model", str(onnx_path),
                "--output_file", str(output_path)
            ]
            
            process = subprocess.Popen(
                cmd, 
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            
            stdout, stderr = process.communicate(input=text)
            
            if process.returncode != 0:
                raise RuntimeError(f"Piper binary failed (code {process.returncode}): {stderr}")
            
            if not Path(output_path).exists() or Path(output_path).stat().st_size == 0:
                 raise RuntimeError(f"Piper binary produced empty file: {stderr}")

            return output_path

        except Exception as e:
            logger.error(f"Piper generation failed: {e}")
            raise # Propagate to facade for fallback

    def _download_piper_model(self, model_name, dest_dir):
        """Downloads .onnx and .json from Hugging Face"""
        try:
            parts = model_name.split("-")
            lang_region, voice, quality = parts[0], parts[1], parts[2]
            lang_code = lang_region.split("_")[0]
            
            base_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/{lang_code}/{lang_region}/{voice}/{quality}/{model_name}"
            
            for ext in [".onnx", ".onnx.json"]:
                url = base_url + ext
                logger.info(f"Downloading {url}...")
                r = requests.get(url)
                r.raise_for_status()
                with open(dest_dir / (model_name + ext), "wb") as f:
                    f.write(r.content)
                    
        except Exception as e:
             raise RuntimeError(f"Could not download model {model_name}: {e}")

    def _download_piper_binary(self, dest_dir, system):
        """Downloads Piper binary for the current platform."""
        if system == "Windows":
            url = "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip"
        else:
            # Linux (Docker)
            url = "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz"

        logger.info(f"Downloading Piper binary from {url}...")
        try:
            r = requests.get(url)
            r.raise_for_status()
            
            import tarfile
            file_obj = io.BytesIO(r.content)
            
            if url.endswith(".zip"):
                 with zipfile.ZipFile(file_obj) as z:
                    z.extractall(dest_dir)
            elif url.endswith(".tar.gz"):
                 with tarfile.open(fileobj=file_obj, mode="r:gz") as t:
                    t.extractall(dest_dir)
                    
        except Exception as e:
            raise RuntimeError(f"Failed to download Piper binary: {e}")
