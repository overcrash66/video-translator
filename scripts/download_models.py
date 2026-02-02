import sys
import os
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelDownloader")

MODELS = {
    "wav2lip_gan.pth": {
        "urls": [
            "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/Wav2Lip/wav2lip_gan.pth",
            "https://huggingface.co/rippertnt/wav2lip/resolve/main/checkpoints/wav2lip_gan.pth",
            "https://huggingface.co/vbalnt/wav2lip-hq/resolve/main/wav2lip_gan.pth"
        ],
        "dest": "models/wav2lip/wav2lip_gan.pth"
    },
    "wav2lip.pth": {
        "urls": [
            "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/Wav2Lip/wav2lip.pth",
            "https://huggingface.co/rippertnt/wav2lip/resolve/main/checkpoints/wav2lip.pth"
        ],
        "dest": "models/wav2lip/wav2lip.pth"
    },
    "GFPGANv1.4.pth": {
        "urls": [
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
        ],
        "dest": "models/gfpgan/GFPGANv1.4.pth"
    }
}

def download_file(urls, dest_path):
    if dest_path.exists():
        logger.info(f"File already exists: {dest_path}")
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    last_error = None
    for url in urls:
        logger.info(f"Attempting download from: {url}")
        try:
            with requests.get(url, stream=True) as r:
                # 1. Handle Permission/Auth Errors -> Skip gracefully but try next mirror
                if r.status_code in [401, 403]:
                    logger.warning(f"⚠️ SKIPPING {dest_path.name}: Permission/Auth failed (HTTP {r.status_code}).")
                    last_error = f"HTTP {r.status_code}"
                    continue

                r.raise_for_status()
                
                with open(dest_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            logger.info(f"✅ Downloaded: {dest_path}")
            return # Success, stop trying mirrors

        except Exception as e:
            logger.warning(f"❌ Failed mirror {url}: {e}")
            last_error = e
            # Cleanup partial file
            if dest_path.exists():
                try: os.remove(dest_path)
                except: pass
            continue # Try next mirror

    # If we get here, ALL mirrors failed
    logger.critical(f"❌ ALL MIRRORS FAILED for {dest_path.name}. Last error: {last_error}")
    sys.exit(1)

def main():
    base_dir = Path(__file__).parent.parent
    logger.info(f"Base Directory: {base_dir}")
    
    for name, info in MODELS.items():
        # Handle relative path from project root
        dest = base_dir / info["dest"]
        download_file(info["urls"], dest)

if __name__ == "__main__":
    main()
