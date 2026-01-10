# AI Video Translator (Local)

An advanced, locally-run video translation pipeline that separates vocals, transcribes speech, translates text, generates new speech, and synchronizes it back to the video while preserving the original background audio and video quality.

## 🚀 Key Features

*   **Vocal Separation**: Uses **HDemucs** (Meta's Hybrid Demucs) to cleanly separate speech from background music/sfx. Optimized with chunking to handle long videos on limited GPU memory.
*   **Precision Transcription**: Powered by **OpenAI Whisper** for state-of-the-art speech-to-text with accurate timestamps.
*   **Multi-Language Translation**: Uses **Google Translate** (via `deep-translator`) to robustly translate content between 16+ languages (English, Spanish, French, German, Japanese, Chinese, etc.).
*   **Neural TTS**: Integrated **Edge-TTS** (Microsoft Edge Online) for high-quality, natural-sounding speech generation without heavy local model weights.
*   **Smart Synchronization**: Automatically stretches or squeezes generated speech to match the original timing, with safeguards against extreme distortion.
*   **GPU Optimized**: Custom memory management includes aggressive model offloading and audio chunking to prevent CUDA Out-of-Memory errors.
*   **Friendly UI**: Easy-to-use **Gradio** web interface.

## 🛠️ Prerequisites

*   **Python 3.10+**
*   **FFmpeg**: Must be installed and accessible in your system's PATH.
    *   *Windows*: `winget install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/).
*   **NVIDIA GPU** (Recommended): For faster HDemucs and Whisper inference. CPU is supported but slower.

## 📦 Installation

1.  **Clone the repository** (or extract files):
    ```bash
    cd video-translator
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Additional Requirements**:
    If not already installed via requirements, ensure you have:
    ```bash
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118  # Adjust for your CUDA version
    pip install openai-whisper deep-translator edge-tts gradio soundfile ffmpeg-python
    ```

## 🖥️ Usage

1.  **Run the Application**:
    ```bash
    python app.py
    ```

2.  **Open Interface**:
    Click the local URL provided in the console (usually `http://127.0.0.1:7860`).

3.  **Translate a Video**:
    *   **Upload Video**: Select an MP4/MKV/MOV file.
    *   **Select Target Language**: Choose from the dropdown (e.g., Spanish, Japanese).
    *   **Select Audio Model**: Keep default "Torchaudio HDemucs".
    *   **Click Submit**: The progress bar will track the stages (Separating -> Transcribing -> Translating -> Synthesizing -> Mixing).

4.  **Output**:
    The translated video will appear in the output component for download/playback. Files are saved to `output/`.

## ⚙️ Configuration

*   **Directory Structure**:
    *   `temp/`: Stores intermediate files (vocals, separated tracks). Cleared/Managed during runs.
    *   `output/`: Stores final processed videos.
*   **Env Variables** (Optional):
    *   `HF_TOKEN`: HuggingFace token if you switch to using gated models (not required for current default pipeline).

## 🧩 Pipeline Architecture

1.  **Extract**: FFmpeg extracts audio from input video.
2.  **Separate**: HDemucs splits audio into `vocals` and `accompaniment`.
3.  **Transcribe**: Whisper converts `vocals` to text segments with start/end times.
4.  **Translate**: Segments are translated text-to-text.
5.  **Synthesize (TTS)**: Edge-TTS generates speech for each translated segment.
6.  **Synchronize**: generated clips are time-stretched to fit original segment duration.
7.  **Mix**: New vocals are mixed with original `accompaniment`.
8.  **Merge**: Final mix is replaced into the original video container.

## 🔧 Troubleshooting

*   **"TorchCodec is required"**: This project uses `soundfile` backend to avoid this error. If you see it, ensure `soundfile` is installed.
*   **CUDA Out of Memory**: The system uses chunking (10s segments) for separation. If you still OOM, try closing other GPU-heavy apps.
*   **Video not playing in browser**: Gradio/Browser compatibility. Use VLC or Media Player for the downloaded file.

## 📄 License

[MIT License](LICENSE) (or appropriate license for your project).
