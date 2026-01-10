# Project Plan: Local AI Video Dubbing & Translation Tool

## 1. Project Overview
**Goal**: Build a locally hosted web tool that takes a video input, isolates the speech from the background, translates the speech to a target language, and re-dubs the video.

**Critical Requirements**:
- **Local Execution**: Must run entirely on the user's machine (Python/GPU).
- **Audio Separation**: Use `facebook/sam-audio-large` to isolate speech from background noise.
- **Voice Cloning**: The new translated audio should sound similar to the original speaker (if possible) or use a high-quality TTS.
- **Synchronization**: The new audio must match the timing of the original speech segments to avoid desynchronization.

## 2. Tech Stack & Libraries
- **Language**: Python 3.10+
- **UI Framework**: Gradio (for the web interface).
- **Audio Separation**: `facebook/sam-audio-large` (via Hugging Face Transformers).
- **Transcription & Timestamping**: OpenAI Whisper (Model: `large-v3`).
- **Translation**: DeepL (API) or NLLB-200 (Local model).
- **Text-to-Speech (TTS)**: Coqui TTS (`XTTS-v2`) for voice cloning and multilingual support.
- **Audio/Video Processing**: FFmpeg (via `ffmpeg-python` or subprocess).
- **Audio Time-Stretching**: Rubberband or FFmpeg `atempo` filter (for synchronization).

## 3. Architecture Pipeline
1. **Input**: User uploads `video.mp4`.
2. **Extraction**: FFmpeg extracts `full_audio.wav`.
3. **Separation (SAM-Audio)**:
   - **Input**: `full_audio.wav` + Text Prompt: "Speech" or "Human voice".
   - **Output 1**: `vocals.wav` (Speech).
   - **Output 2**: `background.wav` (Residuals).
4. **Transcription (Whisper)**:
   - **Input**: `vocals.wav`.
   - **Output**: List of segments `[{start, end, text}, ...]`.
5. **Translation & Dubbing (Loop per segment)**:
   - Translate text -> Generate Audio (TTS) -> Check Duration.
   - **Sync Logic**: Stretch/Squeeze new audio to match `(end - start)` of original segment.
6. **Merging**:
   - Overlay synced translated audio onto `background.wav`.
   - Mux combined audio back into video.

## 4. Step-by-Step Implementation Plan

### Phase 1: Environment & Basic UI
**Task**: Set up the Python environment and create a simple Gradio interface.
* **Requirements**:
  - Install `gradio`, `torch`, `torchaudio`, `transformers`.
  - Create `app.py`.
  - **UI Inputs**: Video File, Target Language Dropdown.
  - **UI Outputs**: Processed Video File.
  - **Verification**: Ensure UI loads and can accept/return a video file.

### Phase 2: Audio Separation (The SAM-Audio Integration)
**Task**: Implement the isolation of speech using the requested model.
* **Requirements**:
  - Load `facebook/sam-audio-large`. **Note**: This requires a Hugging Face Access Token.
  - Implement a function `separate_audio(audio_path)`.
  - **Prompting Strategy**: Use the text prompt "speech" or "person speaking" to generate the mask.
  - Save two files: `vocals.wav` and `accompaniment.wav`.
  - **Fallback**: If `sam-audio-large` fails on long files, implement a fallback using Demucs (standard music separation model).

### Phase 3: Transcription & Translation
**Task**: Get text and precise timestamps.
* **Requirements**:
  - Use Whisper (`Large-v3`) on `vocals.wav`.
  - **Crucial**: We need `word_timestamps=True` or robust segment timestamps.
  - Implement a translation function (e.g., using `googletrans` library for prototype or NLLB for local high-quality).
  - Store data structure: `[ {original_text, translated_text, start_time, end_time, duration} ]`.

### Phase 4: TTS & Synchronization Engine (Critical)
**Task**: Generate audio and force it to fit the time slot.
* **Requirements**:
  - Use Coqui `XTTS-v2`. Use `vocals.wav` as the reference for Voice Cloning.
  - **The Sync Algorithm**:
    1. Calculate `target_duration = segment.end - segment.start`.
    2. Generate TTS audio -> `generated_duration`.
    3. Calculate `speed_factor = generated_duration / target_duration`.
    4. If `speed_factor != 1.0`: Apply FFmpeg `atempo` filter to stretch/squeeze audio to fit `target_duration`.
  - Concatenate all synced segments into a single `translated_track.wav`, inserting silence between segments where necessary.

### Phase 5: Final Assembly
**Task**: Mix audio and render video.
* **Requirements**:
  - Mix `translated_track.wav` (volume 1.0) + `accompaniment.wav` (volume 0.8).
  - Use FFmpeg to replace the video's audio track with this new mix.
  - Display the result in the Gradio UI.

## 5. Instructions for the AI Coder
- **Handling sam-audio-large**: This model is gated. Ensure the code allows the user to input their `HF_TOKEN` or looks for it in environment variables.
- **Sync Logic**: Do not simply overlay the audio. You must calculate the duration of the original sentence and time-stretch the new audio. If the new audio is too fast (>1.5x), warn the user or accept the slight quality drop.
- **File Management**: Use a `temp/` directory to store the intermediate `.wav` chunks to avoid memory overflow. Clean up after processing.
- **GPU Usage**: Ensure all models (Whisper, SAM, TTS) move to `cuda` if available.