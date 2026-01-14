import ffmpeg
import logging
from pathlib import Path
from src.utils import config
import torchaudio
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    def extract_audio(self, video_path, output_path=None):
        if not output_path:
            output_path = str(Path(video_path).with_suffix(".wav"))
        
        try:
            (
                ffmpeg
                .input(video_path)
                .output(output_path, ac=1, ar=24000) # Mono 24k for consistency with TTS
                .overwrite_output()
                .run(quiet=True)
            )
            return output_path
        except ffmpeg.Error as e:
            logger.error(f"Extract audio failed: {e.stderr.decode() if e.stderr else str(e)}")
            return None

    def mix_tracks(self, vocal_track, background_track, output_path, vocal_volume=1.0, bg_volume=0.8):
        """
        Mixes vocal and background tracks using torchaudio to ensure alignment and quality.
        """
        try:
            import soundfile as sf
            import numpy as np
            
            # Load vocals
            v_np, v_sr = sf.read(str(vocal_track))
            if v_np.ndim == 1:
                v_np = v_np[np.newaxis, :]
            else:
                v_np = v_np.T
            v_wav = torch.from_numpy(v_np).float()
            
            # Load background
            b_np, b_sr = sf.read(str(background_track))
            if b_np.ndim == 1:
                b_np = b_np[np.newaxis, :]
            else:
                b_np = b_np.T
            b_wav = torch.from_numpy(b_np).float()
            
            # Resample if mismatch
            if v_sr != b_sr:
                resampler = torchaudio.transforms.Resample(b_sr, v_sr)
                b_wav = resampler(b_wav)
                b_sr = v_sr
            
            # Ensure lengths match (pad or trim background to vocal track? or vice versa?)
            # Usually background is the reference duration (from original file).
            # But here vocal_track is the *new* generated track.
            # We should probably respect the VIDEO duration or the Background duration.
            
            max_len = max(v_wav.shape[1], b_wav.shape[1])
            
            # Pad vocals
            if v_wav.shape[1] < max_len:
                v_wav = torch.cat([v_wav, torch.zeros(v_wav.shape[0], max_len - v_wav.shape[1])], dim=1)
            # Pad background
            if b_wav.shape[1] < max_len:
                b_wav = torch.cat([b_wav, torch.zeros(b_wav.shape[0], max_len - b_wav.shape[1])], dim=1)
                
            # Truncate to min? No, usually extend.
            
            # Mix
            final_mix = (v_wav * vocal_volume) + (b_wav * bg_volume)
            
            # Normalize to avoid clipping
            max_val = torch.abs(final_mix).max()
            if max_val > 0.95:
                final_mix = final_mix / max_val * 0.95
                
            # Save using soundfile
            # final_mix is [Channels, Time], Soundfile wants [Time, Channels]
            final_mix_np = final_mix.detach().cpu().numpy().T
            sf.write(str(output_path), final_mix_np, v_sr)
            return output_path
        except Exception as e:
            logger.error(f"Mixing failed: {e}")
            return None

    def replace_audio(self, video_path, new_audio_path, output_video_path):
        try:
            video = ffmpeg.input(video_path)
            audio = ffmpeg.input(new_audio_path)
            
            # Attempt 1: Copy video stream (Fastest)
            try:
                (
                    ffmpeg
                    .output(video.video, audio, output_video_path, vcodec='copy', acodec='aac', strict='experimental')
                    .overwrite_output()
                    .run(quiet=True)
                )
            except ffmpeg.Error as e:
                logger.warning(f"Replace audio (copy) failed: {e.stderr.decode() if e.stderr else str(e)}. Retrying with re-encode...")
                # Attempt 2: Re-encode video (Compatible)
                (
                    ffmpeg
                    .output(video.video, audio, output_video_path, vcodec='libx264', acodec='aac', strict='experimental', preset='fast')
                    .overwrite_output()
                    .run(quiet=True)
                )

            return output_video_path
        except ffmpeg.Error as e:
             logger.error(f"Replace audio failed (both copy and re-encode): {e.stderr.decode() if e.stderr else str(e)}")
             return None

if __name__ == "__main__":
    pass
