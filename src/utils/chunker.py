import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import List
import soundfile as sf
import ffmpeg

from src.utils import config

logger = logging.getLogger(__name__)

class VideoChunker:
    """
    Handles splitting long videos/audio into chunks and merging them back.
    This prevents OOM errors and timeouts with very long inputs.
    """
    
    def __init__(self, max_duration_sec: int = config.CHUNK_DURATION, overlap_sec: int = config.CHUNK_OVERLAP):
        self.max_duration = max_duration_sec
        self.overlap = overlap_sec

    def should_chunk(self, file_path: Path) -> bool:
        """
        Determines if a file needs chunking based on duration.
        """
        try:
            probe = ffmpeg.probe(str(file_path))
            duration = float(probe['format']['duration'])
            return duration > (self.max_duration * 1.5) # Only chunk if significantly longer
        except Exception as e:
            logger.warning(f"Failed to probe file duration for chunking check: {e}")
            return False

    def split_video(self, video_path: Path) -> List[Path]:
        """
        Splits a video file into chunks using FFmpeg segment muxer.
        """
        output_pattern = config.TEMP_DIR / f"{video_path.stem}_chunk_%03d{video_path.suffix}"
        
        logger.info(f"Splitting video {video_path.name} into {self.max_duration}s chunks...")
        
        try:
            # Use segment muxer for fast, copy-based splitting (no re-encoding)
            # Note: overlap is tricky with segment muxer copy. 
            # Ideally we want exact times.
            # For simplicity in V1, we split strictly by time.
            # We add a small overlap if we were re-encoding, but 'copy' is safest for quality.
            # To handle overlap correctly for lip-sync, we might need to be smarter.
            # For now, let's stick to strict segmentation to avoid sync drift.
            
            (
                ffmpeg
                .input(str(video_path))
                .output(
                    str(output_pattern), 
                    c='copy', 
                    f='segment', 
                    segment_time=self.max_duration,
                    reset_timestamps=1 
                )
                .run(cmd='ffmpeg', capture_stdout=True, capture_stderr=True)
            )
            
            # Collect generated files
            chunks = sorted(config.TEMP_DIR.glob(f"{video_path.stem}_chunk_*{video_path.suffix}"))
            logger.info(f"Created {len(chunks)} video chunks.")
            return chunks
            
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg splitting failed: {e.stderr.decode() if e.stderr else str(e)}")
            raise

    def split_audio(self, audio_path: Path, video_chunks: List[Path]) -> List[Path]:
        """
        Splits an audio file to match the durations of video chunks.
        This ensures audio processing aligns with video chunks.
        """
        # We need to know exact duration of each video chunk to match audio
        chunk_durations = []
        for v in video_chunks:
            probe = ffmpeg.probe(str(v))
            chunk_durations.append(float(probe['format']['duration']))
            
        audio_chunks = []
        start_time = 0.0
        
        logger.info(f"Splitting audio {audio_path.name} to match {len(video_chunks)} video chunks...")
        
        for i, duration in enumerate(chunk_durations):
            out_name = config.TEMP_DIR / f"{audio_path.stem}_chunk_{i:03d}{audio_path.suffix}"
            
            try:
                # Extract segment
                # We used reset_timestamps=1 for video, so audio should be just cut
                (
                    ffmpeg
                    .input(str(audio_path), ss=start_time, t=duration)
                    .output(str(out_name), acodec='pcm_s16le') # Force wav/pcm for safety
                    .overwrite_output()
                    .run(cmd='ffmpeg', capture_stdout=True, capture_stderr=True)
                )
                audio_chunks.append(out_name)
                start_time += duration
                
            except ffmpeg.Error as e:
                logger.error(f"Audio chunking failed for segment {i}: {e.stderr.decode()}")
                raise
                
        return audio_chunks

    def merge_videos(self, chunk_paths: List[Path], output_path: Path) -> Path:
        """
        Merges video chunks using FFmpeg concat demuxer.
        """
        logger.info(f"Merging {len(chunk_paths)} video chunks into {output_path.name}...")
        
        # Create input file list
        list_file = config.TEMP_DIR / "merge_list.txt"
        with open(list_file, 'w', encoding='utf-8') as f:
            for p in chunk_paths:
                # FFmpeg concat requires forward slashes and escaped paths
                safe_path = str(p.resolve()).replace('\\', '/')
                f.write(f"file '{safe_path}'\n")
        
        try:
            (
                ffmpeg
                .input(str(list_file), format='concat', safe=0)
                .output(str(output_path), c='copy')
                .overwrite_output()
                .run(cmd='ffmpeg', capture_stdout=True, capture_stderr=True)
            )
            
            # Clean up list file
            if list_file.exists():
                list_file.unlink()
                
            return output_path
            
        except ffmpeg.Error as e:
            logger.error(f"Video merging failed: {e.stderr.decode()}")
            raise

    def merge_audio(self, chunk_paths: List[Path], output_path: Path) -> Path:
        """
        Merges audio chunks. 
        For V1 simple concatenation (since we did strict cuts).
        V2 could add crossfades if we switch to overlapping chunks.
        """
        logger.info(f"Merging {len(chunk_paths)} audio chunks...")
        
        # Similar to video merge, concat demuxer is safest/fastest for WAV
        list_file = config.TEMP_DIR / "audio_merge_list.txt"
        with open(list_file, 'w', encoding='utf-8') as f:
            for p in chunk_paths:
                safe_path = str(p.resolve()).replace('\\', '/')
                f.write(f"file '{safe_path}'\n")
                
        try:
            (
                ffmpeg
                .input(str(list_file), format='concat', safe=0)
                .output(str(output_path), c='copy')
                .overwrite_output()
                .run(cmd='ffmpeg', capture_stdout=True, capture_stderr=True)
            )
             # Clean up
            if list_file.exists():
                list_file.unlink()
                
            return output_path
            
        except ffmpeg.Error as e:
            logger.error(f"Audio merging failed: {e.stderr.decode()}")
            raise
