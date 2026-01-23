from pathlib import Path
from typing import Optional, Dict, Set

class SessionContext:
    """
    Encapsulates state for a single video translation session.
    Tracks speaker-to-voice assignments and fallback references.
    """
    def __init__(self):
        self.speaker_voice_map: Dict[str, str] = {}
        self.used_voices: Set[str] = set()
        self.last_valid_reference_wav: Optional[str] = None
        self.profiles_dir: Optional[Path] = None

    def get_voice(self, speaker_id: str) -> Optional[str]:
        return self.speaker_voice_map.get(speaker_id)

    def assign_voice(self, speaker_id: str, voice: str):
        self.speaker_voice_map[speaker_id] = voice
        self.used_voices.add(voice)

    def is_voice_used(self, voice: str) -> bool:
        return voice in self.used_voices
    
    def reset(self):
        self.speaker_voice_map.clear()
        self.used_voices.clear()
        self.last_valid_reference_wav = None
        self.profiles_dir = None
