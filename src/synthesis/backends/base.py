from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Callable

class TTSBackend(ABC):
    """
    Abstract Base Class for TTS backends.
    """
    
    @abstractmethod
    def generate(self, 
                 text: str, 
                 output_path: str | Path,
                 language: str = "en", 
                 speaker_wav: str | Path | None = None,
                 **kwargs) -> str | Path | None:
        """
        Generates audio for the given text.
        
        Args:
           text: Text to synthesize
           output_path: Path to write the output audio
           language: Target language code
           speaker_wav: Path to reference audio (for cloning)
           **kwargs: Additional model-specific parameters
           
        Returns:
            Path to the generated file, or None if failed.
        """
        pass

    def generate_batch(self, tasks: list) -> list:
        """
        Generates audio for a batch of tasks.
        Default implementation: Sequential loop.
        
        Args:
            tasks: List of dicts, each containing args for generate()
                   e.g. [{'text': '...', 'output_path': '...', ...}]
                   
        Returns:
            List of (output_path or None) in same order.
        """
        results = []
        for task in tasks:
            try:
                res = self.generate(**task)
                results.append(res)
            except Exception as e:
                # Log but don't break whole batch? Or depend on generate handling?
                # generate usually raises specific errors.
                # We'll stick to generate's behavior (which propagates exceptions to facade)
                # But for batching, maybe we want to continue?
                # Let's let exceptions bubble up for now in default impl.
                raise e
        return results

    def load_model(self):
        pass

    def unload_model(self):
        pass
