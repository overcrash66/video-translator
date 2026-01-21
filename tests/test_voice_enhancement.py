import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import os

# Mock voicefixer before importing the module under test
sys.modules['voicefixer'] = MagicMock()

from src.processing.voice_enhancement import VoiceEnhancer

class TestVoiceEnhancement(unittest.TestCase):
    def setUp(self):
        self.enhancer = VoiceEnhancer()

    @patch('src.processing.voice_enhancement.VoiceFixer')
    def test_initialization(self, MockVoiceFixer):
        """Test that VoiceFixer is not loaded until requested."""
        self.assertIsNone(self.enhancer.model)
        
        self.enhancer.load_model()
        MockVoiceFixer.assert_called_once()
        self.assertIsNotNone(self.enhancer.model)

    @patch('src.processing.voice_enhancement.VoiceFixer')
    def test_enhance_audio(self, MockVoiceFixer):
        """Test the enhance_audio method calls the underlying model correctly."""
        # Setup mock
        mock_model_instance = MagicMock()
        MockVoiceFixer.return_value = mock_model_instance
        
        # Create dummy input file
        input_path = Path("test_input.wav")
        output_path = Path("test_output.wav")
        # Ensure input exists
        with open(input_path, 'w') as f:
            f.write("dummy audio")
            
        try:
            # We mock the restore method to create the output file
            def side_effect(input, output, cuda, mode):
                with open(output, 'w') as f:
                    f.write("enhanced audio")
            
            mock_model_instance.restore.side_effect = side_effect
            
            self.enhancer.enhance_audio(input_path, output_path, cuda=False)
            
            # Update: verification
            mock_model_instance.restore.assert_called_once_with(
                input=str(input_path), 
                output=str(output_path), 
                cuda=False, 
                mode=0
            )
            self.assertTrue(output_path.exists())
            
        finally:
            # Cleanup
            if input_path.exists():
                os.remove(input_path)
            if output_path.exists():
                os.remove(output_path)

    def test_unload_model(self):
        """Test that unload_model works."""
        # Fake a loaded model
        self.enhancer.model = MagicMock()
        self.enhancer.unload_model()
        self.assertIsNone(self.enhancer.model)

if __name__ == '__main__':
    unittest.main()
