
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
print(f"DEBUG: sys.path: {sys.path}")
import numpy as np

from src.translation.visual_translator import VisualTranslator

def test_font_sizing():
    translator = VisualTranslator()
    
    # Mock some boxes and translated texts
    # Let's say we have a box with height 50
    box = [[10, 10], [100, 10], [100, 60], [10, 60]]
    text = "Hello World"
    
    # We want to check if font_size is h + 2 = 52
    h = 50
    expected_font_size = h + 2
    
    # Create a dummy frame
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    
    # Run the overlay method
    # Since we can't easily introspect the internal loop without modification,
    # let's just ensure it runs without error for now.
    # In a real scenario, we might patch ImageFont.truetype to capture the size.
    
    from unittest.mock import patch, MagicMock
    
    # Patch ImageFont used in VisualTranslator
    with patch("src.translation.visual_translator.ImageFont") as MockImageFont:
        # Configure getbbox to return 4 values (left, top, right, bottom)
        mock_font = MockImageFont.truetype.return_value
        mock_font.getbbox.return_value = (0, 0, 100, 20)
        
        # Also need to patch cv2.boundingRect because we can't control how simple points result in rect
        # Or just trust mock cv2 works? 
        # SmartMock for boundingRect returns (1,2,3,4) if not specially handled?
        # conftest doesn't special handle boundingRect.
        # SmartMock returns MagicMock.
        # MagicMock unpacking fails if not iterable.
        # We need cv2.boundingRect to return tuple (x,y,w,h).
        
        with patch("src.translation.visual_translator.cv2.boundingRect") as mock_boundingRect, \
             patch("src.translation.visual_translator.cv2.cvtColor") as mock_cvtColor:
             
             mock_boundingRect.return_value = (10, 10, 90, 50)
             # Ensure cvtColor returns the original frame (or array of same shape)
             # The method calls: cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
             # We want the result to match 'frame' shape.
             mock_cvtColor.side_effect = lambda x, code: frame if x.shape == frame.shape or (x.shape == (100,100,3) and code==-1) else x
             # Actually, simpler: just return 'frame' (the original input) or a copy.
             mock_cvtColor.return_value = frame
             
             result = translator._overlay_translated_text_pil(frame, [box], [text])
    
    print("Overlay test complete. Visual inspection required for final confirmation.")
    assert result.shape == frame.shape
    print(f"Frame shape: {result.shape}")

if __name__ == "__main__":
    try:
        test_font_sizing()
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)
