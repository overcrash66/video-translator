import pytest
from unittest.mock import MagicMock, patch, ANY
from src.synthesis.backends.vibevoice_tts import VibeVoiceBackend, VibeVoiceWrapper

@pytest.fixture
def mock_vibevoice_package():
    mock_vibevoice = MagicMock()
    mock_modular = MagicMock()
    mock_inference = MagicMock()
    mock_processor_pkg = MagicMock()
    mock_processor_cls = MagicMock()

    modules = {
        'vibevoice': mock_vibevoice,
        'vibevoice.modular': mock_modular,
        'vibevoice.modular.modeling_vibevoice_inference': mock_inference,
        'vibevoice.processor': mock_processor_pkg,
        'vibevoice.processor.vibevoice_processor': mock_processor_cls,
    }
    
    with patch.dict('sys.modules', modules):
        yield

@pytest.fixture
def backend(mock_vibevoice_package):
    return VibeVoiceBackend(model_version="vibevoice")

def test_initialization(backend):
    assert backend.model_version == "vibevoice"
    assert backend.wrapper is None

def test_backend_initialization_7b(mock_vibevoice_package):
    backend = VibeVoiceBackend(model_version="vibevoice-7b")
    assert backend.model_version == "vibevoice-7b"

@patch('src.synthesis.vibevoice_tts.VibeVoiceWrapper.load_model')
def test_load_model(mock_load, backend):
    backend.load_model()
    assert backend.wrapper is not None
    assert backend.wrapper.model_name == "vibevoice"
    mock_load.assert_called_once()
    
    # Second call should be no-op regarding wrapper creation
    wrapper_instance = backend.wrapper
    backend.load_model()
    assert backend.wrapper is wrapper_instance
    assert mock_load.call_count == 2 # Wrapper.load_model is called again, which has its own check

@patch('src.synthesis.vibevoice_tts.VibeVoiceWrapper.unload_model')
def test_unload_model(mock_unload, backend):
    # Setup
    backend.load_model()
    assert backend.wrapper is not None
    
    backend.unload_model()
    mock_unload.assert_called_once()

@patch('src.synthesis.vibevoice_tts.VibeVoiceWrapper.generate_speech')
def test_generate_delegation(mock_generate, backend):
    backend.generate("Hello world", "output.wav", language="en", speaker_id="Alice")
    
    mock_generate.assert_called_once_with(
        text="Hello world",
        output_path="output.wav",
        language="en",
        speaker_name="Alice"
    )

@patch('src.synthesis.vibevoice_tts.VibeVoiceWrapper.generate_speech')
def test_generate_default_speaker(mock_generate, backend):
    # If no speaker_id provided, should default
    backend.generate("Hello world", "output.wav")
    
    mock_generate.assert_called_once()
    args, kwargs = mock_generate.call_args
    assert kwargs['speaker_name'] == "Alice" # Default we set in backend

@patch('src.synthesis.vibevoice_tts.VibeVoiceWrapper.generate_speech')
def test_generate_intializes_wrapper(mock_generate, backend):
    assert backend.wrapper is None
    backend.generate("test", "out.wav")
    assert backend.wrapper is not None
