# Language Code Mapping
# Maps display names or variations to standardized ISO codes (mostly)
LANGUAGE_CODE_MAP = {
    "Auto Detect": "auto",
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Polish": "pl",
    "Turkish": "tr",
    "Russian": "ru", 
    "Dutch": "nl", 
    "Czech": "cs", 
    "Arabic": "ar",
    "Chinese (Simplified)": "zh", 
    "Chinese": "zh", # Fallback
    "Japanese": "ja", 
    "Korean": "ko",
    "Hindi": "hi"
}

def get_language_code(name: str) -> str:
    """Returns the internal language code for a given display name."""
    # Check if it's already a valid code
    if name in LANGUAGE_CODE_MAP.values():
        return name
    return LANGUAGE_CODE_MAP.get(name, "en")

def get_language_name(code: str) -> str:

    """Returns the display name for a given language code (reverse lookup)."""
    for name, lang_code in LANGUAGE_CODE_MAP.items():
        if lang_code == code:
            return name
    return "English" # Default fallback


# Edge-TTS Voice Mapping
# Structure: { lang_code: { "Female": [...], "Male": [...] } }
EDGE_TTS_VOICE_MAP = {
    "en": {
        "Female": ["en-US-AriaNeural", "en-US-JennyNeural", "en-GB-SoniaNeural"],
        "Male": ["en-US-GuyNeural", "en-US-ChristopherNeural", "en-GB-RyanNeural"]
    },
    "es": {
        "Female": ["es-ES-ElviraNeural", "es-MX-DaliaNeural"],
        "Male": ["es-ES-AlvaroNeural", "es-MX-JorgeNeural"]
    },
    "fr": {
        "Female": ["fr-FR-DeniseNeural", "fr-CA-SylvieNeural"],
        "Male": ["fr-FR-HenriNeural", "fr-CA-JeanNeural"]
    },
    "de": {
        "Female": ["de-DE-KatjaNeural", "de-AT-IngridNeural"],
        "Male": ["de-DE-ConradNeural", "de-AT-JonasNeural"]
    },
    "it": {
        "Female": ["it-IT-ElsaNeural", "it-IT-IsabellaNeural"],
        "Male": ["it-IT-DiegoNeural", "it-IT-GiuseppeNeural"]
    },
    "pt": {
        "Female": ["pt-BR-FranciscaNeural", "pt-PT-RaquelNeural"],
        "Male": ["pt-BR-AntonioNeural", "pt-PT-DuarteNeural"]
    },
    "pl": {
        "Female": ["pl-PL-ZofiaNeural", "pl-PL-AgnieszkaNeural"],
        "Male": ["pl-PL-MarekNeural"]
    },
    "tr": {
        "Female": ["tr-TR-EmelNeural"],
        "Male": ["tr-TR-AhmetNeural"]
    },
    "ru": {
        "Female": ["ru-RU-SvetlanaNeural", "ru-RU-DariyaNeural"],
        "Male": ["ru-RU-DmitryNeural"]
    },
    "nl": {
        "Female": ["nl-NL-ColetteNeural", "nl-NL-FennaNeural"],
        "Male": ["nl-NL-MaartenNeural"]
    },
    "cs": {
        "Female": ["cs-CZ-VlastaNeural"],
        "Male": ["cs-CZ-AntoninNeural"]
    },
    "ar": {
        "Female": ["ar-SA-ZariyahNeural", "ar-EG-SalmaNeural"],
        "Male": ["ar-SA-HamedNeural", "ar-EG-ShakirNeural"]
    },
    "zh": {
        "Female": ["zh-CN-XiaoxiaoNeural", "zh-CN-XiaoyiNeural", "zh-CN-XiaochenNeural"],
        "Male": ["zh-CN-YunxiNeural", "zh-CN-YunjianNeural", "zh-CN-YunyeNeural"]
    },
    "ja": {
        "Female": ["ja-JP-NanamiNeural", "ja-JP-AoiNeural"],
        "Male": ["ja-JP-KeitaNeural", "ja-JP-DaichiNeural"]
    },
    "ko": {
        "Female": ["ko-KR-SunHiNeural", "ko-KR-JiMinNeural"],
        "Male": ["ko-KR-InJoonNeural", "ko-KR-BongJinNeural"]
    },
    "hi": {
        "Female": ["hi-IN-SwaraNeural"],
        "Male": ["hi-IN-MadhurNeural"]
    }
}

# Piper TTS Model Mapping
# Maps language code to default model name
PIPER_MODEL_MAP = {
    "en": "en_US-lessac-high",
    "es": "es_ES-sharvard-medium",
    "fr": "fr_FR-siwis-medium",
    "de": "de_DE-thorsten-medium",
    "it": "it_IT-riccardo-x_low", 
}
