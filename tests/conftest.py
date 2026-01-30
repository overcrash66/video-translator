import sys
try:
    import ctranslate2
    print("DEBUG: ctranslate2 pre-imported in tests/conftest.py")
except ImportError:
    pass
