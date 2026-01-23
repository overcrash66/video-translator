"""Tests for config.py path validation."""
import pytest
from pathlib import Path
from src.utils.config import validate_path
import os

def test_validate_path_exists(tmp_path):
    # Create a temporary file
    p = tmp_path / "hello.txt"
    p.write_text("content")
    
    result = validate_path(str(p), must_exist=True)
    assert result.exists()
    assert result.resolve() == p.resolve()

def test_validate_path_not_exists_raises():
    with pytest.raises(FileNotFoundError):
        validate_path("I:/nonexistent/file.py", must_exist=True)

def test_validate_path_resolves():
    # Relative path
    result = validate_path("context/../app.py", must_exist=False)
    assert result.is_absolute()
    
def test_validate_path_allowed_dirs(tmp_path):
    # Allowed directory
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    
    # File inside allowed
    p = allowed_dir / "file.txt"
    p.touch()
    
    result = validate_path(p, allowed_dirs=[allowed_dir])
    assert result == p.resolve()

def test_validate_path_outside_allowed_raises(tmp_path):
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    
    forbidden_dir = tmp_path / "forbidden"
    forbidden_dir.mkdir()
    p = forbidden_dir / "secret.txt"
    
    with pytest.raises(ValueError):
        validate_path(p, allowed_dirs=[allowed_dir])
