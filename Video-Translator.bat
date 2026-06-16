@echo off
set CURRENT_DIR=%CD%
echo ***** Current directory: %CURRENT_DIR% *****
set PYTHONPATH=%CURRENT_DIR%

rem Use venv Python directly (avoids PATH resolution issues)
set VENV_PYTHON=%CURRENT_DIR%\venv\Scripts\python.exe
if not exist "%VENV_PYTHON%" (
    echo ERROR: venv Python not found at %VENV_PYTHON%
    pause
    exit /b 1
)

"%VENV_PYTHON%" app.py
pause