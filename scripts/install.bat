@echo off
REM =============================================================================
REM Video Translator — Windows Installer
REM =============================================================================
REM Usage:
REM   scripts\install.bat              -- Auto-detect
REM   scripts\install.bat --cpu        -- CPU-only
REM   scripts\install.bat --cuda 12.8  -- Force CUDA 12.8 (RTX 50 series)
REM =============================================================================

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set VENV_DIR=%PROJECT_ROOT%\venv
set PYTHON_EXE=%VENV_DIR%\Scripts\python.exe

echo === Video Translator Installer ===
echo Project root: %PROJECT_ROOT%

REM Check if venv exists
if not exist "%VENV_DIR%" (
    echo Creating virtual environment...
    
    REM Try py -3.10 first, then python
    where py >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        py -3.10 -m venv "%VENV_DIR%"
        if %ERRORLEVEL% neq 0 (
            echo Python 3.10 not found, trying default python...
            python -m venv "%VENV_DIR%"
        )
    ) else (
        python -m venv "%VENV_DIR%"
    )
    
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to create virtual environment!
        echo Make sure Python 3.10+ is installed.
        pause
        exit /b 1
    )
)

REM Run the installer
echo Running installer...
"%PYTHON_EXE%" "%SCRIPT_DIR%install.py" %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo Installation completed with warnings. Check output above.
)

pause
