@echo off
setlocal

echo ========================================================
echo Building ONNX Runtime with 5D GridSample Support (Option B)
echo ========================================================

REM Check prerequisites
git --version >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed or not in PATH.
    pause
    exit /b 1
)

cmake --version >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] CMake is not installed or not in PATH.
    pause
    exit /b 1
)

nvcc --version >nul 2>nul
if %errorlevel% equ 0 goto :found_nvcc

echo [WARNING] NVCC not found in PATH. Checking common locations...

if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe" (
    set "CUDA_BIN_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
    goto :add_cuda
)
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\nvcc.exe" (
    set "CUDA_BIN_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
    goto :add_cuda
)
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin\nvcc.exe" (
    set "CUDA_BIN_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin"
    goto :add_cuda
)

echo [ERROR] NVCC (CUDA Compiler) is not installed or not in PATH.
echo Please install CUDA Toolkit (preferably 11.8 or 12.x).
pause
exit /b 1

:add_cuda
set "PATH=%PATH%;%CUDA_BIN_PATH%"
echo Found CUDA at %CUDA_BIN_PATH%. Added to PATH.

:found_nvcc
echo.
echo [1/4] Cloning ONNX Runtime (liqun/ImageDecoder-cuda branch)...
if not exist "onnxruntime_src" (
    git clone --recursive -b liqun/ImageDecoder-cuda https://github.com/microsoft/onnxruntime onnxruntime_src
) else (
    echo Repo already cloned. Skipping.
)

cd onnxruntime_src

echo.
echo [2/4] configuring Build Variables...
echo Please ensure you have Visual Studio Build Tools installed.
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    echo Initializing VS environment...
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
)

REM Set CUDA params - adjust these if your paths differ!
set "CUDA_VER=12.2"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VER%"
set "CUDA_PATH=%CUDA_HOME%"
set "CUDA_PATH_V12_2=%CUDA_HOME%"
set "CUDNN_HOME=I:\video-translator\cudnn_temp"

echo Target CUDA Version: %CUDA_VER%
echo CUDA Home: %CUDA_HOME%

if not exist "%CUDA_HOME%" (
    echo [WARNING] CUDA %CUDA_VER% path NOT found.
    set "CUDA_VER=12.2"
    set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2"
)
echo Final CUDA Home: %CUDA_HOME%

echo.
echo [3/4] Building from source (This will take a long time)...
set CL=/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
set NVCC_PREPEND_FLAGS=-allow-unsupported-compiler
call build.bat --config Release ^
    --build_shared_lib ^
    --parallel ^
    --use_cuda ^
    --cuda_version %CUDA_VER% ^
    --cuda_home "%CUDA_HOME%" ^
    --cudnn_home "%CUDNN_HOME%" ^
    --cmake_extra_defines CMAKE_POLICY_VERSION_MINIMUM=3.5 onnxruntime_ENABLE_CPUINFO=OFF "onnxruntime_CMAKE_CUDA_ARCHITECTURES=60;70;75;80;86;89;90;compute_90" ^
    --build_wheel ^
    --skip_submodule_sync ^
    --skip_tests

if %errorlevel% neq 0 (
    echo [ERROR] Build failed. Please check logs.
    cd ..
    pause
    exit /b 1
)

echo.
echo [4/4] Installing Python Wheel...
cd build\Windows\Release\Release\dist
for %%f in (*.whl) do (
    echo Installing %%f...
    pip install "%%f" --force-reinstall
)

cd ..\..\..\..\..
echo.
echo ========================================================
echo Build and Installation Complete!
echo You can now use LivePortrait with GPU acceleration.
echo ========================================================
pause
