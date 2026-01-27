# Scripts/setup_tensorrt.ps1
# Setup script for LivePortrait TensorRT Acceleration

Write-Host "Setting up TensorRT Environment for LivePortrait..." -ForegroundColor Cyan

# 1. Install tensorrt python package
Write-Host "Step 1: Installing 'tensorrt' python package..." -ForegroundColor Yellow
pip install tensorrt
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install tensorrt. Please check your pip configuration." -ForegroundColor Red
    exit 1
}
Write-Host "TensorRT package installed." -ForegroundColor Green

# 2. Instructions for Plugin Compilation
Write-Host "`nStep 2: Building 'grid-sample3d-trt-plugin'..." -ForegroundColor Yellow
Write-Host "This step requires: Git, CMake, Visual Studio Build Tools (C++), AND TensorRT SDK (Headers)." -ForegroundColor Gray

$pluginRepo = "https://github.com/SeanWangJS/grid-sample3d-trt-plugin"
$pluginDir = "models/grid-sample3d-trt-plugin"

if (-not (Test-Path $pluginDir)) {
    Write-Host "Cloning plugin repo to $pluginDir..."
    git clone $pluginRepo $pluginDir
} else {
    Write-Host "Plugin repo already exists at $pluginDir."
}

# Check for TensorRT SDK Headers (NvInfer.h)
$trtRoot = $env:TensorRT_ROOT
if (-not $trtRoot -or -not (Test-Path "$trtRoot/include/NvInfer.h")) {
    Write-Host "`n[IMPORTANT] TensorRT Headers (NvInfer.h) not found in TensorRT_ROOT." -ForegroundColor Red
    Write-Host "The 'tensorrt' pip package usually does NOT contain headers required for compilation."
    Write-Host "Please download the TensorRT SDK (Zip/Installer) from NVIDIA:"
    Write-Host "  https://developer.nvidia.com/tensorrt"
    Write-Host "Extract it, and set TensorRT_ROOT environment variable, OR enter the path below."
    
    $trtInput = Read-Host "`nEnter path to TensorRT SDK (e.g. C:\TensorRT-10.x.x.x):"
    if ($trtInput -and (Test-Path "$trtInput/include/NvInfer.h")) {
        $trtRoot = $trtInput
        $env:TensorRT_ROOT = $trtInput # Set for session
    } else {
        Write-Host "Invalid path or headers not found. Cannot compile plugin." -ForegroundColor Red
        Write-Host "Please setup TensorRT SDK and run this script again."
        exit 1
    }
}
Write-Host "Found TensorRT headers at: $trtRoot/include" -ForegroundColor Green

Write-Host "`nTo compile the plugin, please run the following commands manually (or allow this script to try):" -ForegroundColor Cyan
Write-Host "  cd $pluginDir"
Write-Host "  mkdir build"
Write-Host "  cd build"
Write-Host "  cmake .. -DCMAKE_BUILD_TYPE=Release -DTensorRT_ROOT='$trtRoot'"
Write-Host "  cmake --build . --config Release"
Write-Host "`nNote: Ensure 'nvcc' (CUDA Toolkit) is in your PATH."

$response = Read-Host "Do you want to attempt compilation now? (y/n)"
if ($response -eq 'y') {
    Push-Location $pluginDir
    if (-not (Test-Path "build")) { mkdir build }
    Set-Location build
    
    try {
        cmake .. -DCMAKE_BUILD_TYPE=Release "-DTensorRT_ROOT=$trtRoot"
        if ($LASTEXITCODE -eq 0) {
            cmake --build . --config Release
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Plugin compiled successfully!" -ForegroundColor Green
                # Copy DLL to model dir for easy access
                $dllSource = "Release/grid_sample_3d_plugin.dll"
                if (Test-Path $dllSource) {
                    Copy-Item $dllSource -Destination "../../live_portrait_onnx/" -Force
                    Write-Host "Copied plugin DLL to models/live_portrait_onnx/" -ForegroundColor Green
                }
            } else {
                Write-Host "Build failed." -ForegroundColor Red
            }
        } else {
            Write-Host "CMake configuration failed. Check prerequisites." -ForegroundColor Red
        }
    } catch {
        Write-Host "An error occurred during compilation: $_" -ForegroundColor Red
    } finally {
        Pop-Location
    }
} else {
    Write-Host "Skipping compilation. Please compile manually later." -ForegroundColor Gray
}

Write-Host "`nTensorRT setup process finished." -ForegroundColor Cyan
