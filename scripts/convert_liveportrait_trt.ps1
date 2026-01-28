# scripts/convert_liveportrait_trt.ps1
# Automates the conversion of LivePortrait ONNX models to TensorRT engines.

$ErrorActionPreference = "Stop"

# Paths
$BaseDir = Resolve-Path "$PSScriptRoot\.."
$ModelDir = "$BaseDir\models\live_portrait_onnx\liveportrait_onnx"
$PluginPath = "$BaseDir\models\live_portrait_onnx\grid_sample_3d_plugin.dll"

# Check Python
$PythonExe = "$BaseDir\venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Host "Error: venv python not found at $PythonExe" -ForegroundColor Red
    exit 1
}

$BuilderScript = "$PSScriptRoot\build_engine.py"
if (-not (Test-Path $BuilderScript)) {
     Write-Host "Error: Builder script not found at $BuilderScript" -ForegroundColor Red
     exit 1
}

if (-not (Test-Path $ModelDir)) {
    Write-Host "Error: Model directory not found: $ModelDir" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $PluginPath)) {
    Write-Host "Error: Plugin DLL not found: $PluginPath" -ForegroundColor Red
    Write-Host "Please run setup_tensorrt.ps1 first to compile the plugin."
    exit 1
}

Write-Host "Found models at: $ModelDir" -ForegroundColor Cyan
Write-Host "Found plugin at: $PluginPath" -ForegroundColor Cyan

# Function to convert
function Convert-Model {
    param (
        [string]$Name,
        [string]$OnnxFile,
        [string]$EngineFile,
        [string]$PluginArg = ""
    )
    
    if (Test-Path $EngineFile) {
        Write-Host "Engine for $Name already exists (Delete to rebuild). Skipping." -ForegroundColor Yellow
        return
    }

    Write-Host "`nConverting $Name..." -ForegroundColor Green
    # Use python script
    $Cmd = "& `"$PythonExe`" `"$BuilderScript`" --onnx `"$OnnxFile`" --saveEngine `"$EngineFile`" --fp16 $PluginArg"
    
    Write-Host "Running: $Cmd" -ForegroundColor DarkGray
    
    # Execute
    Invoke-Expression $Cmd
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to convert $Name" -ForegroundColor Red
        exit 1
    }
}

# 1. Appearance Feature Extractor
Convert-Model -Name "Appearance Feature Extractor" `
              -OnnxFile "$ModelDir\appearance_feature_extractor.onnx" `
              -EngineFile "$ModelDir\appearance_feature_extractor.engine"

# 2. Motion Extractor
Convert-Model -Name "Motion Extractor" `
              -OnnxFile "$ModelDir\motion_extractor.onnx" `
              -EngineFile "$ModelDir\motion_extractor.engine"

# 3. Warping Spade (Requires Plugin AND ONNX patching)
# The ONNX model uses "GridSample" op, but the plugin is "GridSample3D".
# We need to patch the model first.
$WarpingOnnx = "$ModelDir\warping_spade.onnx"
$WarpingPatchedOnnx = "$ModelDir\warping_spade_patched.onnx"
$PatcherScript = "$PSScriptRoot\patch_onnx_gridsample3d.py"

if (-not (Test-Path $WarpingPatchedOnnx)) {
    Write-Host "`nPatching warping_spade.onnx for GridSample3D plugin..." -ForegroundColor Cyan
    & "$PythonExe" "$PatcherScript" --input "$WarpingOnnx" --output "$WarpingPatchedOnnx"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to patch warping_spade.onnx" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Patched ONNX for warping_spade already exists. Skipping patch." -ForegroundColor Yellow
}

Convert-Model -Name "Warping Spade" `
              -OnnxFile "$WarpingPatchedOnnx" `
              -EngineFile "$ModelDir\warping_spade.engine" `
              -PluginArg "--plugins `"$PluginPath`""

Write-Host "`nAll models converted successfully!" -ForegroundColor Green
