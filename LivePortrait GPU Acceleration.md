# LivePortrait GPU Acceleration

## Problem Statement
LivePortrait inference is running on **CPU** (~2.6s/frame) instead of GPU. The `warping_spade` module is forced to CPU because ONNX Runtime's CUDA kernel doesn't support **5D GridSample** operations.

## TensorRT Integration (Implemented)

> [!IMPORTANT]
> **Status**: Implemented. Requires manual setup of TensorRT environment and plugin compilation.

**Prerequisites:**
1.  **NVIDIA GPU** (RTX 30 series or newer recommended).
2.  **CUDA Toolkit** (Ensure `nvcc` is in PATH).
3.  **Visual Studio Build Tools** (C++ Desktop Development workload).
4.  **TensorRT SDK** (Required for compilation headers).
    - Download ZIP from [NVIDIA Developer](https://developer.nvidia.com/tensorrt).
    - Extract to a known location (e.g., `C:\TensorRT-10.x.x.x`).

**Setup Instructions:**

1.  **Run the Setup Script**:
    ```powershell
    .\scripts\setup_tensorrt.ps1
    ```
    - The script will ask for the path to your extracted TensorRT SDK if it cannot find the headers.
    - It will then compile the plugin using `cmake`.

2.  **Convert Models**:
    You need to convert the ONNX models to TensorRT engines (`.engine`).
    You need to convert the ONNX models to TensorRT engines (`.engine`).
    We have provided a script to automate this using the installed `tensorrt` Python package.

    ```powershell
    .\scripts\convert_liveportrait_trt.ps1
    ```
    
    *This process will take a few minutes. It performs FP16 conversion and correctly loads the `warping_spade` plugin.*

3.  **Enable in UI**:
    - Launch the app (`python app.py`).
    - Select **"Enable Lip-Sync"**.
    - Choose **"LivePortrait (High Quality - Slow)"**.
    - A new dropdown **"LivePortrait Acceleration"** will appear. Select **"tensorrt"**.

**Automatic Fallback:**
If TensorRT is not found or engines are missing, the system will automatically fallback to ONNX Runtime (CPU for warping, GPU for others).
