
Write-Host "Upgrading everything to CUDA 13.0 Nightly (Active Branch)..." -ForegroundColor Cyan

# 1. Clean Slate
Write-Host "Uninstalling OLD versions..." -ForegroundColor Yellow
.\venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio

# 2. Install EVERYTHING from CU130
Write-Host "Installing Torch+Vision+Audio (CU130)..." -ForegroundColor Yellow
.\venv\Scripts\python.exe -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130 --no-cache-dir

# 3. Verification
Write-Host "Verifying..." -ForegroundColor Yellow
.\venv\Scripts\python.exe -c "import torch; print(f'Torch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); import torchvision; print(f'Vision: {torchvision.__version__}'); import torchaudio; print(f'Audio: {torchaudio.__version__}')"

Write-Host "Done!" -ForegroundColor Cyan
