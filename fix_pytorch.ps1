
Write-Host "Starting PyTorch Re-installation Process..." -ForegroundColor Cyan

# 1. Force Uninstall
Write-Host "Uninstalling existing PyTorch packages..." -ForegroundColor Yellow
.\venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio

# 2. Clean Cache
Write-Host "Cleaning pip cache..." -ForegroundColor Yellow
.\venv\Scripts\python.exe -m pip cache purge

# 3. Install Correct Version
Write-Host "Installing PyTorch Nightly (CUDA 12.4)..." -ForegroundColor Yellow

# Manually cleanup residual folders just in case
if (Test-Path ".\venv\Lib\site-packages\torch") { Remove-Item ".\venv\Lib\site-packages\torch" -Recurse -Force -ErrorAction SilentlyContinue }

# Install Torch execution FIRST to ensure core library is present
Write-Host "Part A: Installing Torch Core..."
.\venv\Scripts\python.exe -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124 --no-cache-dir

# Install dependencies SECOND
Write-Host "Part B: Installing Vision and Audio..."
.\venv\Scripts\python.exe -m pip install --pre torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124 --no-cache-dir

# 4. Verification
Write-Host "Verifying installation..." -ForegroundColor Yellow
.\venv\Scripts\python.exe -c "import torch; print(f'Torch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

Write-Host "Done!" -ForegroundColor Cyan
