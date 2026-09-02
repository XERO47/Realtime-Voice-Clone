$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot
$detectorPython = Join-Path $projectRoot '.venv-detector\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $detectorPython)) {
    uv venv .venv-detector --python 3.11 --system-site-packages
}
uv pip install --python $detectorPython -r detector_requirements.txt
& $detectorPython -c 'import torch' 2>$null
if ($LASTEXITCODE -ne 0) {
    uv pip install --python $detectorPython torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
}
uv pip install --python $detectorPython torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu --no-deps
Write-Host 'Detector environment ready.'
