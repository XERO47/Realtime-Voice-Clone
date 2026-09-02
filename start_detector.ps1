$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot
$detectorPython = Join-Path $projectRoot '.venv-detector\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $detectorPython)) {
    throw 'Detector environment not found. Run setup_detector.ps1 first.'
}
$env:USE_TF = '0'
$env:USE_FLAX = '0'
& $detectorPython -m uvicorn detector_service.main:app --host 127.0.0.1 --port 8001
