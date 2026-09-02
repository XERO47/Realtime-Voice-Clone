$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$python = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { 'python' }
$detectorPython = Join-Path $projectRoot '.venv-detector\Scripts\python.exe'
$checkpoint = Join-Path $projectRoot '..\best_telephony_detector(2).pth'
if ((Test-Path -LiteralPath $detectorPython) -and (Test-Path -LiteralPath $checkpoint)) {
    try {
        Invoke-RestMethod -Uri 'http://127.0.0.1:8001/health' -TimeoutSec 1 | Out-Null
    } catch {
        $detectorArguments = @('-m','uvicorn','detector_service.main:app','--host','127.0.0.1','--port','8001')
        Start-Process -FilePath $detectorPython -ArgumentList $detectorArguments -WorkingDirectory $projectRoot -WindowStyle Hidden
        Start-Sleep -Seconds 2
    }
}
& $python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
