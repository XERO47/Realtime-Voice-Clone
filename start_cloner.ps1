$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot '.venv-cloner\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Cloner environment is missing. Run setup_cloner.ps1 first.'
}
$env:HF_HOME = Join-Path $projectRoot 'models\chatterbox-cache'
$env:NUMBA_CACHE_DIR = Join-Path $projectRoot 'models\numba-cache'
& $python -m uvicorn cloner_service.main:app --host 0.0.0.0 --port 8002

