$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot
$env:UV_CACHE_DIR = Join-Path $projectRoot '.uv-cache'
uv venv .venv-cloner --python 3.11
uv pip install --python .venv-cloner\Scripts\python.exe fastapi uvicorn python-multipart
uv pip install --python .venv-cloner\Scripts\python.exe --upgrade 'chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox.git@master'
Write-Host 'Cloner ready. Run .\start_cloner.ps1 and open http://127.0.0.1:8002/'

