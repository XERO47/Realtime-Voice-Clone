$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$modelDir = Join-Path $projectRoot 'models\tts'
New-Item -ItemType Directory -Path $modelDir -Force | Out-Null

$modelUrl = 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.onnx'
$voicesUrl = 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin'
$modelPath = Join-Path $modelDir 'kokoro-v1.0.onnx'
$voicesPath = Join-Path $modelDir 'voices-v1.0.bin'

if (-not (Test-Path -LiteralPath $modelPath)) { curl.exe -L $modelUrl -o $modelPath }
if (-not (Test-Path -LiteralPath $voicesPath)) { curl.exe -L $voicesUrl -o $voicesPath }

Get-Item -LiteralPath $modelPath,$voicesPath | Select-Object Name,Length,LastWriteTime
