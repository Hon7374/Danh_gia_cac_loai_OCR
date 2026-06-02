$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceRoot = Split-Path -Parent $projectRoot
$tempDir = Join-Path $workspaceRoot "temp"
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

$env:TEMP = $tempDir
$env:TMP = $tempDir
$env:HF_HOME = Join-Path $workspaceRoot "cache\huggingface"
$env:TRANSFORMERS_CACHE = Join-Path $workspaceRoot "cache\huggingface\hub"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $workspaceRoot "cache\huggingface\hub"
$env:PADDLE_HOME = Join-Path $workspaceRoot "cache\paddle"
$env:PADDLEOCR_HOME = Join-Path $workspaceRoot "cache\paddleocr"
$env:PADDLE_PDX_CACHE_HOME = Join-Path $workspaceRoot "cache\paddlex"
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"
$env:PADDLE_PDX_MODEL_SOURCE = "huggingface"

Set-Location $projectRoot
& (Join-Path $projectRoot ".venv\Scripts\python.exe") -m uvicorn app.main:app --host 127.0.0.1 --port 8000
