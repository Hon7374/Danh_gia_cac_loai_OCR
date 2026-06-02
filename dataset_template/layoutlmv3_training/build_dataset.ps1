$ErrorActionPreference = "Stop"

$DatasetDir = $PSScriptRoot
$ProjectRoot = Resolve-Path (Join-Path $DatasetDir "..\..")
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$BuildScript = Join-Path $ProjectRoot "scripts\build_layoutlmv3_dataset.py"

if (!(Test-Path $Python)) {
  throw "Khong tim thay Python venv: $Python"
}

& $Python $BuildScript `
  --dataset-dir $DatasetDir `
  --max-pages 1 `
  --clean-rendered
