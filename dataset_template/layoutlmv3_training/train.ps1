$ErrorActionPreference = "Stop"

$DatasetDir = $PSScriptRoot
$ProjectRoot = Resolve-Path (Join-Path $DatasetDir "..\..")
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$TrainScript = Join-Path $ProjectRoot "scripts\train_layoutlmv3_fields.py"
$TrainJsonl = Join-Path $DatasetDir "train.jsonl"
$EvalJsonl = Join-Path $DatasetDir "eval.jsonl"
$OutputDir = Join-Path $ProjectRoot "models\layoutlmv3-congvan-token-classification"

if (!(Test-Path $Python)) {
  throw "Khong tim thay Python venv: $Python"
}

if (!(Test-Path $TrainJsonl)) {
  throw "Khong tim thay train.jsonl: $TrainJsonl"
}

$TrainHasData = [bool](Get-Content $TrainJsonl | Where-Object { $_.Trim() })
if (!$TrainHasData) {
  throw "train.jsonl dang trong. Hay bo anh vao pages/ va dien du lieu JSONL vao train.jsonl truoc khi train. Xem example.jsonl de lay format."
}

$ArgsList = @(
  $TrainScript,
  "--train-jsonl", $TrainJsonl,
  "--output-dir", $OutputDir,
  "--epochs", "20",
  "--batch-size", "1",
  "--eval-ratio", "0"
)

$EvalHasData = $false
if (Test-Path $EvalJsonl) {
  $EvalHasData = [bool](Get-Content $EvalJsonl | Where-Object { $_.Trim() })
}
if ($EvalHasData) {
  $ArgsList += @("--eval-jsonl", $EvalJsonl)
}

& $Python @ArgsList
