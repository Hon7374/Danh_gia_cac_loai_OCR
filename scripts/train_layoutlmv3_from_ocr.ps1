param(
  [string]$SourceDir = "OCR",
  [string]$DatasetDir = "dataset_template\layoutlmv3_training",
  [string]$OutputDir = "models\layoutlmv3-congvan-token-classification",
  [string]$BaseModel = "",
  [int]$MaxPages = 1,
  [double]$EvalRatio = 0.15,
  [double]$Epochs = 8,
  [int]$BatchSize = 1,
  [int]$GradientAccumulationSteps = 1,
  [double]$LearningRate = 0.00002,
  [int]$Seed = 42,
  [switch]$Cpu,
  [switch]$Fp16,
  [switch]$NoCleanRendered
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$BuildScript = Join-Path $ProjectRoot "scripts\build_layoutlmv3_dataset.py"
$TrainScript = Join-Path $ProjectRoot "scripts\train_layoutlmv3_fields.py"
$DatasetPath = Join-Path $ProjectRoot $DatasetDir
$SourcePath = Join-Path $ProjectRoot $SourceDir
$OutputPath = Join-Path $ProjectRoot $OutputDir
$TrainJsonl = Join-Path $DatasetPath "train.jsonl"
$EvalJsonl = Join-Path $DatasetPath "eval.jsonl"

if (!(Test-Path $Python)) {
  throw "Khong tim thay Python venv: $Python"
}
if (!(Test-Path $SourcePath)) {
  throw "Khong tim thay folder du lieu OCR: $SourcePath"
}

if (!$BaseModel) {
  $LocalConfig = Join-Path $OutputPath "config.json"
  if (Test-Path $LocalConfig) {
    $BaseModel = $OutputPath
  } else {
    $BaseModel = "microsoft/layoutlmv3-base"
  }
}

New-Item -ItemType Directory -Force -Path $DatasetPath | Out-Null

$BuildArgs = @(
  $BuildScript,
  "--dataset-dir", $DatasetPath,
  "--source-dir", $SourcePath,
  "--recursive",
  "--max-pages", "$MaxPages",
  "--eval-ratio", "$EvalRatio",
  "--seed", "$Seed",
  "--min-labeled-tokens", "1"
)
if (!$NoCleanRendered) {
  $BuildArgs += "--clean-rendered"
}

Write-Host "== Build LayoutLMv3 dataset from $SourcePath =="
& $Python @BuildArgs
if ($LASTEXITCODE -ne 0) {
  throw "Build dataset failed with exit code $LASTEXITCODE"
}

$EvalHasData = $false
if (Test-Path $EvalJsonl) {
  $EvalHasData = [bool](Get-Content $EvalJsonl | Where-Object { $_.Trim() })
}

$TrainArgs = @(
  $TrainScript,
  "--train-jsonl", $TrainJsonl,
  "--output-dir", $OutputPath,
  "--base-model", $BaseModel,
  "--epochs", "$Epochs",
  "--batch-size", "$BatchSize",
  "--gradient-accumulation-steps", "$GradientAccumulationSteps",
  "--learning-rate", "$LearningRate",
  "--eval-ratio", "0",
  "--save-total-limit", "2",
  "--seed", "$Seed"
)
if ($EvalHasData) {
  $TrainArgs += @("--eval-jsonl", $EvalJsonl)
}
if ($Cpu) {
  $TrainArgs += "--cpu"
}
if ($Fp16) {
  $TrainArgs += "--fp16"
}

Write-Host "== Train LayoutLMv3 checkpoint =="
Write-Host "Base model: $BaseModel"
& $Python @TrainArgs
if ($LASTEXITCODE -ne 0) {
  throw "Train failed with exit code $LASTEXITCODE"
}

Write-Host "== Done =="
Write-Host "Model: $OutputPath"
Write-Host "Dataset manifest: $(Join-Path $DatasetPath 'build_manifest.json')"
Write-Host "Training summary: $(Join-Path $OutputPath 'training_summary.json')"
