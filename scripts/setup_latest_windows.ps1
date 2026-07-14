param(
    [switch]$Full,
    [switch]$NoTesseract
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "=== Setup OCR Full Demo - Windows latest ===" -ForegroundColor Cyan

if (-not (Test-Path ".venv")) {
    py -3 -m venv .venv
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
& .\.venv\Scripts\python.exe -m pip install -U -r requirements.txt

if ($Full) {
    Write-Host "Installing/upgrading optional OCR packages. This can take long." -ForegroundColor Yellow
    & .\.venv\Scripts\python.exe -m pip install -U -r requirements-optional-ocr.txt
    Write-Host "Applying OCR compatibility fixes for Windows CPU runtime." -ForegroundColor Yellow
    & .\.venv\Scripts\python.exe -m pip install -U "einops>=0.8.1" "chardet<6" "numpy==2.2.1" "pillow==10.2.0"
}

if (-not $NoTesseract) {
    $tess = Get-Command tesseract -ErrorAction SilentlyContinue
    if (-not $tess) {
        Write-Host "Tesseract not found. Trying winget install UB-Mannheim.TesseractOCR ..." -ForegroundColor Yellow
        $winget = Get-Command winget -ErrorAction SilentlyContinue
        if ($winget) {
            winget install --id UB-Mannheim.TesseractOCR -e --accept-package-agreements --accept-source-agreements
        } else {
            Write-Host "winget not found. Please install UB Mannheim Tesseract manually from official tessdoc downloads." -ForegroundColor Red
        }
    }
}

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}

& .\.venv\Scripts\python.exe scripts\repair_venv_launchers.py

Write-Host "\nEnvironment check:" -ForegroundColor Cyan
& .\.venv\Scripts\python.exe scripts\check_environment.py
Write-Host "Done. Run: .\run_windows.bat" -ForegroundColor Green
