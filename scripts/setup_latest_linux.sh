#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FULL="${FULL:-0}"
NO_TESSERACT="${NO_TESSERACT:-0}"

echo "=== Setup OCR Full Demo - Linux latest ==="
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -U -r requirements.txt

if [ "$FULL" = "1" ]; then
  echo "Installing/upgrading optional OCR packages. This can take long."
  python -m pip install -U -r requirements-optional-ocr.txt
  echo "Applying OCR compatibility fixes."
  python -m pip install -U "einops>=0.8.1" "chardet<6" "numpy==2.2.1" "pillow==10.2.0"
fi

if [ "$NO_TESSERACT" != "1" ] && ! command -v tesseract >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y tesseract-ocr tesseract-ocr-vie tesseract-ocr-eng
  else
    echo "No apt-get found. Install tesseract manually for your OS."
  fi
fi

[ -f .env ] || cp .env.example .env
python scripts/check_environment.py
echo "Done. Run: bash run_linux.sh"
