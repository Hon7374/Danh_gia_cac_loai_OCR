#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -U -r requirements.txt
else
  source .venv/bin/activate
fi
[ -f .env ] || cp .env.example .env
python scripts/check_environment.py
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
