@echo off
setlocal
cd /d %~dp0

if not exist .venv (
  echo [INFO] Chua co .venv, dang cai moi moi truong co ban...
  py -3 -m venv .venv
  call .venv\Scripts\activate.bat
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -U -r requirements.txt
) else (
  call .venv\Scripts\activate.bat
)

if not exist .env copy .env.example .env >nul
python scripts\check_environment.py
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
