@echo off
setlocal
cd /d %~dp0

if not exist .venv (
  echo [INFO] Chua co .venv, dang cai moi moi truong co ban...
  py -3 -m venv .venv
  .venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
  .venv\Scripts\python.exe -m pip install -U -r requirements.txt
)

if not exist .env copy .env.example .env >nul
.venv\Scripts\python.exe scripts\check_environment.py
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
