@echo off
cd /d %~dp0
powershell -ExecutionPolicy Bypass -File scripts\setup_latest_windows.ps1 %*
