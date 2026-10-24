@echo off
title MinuteAI - AI Meeting Minutes Generator
echo =================================================================
echo   Launching MinuteAI Application...
echo =================================================================

cd /d "%~dp0"

if exist ".venv312\Scripts\python.exe" (
    .venv312\Scripts\python.exe run_app.py
) else if exist "..\.venv\Scripts\python.exe" (
    ..\.venv\Scripts\python.exe run_app.py
) else (
    python run_app.py
)
pause
