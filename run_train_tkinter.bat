@echo off
cd /d "%~dp0"
if exist venv\Scripts\python.exe (
  venv\Scripts\python.exe train_models.py
) else (
  py -3 train_models.py
)
if errorlevel 1 pause
