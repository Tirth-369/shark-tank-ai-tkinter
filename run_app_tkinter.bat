@echo off
cd /d "%~dp0"
if exist venv\Scripts\python.exe (
  venv\Scripts\python.exe tkinter_app.py
) else (
  py -3 tkinter_app.py
)
if errorlevel 1 pause
