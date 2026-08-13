@echo off
echo Starting Skin Disease Detection Backend Server...
cd /d "%~dp0backend"
call venv\Scripts\activate.bat
python main.py
pause
