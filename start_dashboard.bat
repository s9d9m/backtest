@echo off
REM ORB Lab: one-step launcher for Windows. Installs what is needed (first time only) and opens the dashboard.
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Python is not installed. Install Python 3.12 from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^) and run this again.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo First run: setting up ^(this takes a few minutes^)...
  py -3 -m venv .venv
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\python -m pip install -r requirements.txt
)
echo Starting ORB Lab... your browser will open at http://localhost:8501 ^(leave this window open; close it to stop^).
.venv\Scripts\python -m streamlit run app.py --server.port 8501
pause
