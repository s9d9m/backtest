#!/usr/bin/env bash
# ORB Lab: one-step launcher for macOS / Linux. Installs what is needed (first time only) and opens the dashboard.
set -e
cd "$(dirname "$0")"
PY=""
for c in python3.12 python3.13 python3.11 python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "Python is not installed. Install Python 3.12 from https://www.python.org/downloads/ and run this again."
  read -r -p "Press Enter to close." _; exit 1
fi
if [ ! -x ".venv/bin/python" ]; then
  echo "First run: setting up (this takes a few minutes)..."
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi
echo "Starting ORB Lab... your browser will open at http://localhost:8501 (leave this window open; close it to stop)."
.venv/bin/python -m streamlit run app.py --server.port 8501
