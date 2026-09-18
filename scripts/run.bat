@echo off
rem Start the Crop Disease Detection app on Windows (creates the environment on first run).
cd /d "%~dp0\.."

where uv >nul 2>nul
if %errorlevel%==0 (
  uv sync --extra dev || exit /b 1
  set PY=uv run python
) else (
  if not exist .venv (
    python -m venv .venv || exit /b 1
    .venv\Scripts\pip install --upgrade pip
    .venv\Scripts\pip install -r requirements.txt || exit /b 1
  )
  set PY=.venv\Scripts\python
)

if "%CROP_PORT%"=="" set CROP_PORT=5001
echo Open http://localhost:%CROP_PORT%/
%PY% backend\app.py
