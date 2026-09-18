@echo off
REM Crop Disease Detection System - Quick Start Script (Windows)
REM This script starts the backend server

echo =========================================
echo Crop Disease Detection System
echo =========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo WARNING: Virtual environment not found!
    echo Creating virtual environment...
    python -m venv venv

    echo Installing dependencies...
    call venv\Scripts\activate
    cd backend
    pip install -r requirements.txt
    cd ..
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Check if in correct directory
if not exist "backend\app.py" (
    echo ERROR: backend\app.py not found!
    echo Please run this script from the crop_monitoring_app directory
    pause
    exit /b 1
)

REM Start backend server
echo.
echo Starting backend server...
echo Backend will be available at: http://localhost:5000
echo Open frontend at: frontend\index.html
echo.
echo Press Ctrl+C to stop the server
echo.

cd backend
python app.py
