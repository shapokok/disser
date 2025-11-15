#!/bin/bash

# Crop Disease Detection System - Quick Start Script
# This script starts the backend server

echo "========================================="
echo "Crop Disease Detection System"
echo "========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found!"
    echo "Creating virtual environment..."
    python3 -m venv venv

    echo "Installing dependencies..."
    source venv/bin/activate
    cd backend
    pip install -r requirements.txt
    cd ..
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if in correct directory
if [ ! -f "backend/app.py" ]; then
    echo "❌ Error: backend/app.py not found!"
    echo "Please run this script from the crop_monitoring_app directory"
    exit 1
fi

# Start backend server
echo ""
echo "🚀 Starting backend server..."
echo "Backend will be available at: http://localhost:5000"
echo "Open frontend at: frontend/index.html"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd backend
python app.py
