#!/bin/bash

echo "=========================================="
echo "Face Recognition Attendance System"
echo "=========================================="
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for Python 3.11
if ! command_exists python3.11; then
    echo "❌ Error: Python 3.11 is required but not found."
    echo "Please install Python 3.11 and try again."
    exit 1
fi

# Check for Node.js
if ! command_exists node; then
    echo "❌ Error: Node.js is required but not found."
    echo "Please install Node.js and try again."
    exit 1
fi

echo "✅ Python 3.11 found"
echo "✅ Node.js found"
echo ""

# Setup backend
echo "📦 Setting up backend..."
cd "$(dirname "$0")/backend"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3.11 -m venv venv
    echo "Installing Python dependencies (this may take a few minutes)..."
    ./venv/bin/pip install -q fastapi 'uvicorn[standard]' python-multipart pydantic pydantic-settings pyyaml opencv-python numpy pandas openpyxl python-dateutil insightface onnxruntime
    echo "✅ Backend dependencies installed"
else
    echo "✅ Backend virtual environment exists"
fi

# Setup frontend
echo ""
echo "📦 Setting up frontend..."
cd ../frontend

if [ ! -d "node_modules" ]; then
    echo "Installing Node dependencies (this may take a few minutes)..."
    npm install
    echo "✅ Frontend dependencies installed"
else
    echo "✅ Frontend dependencies exist"
fi

cd ..

echo ""
echo "=========================================="
echo "🚀 Starting servers..."
echo "=========================================="
echo ""
echo "Backend will be available at: http://localhost:8000"
echo "Frontend will be available at: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Start backend in background
cd backend
./venv/bin/python main.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend in background
cd ../frontend
npm run dev &
FRONTEND_PID=$!

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

# Trap Ctrl+C
trap cleanup INT

# Wait for both processes
wait
