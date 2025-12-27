#!/bin/bash

cd "$(dirname "$0")/backend"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment with Python 3.11..."
    python3.11 -m venv venv
    echo "Installing dependencies..."
    ./venv/bin/pip install -q fastapi 'uvicorn[standard]' python-multipart pydantic pydantic-settings pyyaml opencv-python numpy pandas openpyxl python-dateutil insightface onnxruntime
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Starting backend server..."
echo "Backend will be available at http://localhost:8000"
python main.py
