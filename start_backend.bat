@echo off
cd /d "%~dp0\backend"

if not exist "venv" (
    echo Creating virtual environment with Python...
    python -m venv venv
    echo Installing dependencies...
    venv\Scripts\pip install -q fastapi "uvicorn[standard]" python-multipart pydantic pydantic-settings pyyaml opencv-python numpy pandas openpyxl python-dateutil insightface onnxruntime
)

echo Activating virtual environment...
call venv\Scripts\activate

echo Starting backend server...
echo Backend will be available at http://localhost:8000
python main.py
