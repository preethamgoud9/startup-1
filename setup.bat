@echo off
REM Setup script for Face Recognition Attendance System (Windows)
setlocal enabledelayedexpansion

set ROOT_DIR=%~dp0

echo [INFO] Checking Python 3.11...
set PYTHON=
for %%P in (python3.11 python py) do (
    where %%P >nul 2>&1
    if !errorlevel! == 0 (
        for /f "tokens=2 delims= " %%V in ('%%P --version 2^>^&1') do (
            echo %%V | findstr /b "3.11" >nul
            if !errorlevel! == 0 (
                set PYTHON=%%P
                goto :found_python
            )
        )
    )
)

REM Try py launcher with version flag
where py >nul 2>&1
if %errorlevel% == 0 (
    for /f "tokens=2 delims= " %%V in ('py -3.11 --version 2^>^&1') do (
        echo %%V | findstr /b "3.11" >nul
        if !errorlevel! == 0 (
            set PYTHON=py -3.11
            goto :found_python
        )
    )
)

echo [ERROR] Python 3.11 is required but not found.
echo   Download from: https://www.python.org/downloads/release/python-3119/
echo   Make sure to check "Add Python to PATH" during installation.
exit /b 1

:found_python
echo [INFO] Found Python 3.11 (%PYTHON%)

echo [INFO] Checking Node.js...
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js is required but not found.
    echo   Download from: https://nodejs.org ^(v20 or higher^)
    exit /b 1
)
for /f "tokens=1 delims=." %%V in ('node -v') do set NODE_VER=%%V
set NODE_VER=%NODE_VER:v=%
if %NODE_VER% LSS 20 (
    echo [ERROR] Node.js 20+ is required. Update from https://nodejs.org
    exit /b 1
)
echo [INFO] Found Node.js

REM ── Backend setup ──
echo [INFO] Setting up backend...
cd /d "%ROOT_DIR%backend"

if not exist "venv" (
    echo [INFO] Creating virtual environment...
    %PYTHON% -m venv venv
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo [INFO] Installing Python dependencies...
pip install --upgrade pip -q
pip install -r requirements.txt -q

REM ── Create data directories ──
if not exist "data\attendance\exports" mkdir data\attendance\exports
if not exist "data\embeddings" mkdir data\embeddings

REM ── Download face recognition model (antelopev2) ──
set ANTELOPE_DIR=%USERPROFILE%\.insightface\models\antelopev2
set RELEASE_URL=https://github.com/preethamgoud2912/production_1/releases/download/v1.0.0/antelopev2.zip
set TMP_ZIP=%TEMP%\antelopev2.zip

if exist "%ANTELOPE_DIR%\glintr100.onnx" (
    echo [INFO] antelopev2 model already installed
) else (
    echo [INFO] Downloading antelopev2 face recognition model (~344MB)...
    echo [INFO] This is a one-time download...
    if not exist "%USERPROFILE%\.insightface\models" mkdir "%USERPROFILE%\.insightface\models"

    curl -L --progress-bar -o "%TMP_ZIP%" "%RELEASE_URL%"
    if !errorlevel! neq 0 (
        echo [WARN] antelopev2 download failed. Falling back to buffalo_l...
        %PYTHON% -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); app.prepare(ctx_id=-1, det_size=(640, 640)); print('buffalo_l downloaded')"
        goto :model_done
    )

    echo [INFO] Extracting model files...
    powershell -command "Expand-Archive -Force '%TMP_ZIP%' '%USERPROFILE%\.insightface\models\'"
    del "%TMP_ZIP%"

    if exist "%ANTELOPE_DIR%\glintr100.onnx" (
        echo [INFO] antelopev2 model installed successfully!
    ) else (
        echo [WARN] Extraction failed. Falling back to buffalo_l...
        %PYTHON% -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); app.prepare(ctx_id=-1, det_size=(640, 640)); print('buffalo_l downloaded')"
    )
)
:model_done

call deactivate
cd /d "%ROOT_DIR%"

REM ── Frontend setup ──
echo [INFO] Setting up frontend...
cd /d "%ROOT_DIR%frontend"

if not exist ".env" (
    echo [INFO] Creating frontend .env from example...
    copy .env.example .env >nul
)

echo [INFO] Installing Node.js dependencies...
call npm install --silent

cd /d "%ROOT_DIR%"

echo.
echo [INFO] Setup complete!
echo.
echo   To start the app, run:
echo.
echo     start.bat
echo.
echo   Or start each server manually:
echo.
echo     Terminal 1 - Backend:
echo       cd backend
echo       venv\Scripts\activate
echo       python main.py
echo.
echo     Terminal 2 - Frontend:
echo       cd frontend
echo       npm run dev
echo.
echo   Then open http://localhost:5173 in your browser.
echo.
