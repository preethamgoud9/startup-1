@echo off
REM Start both backend and frontend servers (Windows)
setlocal

set ROOT_DIR=%~dp0

REM Check if setup has been run
if not exist "%ROOT_DIR%backend\venv" (
    echo First run detected. Running setup...
    call "%ROOT_DIR%setup.bat"
)

if not exist "%ROOT_DIR%frontend\node_modules" (
    echo Frontend dependencies missing. Running setup...
    call "%ROOT_DIR%setup.bat"
)

echo Starting backend server...
start "Backend" cmd /c "cd /d %ROOT_DIR%backend && call venv\Scripts\activate.bat && python main.py"

echo Starting frontend server...
start "Frontend" cmd /c "cd /d %ROOT_DIR%frontend && npm run dev"

echo.
echo ========================================
echo   App is running!
echo   Frontend: http://localhost:5173
echo   Backend:  http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo ========================================
echo.
echo Close the Backend and Frontend terminal windows to stop.
echo.
pause
