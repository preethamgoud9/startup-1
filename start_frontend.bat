@echo off
cd /d "%~dp0\frontend"

if not exist "node_modules" (
    echo Installing frontend dependencies...
    call npm install
)

echo Starting frontend server...
echo Frontend will be available at http://localhost:5173
call npm run dev
