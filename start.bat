@echo off
echo Starting Face Recognition Attendance System...
echo.

start "Backend Server" cmd /k "%~dp0\start_backend.bat"
timeout /t 5 /nobreak >nul
start "Frontend Server" cmd /k "%~dp0\start_frontend.bat"

echo.
echo Servers are starting...
echo Backend: http://localhost:8000
echo Frontend: http://localhost:5173
echo.
echo Press any key to stop all servers...
pause >nul

taskkill /FI "WindowTitle eq Backend Server*" /T /F
taskkill /FI "WindowTitle eq Frontend Server*" /T /F
