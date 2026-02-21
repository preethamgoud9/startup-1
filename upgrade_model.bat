@echo off
REM Download and install antelopev2 (ResNet100) model for higher accuracy.
REM Optional — the system works with buffalo_l (ResNet50) out of the box.
setlocal

set MODEL_DIR=%USERPROFILE%\.insightface\models\antelopev2
set RELEASE_URL=https://github.com/preethamgoud2912/production_1/releases/download/v1.0.0/antelopev2.zip
set TMP_ZIP=%TEMP%\antelopev2.zip

if exist "%MODEL_DIR%\glintr100.onnx" (
    echo [INFO] antelopev2 model is already installed.
    exit /b 0
)

echo [INFO] Downloading antelopev2 model (~344MB)...
echo [INFO] This is a one-time download for the high-accuracy ResNet100 model.
echo.

curl -L --progress-bar -o "%TMP_ZIP%" "%RELEASE_URL%"
if %errorlevel% neq 0 (
    echo [ERROR] Download failed. Make sure curl is available or download manually from:
    echo   %RELEASE_URL%
    exit /b 1
)

echo [INFO] Extracting model files...
if not exist "%USERPROFILE%\.insightface\models" mkdir "%USERPROFILE%\.insightface\models"

powershell -command "Expand-Archive -Force '%TMP_ZIP%' '%USERPROFILE%\.insightface\models\'"

del "%TMP_ZIP%"

if exist "%MODEL_DIR%\glintr100.onnx" (
    echo [INFO] antelopev2 model installed successfully!
    echo [INFO] Restart the backend to use the upgraded model.
) else (
    echo [ERROR] Installation failed. Model files not found.
    exit /b 1
)
