# Windows Setup Guide

This guide explains how to run the Face Recognition Attendance System on Windows.

## Prerequisites

### Required Software
- **Python 3.11** - [Download](https://www.python.org/downloads/)
  - ⚠️ During installation, check "Add Python to PATH"
- **Node.js 16+** - [Download](https://nodejs.org/)
- **Git** (optional) - [Download](https://git-scm.com/download/win)

### Verify Installation
Open Command Prompt and verify:
```cmd
python --version
node --version
npm --version
```

---

## Quick Start

### Method 1: Using Batch Files (Recommended)

1. **Navigate to project directory:**
   ```cmd
   cd C:\path\to\face_recognition
   ```

2. **Run the system:**
   ```cmd
   start.bat
   ```

3. **Access the application:**
   - Open browser: **http://localhost:5173**

4. **Stop the system:**
   - Press any key in the main window

### Method 2: Manual Start

**Terminal 1 - Backend:**
```cmd
cd backend
python -m venv venv
venv\Scripts\activate
pip install fastapi "uvicorn[standard]" python-multipart pydantic pydantic-settings pyyaml opencv-python numpy pandas openpyxl python-dateutil insightface onnxruntime
python main.py
```

**Terminal 2 - Frontend:**
```cmd
cd frontend
npm install
npm run dev
```

---

## Network Access (Access from Other Devices)

### Step 1: Find Your Windows PC's IP Address
```cmd
ipconfig
```
Look for "IPv4 Address" under your active network adapter (e.g., `192.168.1.100`)

### Step 2: Configure Frontend
1. Navigate to `frontend` folder
2. Copy `.env.example` to `.env`
3. Edit `.env`:
   ```
   VITE_API_URL=http://YOUR_IP:8000/api
   ```
   Example: `VITE_API_URL=http://192.168.1.100:8000/api`

### Step 3: Allow Firewall Access
Windows Firewall will prompt you to allow Python and Node.js. Click **"Allow access"** for both.

Or manually add firewall rules:
```cmd
netsh advfirewall firewall add rule name="Face Recognition Backend" dir=in action=allow protocol=TCP localport=8000
netsh advfirewall firewall add rule name="Face Recognition Frontend" dir=in action=allow protocol=TCP localport=5173
```

### Step 4: Access from Other Devices
- **Your PC:** `http://localhost:5173`
- **Other devices:** `http://YOUR_IP:5173`
  - Example: `http://192.168.1.100:5173`

---

## Camera Configuration

### USB Webcam
Default configuration uses the first available webcam (index 0).

To use a different USB camera, edit `backend\config.yaml`:
```yaml
camera:
  rtsp_url: ""
  usb_device_id: 1  # Change to 1, 2, 3, etc.
```

### Mobile Phone Camera
1. Install "IP Webcam" app on Android
2. Start server in the app
3. Note the IP address (e.g., `192.168.1.50:8080`)
4. Edit `backend\config.yaml`:
   ```yaml
   camera:
     rtsp_url: "http://192.168.1.50:8080/videofeed"
   ```

### CCTV/IP Camera
See `CONNECT_TO_CCTV.md` for detailed instructions.

---

## Troubleshooting

### Python Not Found
- Reinstall Python and check "Add Python to PATH"
- Or use full path: `C:\Python311\python.exe`

### Port Already in Use
**Backend (port 8000):**
```cmd
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**Frontend (port 5173):**
```cmd
netstat -ano | findstr :5173
taskkill /PID <PID> /F
```

### Virtual Environment Activation Error (PowerShell)
If you get "execution policy" error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### OpenCV/Camera Issues
- Install Visual C++ Redistributable: [Download](https://aka.ms/vs/17/release/vc_redist.x64.exe)
- Update camera drivers

### Slow Performance
- Close unnecessary applications
- Consider using GPU acceleration (requires CUDA-compatible GPU)
- Edit `backend\config.yaml`:
  ```yaml
  face_recognition:
    device: "cuda"  # Change from "cpu" to "cuda" if you have NVIDIA GPU
  ```

---

## Differences from Mac/Linux

| Feature | Mac/Linux | Windows |
|---------|-----------|---------|
| Shell scripts | `.sh` | `.bat` |
| Python command | `python3.11` | `python` |
| Virtual env activate | `source venv/bin/activate` | `venv\Scripts\activate` |
| Path separator | `/` | `\` |
| Kill process | `lsof -ti:8000 \| xargs kill -9` | `taskkill /F /PID <PID>` |

---

## GPU Acceleration (Optional)

For better performance with NVIDIA GPU:

1. **Install CUDA Toolkit:** [Download](https://developer.nvidia.com/cuda-downloads)

2. **Install GPU-enabled ONNX Runtime:**
   ```cmd
   venv\Scripts\activate
   pip uninstall onnxruntime
   pip install onnxruntime-gpu
   ```

3. **Update config:**
   ```yaml
   face_recognition:
     device: "cuda"
   ```

---

## Starting on System Boot (Optional)

### Using Task Scheduler
1. Open Task Scheduler
2. Create Basic Task
3. Trigger: "When the computer starts"
4. Action: "Start a program"
5. Program: `C:\path\to\face_recognition\start.bat`

### Using Startup Folder
1. Press `Win + R`
2. Type: `shell:startup`
3. Create shortcut to `start.bat` in this folder

---

## Stopping the System

### If using start.bat
- Press any key in the main window

### Manual Stop
- Press `Ctrl+C` in each terminal window
- Or close the terminal windows

### Force Stop
```cmd
taskkill /F /IM python.exe
taskkill /F /IM node.exe
```

---

## File Paths

Windows uses backslashes (`\`) but Python accepts forward slashes (`/`) in most cases.

The application automatically handles path differences between Windows and Mac/Linux.

---

## Additional Notes

- **Antivirus:** Some antivirus software may flag Python scripts. Add exception if needed.
- **Updates:** Keep Python, Node.js, and dependencies updated
- **Backup:** Regularly backup the `data` folder containing embeddings and attendance records
- **Security:** Don't expose the application to the internet without proper security measures

---

## Support

For issues specific to Windows, check:
- Python installation: `python --version`
- Node.js installation: `node --version`
- Firewall settings: Windows Defender Firewall
- Network connectivity: `ping YOUR_IP`
