# How to Run - Face Recognition Attendance System

## Prerequisites
- Python 3.11
- Node.js 16+

## Quick Start

### 1. Navigate to Project Directory
```bash
cd /Users/daivik2/Desktop/face_recognition
```

### 2. Run the System
```bash
./start.sh ## this is command to start the website
```

That's it! The script will:
- ✅ Check requirements
- ✅ Install dependencies (first run only)
- ✅ Start both servers

### 3. Access the Application

**Local Access (same computer):**
- Open your browser: **http://localhost:5173**

**Network Access (from another laptop/device):**
1. Find your computer's IP address:
   ```bash
   # On Mac/Linux
   ifconfig | grep "inet " | grep -v 127.0.0.1
   # Or simpler:
   ipconfig getifaddr en0
   ```
2. On the other device, open browser: **http://YOUR_IP_ADDRESS:5173**
   - Example: `http://192.168.1.100:5173`
3. Make sure both devices are on the same network (WiFi/LAN)

### 4. Stop the System
Press `Ctrl+C` in the terminal

---

## Alternative: Run Servers Separately

### Backend Only
```bash
./start_backend.sh
```
Access at: http://localhost:8000

### Frontend Only (in new terminal)
```bash
./start_frontend.sh
```
Access at: http://localhost:5173

---

## First Time Setup
The first run takes 2-3 minutes to install dependencies. Subsequent runs start in seconds.

## Network Access Configuration

**For accessing from other devices on your network:**

The servers are already configured to accept network connections. However, if you're accessing from another device, you need to configure the frontend to connect to your computer's IP:

1. Create a `.env` file in the `frontend` directory:
   ```bash
   cd frontend
   cp .env.example .env
   ```

2. Edit `.env` and replace `localhost` with your computer's IP address:
   ```
   VITE_API_URL=http://YOUR_IP_ADDRESS:8000/api
   ```
   Example: `VITE_API_URL=http://192.168.1.100:8000/api`

3. Restart the application

**Firewall Note:** If you can't connect from another device, check your firewall settings to allow connections on ports 5173 and 8000.

---

## Troubleshooting

**Backend not starting?**
```bash
lsof -ti:8000 | xargs kill -9
./start_backend.sh
```

**Frontend not starting?**
```bash
lsof -ti:5173 | xargs kill -9
./start_frontend.sh
```

**Need to reinstall?**
```bash
rm -rf backend/venv frontend/node_modules
./start.sh
```
