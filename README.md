# Surveillance System

Real-time face recognition surveillance and attendance system built with FastAPI and React. Uses InsightFace (SCRFD + ArcFace) to detect and identify individuals from live camera feeds, automatically marking attendance.

## Prerequisites

| Requirement | Version | Install |
|-------------|---------|---------|
| **Python**  | 3.11    | [python.org](https://www.python.org/downloads/release/python-3119/) or `brew install python@3.11` |
| **Node.js** | 20+     | [nodejs.org](https://nodejs.org) or `brew install node` |
| **Webcam**  | USB     | Built-in or external |

> **Why Python 3.11?** The `onnxruntime` package (used for face detection/recognition inference) requires Python 3.11. Newer versions are not yet supported.

## Quick Start

### macOS / Linux

```bash
git clone https://github.com/preethamgoud2912/production_1.git
cd production_1
./setup.sh    # one-time: installs dependencies + downloads model
./start.sh    # starts both servers
```

### Windows

```powershell
git clone https://github.com/preethamgoud2912/production_1.git
cd production_1
setup.bat     # one-time: installs dependencies + downloads model
start.bat     # starts both servers
```

Then open **http://localhost:5173** in your browser.

Default login: `admin` / `admin123`

## Manual Setup

If you prefer to set things up step by step:

### 1. Backend

```bash
cd backend

# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt

# Start the server
python main.py
```

The backend runs at **http://localhost:8000**. API docs at **http://localhost:8000/docs**.

### 2. Frontend (new terminal)

```bash
cd frontend

# Copy environment config
cp .env.example .env

# Install dependencies and start
npm install
npm run dev
```

The frontend runs at **http://localhost:5173**.

## How It Works

1. **Enroll Students** - Capture 15 face images per student from different angles
2. **Live Recognition** - Start the camera feed; the system detects and identifies enrolled students in real-time
3. **Attendance** - Recognized students are automatically marked present (once per day)

## Face Recognition Model

The setup script automatically downloads **antelopev2** (ResNet100, ~344MB) — the highest accuracy model available. This is a one-time download during setup.

If the download fails (e.g. network issues), the system falls back to **buffalo_l** (ResNet50) which still provides good accuracy.

## Configuration

Edit `backend/config.yaml` to adjust settings:

```yaml
face_recognition:
  det_size: [1280, 1280]       # detection resolution (higher = better range, more CPU)
  recognition_threshold: 0.36   # matching strictness (lower = stricter)

camera:
  usb_device_id: 0             # camera device index
  fps_limit: 2                 # frames processed per second
```

### Performance Tuning

| Goal | Setting |
|------|---------|
| Better accuracy at distance | Increase `det_size` to `[1920, 1920]` |
| Faster on weak hardware | Decrease `det_size` to `[640, 640]`, increase `frame_skip` |
| Stricter matching | Lower `recognition_threshold` to `0.30` |
| More lenient matching | Raise `recognition_threshold` to `0.45` |

## Project Structure

```
production_1/
├── backend/
│   ├── app/
│   │   ├── api/            # REST endpoints
│   │   ├── core/           # config, security
│   │   ├── models/         # Pydantic schemas
│   │   ├── services/       # face engine, attendance, enrollment
│   │   └── utils/          # logging
│   ├── data/               # attendance records + face embeddings (gitignored)
│   ├── main.py             # FastAPI app entry point
│   ├── config.yaml         # app configuration
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/          # React page components
│   │   ├── components/     # shared UI components
│   │   ├── services/       # API client (axios)
│   │   ├── context/        # auth context
│   │   └── hooks/          # custom React hooks
│   ├── .env.example        # environment template
│   └── package.json
├── setup.sh / setup.bat              # one-time setup
├── start.sh / start.bat              # start both servers
├── upgrade_model.sh / upgrade_model.bat  # optional: install antelopev2
└── README.md
```

## Troubleshooting

**`onnxruntime` won't install** - Make sure you're using Python 3.11, not 3.12+.

**Model fails to download** - Download manually:
```bash
mkdir -p ~/.insightface/models
python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']).prepare(ctx_id=-1, det_size=(640,640))"
```

**Camera not detected** - Check `usb_device_id` in `config.yaml`. Try `0`, `1`, or `2`.

**Port already in use** - Kill existing processes:
```bash
# macOS/Linux
lsof -ti:8000 | xargs kill -9
lsof -ti:5173 | xargs kill -9

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

## Tech Stack

- **Backend**: FastAPI, InsightFace, OpenCV, NumPy, ONNX Runtime
- **Frontend**: React 19, TypeScript, Vite, Recharts, Axios
- **Auth**: JWT (python-jose + bcrypt)
- **Storage**: JSON files (attendance), NPZ (face embeddings)
