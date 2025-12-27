# Face Recognition Attendance System

A production-grade, real-time face recognition attendance system built with FastAPI and React. This system uses state-of-the-art InsightFace models (SCRFD detector + ArcFace embeddings) to detect and identify students from live camera feeds, automatically marking attendance with timestamps.

## 🎯 Features

### Core Functionality
- **Real-time Face Detection & Recognition**: Live camera feed processing with face detection and identification
- **Automatic Attendance Marking**: One entry per student per day with timestamp
- **Student Enrollment**: Guided 15-image capture process with pose instructions
- **Attendance Management**: View daily records and export to Excel format
- **High Accuracy**: ArcFace 512D embeddings with configurable similarity thresholds
- **Enhanced Detection Range**: 1280x1280 detection size for long-distance face recognition

### Technical Highlights
- **On-Device Processing**: No cloud dependencies, all processing runs locally
- **CPU Optimized**: Efficient inference on CPU with configurable frame rates
- **Persistent Storage**: Face embeddings stored as .npz files, attendance in JSON format
- **Modern UI**: Clean, responsive React interface with real-time updates
- **Full-Stack Architecture**: FastAPI backend with React TypeScript frontend

## 📋 System Requirements

### Hardware
- **Minimum**: 8GB RAM, Quad-core CPU
- **Recommended**: 16GB RAM, Multi-core CPU
- **Camera**: USB webcam or RTSP-compatible IP camera

### Software
- **Python**: 3.11 (required for onnxruntime compatibility)
- **Node.js**: 16+ with npm
- **Operating System**: macOS, Linux, or Windows

## 🚀 Quick Start

### Option 1: One-Command Start (Recommended)

```bash
cd /Users/daivik2/Desktop/face_recognition
./start.sh
```

This single script will:
- ✅ Check for Python 3.11 and Node.js
- ✅ Create virtual environment (if needed)
- ✅ Install all dependencies (if needed)
- ✅ Start both backend and frontend servers
- ✅ Open the application at `http://localhost:5173`

Press `Ctrl+C` to stop both servers.

### Option 2: Start Servers Separately

**Backend:**
```bash
./start_backend.sh
```

**Frontend (in a new terminal):**
```bash
./start_frontend.sh
```

### Option 3: Manual Setup

#### Backend
```bash
cd backend
python3.11 -m venv venv
source venv/bin/activate
pip install fastapi 'uvicorn[standard]' python-multipart pydantic pydantic-settings pyyaml opencv-python numpy pandas openpyxl python-dateutil insightface onnxruntime
python main.py
```

#### Frontend (in a new terminal)
```bash
cd frontend
npm install
npm run dev
```

### Access the Application
Open your browser and navigate to `http://localhost:5173`

## 📖 User Guide

### Enrolling Students

1. Click **"Enroll Student"** in the navigation bar
2. Fill in student details:
   - Student ID (e.g., `101`)
   - Full Name (e.g., `John Doe`)
   - Class (e.g., `10A`)
3. Click **"Start Enrollment"**
4. Allow webcam access when prompted
5. Follow the on-screen pose instructions:
   - Center (3 images)
   - Left (3 images)
   - Right (3 images)
   - Up (3 images)
   - Down (3 images)
6. Click **"Capture Image"** for each pose
7. Wait for "Enrollment completed successfully!" message

### Live Recognition

1. Click **"Live Recognition"** in the navigation bar
2. Click **"Start Camera"**
3. Allow webcam access if prompted
4. The system will:
   - Detect faces in real-time (green box for recognized, red for unknown)
   - Display student name and confidence score
   - Automatically mark attendance (once per day)
5. Click **"Stop Camera"** when done

### Viewing Attendance

1. Click **"Attendance"** in the navigation bar
2. View today's attendance records
3. Use the date picker to view records from other dates
4. Click **"Export Excel"** to download attendance report
5. Click **"Refresh"** to update the data

## ⚙️ Configuration

### Backend Configuration (`backend/config.yaml`)

```yaml
face_recognition:
  detector: "SCRFD"              # Face detection model
  embedding_model: "ArcFace"     # Face recognition model
  embedding_dim: 512             # Embedding dimension
  det_size: [1280, 1280]         # Detection input size (higher = better range)
  min_detection_score: 0.35      # Minimum face detection confidence
  recognition_threshold: 0.35    # Face matching threshold (lower = stricter)
  device: "cpu"                  # Processing device

camera:
  rtsp_url: ""                   # RTSP camera URL (optional)
  usb_device_id: 0               # USB camera ID (default: 0)
  fps_limit: 2                   # Frame processing rate
  frame_skip: 1                  # Process every Nth frame

enrollment:
  required_images: 15            # Total images per student
  poses:                         # Required poses
    - "center"
    - "left"
    - "right"
    - "up"
    - "down"
  min_face_quality: 0.7          # Minimum face quality score

attendance:
  data_dir: "data/attendance"    # Attendance records directory
  export_dir: "data/attendance/exports"  # Excel exports directory

embeddings:
  storage_dir: "data/embeddings" # Face embeddings directory
  gallery_file: "gallery.npz"    # Embeddings database file
```

### Adjusting Detection Range

To detect faces at greater distances, increase the `det_size`:
- **640x640**: Standard range (default for most systems)
- **1280x1280**: Extended range (current setting, 2-3x better)
- **1920x1920**: Maximum range (requires more CPU power)

### Adjusting Recognition Sensitivity

- **Lower threshold (0.3-0.35)**: Stricter matching, fewer false positives
- **Higher threshold (0.4-0.5)**: More lenient matching, may increase false positives

## 🏗️ Project Structure

```
face_recognition/
├── backend/
│   ├── app/
│   │   ├── api/              # API endpoints
│   │   │   ├── health.py
│   │   │   ├── enrollment.py
│   │   │   ├── recognition.py
│   │   │   └── attendance.py
│   │   ├── core/             # Core configuration
│   │   │   └── config.py
│   │   ├── models/           # Data models
│   │   │   └── schemas.py
│   │   ├── services/         # Business logic
│   │   │   ├── face_engine.py
│   │   │   ├── enrollment_service.py
│   │   │   ├── recognition_service.py
│   │   │   └── attendance_service.py
│   │   └── utils/            # Utilities
│   │       └── logger.py
│   ├── data/                 # Data storage
│   │   ├── embeddings/       # Face embeddings
│   │   └── attendance/       # Attendance records
│   ├── main.py               # Application entry point
│   ├── config.yaml           # Configuration file
│   └── requirements.txt      # Python dependencies
│
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   │   └── Navbar.tsx
│   │   ├── pages/            # Page components
│   │   │   ├── LiveRecognition.tsx
│   │   │   ├── EnrollStudent.tsx
│   │   │   └── AttendanceDashboard.tsx
│   │   ├── services/         # API client
│   │   │   └── api.ts
│   │   ├── hooks/            # Custom hooks
│   │   │   └── useWebcam.ts
│   │   ├── App.tsx           # Main app component
│   │   ├── App.css           # Global styles
│   │   └── index.css         # Base styles
│   └── package.json          # Node dependencies
│
└── README.md                 # This file
```

## 🔧 API Documentation

### Base URL
`http://localhost:8000/api`

### Endpoints

#### Health Check
```
GET /health
Response: { "status": "healthy", "timestamp": "...", "version": "1.0.0" }
```

#### Enrollment
```
POST /enroll/start
Body: { "student_id": "101", "name": "John Doe", "class": "10A" }

POST /enroll/capture
Body: { "student_id": "101", "image": "base64_image_data", "pose": "center" }

POST /enroll/complete
Body: { "student_id": "101" }

POST /enroll/cancel
Body: { "student_id": "101" }
```

#### Recognition
```
POST /recognition/start
Body: { "device_id": 0 }

GET /recognition/live
Response: { "frame": "base64_image", "results": [...] }

POST /recognition/stop
```

#### Attendance
```
GET /attendance/today
Response: [{ "student_id": "101", "name": "John Doe", ... }]

GET /attendance/date?date=2025-12-27
Response: [{ "student_id": "101", "name": "John Doe", ... }]

POST /attendance/mark
Body: { "student_id": "101", "name": "John Doe", "class": "10A", "confidence": 0.85 }

GET /attendance/export?date=2025-12-27
Response: Excel file download
```

### Interactive API Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🛠️ Troubleshooting

### Backend Issues

**Issue**: `onnxruntime` installation fails
- **Solution**: Ensure you're using Python 3.11. Python 3.14+ is not compatible.
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install onnxruntime
```

**Issue**: InsightFace models not downloading
- **Solution**: Download manually:
```bash
mkdir -p ~/.insightface/models
curl -L -o ~/.insightface/models/buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
unzip ~/.insightface/models/buffalo_l.zip -d ~/.insightface/models/
rm ~/.insightface/models/buffalo_l.zip
```

**Issue**: Port 8000 already in use
- **Solution**: Kill the existing process:
```bash
lsof -ti:8000 | xargs kill -9
```

### Frontend Issues

**Issue**: Port 5173 already in use
- **Solution**: Kill the existing process:
```bash
lsof -ti:5173 | xargs kill -9
```

**Issue**: `npm run dev` fails with "Missing script"
- **Solution**: Reinstall dependencies:
```bash
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### Camera Issues

**Issue**: Camera not detected
- **Solution**: Check camera permissions and device ID in `config.yaml`

**Issue**: Low frame rate
- **Solution**: Reduce `det_size` in `config.yaml` or increase `frame_skip`

## 📊 Performance Optimization

### For Better Accuracy
- Increase `det_size` to 1920x1920
- Lower `recognition_threshold` to 0.3
- Ensure good lighting during enrollment
- Capture enrollment images from multiple angles

### For Better Speed
- Decrease `det_size` to 640x640
- Increase `frame_skip` to 2 or 3
- Lower `fps_limit` to 1
- Use USB camera instead of RTSP

### For Edge Devices (Raspberry Pi)
```yaml
face_recognition:
  det_size: [640, 640]
  device: "cpu"
camera:
  fps_limit: 1
  frame_skip: 2
```

## 🔒 Security Considerations

- Face embeddings are stored locally in `.npz` format
- No data is sent to external servers
- Attendance records are stored in JSON format
- Excel exports contain only attendance data
- Camera feed is processed in real-time and not recorded

## 📝 Data Storage

### Face Embeddings
- **Location**: `backend/data/embeddings/gallery.npz`
- **Format**: NumPy compressed array
- **Contents**: Student ID, name, class, and 512D embedding vector

### Attendance Records
- **Location**: `backend/data/attendance/attendance_YYYY-MM-DD.json`
- **Format**: JSON
- **Contents**: Student ID, name, class, timestamp, confidence score

### Excel Exports
- **Location**: `backend/data/attendance/exports/`
- **Format**: .xlsx
- **Contents**: Formatted attendance report with student details

## 🤝 Contributing

This is a production-ready system. For modifications:
1. Backend changes: Modify files in `backend/app/`
2. Frontend changes: Modify files in `frontend/src/`
3. Configuration: Update `backend/config.yaml`
4. Restart servers to apply changes

## 📄 License

This project is for educational and institutional use.

## 🙏 Acknowledgments

- **InsightFace**: Face detection and recognition models
- **FastAPI**: Modern Python web framework
- **React**: Frontend framework
- **Vite**: Build tool and dev server

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review API documentation at `http://localhost:8000/docs`
3. Check browser console for frontend errors
4. Check backend logs for API errors

---

**Built with ❤️ for automated attendance management**
