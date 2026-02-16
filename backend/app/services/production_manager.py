"""
Production Manager - Handles multi-camera streams with GPU acceleration.
Supports up to 32+ cameras with auto-reconnect, face recognition, and auto-attendance.
Integrates optimized detection and scalable processing.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Callable, Any
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import queue

import cv2
import numpy as np

from app.services.optimized_detector import OptimizedDetector, DetectionQuality, DetectedFace
from app.services.scalable_processor import (
    ScalableProcessor, ProcessingPriority, ProcessingResult, ScalableConfig,
    MotionDetector,
)
from app.services.embedding_stabilizer import EmbeddingStabilizer, StabilizerConfig

logger = logging.getLogger(__name__)


class CameraStatus(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    PAUSED = "paused"


class ProcessingMode(str, Enum):
    CPU = "cpu"
    GPU_CUDA = "cuda"
    GPU_OPENCL = "opencl"


@dataclass
class CameraConfig:
    """Configuration for a single camera."""
    id: int
    name: str
    stream_url: str
    enabled: bool = True
    fps_limit: int = 5
    resolution_scale: float = 1.0
    detection_enabled: bool = True
    recording_enabled: bool = False


@dataclass
class CameraState:
    """Runtime state for a single camera."""
    config: CameraConfig
    status: CameraStatus = CameraStatus.DISCONNECTED
    last_frame: Optional[np.ndarray] = None
    last_frame_time: float = 0
    fps: float = 0
    error_message: Optional[str] = None
    detection_count: int = 0
    frame_count: int = 0
    # Reconnection state
    reconnect_attempts: int = 0
    max_reconnect_attempts: int = 50
    last_successful_frame_time: float = 0
    consecutive_empty_frames: int = 0
    frozen_frame_hash: Optional[int] = None


@dataclass
class ProductionConfig:
    """Production deployment configuration."""
    max_cameras: int = 32
    processing_mode: ProcessingMode = ProcessingMode.CPU
    gpu_device_id: int = 0
    worker_threads: int = 4
    frame_buffer_size: int = 2
    detection_interval_ms: int = 200
    enable_recording: bool = False
    recording_path: str = "data/recordings"
    enable_alerts: bool = True
    alert_cooldown_seconds: int = 30
    # Reconnection config
    reconnect_base_delay: float = 1.0
    reconnect_max_delay: float = 30.0
    reconnect_max_attempts: int = 50
    frozen_frame_threshold: int = 30
    empty_frame_tolerance: int = 50


class ProductionManager:
    """
    Manages multiple camera streams for production deployment.
    Supports GPU acceleration, auto-reconnect, face recognition, and auto-attendance.
    """

    def __init__(self):
        self.config = ProductionConfig()
        self.cameras: Dict[int, CameraState] = {}
        self.running = False
        self.lock = threading.RLock()
        self.executor: Optional[ThreadPoolExecutor] = None
        self.frame_queues: Dict[int, queue.Queue] = {}
        self._processing_threads: Dict[int, threading.Thread] = {}
        self._gpu_available = self._check_gpu_availability()

        # Advanced components (initialized lazily)
        self._optimized_detector: Optional[OptimizedDetector] = None
        self._scalable_processor: Optional[ScalableProcessor] = None
        self._embedding_stabilizer: Optional[EmbeddingStabilizer] = None
        self._detection_quality = DetectionQuality.BALANCED

        # Face engine and attendance
        self._face_engine: Optional[Any] = None
        self._attendance_service: Optional[Any] = None

        # Recognition state
        self._recognition_thread: Optional[threading.Thread] = None
        self._recognition_callbacks: List[Callable] = []
        self._latest_detections: Dict[int, List[DetectedFace]] = {}
        self._latest_recognitions: Dict[int, List[dict]] = {}
        self._attendance_cooldowns: Dict[str, float] = {}
        self._recent_attendance_log: deque = deque(maxlen=100)
        self._motion_detector = MotionDetector(threshold=25.0)
        self._last_detection_times: Dict[int, float] = {}

    def _check_gpu_availability(self) -> dict:
        """Check available GPU acceleration options."""
        result = {
            "cuda_available": False,
            "cuda_devices": [],
            "opencl_available": False,
            "opencl_devices": [],
        }

        try:
            cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
            if cuda_count > 0:
                result["cuda_available"] = True
                for i in range(cuda_count):
                    cv2.cuda.setDevice(i)
                    result["cuda_devices"].append({
                        "id": i,
                        "name": f"CUDA Device {i}",
                    })
        except Exception as e:
            logger.debug(f"CUDA not available: {e}")

        try:
            if cv2.ocl.haveOpenCL():
                result["opencl_available"] = True
                cv2.ocl.setUseOpenCL(True)
                result["opencl_devices"].append({
                    "id": 0,
                    "name": "Default OpenCL Device",
                })
        except Exception as e:
            logger.debug(f"OpenCL not available: {e}")

        return result

    def get_gpu_info(self) -> dict:
        """Get GPU availability and device information."""
        return {
            **self._gpu_available,
            "current_mode": self.config.processing_mode.value,
            "gpu_device_id": self.config.gpu_device_id,
            "detection_quality": self._detection_quality.value,
        }

    def initialize_advanced_processing(self, face_engine):
        """Initialize optimized detector, scalable processor, stabilizer, and face engine."""
        try:
            self._face_engine = face_engine
            self._optimized_detector = OptimizedDetector(face_engine)
            self._optimized_detector.set_quality(self._detection_quality)

            self._scalable_processor = ScalableProcessor(face_engine, self._optimized_detector)
            self._scalable_processor.add_result_callback(self._on_detection_result)

            # Embedding stabilizer for temporal aggregation (long-range accuracy)
            from app.core.config import settings
            stabilizer_config = StabilizerConfig(
                enabled=settings.face_recognition.stabilizer_enabled,
                min_frames_for_recognition=settings.face_recognition.stabilizer_min_frames,
                alpha_base=settings.face_recognition.stabilizer_alpha,
                min_consistency_sim=settings.face_recognition.stabilizer_min_consistency,
            )
            self._embedding_stabilizer = EmbeddingStabilizer(stabilizer_config)
            logger.info(
                f"Embedding stabilizer initialized "
                f"(enabled={stabilizer_config.enabled}, "
                f"min_frames={stabilizer_config.min_frames_for_recognition})"
            )

            logger.info("Advanced processing components initialized")
        except Exception as e:
            logger.error(f"Failed to initialize advanced processing: {e}")

    def set_attendance_service(self, attendance_service):
        """Inject attendance service for auto-marking."""
        self._attendance_service = attendance_service
        logger.info("Attendance service connected to production manager")

    def set_detection_quality(self, quality: str) -> dict:
        """Set detection quality preset."""
        try:
            new_quality = DetectionQuality(quality)
            self._detection_quality = new_quality

            if self._optimized_detector:
                self._optimized_detector.set_quality(new_quality)

            return {"success": True, "quality": new_quality.value}
        except ValueError:
            return {"success": False, "error": f"Invalid quality: {quality}"}

    def get_detection_qualities(self) -> List[dict]:
        """Get available detection quality presets."""
        return [
            {"id": "fast", "name": "Fast", "description": "Low latency, good for close range"},
            {"id": "balanced", "name": "Balanced", "description": "Balance between speed and accuracy"},
            {"id": "accurate", "name": "Accurate", "description": "High accuracy, good for long range"},
            {"id": "maximum", "name": "Maximum", "description": "Maximum quality with multi-scale detection"},
        ]

    def _on_detection_result(self, result: ProcessingResult):
        """Callback for detection results from scalable processor."""
        with self.lock:
            self._latest_detections[result.camera_id] = result.faces

            if result.camera_id in self.cameras:
                camera = self.cameras[result.camera_id]
                camera.detection_count += len(result.faces)
                camera.frame_count += 1

        for callback in self._recognition_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Recognition callback error: {e}")

    def add_recognition_callback(self, callback: Callable):
        """Add callback for recognition results."""
        self._recognition_callbacks.append(callback)

    def get_camera_detections(self, camera_id: int) -> List[dict]:
        """Get latest detections for a camera."""
        with self.lock:
            faces = self._latest_detections.get(camera_id, [])
            return [
                {
                    "bbox": face.bbox,
                    "confidence": face.confidence,
                    "face_size": face.face_size,
                    "enhanced": face.enhanced,
                }
                for face in faces
            ]

    def get_camera_recognitions(self, camera_id: int) -> List[dict]:
        """Get latest recognition results for a camera (with student IDs)."""
        with self.lock:
            return self._latest_recognitions.get(camera_id, [])

    def get_recent_attendance_feed(self) -> List[dict]:
        """Get recent auto-attendance events across all cameras."""
        with self.lock:
            return list(self._recent_attendance_log)

    # ─── Stream Reconnection ──────────────────────────────────────────

    def _connect_stream(self, camera_id: int, stream_url: str) -> Optional[cv2.VideoCapture]:
        """Attempt to connect to stream with exponential backoff."""
        camera = self.cameras.get(camera_id)
        if not camera:
            return None

        camera.status = CameraStatus.CONNECTING

        # Exponential backoff delay
        delay = min(
            self.config.reconnect_base_delay * (2 ** camera.reconnect_attempts),
            self.config.reconnect_max_delay,
        )
        if camera.reconnect_attempts > 0:
            logger.info(
                f"Camera {camera_id}: Reconnect attempt {camera.reconnect_attempts}, "
                f"waiting {delay:.1f}s"
            )
            # Sleep in small increments so we can exit quickly on shutdown
            sleep_end = time.time() + delay
            while time.time() < sleep_end and self.running:
                time.sleep(0.5)
            if not self.running:
                return None

        # Force TCP transport for RTSP (more reliable than UDP)
        import os
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|rtsp_flags;prefer_tcp"
        
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Increased buffer for smoother streaming
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15000)  # Increased timeout
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 15000)

        if not cap.isOpened():
            cap.release()
            camera.reconnect_attempts += 1
            camera.error_message = f"Connection failed (attempt {camera.reconnect_attempts})"
            camera.status = CameraStatus.RECONNECTING
            return None

        # Verify we can actually read a frame
        ret, _ = cap.read()
        if not ret:
            cap.release()
            camera.reconnect_attempts += 1
            camera.error_message = f"Stream opened but no frames (attempt {camera.reconnect_attempts})"
            camera.status = CameraStatus.RECONNECTING
            return None

        return cap

    def _should_retry(self, camera: CameraState) -> bool:
        """Check if reconnection should be attempted."""
        if self.config.reconnect_max_attempts == 0:
            return True  # 0 = infinite retries
        return camera.reconnect_attempts < self.config.reconnect_max_attempts

    def _is_frozen_frame(self, camera: CameraState, frame: np.ndarray) -> bool:
        """Detect frozen stream by comparing frame hashes."""
        small = cv2.resize(frame, (16, 16))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        frame_hash = hash(gray.tobytes())

        if camera.frozen_frame_hash == frame_hash:
            return True
        camera.frozen_frame_hash = frame_hash
        return False

    def _compute_health(self, state: CameraState) -> str:
        """Return 'green', 'yellow', or 'red' health status."""
        if state.status == CameraStatus.CONNECTED and state.fps > 0:
            if state.consecutive_empty_frames > 10:
                return "yellow"
            return "green"
        elif state.status in (CameraStatus.RECONNECTING, CameraStatus.CONNECTING):
            return "yellow"
        elif state.status in (CameraStatus.ERROR, CameraStatus.DISCONNECTED):
            return "red"
        return "yellow"

    # ─── Recognition Processing ───────────────────────────────────────

    def _start_recognition_processor(self):
        """Start background thread that processes frames for face recognition."""
        self._recognition_thread = threading.Thread(
            target=self._recognition_loop,
            daemon=True,
            name="RecognitionProcessor",
        )
        self._recognition_thread.start()
        logger.info("Recognition processor started")

    def _recognition_loop(self):
        """Process frames from all cameras for face recognition."""
        while self.running:
            processed_any = False

            for camera_id in list(self.frame_queues.keys()):
                if camera_id not in self.cameras:
                    continue
                camera = self.cameras[camera_id]
                if not camera.config.detection_enabled:
                    continue

                # Detection interval limiting
                now = time.time()
                last_det = self._last_detection_times.get(camera_id, 0)
                if now - last_det < (self.config.detection_interval_ms / 1000.0):
                    continue

                try:
                    frame = self.frame_queues[camera_id].get_nowait()
                except queue.Empty:
                    continue

                # Skip static scenes for performance
                if not self._motion_detector.has_motion(camera_id, frame):
                    continue

                processed_any = True
                self._last_detection_times[camera_id] = now
                self._process_frame_for_recognition(camera_id, frame)

            if not processed_any:
                time.sleep(0.01)

    def _process_frame_for_recognition(self, camera_id: int, frame: np.ndarray):
        """Detect faces using optimized detector and recognize against gallery."""
        if not self._face_engine:
            return

        try:
            recognition_results = []

            # Use optimized detector if available (tiled detection, quality scoring)
            if self._optimized_detector:
                detected_faces = self._optimized_detector.detect(
                    frame, camera_id=camera_id, get_embeddings=True
                )

                # Pass through embedding stabilizer for temporal aggregation
                if self._embedding_stabilizer:
                    stabilized = self._embedding_stabilizer.process(
                        camera_id, detected_faces
                    )
                    for entry in stabilized:
                        face = entry["face"]
                        embedding = entry["stabilized_embedding"]

                        if embedding is None:
                            continue

                        student_id, confidence = self._face_engine.recognize(
                            embedding, quality_score=face.quality_score
                        )

                        result = {
                            "student_id": student_id,
                            "confidence": float(confidence),
                            "is_known": student_id is not None,
                            "bbox": list(face.bbox),
                            "camera_id": camera_id,
                            "timestamp": datetime.now().isoformat(),
                            "quality": face.quality_score,
                            "stabilizer_frames": entry["consistent_frames"],
                            "track_id": entry["track_id"],
                        }

                        if student_id and student_id in self._face_engine.gallery_metadata:
                            metadata = self._face_engine.gallery_metadata[student_id]
                            result["name"] = metadata.get("name")
                            result["class"] = metadata.get("class")

                        recognition_results.append(result)
                else:
                    # No stabilizer — direct recognition (original path)
                    for face in detected_faces:
                        if face.embedding is None:
                            continue

                        student_id, confidence = self._face_engine.recognize(
                            face.embedding, quality_score=face.quality_score
                        )

                        result = {
                            "student_id": student_id,
                            "confidence": float(confidence),
                            "is_known": student_id is not None,
                            "bbox": list(face.bbox),
                            "camera_id": camera_id,
                            "timestamp": datetime.now().isoformat(),
                            "quality": face.quality_score,
                        }

                        if student_id and student_id in self._face_engine.gallery_metadata:
                            metadata = self._face_engine.gallery_metadata[student_id]
                            result["name"] = metadata.get("name")
                            result["class"] = metadata.get("class")

                        recognition_results.append(result)
            else:
                # Fallback to basic detection
                faces = self._face_engine.detect_faces(frame)
                for face in faces:
                    embedding = getattr(face, "normed_embedding", None)
                    if embedding is None:
                        continue

                    embedding = np.asarray(embedding, dtype=np.float32)
                    student_id, confidence = self._face_engine.recognize(embedding)

                    bbox = face.bbox.astype(int).tolist()

                    result = {
                        "student_id": student_id,
                        "confidence": float(confidence),
                        "is_known": student_id is not None,
                        "bbox": bbox,
                        "camera_id": camera_id,
                        "timestamp": datetime.now().isoformat(),
                    }

                    if student_id and student_id in self._face_engine.gallery_metadata:
                        metadata = self._face_engine.gallery_metadata[student_id]
                        result["name"] = metadata.get("name")
                        result["class"] = metadata.get("class")

                    recognition_results.append(result)

            # Store results
            with self.lock:
                self._latest_recognitions[camera_id] = recognition_results
                if camera_id in self.cameras:
                    self.cameras[camera_id].detection_count += len(
                        [r for r in recognition_results if r["is_known"]]
                    )

            # Auto-mark attendance for recognized faces
            for result in recognition_results:
                if result["is_known"] and result["student_id"]:
                    self._handle_attendance(result, camera_id)

        except Exception as e:
            logger.error(f"Camera {camera_id}: Recognition error - {e}")

    def _handle_attendance(self, recognition_result: dict, camera_id: int):
        """Auto-mark attendance with cooldown per student per camera."""
        if not self._attendance_service:
            return

        student_id = recognition_result["student_id"]
        confidence = recognition_result["confidence"]

        # Cooldown check: prevent spam
        cooldown_key = f"{student_id}_{camera_id}"
        current_time = time.time()
        last_marked = self._attendance_cooldowns.get(cooldown_key, 0)

        if current_time - last_marked < self.config.alert_cooldown_seconds:
            return

        name = recognition_result.get("name", student_id)
        class_name = recognition_result.get("class", "Unknown")
        camera_name = (
            self.cameras[camera_id].config.name
            if camera_id in self.cameras
            else f"Camera {camera_id}"
        )

        try:
            success, message = self._attendance_service.mark_attendance(
                student_id=student_id,
                name=name,
                class_name=class_name,
                confidence=confidence,
                camera_id=camera_id,
                camera_name=camera_name,
            )

            if success:
                self._attendance_cooldowns[cooldown_key] = current_time
                log_entry = {
                    "student_id": student_id,
                    "name": name,
                    "class": class_name,
                    "confidence": confidence,
                    "camera_id": camera_id,
                    "camera_name": camera_name,
                    "timestamp": datetime.now().isoformat(),
                    "message": message,
                }
                with self.lock:
                    self._recent_attendance_log.appendleft(log_entry)
                logger.info(f"Auto-attendance: {name} ({student_id}) via {camera_name}")
        except Exception as e:
            logger.error(f"Attendance marking error: {e}")

    # ─── GPU / Processing Mode ────────────────────────────────────────

    def set_processing_mode(self, mode: str, device_id: int = 0) -> dict:
        """Set the processing mode (CPU/CUDA/OpenCL)."""
        try:
            new_mode = ProcessingMode(mode)

            if new_mode == ProcessingMode.GPU_CUDA and not self._gpu_available["cuda_available"]:
                return {"success": False, "error": "CUDA is not available on this system"}

            if new_mode == ProcessingMode.GPU_OPENCL and not self._gpu_available["opencl_available"]:
                return {"success": False, "error": "OpenCL is not available on this system"}

            self.config.processing_mode = new_mode
            self.config.gpu_device_id = device_id

            if new_mode == ProcessingMode.GPU_CUDA:
                cv2.cuda.setDevice(device_id)
                logger.info(f"Set CUDA device to {device_id}")
            elif new_mode == ProcessingMode.GPU_OPENCL:
                cv2.ocl.setUseOpenCL(True)
                logger.info("Enabled OpenCL acceleration")
            else:
                cv2.ocl.setUseOpenCL(False)
                logger.info("Using CPU processing")

            return {"success": True, "mode": new_mode.value, "device_id": device_id}
        except ValueError:
            return {"success": False, "error": f"Invalid processing mode: {mode}"}

    # ─── Configuration ────────────────────────────────────────────────

    def get_config(self) -> dict:
        """Get current production configuration."""
        return {
            "max_cameras": self.config.max_cameras,
            "processing_mode": self.config.processing_mode.value,
            "gpu_device_id": self.config.gpu_device_id,
            "worker_threads": self.config.worker_threads,
            "frame_buffer_size": self.config.frame_buffer_size,
            "detection_interval_ms": self.config.detection_interval_ms,
            "enable_recording": self.config.enable_recording,
            "recording_path": self.config.recording_path,
            "enable_alerts": self.config.enable_alerts,
            "alert_cooldown_seconds": self.config.alert_cooldown_seconds,
        }

    def update_config(self, updates: dict) -> dict:
        """Update production configuration."""
        with self.lock:
            if "max_cameras" in updates:
                self.config.max_cameras = min(updates["max_cameras"], 64)
            if "worker_threads" in updates:
                self.config.worker_threads = max(1, min(updates["worker_threads"], 16))
            if "frame_buffer_size" in updates:
                self.config.frame_buffer_size = max(1, min(updates["frame_buffer_size"], 10))
            if "detection_interval_ms" in updates:
                self.config.detection_interval_ms = max(50, updates["detection_interval_ms"])
            if "enable_recording" in updates:
                self.config.enable_recording = updates["enable_recording"]
            if "recording_path" in updates:
                self.config.recording_path = updates["recording_path"]
            if "enable_alerts" in updates:
                self.config.enable_alerts = updates["enable_alerts"]
            if "alert_cooldown_seconds" in updates:
                self.config.alert_cooldown_seconds = max(5, updates["alert_cooldown_seconds"])

        return self.get_config()

    # ─── Camera CRUD ──────────────────────────────────────────────────

    def add_camera(self, camera_config: dict) -> dict:
        """Add a new camera to the production system."""
        with self.lock:
            camera_id = camera_config.get("id")
            if camera_id is None:
                camera_id = len(self.cameras) + 1
                while camera_id in self.cameras:
                    camera_id += 1

            if camera_id > self.config.max_cameras:
                return {"success": False, "error": f"Maximum cameras ({self.config.max_cameras}) reached"}

            if camera_id in self.cameras:
                return {"success": False, "error": f"Camera {camera_id} already exists"}

            config = CameraConfig(
                id=camera_id,
                name=camera_config.get("name", f"Camera {camera_id}"),
                stream_url=camera_config.get("stream_url", ""),
                enabled=camera_config.get("enabled", True),
                fps_limit=camera_config.get("fps_limit", 5),
                resolution_scale=camera_config.get("resolution_scale", 1.0),
                detection_enabled=camera_config.get("detection_enabled", True),
                recording_enabled=camera_config.get("recording_enabled", False),
            )

            self.cameras[camera_id] = CameraState(config=config)
            self.frame_queues[camera_id] = queue.Queue(maxsize=self.config.frame_buffer_size)

            logger.info(f"Added camera {camera_id}: {config.name}")
            return {"success": True, "camera_id": camera_id, "config": self._camera_to_dict(config)}

    def remove_camera(self, camera_id: int) -> dict:
        """Remove a camera from the production system."""
        with self.lock:
            if camera_id not in self.cameras:
                return {"success": False, "error": f"Camera {camera_id} not found"}

            self._stop_camera_stream(camera_id)

            del self.cameras[camera_id]
            if camera_id in self.frame_queues:
                del self.frame_queues[camera_id]
            if camera_id in self._latest_recognitions:
                del self._latest_recognitions[camera_id]
            self._motion_detector.clear(camera_id)
            if self._embedding_stabilizer:
                self._embedding_stabilizer.clear_camera(camera_id)

            logger.info(f"Removed camera {camera_id}")
            return {"success": True, "camera_id": camera_id}

    def update_camera(self, camera_id: int, updates: dict) -> dict:
        """Update camera configuration."""
        with self.lock:
            if camera_id not in self.cameras:
                return {"success": False, "error": f"Camera {camera_id} not found"}

            camera = self.cameras[camera_id]
            config = camera.config

            if "name" in updates:
                config.name = updates["name"]
            if "stream_url" in updates:
                config.stream_url = updates["stream_url"]
                if camera.status == CameraStatus.CONNECTED:
                    self._stop_camera_stream(camera_id)
                    self._start_camera_stream(camera_id)
            if "enabled" in updates:
                config.enabled = updates["enabled"]
            if "fps_limit" in updates:
                config.fps_limit = max(1, min(updates["fps_limit"], 30))
            if "resolution_scale" in updates:
                config.resolution_scale = max(0.25, min(updates["resolution_scale"], 1.0))
            if "detection_enabled" in updates:
                config.detection_enabled = updates["detection_enabled"]
            if "recording_enabled" in updates:
                config.recording_enabled = updates["recording_enabled"]

            return {"success": True, "camera_id": camera_id, "config": self._camera_to_dict(config)}

    def get_camera(self, camera_id: int) -> Optional[dict]:
        """Get camera state and configuration."""
        with self.lock:
            if camera_id not in self.cameras:
                return None
            return self._camera_state_to_dict(self.cameras[camera_id])

    def get_all_cameras(self) -> List[dict]:
        """Get all cameras with their states."""
        with self.lock:
            return [self._camera_state_to_dict(cam) for cam in self.cameras.values()]

    # ─── Start / Stop ─────────────────────────────────────────────────

    def start_all(self) -> dict:
        """Start all enabled cameras and recognition processor."""
        if self.running:
            return {"success": False, "error": "Production system already running"}

        self.running = True
        self.executor = ThreadPoolExecutor(max_workers=self.config.worker_threads)

        started = 0
        for camera_id, camera in self.cameras.items():
            if camera.config.enabled:
                self._start_camera_stream(camera_id)
                started += 1

        # Start recognition processor thread
        self._start_recognition_processor()

        logger.info(f"Started production system with {started} cameras")
        return {"success": True, "cameras_started": started}

    def stop_all(self) -> dict:
        """Stop all cameras and recognition processor."""
        self.running = False

        for camera_id in list(self.cameras.keys()):
            self._stop_camera_stream(camera_id)

        if self._recognition_thread and self._recognition_thread.is_alive():
            self._recognition_thread.join(timeout=3.0)

        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None

        logger.info("Stopped production system")
        return {"success": True}

    def start_camera(self, camera_id: int) -> dict:
        """Start a specific camera."""
        with self.lock:
            if camera_id not in self.cameras:
                return {"success": False, "error": f"Camera {camera_id} not found"}

            self._start_camera_stream(camera_id)
            return {"success": True, "camera_id": camera_id}

    def stop_camera(self, camera_id: int) -> dict:
        """Stop a specific camera."""
        with self.lock:
            if camera_id not in self.cameras:
                return {"success": False, "error": f"Camera {camera_id} not found"}

            self._stop_camera_stream(camera_id)
            return {"success": True, "camera_id": camera_id}

    def _start_camera_stream(self, camera_id: int):
        """Internal: Start camera stream processing."""
        camera = self.cameras[camera_id]
        camera.status = CameraStatus.CONNECTING
        camera.reconnect_attempts = 0

        thread = threading.Thread(
            target=self._camera_loop,
            args=(camera_id,),
            daemon=True,
            name=f"Camera-{camera_id}",
        )
        self._processing_threads[camera_id] = thread
        thread.start()

    def _stop_camera_stream(self, camera_id: int):
        """Internal: Stop camera stream processing."""
        if camera_id in self.cameras:
            self.cameras[camera_id].status = CameraStatus.DISCONNECTED

        if camera_id in self._processing_threads:
            thread = self._processing_threads[camera_id]
            if thread.is_alive():
                thread.join(timeout=3.0)
            del self._processing_threads[camera_id]

    # ─── Camera Loop with Auto-Reconnect ──────────────────────────────

    def _camera_loop(self, camera_id: int):
        """Main loop with auto-reconnect and health monitoring."""
        camera = self.cameras.get(camera_id)
        if not camera:
            return

        config = camera.config

        while self.running and camera.status != CameraStatus.DISCONNECTED:
            # ── Connect / Reconnect Phase ──
            cap = self._connect_stream(camera_id, config.stream_url)
            if cap is None:
                if not self.running or camera.status == CameraStatus.DISCONNECTED:
                    return
                if not self._should_retry(camera):
                    camera.status = CameraStatus.ERROR
                    camera.error_message = (
                        f"Max reconnect attempts ({camera.reconnect_attempts}) reached"
                    )
                    logger.error(
                        f"Camera {camera_id}: Giving up after "
                        f"{camera.reconnect_attempts} attempts"
                    )
                    return
                continue

            # Connection successful - reset state
            camera.status = CameraStatus.CONNECTED
            camera.error_message = None
            camera.reconnect_attempts = 0
            camera.consecutive_empty_frames = 0
            camera.last_successful_frame_time = time.time()
            logger.info(f"Camera {camera_id}: Connected")

            # ── Capture Phase ──
            frame_interval = 1.0 / config.fps_limit
            last_frame_time = 0
            fps_counter = 0
            fps_start_time = time.time()
            consecutive_failures = 0

            try:
                while camera.status == CameraStatus.CONNECTED and self.running:
                    current_time = time.time()

                    if current_time - last_frame_time < frame_interval:
                        time.sleep(0.01)
                        continue

                    ret, frame = cap.read()
                    if not ret or frame is None:
                        consecutive_failures += 1
                        if consecutive_failures >= self.config.empty_frame_tolerance:
                            logger.warning(
                                f"Camera {camera_id}: {consecutive_failures} empty frames, "
                                f"reconnecting"
                            )
                            break  # Break to outer loop for reconnect
                        time.sleep(0.01)  # Reduced from 0.05 for better responsiveness
                        continue

                    consecutive_failures = 0
                    
                    # Flush buffer to get latest frame (discard old buffered frames)
                    cap.grab()  # Grab and discard one extra frame to keep buffer fresh

                    # Frozen frame detection
                    if self._is_frozen_frame(camera, frame):
                        camera.consecutive_empty_frames += 1
                        if camera.consecutive_empty_frames >= self.config.frozen_frame_threshold:
                            logger.warning(
                                f"Camera {camera_id}: Frozen stream detected, reconnecting"
                            )
                            break
                    else:
                        camera.consecutive_empty_frames = 0

                    # Scale frame if needed
                    if config.resolution_scale < 1.0:
                        new_w = int(frame.shape[1] * config.resolution_scale)
                        new_h = int(frame.shape[0] * config.resolution_scale)
                        frame = cv2.resize(frame, (new_w, new_h))

                    # Update state
                    with self.lock:
                        camera.last_frame = frame
                        camera.last_frame_time = current_time
                        camera.last_successful_frame_time = current_time
                        camera.frame_count += 1

                    # Put frame in queue for recognition processing
                    try:
                        self.frame_queues[camera_id].put_nowait(frame)
                    except queue.Full:
                        pass  # Drop frame if queue is full

                    last_frame_time = current_time
                    fps_counter += 1

                    # Calculate FPS every second
                    if current_time - fps_start_time >= 1.0:
                        camera.fps = fps_counter / (current_time - fps_start_time)
                        fps_counter = 0
                        fps_start_time = current_time

            except Exception as e:
                logger.error(f"Camera {camera_id}: Error in capture loop - {e}")
            finally:
                cap.release()

            # If we broke out while still running, enter reconnect cycle
            if self.running and camera.status != CameraStatus.DISCONNECTED:
                camera.status = CameraStatus.RECONNECTING
                camera.fps = 0
                logger.info(f"Camera {camera_id}: Entering reconnect cycle")

    # ─── Frame / Stats ────────────────────────────────────────────────

    def get_frame(self, camera_id: int) -> Optional[np.ndarray]:
        """Get the latest frame from a camera."""
        with self.lock:
            if camera_id not in self.cameras:
                return None
            return self.cameras[camera_id].last_frame

    def get_system_stats(self) -> dict:
        """Get overall system statistics."""
        with self.lock:
            total_cameras = len(self.cameras)
            connected = sum(
                1 for c in self.cameras.values() if c.status == CameraStatus.CONNECTED
            )
            errored = sum(
                1 for c in self.cameras.values() if c.status == CameraStatus.ERROR
            )
            reconnecting = sum(
                1 for c in self.cameras.values() if c.status == CameraStatus.RECONNECTING
            )
            total_fps = sum(c.fps for c in self.cameras.values())
            total_frames = sum(c.frame_count for c in self.cameras.values())
            total_detections = sum(c.detection_count for c in self.cameras.values())

            return {
                "running": self.running,
                "total_cameras": total_cameras,
                "connected_cameras": connected,
                "reconnecting_cameras": reconnecting,
                "errored_cameras": errored,
                "total_fps": round(total_fps, 1),
                "total_frames_processed": total_frames,
                "total_detections": total_detections,
                "processing_mode": self.config.processing_mode.value,
                "worker_threads": self.config.worker_threads,
                "recent_attendance_count": len(self._recent_attendance_log),
                "stabilizer_stats": (
                    self._embedding_stabilizer.get_stats()
                    if self._embedding_stabilizer
                    else None
                ),
            }

    # ─── Serialization ────────────────────────────────────────────────

    def _camera_to_dict(self, config: CameraConfig) -> dict:
        """Convert CameraConfig to dict."""
        return {
            "id": config.id,
            "name": config.name,
            "stream_url": self._mask_url(config.stream_url),
            "enabled": config.enabled,
            "fps_limit": config.fps_limit,
            "resolution_scale": config.resolution_scale,
            "detection_enabled": config.detection_enabled,
            "recording_enabled": config.recording_enabled,
        }

    def _camera_state_to_dict(self, state: CameraState) -> dict:
        """Convert CameraState to dict."""
        return {
            **self._camera_to_dict(state.config),
            "status": state.status.value,
            "fps": round(state.fps, 1),
            "error_message": state.error_message,
            "detection_count": state.detection_count,
            "frame_count": state.frame_count,
            "has_frame": state.last_frame is not None,
            "reconnect_attempts": state.reconnect_attempts,
            "last_successful_frame_time": state.last_successful_frame_time,
            "health": self._compute_health(state),
        }

    def _mask_url(self, url: str) -> str:
        """Mask credentials in URL."""
        if "@" in url:
            parts = url.split("@")
            protocol = parts[0].split("://")[0] if "://" in parts[0] else "rtsp"
            return f"{protocol}://***:***@{parts[-1]}"
        return url


# Singleton instance
production_manager = ProductionManager()
