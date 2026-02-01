"""
Production Manager - Handles multi-camera streams with GPU acceleration.
Supports up to 32 cameras with configurable processing pipelines.
Integrates optimized detection and scalable processing.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import queue

import cv2
import numpy as np

from app.services.optimized_detector import OptimizedDetector, DetectionQuality, DetectedFace
from app.services.scalable_processor import (
    ScalableProcessor, ProcessingPriority, ProcessingResult, ScalableConfig
)

logger = logging.getLogger(__name__)


class CameraStatus(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
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
    resolution_scale: float = 1.0  # Scale down for performance
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


@dataclass
class ProductionConfig:
    """Production deployment configuration."""
    max_cameras: int = 32
    processing_mode: ProcessingMode = ProcessingMode.CPU
    gpu_device_id: int = 0
    worker_threads: int = 4
    frame_buffer_size: int = 2
    detection_interval_ms: int = 200  # Run detection every N ms
    enable_recording: bool = False
    recording_path: str = "data/recordings"
    enable_alerts: bool = True
    alert_cooldown_seconds: int = 30


class ProductionManager:
    """
    Manages multiple camera streams for production deployment.
    Supports GPU acceleration, scalable processing, and state-of-the-art detection.
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
        self._detection_quality = DetectionQuality.BALANCED
        
        # Recognition callbacks
        self._recognition_callbacks: List[Callable] = []
        
        # Detection results cache
        self._latest_detections: Dict[int, List[DetectedFace]] = {}

    def _check_gpu_availability(self) -> dict:
        """Check available GPU acceleration options."""
        result = {
            "cuda_available": False,
            "cuda_devices": [],
            "opencl_available": False,
            "opencl_devices": [],
        }

        # Check CUDA
        try:
            cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
            if cuda_count > 0:
                result["cuda_available"] = True
                for i in range(cuda_count):
                    cv2.cuda.setDevice(i)
                    props = cv2.cuda.getDevice()
                    result["cuda_devices"].append({
                        "id": i,
                        "name": f"CUDA Device {i}",
                    })
        except Exception as e:
            logger.debug(f"CUDA not available: {e}")

        # Check OpenCL
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
        """Initialize optimized detector and scalable processor."""
        try:
            self._optimized_detector = OptimizedDetector(face_engine)
            self._optimized_detector.set_quality(self._detection_quality)
            
            self._scalable_processor = ScalableProcessor(face_engine, self._optimized_detector)
            self._scalable_processor.add_result_callback(self._on_detection_result)
            
            logger.info("Advanced processing components initialized")
        except Exception as e:
            logger.error(f"Failed to initialize advanced processing: {e}")

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
            
            # Update camera state
            if result.camera_id in self.cameras:
                camera = self.cameras[result.camera_id]
                camera.detection_count += len(result.faces)
                camera.frame_count += 1
        
        # Notify external callbacks
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

            # Apply GPU settings
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

    def add_camera(self, camera_config: dict) -> dict:
        """Add a new camera to the production system."""
        with self.lock:
            camera_id = camera_config.get("id")
            if camera_id is None:
                # Auto-assign ID
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

            # Stop camera if running
            self._stop_camera_stream(camera_id)

            del self.cameras[camera_id]
            if camera_id in self.frame_queues:
                del self.frame_queues[camera_id]

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
                # Reconnect if URL changed
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

    def start_all(self) -> dict:
        """Start all enabled cameras."""
        if self.running:
            return {"success": False, "error": "Production system already running"}

        self.running = True
        self.executor = ThreadPoolExecutor(max_workers=self.config.worker_threads)

        started = 0
        for camera_id, camera in self.cameras.items():
            if camera.config.enabled:
                self._start_camera_stream(camera_id)
                started += 1

        logger.info(f"Started production system with {started} cameras")
        return {"success": True, "cameras_started": started}

    def stop_all(self) -> dict:
        """Stop all cameras."""
        self.running = False

        for camera_id in list(self.cameras.keys()):
            self._stop_camera_stream(camera_id)

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

        thread = threading.Thread(
            target=self._camera_loop,
            args=(camera_id,),
            daemon=True,
            name=f"Camera-{camera_id}"
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
                thread.join(timeout=2.0)
            del self._processing_threads[camera_id]

    def _camera_loop(self, camera_id: int):
        """Internal: Main loop for camera stream processing."""
        camera = self.cameras[camera_id]
        config = camera.config

        cap = cv2.VideoCapture(config.stream_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            camera.status = CameraStatus.ERROR
            camera.error_message = "Failed to open stream"
            logger.error(f"Camera {camera_id}: Failed to open stream")
            return

        camera.status = CameraStatus.CONNECTED
        camera.error_message = None
        logger.info(f"Camera {camera_id}: Connected")

        frame_interval = 1.0 / config.fps_limit
        last_frame_time = 0
        fps_counter = 0
        fps_start_time = time.time()

        try:
            while camera.status == CameraStatus.CONNECTED and self.running:
                current_time = time.time()

                # Rate limiting
                if current_time - last_frame_time < frame_interval:
                    time.sleep(0.01)
                    continue

                ret, frame = cap.read()
                if not ret or frame is None:
                    camera.error_message = "No frames received"
                    time.sleep(0.1)
                    continue

                # Scale frame if needed
                if config.resolution_scale < 1.0:
                    new_width = int(frame.shape[1] * config.resolution_scale)
                    new_height = int(frame.shape[0] * config.resolution_scale)
                    frame = cv2.resize(frame, (new_width, new_height))

                # Update state
                with self.lock:
                    camera.last_frame = frame
                    camera.last_frame_time = current_time
                    camera.frame_count += 1

                # Put frame in queue for processing
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
            camera.status = CameraStatus.ERROR
            camera.error_message = str(e)
            logger.error(f"Camera {camera_id}: Error - {e}")
        finally:
            cap.release()
            if camera.status != CameraStatus.ERROR:
                camera.status = CameraStatus.DISCONNECTED

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
            connected = sum(1 for c in self.cameras.values() if c.status == CameraStatus.CONNECTED)
            errored = sum(1 for c in self.cameras.values() if c.status == CameraStatus.ERROR)
            total_fps = sum(c.fps for c in self.cameras.values())
            total_frames = sum(c.frame_count for c in self.cameras.values())
            total_detections = sum(c.detection_count for c in self.cameras.values())

            return {
                "running": self.running,
                "total_cameras": total_cameras,
                "connected_cameras": connected,
                "errored_cameras": errored,
                "total_fps": round(total_fps, 1),
                "total_frames_processed": total_frames,
                "total_detections": total_detections,
                "processing_mode": self.config.processing_mode.value,
                "worker_threads": self.config.worker_threads,
            }

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
