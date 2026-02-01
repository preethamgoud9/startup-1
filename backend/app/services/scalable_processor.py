"""
Scalable Multi-Camera Processor - Handles 32+ cameras with optimal resource utilization.
Implements batch processing, priority queues, and adaptive load balancing.
"""

import logging
import threading
import time
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Any
from enum import Enum
from collections import deque
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ProcessingPriority(int, Enum):
    """Camera processing priority levels."""
    CRITICAL = 0    # Always process (entrance, security areas)
    HIGH = 1        # Process frequently
    NORMAL = 2      # Standard processing
    LOW = 3         # Process when resources available
    BACKGROUND = 4  # Process only when idle


@dataclass
class CameraStream:
    """Camera stream with processing state."""
    camera_id: int
    name: str
    stream_url: str
    priority: ProcessingPriority = ProcessingPriority.NORMAL
    
    # Processing state
    enabled: bool = True
    fps_limit: int = 10
    skip_frames: int = 0  # Frames to skip between detections
    
    # Adaptive state
    current_skip: int = 0
    last_detection_time: float = 0
    avg_faces_detected: float = 0
    detection_history: deque = field(default_factory=lambda: deque(maxlen=30))
    
    # Performance metrics
    frames_processed: int = 0
    detections_total: int = 0
    processing_time_avg: float = 0
    last_frame_time: float = 0


@dataclass
class ProcessingResult:
    """Result from processing a frame."""
    camera_id: int
    frame_id: int
    faces: List[Any]
    processing_time: float
    timestamp: float
    frame: Optional[np.ndarray] = None


@dataclass
class ScalableConfig:
    """Configuration for scalable processing."""
    # Resource limits
    max_cameras: int = 32
    max_concurrent_detections: int = 4  # Parallel detection threads
    max_batch_size: int = 8             # Frames to batch process
    
    # Memory management
    frame_buffer_per_camera: int = 2
    max_memory_mb: int = 4096           # Max memory usage
    
    # Adaptive processing
    enable_adaptive_skip: bool = True
    target_system_fps: int = 100        # Total FPS across all cameras
    min_camera_fps: int = 2             # Minimum FPS per camera
    max_camera_fps: int = 15            # Maximum FPS per camera
    
    # Load balancing
    enable_load_balancing: bool = True
    rebalance_interval_sec: float = 5.0
    
    # Quality preservation
    preserve_quality_threshold: float = 0.8  # Don't skip if detection rate high
    motion_detection_threshold: float = 25.0  # Skip if no motion


class AdaptiveLoadBalancer:
    """
    Dynamically balances processing load across cameras.
    Ensures critical cameras get priority while maintaining overall throughput.
    """

    def __init__(self, config: ScalableConfig):
        self.config = config
        self.camera_stats: Dict[int, dict] = {}
        self.last_rebalance = 0
        self.lock = threading.Lock()

    def update_stats(self, camera_id: int, faces_detected: int, processing_time: float):
        """Update camera statistics."""
        with self.lock:
            if camera_id not in self.camera_stats:
                self.camera_stats[camera_id] = {
                    "faces_history": deque(maxlen=30),
                    "time_history": deque(maxlen=30),
                    "skip_frames": 0,
                    "priority_boost": 0,
                }
            
            stats = self.camera_stats[camera_id]
            stats["faces_history"].append(faces_detected)
            stats["time_history"].append(processing_time)

    def get_skip_frames(self, camera: CameraStream) -> int:
        """Calculate optimal frame skip for a camera."""
        if not self.config.enable_adaptive_skip:
            return camera.skip_frames
        
        with self.lock:
            if camera.camera_id not in self.camera_stats:
                return 0
            
            stats = self.camera_stats[camera.camera_id]
            
            # High detection rate = don't skip
            if stats["faces_history"]:
                avg_faces = np.mean(stats["faces_history"])
                if avg_faces > self.config.preserve_quality_threshold:
                    return 0
            
            # Calculate based on priority and load
            base_skip = {
                ProcessingPriority.CRITICAL: 0,
                ProcessingPriority.HIGH: 1,
                ProcessingPriority.NORMAL: 2,
                ProcessingPriority.LOW: 4,
                ProcessingPriority.BACKGROUND: 8,
            }.get(camera.priority, 2)
            
            # Adjust based on system load
            total_cameras = len(self.camera_stats)
            if total_cameras > 16:
                base_skip = int(base_skip * 1.5)
            if total_cameras > 24:
                base_skip = int(base_skip * 2)
            
            return min(base_skip, 10)

    def rebalance(self, cameras: List[CameraStream]) -> Dict[int, int]:
        """Rebalance processing across all cameras."""
        current_time = time.time()
        if current_time - self.last_rebalance < self.config.rebalance_interval_sec:
            return {}
        
        self.last_rebalance = current_time
        
        with self.lock:
            # Calculate total processing budget
            target_total_fps = self.config.target_system_fps
            num_cameras = len(cameras)
            
            if num_cameras == 0:
                return {}
            
            # Distribute FPS based on priority
            priority_weights = {
                ProcessingPriority.CRITICAL: 4.0,
                ProcessingPriority.HIGH: 2.0,
                ProcessingPriority.NORMAL: 1.0,
                ProcessingPriority.LOW: 0.5,
                ProcessingPriority.BACKGROUND: 0.25,
            }
            
            total_weight = sum(priority_weights.get(c.priority, 1.0) for c in cameras if c.enabled)
            
            allocations = {}
            for camera in cameras:
                if not camera.enabled:
                    continue
                
                weight = priority_weights.get(camera.priority, 1.0)
                allocated_fps = (weight / total_weight) * target_total_fps
                
                # Clamp to limits
                allocated_fps = max(self.config.min_camera_fps, min(self.config.max_camera_fps, allocated_fps))
                
                # Convert FPS to skip frames
                if allocated_fps > 0:
                    skip = max(0, int(camera.fps_limit / allocated_fps) - 1)
                else:
                    skip = 10
                
                allocations[camera.camera_id] = skip
            
            return allocations


class MotionDetector:
    """Fast motion detection to skip processing static scenes."""

    def __init__(self, threshold: float = 25.0):
        self.threshold = threshold
        self.prev_frames: Dict[int, np.ndarray] = {}

    def has_motion(self, camera_id: int, frame: np.ndarray) -> bool:
        """Check if frame has significant motion."""
        # Convert to grayscale and resize for speed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 120))
        
        if camera_id not in self.prev_frames:
            self.prev_frames[camera_id] = small
            return True  # First frame, assume motion
        
        # Calculate frame difference
        diff = cv2.absdiff(self.prev_frames[camera_id], small)
        motion_score = np.mean(diff)
        
        self.prev_frames[camera_id] = small
        
        return motion_score > self.threshold

    def clear(self, camera_id: int):
        """Clear motion history for a camera."""
        if camera_id in self.prev_frames:
            del self.prev_frames[camera_id]


class BatchProcessor:
    """
    Processes frames in batches for GPU efficiency.
    Groups frames from multiple cameras for parallel processing.
    """

    def __init__(self, detector, max_batch_size: int = 8):
        self.detector = detector
        self.max_batch_size = max_batch_size
        self.pending_frames: List[tuple] = []  # (camera_id, frame, frame_id)
        self.lock = threading.Lock()

    def add_frame(self, camera_id: int, frame: np.ndarray, frame_id: int):
        """Add frame to batch queue."""
        with self.lock:
            self.pending_frames.append((camera_id, frame, frame_id))

    def process_batch(self) -> List[ProcessingResult]:
        """Process all pending frames."""
        with self.lock:
            if not self.pending_frames:
                return []
            
            batch = self.pending_frames[:self.max_batch_size]
            self.pending_frames = self.pending_frames[self.max_batch_size:]
        
        results = []
        for camera_id, frame, frame_id in batch:
            start_time = time.time()
            
            try:
                faces = self.detector.detect(frame, camera_id=camera_id, get_embeddings=True)
                processing_time = time.time() - start_time
                
                results.append(ProcessingResult(
                    camera_id=camera_id,
                    frame_id=frame_id,
                    faces=faces,
                    processing_time=processing_time,
                    timestamp=time.time(),
                ))
            except Exception as e:
                logger.error(f"Batch processing error for camera {camera_id}: {e}")
        
        return results

    def pending_count(self) -> int:
        """Get number of pending frames."""
        with self.lock:
            return len(self.pending_frames)


class ScalableProcessor:
    """
    Main processor for handling 32+ cameras efficiently.
    Combines all optimization techniques for maximum throughput.
    """

    def __init__(self, face_engine, optimized_detector):
        self.face_engine = face_engine
        self.detector = optimized_detector
        self.config = ScalableConfig()
        
        # Components
        self.load_balancer = AdaptiveLoadBalancer(self.config)
        self.motion_detector = MotionDetector(self.config.motion_detection_threshold)
        self.batch_processor = BatchProcessor(self.detector, self.config.max_batch_size)
        
        # Camera management
        self.cameras: Dict[int, CameraStream] = {}
        self.camera_captures: Dict[int, cv2.VideoCapture] = {}
        self.frame_buffers: Dict[int, deque] = {}
        
        # Threading
        self.running = False
        self.lock = threading.RLock()
        self.executor: Optional[ThreadPoolExecutor] = None
        self.capture_threads: Dict[int, threading.Thread] = {}
        self.processing_thread: Optional[threading.Thread] = None
        
        # Results
        self.result_callbacks: List[Callable[[ProcessingResult], None]] = []
        self.latest_results: Dict[int, ProcessingResult] = {}
        
        # Performance tracking
        self.total_frames_processed = 0
        self.total_detections = 0
        self.start_time = 0
        self.processing_times = deque(maxlen=100)

    def add_camera(
        self,
        camera_id: int,
        name: str,
        stream_url: str,
        priority: ProcessingPriority = ProcessingPriority.NORMAL,
        fps_limit: int = 10,
    ) -> bool:
        """Add a camera to the processor."""
        with self.lock:
            if len(self.cameras) >= self.config.max_cameras:
                logger.error(f"Maximum cameras ({self.config.max_cameras}) reached")
                return False
            
            if camera_id in self.cameras:
                logger.warning(f"Camera {camera_id} already exists")
                return False
            
            self.cameras[camera_id] = CameraStream(
                camera_id=camera_id,
                name=name,
                stream_url=stream_url,
                priority=priority,
                fps_limit=fps_limit,
            )
            
            self.frame_buffers[camera_id] = deque(maxlen=self.config.frame_buffer_per_camera)
            
            logger.info(f"Added camera {camera_id}: {name} (priority: {priority.name})")
            
            # Start capture if running
            if self.running:
                self._start_camera_capture(camera_id)
            
            return True

    def remove_camera(self, camera_id: int) -> bool:
        """Remove a camera from the processor."""
        with self.lock:
            if camera_id not in self.cameras:
                return False
            
            self._stop_camera_capture(camera_id)
            
            del self.cameras[camera_id]
            if camera_id in self.frame_buffers:
                del self.frame_buffers[camera_id]
            if camera_id in self.latest_results:
                del self.latest_results[camera_id]
            
            self.motion_detector.clear(camera_id)
            
            logger.info(f"Removed camera {camera_id}")
            return True

    def set_camera_priority(self, camera_id: int, priority: ProcessingPriority):
        """Set camera processing priority."""
        with self.lock:
            if camera_id in self.cameras:
                self.cameras[camera_id].priority = priority
                logger.info(f"Camera {camera_id} priority set to {priority.name}")

    def start(self):
        """Start processing all cameras."""
        if self.running:
            logger.warning("Processor already running")
            return
        
        self.running = True
        self.start_time = time.time()
        
        # Start thread pool
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_detections + len(self.cameras)
        )
        
        # Start capture threads for all cameras
        for camera_id in self.cameras:
            self._start_camera_capture(camera_id)
        
        # Start main processing thread
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
            name="ProcessingLoop"
        )
        self.processing_thread.start()
        
        logger.info(f"Started scalable processor with {len(self.cameras)} cameras")

    def stop(self):
        """Stop all processing."""
        self.running = False
        
        # Stop all capture threads
        for camera_id in list(self.capture_threads.keys()):
            self._stop_camera_capture(camera_id)
        
        # Wait for processing thread
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        
        logger.info("Stopped scalable processor")

    def _start_camera_capture(self, camera_id: int):
        """Start capture thread for a camera."""
        if camera_id in self.capture_threads:
            return
        
        thread = threading.Thread(
            target=self._capture_loop,
            args=(camera_id,),
            daemon=True,
            name=f"Capture-{camera_id}"
        )
        self.capture_threads[camera_id] = thread
        thread.start()

    def _stop_camera_capture(self, camera_id: int):
        """Stop capture thread for a camera."""
        if camera_id in self.camera_captures:
            try:
                self.camera_captures[camera_id].release()
            except:
                pass
            del self.camera_captures[camera_id]
        
        if camera_id in self.capture_threads:
            # Thread will exit when running=False or camera removed
            del self.capture_threads[camera_id]

    def _capture_loop(self, camera_id: int):
        """Capture frames from a camera."""
        camera = self.cameras.get(camera_id)
        if not camera:
            return
        
        # Open capture
        cap = cv2.VideoCapture(camera.stream_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_id}: {camera.stream_url}")
            return
        
        self.camera_captures[camera_id] = cap
        
        frame_interval = 1.0 / camera.fps_limit
        last_capture_time = 0
        frame_count = 0
        
        logger.info(f"Camera {camera_id} capture started")
        
        try:
            while self.running and camera_id in self.cameras:
                current_time = time.time()
                
                # Rate limiting
                if current_time - last_capture_time < frame_interval:
                    time.sleep(0.001)
                    continue
                
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                last_capture_time = current_time
                
                # Get adaptive skip frames
                skip_frames = self.load_balancer.get_skip_frames(camera)
                
                # Skip frames based on load balancing
                if frame_count % (skip_frames + 1) != 0:
                    continue
                
                # Check for motion (skip static scenes)
                if not self.motion_detector.has_motion(camera_id, frame):
                    continue
                
                # Add to buffer
                with self.lock:
                    if camera_id in self.frame_buffers:
                        self.frame_buffers[camera_id].append((frame, frame_count, current_time))
                
        except Exception as e:
            logger.error(f"Camera {camera_id} capture error: {e}")
        finally:
            cap.release()
            logger.info(f"Camera {camera_id} capture stopped")

    def _processing_loop(self):
        """Main processing loop - processes frames from all cameras."""
        logger.info("Processing loop started")
        
        while self.running:
            try:
                # Collect frames from all cameras based on priority
                frames_to_process = []
                
                with self.lock:
                    # Sort cameras by priority
                    sorted_cameras = sorted(
                        self.cameras.values(),
                        key=lambda c: c.priority.value
                    )
                    
                    for camera in sorted_cameras:
                        if not camera.enabled:
                            continue
                        
                        buffer = self.frame_buffers.get(camera.camera_id)
                        if buffer and len(buffer) > 0:
                            frame_data = buffer.popleft()
                            frames_to_process.append((camera.camera_id, *frame_data))
                            
                            # Limit batch size
                            if len(frames_to_process) >= self.config.max_batch_size:
                                break
                
                if not frames_to_process:
                    time.sleep(0.01)
                    continue
                
                # Process frames
                for camera_id, frame, frame_id, timestamp in frames_to_process:
                    self._process_frame(camera_id, frame, frame_id)
                
                # Periodic rebalancing
                allocations = self.load_balancer.rebalance(list(self.cameras.values()))
                if allocations:
                    for cam_id, skip in allocations.items():
                        if cam_id in self.cameras:
                            self.cameras[cam_id].skip_frames = skip
                
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                time.sleep(0.1)
        
        logger.info("Processing loop stopped")

    def _process_frame(self, camera_id: int, frame: np.ndarray, frame_id: int):
        """Process a single frame."""
        start_time = time.time()
        
        try:
            # Run detection
            faces = self.detector.detect(frame, camera_id=camera_id, get_embeddings=True)
            
            processing_time = time.time() - start_time
            
            # Create result
            result = ProcessingResult(
                camera_id=camera_id,
                frame_id=frame_id,
                faces=faces,
                processing_time=processing_time,
                timestamp=time.time(),
                frame=frame,
            )
            
            # Update stats
            self.load_balancer.update_stats(camera_id, len(faces), processing_time)
            
            with self.lock:
                self.latest_results[camera_id] = result
                self.total_frames_processed += 1
                self.total_detections += len(faces)
                self.processing_times.append(processing_time)
                
                if camera_id in self.cameras:
                    self.cameras[camera_id].frames_processed += 1
                    self.cameras[camera_id].detections_total += len(faces)
            
            # Notify callbacks
            for callback in self.result_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
            
        except Exception as e:
            logger.error(f"Frame processing error for camera {camera_id}: {e}")

    def add_result_callback(self, callback: Callable[[ProcessingResult], None]):
        """Add callback for processing results."""
        self.result_callbacks.append(callback)

    def get_latest_result(self, camera_id: int) -> Optional[ProcessingResult]:
        """Get latest processing result for a camera."""
        with self.lock:
            return self.latest_results.get(camera_id)

    def get_stats(self) -> dict:
        """Get processor statistics."""
        with self.lock:
            uptime = time.time() - self.start_time if self.start_time > 0 else 0
            avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
            
            camera_stats = []
            for cam_id, camera in self.cameras.items():
                camera_stats.append({
                    "id": cam_id,
                    "name": camera.name,
                    "priority": camera.priority.name,
                    "enabled": camera.enabled,
                    "frames_processed": camera.frames_processed,
                    "detections": camera.detections_total,
                    "skip_frames": camera.skip_frames,
                })
            
            return {
                "running": self.running,
                "uptime_seconds": uptime,
                "total_cameras": len(self.cameras),
                "total_frames_processed": self.total_frames_processed,
                "total_detections": self.total_detections,
                "avg_processing_time_ms": avg_processing_time * 1000,
                "effective_fps": self.total_frames_processed / uptime if uptime > 0 else 0,
                "cameras": camera_stats,
                "detector_stats": self.detector.get_stats(),
            }
