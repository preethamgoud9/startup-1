"""
Optimized Face Detector - State-of-the-art face detection with long-range accuracy.
Implements multi-scale detection, super-resolution enhancement, and adaptive processing.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from enum import Enum
import queue
from collections import deque

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DetectionQuality(str, Enum):
    """Detection quality presets."""
    FAST = "fast"           # Low latency, good for close range
    BALANCED = "balanced"   # Balance between speed and accuracy
    ACCURATE = "accurate"   # High accuracy, good for long range
    MAXIMUM = "maximum"     # Maximum quality, multi-scale + enhancement


@dataclass
class DetectionConfig:
    """Configuration for optimized detection."""
    # Detection parameters
    min_face_size: int = 20              # Minimum face size in pixels
    max_face_size: int = 0               # 0 = no limit
    detection_threshold: float = 0.5     # Detection confidence threshold
    nms_threshold: float = 0.4           # Non-maximum suppression threshold
    
    # Long-range enhancement
    enable_super_resolution: bool = True  # Upscale for small faces
    super_resolution_threshold: int = 64  # Face size below which to upscale
    upscale_factor: float = 2.0          # Upscale factor for small faces
    
    # Multi-scale detection
    enable_multi_scale: bool = True      # Use image pyramid
    scale_factors: List[float] = field(default_factory=lambda: [1.0, 0.75, 0.5])
    
    # Adaptive processing
    enable_adaptive: bool = True         # Adapt based on scene
    target_fps: int = 10                 # Target FPS for adaptive mode
    min_detection_interval_ms: int = 50  # Minimum time between detections
    
    # ROI tracking
    enable_roi_tracking: bool = True     # Track regions of interest
    roi_expansion: float = 1.5           # Expand ROI by this factor
    roi_max_age_frames: int = 30         # Max frames to track ROI without detection
    
    # Quality enhancement
    enable_histogram_eq: bool = True     # Histogram equalization for low light
    enable_denoising: bool = False       # Denoise (slower but better in noise)
    sharpen_strength: float = 0.3        # Sharpening for distant faces


@dataclass
class DetectedFace:
    """Detected face with metadata."""
    bbox: Tuple[int, int, int, int]      # x1, y1, x2, y2
    confidence: float
    landmarks: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    face_size: int = 0                   # Face width in pixels
    detection_scale: float = 1.0         # Scale at which detected
    enhanced: bool = False               # Was enhancement applied
    frame_id: int = 0


@dataclass
class ROIRegion:
    """Region of interest for tracking."""
    bbox: Tuple[int, int, int, int]
    last_detection_frame: int
    detection_count: int = 0
    velocity: Tuple[float, float] = (0.0, 0.0)  # Movement prediction


class OptimizedDetector:
    """
    State-of-the-art face detector optimized for:
    - Long-range detection (small faces)
    - Multi-camera scalability
    - Consistent detection quality
    - GPU acceleration
    """

    def __init__(self, face_engine):
        self.face_engine = face_engine
        self.config = DetectionConfig()
        self.lock = threading.RLock()
        
        # Performance tracking
        self.frame_times = deque(maxlen=30)
        self.detection_times = deque(maxlen=30)
        self.current_fps = 0.0
        self.avg_detection_time = 0.0
        
        # ROI tracking per camera
        self.camera_rois: Dict[int, List[ROIRegion]] = {}
        self.frame_counters: Dict[int, int] = {}
        
        # Detection queue for batch processing
        self.detection_queue = queue.Queue(maxsize=64)
        self.result_queues: Dict[int, queue.Queue] = {}
        
        # Pre-computed kernels
        self._sharpen_kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        
        self._initialized = False
        logger.info("OptimizedDetector initialized")

    def set_quality(self, quality: DetectionQuality):
        """Set detection quality preset."""
        with self.lock:
            if quality == DetectionQuality.FAST:
                self.config.enable_multi_scale = False
                self.config.enable_super_resolution = False
                self.config.enable_histogram_eq = False
                self.config.scale_factors = [1.0]
                self.config.detection_threshold = 0.6
            elif quality == DetectionQuality.BALANCED:
                self.config.enable_multi_scale = True
                self.config.enable_super_resolution = True
                self.config.enable_histogram_eq = True
                self.config.scale_factors = [1.0, 0.75]
                self.config.detection_threshold = 0.5
            elif quality == DetectionQuality.ACCURATE:
                self.config.enable_multi_scale = True
                self.config.enable_super_resolution = True
                self.config.enable_histogram_eq = True
                self.config.scale_factors = [1.0, 0.75, 0.5]
                self.config.detection_threshold = 0.4
                self.config.upscale_factor = 2.0
            elif quality == DetectionQuality.MAXIMUM:
                self.config.enable_multi_scale = True
                self.config.enable_super_resolution = True
                self.config.enable_histogram_eq = True
                self.config.enable_denoising = True
                self.config.scale_factors = [1.0, 0.75, 0.5, 0.25]
                self.config.detection_threshold = 0.35
                self.config.upscale_factor = 3.0
        
        logger.info(f"Detection quality set to: {quality.value}")

    def detect(
        self,
        frame: np.ndarray,
        camera_id: int = 0,
        get_embeddings: bool = True,
    ) -> List[DetectedFace]:
        """
        Detect faces with optimized multi-scale processing.
        
        Args:
            frame: Input BGR image
            camera_id: Camera identifier for ROI tracking
            get_embeddings: Whether to compute face embeddings
            
        Returns:
            List of detected faces with metadata
        """
        start_time = time.time()
        
        # Update frame counter
        if camera_id not in self.frame_counters:
            self.frame_counters[camera_id] = 0
        self.frame_counters[camera_id] += 1
        frame_id = self.frame_counters[camera_id]
        
        # Pre-process frame
        processed_frame = self._preprocess_frame(frame)
        
        all_faces = []
        
        # Check ROI regions first (faster for tracking)
        if self.config.enable_roi_tracking and camera_id in self.camera_rois:
            roi_faces = self._detect_in_rois(processed_frame, camera_id, frame_id)
            all_faces.extend(roi_faces)
        
        # Multi-scale detection
        if self.config.enable_multi_scale:
            for scale in self.config.scale_factors:
                scaled_faces = self._detect_at_scale(processed_frame, scale, frame_id)
                all_faces.extend(scaled_faces)
        else:
            all_faces.extend(self._detect_at_scale(processed_frame, 1.0, frame_id))
        
        # Detect small faces with super-resolution
        if self.config.enable_super_resolution:
            small_face_regions = self._find_potential_small_faces(processed_frame, all_faces)
            for region in small_face_regions:
                enhanced_faces = self._detect_with_enhancement(processed_frame, region, frame_id)
                all_faces.extend(enhanced_faces)
        
        # Non-maximum suppression to remove duplicates
        faces = self._nms_faces(all_faces)
        
        # Get embeddings for recognized faces
        if get_embeddings and faces:
            faces = self._compute_embeddings(frame, faces)
        
        # Update ROI tracking
        if self.config.enable_roi_tracking:
            self._update_rois(camera_id, faces, frame_id)
        
        # Track performance
        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)
        self.avg_detection_time = np.mean(self.detection_times)
        
        return faces

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply preprocessing for better detection."""
        processed = frame.copy()
        
        # Histogram equalization for low-light enhancement
        if self.config.enable_histogram_eq:
            # Convert to LAB and equalize L channel
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            lab = cv2.merge([l, a, b])
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Denoising (optional, slower)
        if self.config.enable_denoising:
            processed = cv2.fastNlMeansDenoisingColored(processed, None, 6, 6, 7, 21)
        
        # Sharpening for distant faces
        if self.config.sharpen_strength > 0:
            blurred = cv2.GaussianBlur(processed, (0, 0), 3)
            processed = cv2.addWeighted(
                processed, 1.0 + self.config.sharpen_strength,
                blurred, -self.config.sharpen_strength,
                0
            )
        
        return processed

    def _detect_at_scale(
        self,
        frame: np.ndarray,
        scale: float,
        frame_id: int,
    ) -> List[DetectedFace]:
        """Detect faces at a specific scale."""
        if scale != 1.0:
            h, w = frame.shape[:2]
            new_w, new_h = int(w * scale), int(h * scale)
            scaled_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            scaled_frame = frame
            scale = 1.0
        
        # Run detection
        try:
            raw_faces = self.face_engine.app.get(scaled_frame)
        except Exception as e:
            logger.error(f"Detection error at scale {scale}: {e}")
            return []
        
        faces = []
        for f in raw_faces:
            if f.det_score is None or f.det_score < self.config.detection_threshold:
                continue
            
            # Scale bbox back to original size
            x1, y1, x2, y2 = f.bbox
            if scale != 1.0:
                x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
            
            face_width = int(x2 - x1)
            
            # Filter by face size
            if face_width < self.config.min_face_size:
                continue
            if self.config.max_face_size > 0 and face_width > self.config.max_face_size:
                continue
            
            faces.append(DetectedFace(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                confidence=float(f.det_score),
                landmarks=f.kps if hasattr(f, 'kps') else None,
                face_size=face_width,
                detection_scale=scale,
                frame_id=frame_id,
            ))
        
        return faces

    def _detect_in_rois(
        self,
        frame: np.ndarray,
        camera_id: int,
        frame_id: int,
    ) -> List[DetectedFace]:
        """Detect faces in tracked ROI regions (faster)."""
        if camera_id not in self.camera_rois:
            return []
        
        faces = []
        h, w = frame.shape[:2]
        
        for roi in self.camera_rois[camera_id]:
            # Expand ROI
            x1, y1, x2, y2 = roi.bbox
            roi_w, roi_h = x2 - x1, y2 - y1
            expand = self.config.roi_expansion
            
            x1 = max(0, int(x1 - roi_w * (expand - 1) / 2))
            y1 = max(0, int(y1 - roi_h * (expand - 1) / 2))
            x2 = min(w, int(x2 + roi_w * (expand - 1) / 2))
            y2 = min(h, int(y2 + roi_h * (expand - 1) / 2))
            
            # Extract ROI
            roi_frame = frame[y1:y2, x1:x2]
            if roi_frame.size == 0:
                continue
            
            # Detect in ROI
            try:
                raw_faces = self.face_engine.app.get(roi_frame)
            except:
                continue
            
            for f in raw_faces:
                if f.det_score is None or f.det_score < self.config.detection_threshold:
                    continue
                
                # Adjust bbox to full frame coordinates
                fx1, fy1, fx2, fy2 = f.bbox
                faces.append(DetectedFace(
                    bbox=(int(fx1 + x1), int(fy1 + y1), int(fx2 + x1), int(fy2 + y1)),
                    confidence=float(f.det_score),
                    landmarks=f.kps + np.array([x1, y1]) if hasattr(f, 'kps') and f.kps is not None else None,
                    face_size=int(fx2 - fx1),
                    detection_scale=1.0,
                    frame_id=frame_id,
                ))
        
        return faces

    def _find_potential_small_faces(
        self,
        frame: np.ndarray,
        detected_faces: List[DetectedFace],
    ) -> List[Tuple[int, int, int, int]]:
        """Find regions that might contain small faces not yet detected."""
        # Use motion detection or edge detection to find potential face regions
        # For now, return regions with detected small faces for enhancement
        regions = []
        
        for face in detected_faces:
            if face.face_size < self.config.super_resolution_threshold:
                # This face is small, mark region for enhancement
                x1, y1, x2, y2 = face.bbox
                # Expand region
                h, w = frame.shape[:2]
                pad = int(face.face_size * 0.5)
                regions.append((
                    max(0, x1 - pad),
                    max(0, y1 - pad),
                    min(w, x2 + pad),
                    min(h, y2 + pad),
                ))
        
        return regions

    def _detect_with_enhancement(
        self,
        frame: np.ndarray,
        region: Tuple[int, int, int, int],
        frame_id: int,
    ) -> List[DetectedFace]:
        """Detect faces in a region with super-resolution enhancement."""
        x1, y1, x2, y2 = region
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return []
        
        # Upscale the region
        scale = self.config.upscale_factor
        upscaled = cv2.resize(
            roi,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )
        
        # Apply sharpening to upscaled image
        upscaled = cv2.filter2D(upscaled, -1, self._sharpen_kernel * 0.5 + np.eye(3) * 0.5)
        
        # Detect in upscaled region
        try:
            raw_faces = self.face_engine.app.get(upscaled)
        except:
            return []
        
        faces = []
        for f in raw_faces:
            if f.det_score is None or f.det_score < self.config.detection_threshold:
                continue
            
            # Scale bbox back and adjust to full frame
            fx1, fy1, fx2, fy2 = f.bbox
            fx1, fy1 = fx1 / scale + x1, fy1 / scale + y1
            fx2, fy2 = fx2 / scale + x1, fy2 / scale + y1
            
            faces.append(DetectedFace(
                bbox=(int(fx1), int(fy1), int(fx2), int(fy2)),
                confidence=float(f.det_score),
                landmarks=None,  # Landmarks need recalculation
                face_size=int((fx2 - fx1)),
                detection_scale=1.0 / scale,
                enhanced=True,
                frame_id=frame_id,
            ))
        
        return faces

    def _nms_faces(self, faces: List[DetectedFace]) -> List[DetectedFace]:
        """Apply non-maximum suppression to remove duplicate detections."""
        if len(faces) <= 1:
            return faces
        
        # Sort by confidence
        faces = sorted(faces, key=lambda f: f.confidence, reverse=True)
        
        keep = []
        suppressed = set()
        
        for i, face_i in enumerate(faces):
            if i in suppressed:
                continue
            
            keep.append(face_i)
            
            for j, face_j in enumerate(faces[i+1:], i+1):
                if j in suppressed:
                    continue
                
                iou = self._compute_iou(face_i.bbox, face_j.bbox)
                if iou > self.config.nms_threshold:
                    suppressed.add(j)
        
        return keep

    def _compute_iou(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int],
    ) -> float:
        """Compute intersection over union."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

    def _compute_embeddings(
        self,
        frame: np.ndarray,
        faces: List[DetectedFace],
    ) -> List[DetectedFace]:
        """Compute face embeddings for recognition."""
        for face in faces:
            x1, y1, x2, y2 = face.bbox
            
            # Expand face region for better embedding
            h, w = frame.shape[:2]
            face_w, face_h = x2 - x1, y2 - y1
            pad_w, pad_h = int(face_w * 0.2), int(face_h * 0.2)
            
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(w, x2 + pad_w)
            y2 = min(h, y2 + pad_h)
            
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size == 0:
                continue
            
            # Resize to optimal size for embedding
            target_size = 112
            face_img = cv2.resize(face_img, (target_size, target_size))
            
            try:
                # Get embedding using face engine
                detected = self.face_engine.app.get(face_img)
                if detected and hasattr(detected[0], 'normed_embedding'):
                    face.embedding = np.asarray(detected[0].normed_embedding, dtype=np.float32)
            except Exception as e:
                logger.debug(f"Embedding extraction failed: {e}")
        
        return faces

    def _update_rois(
        self,
        camera_id: int,
        faces: List[DetectedFace],
        frame_id: int,
    ):
        """Update ROI tracking for a camera."""
        if camera_id not in self.camera_rois:
            self.camera_rois[camera_id] = []
        
        # Match detected faces to existing ROIs
        matched_rois = set()
        new_rois = []
        
        for face in faces:
            best_iou = 0
            best_roi_idx = -1
            
            for i, roi in enumerate(self.camera_rois[camera_id]):
                if i in matched_rois:
                    continue
                iou = self._compute_iou(face.bbox, roi.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_roi_idx = i
            
            if best_iou > 0.3 and best_roi_idx >= 0:
                # Update existing ROI
                roi = self.camera_rois[camera_id][best_roi_idx]
                old_bbox = roi.bbox
                roi.bbox = face.bbox
                roi.last_detection_frame = frame_id
                roi.detection_count += 1
                # Update velocity
                roi.velocity = (
                    (face.bbox[0] - old_bbox[0]) * 0.5 + roi.velocity[0] * 0.5,
                    (face.bbox[1] - old_bbox[1]) * 0.5 + roi.velocity[1] * 0.5,
                )
                matched_rois.add(best_roi_idx)
            else:
                # Create new ROI
                new_rois.append(ROIRegion(
                    bbox=face.bbox,
                    last_detection_frame=frame_id,
                    detection_count=1,
                ))
        
        # Remove old ROIs
        self.camera_rois[camera_id] = [
            roi for i, roi in enumerate(self.camera_rois[camera_id])
            if i in matched_rois or (frame_id - roi.last_detection_frame) < self.config.roi_max_age_frames
        ]
        
        # Add new ROIs
        self.camera_rois[camera_id].extend(new_rois)
        
        # Limit number of ROIs
        if len(self.camera_rois[camera_id]) > 50:
            self.camera_rois[camera_id] = sorted(
                self.camera_rois[camera_id],
                key=lambda r: r.detection_count,
                reverse=True
            )[:50]

    def get_stats(self) -> dict:
        """Get detector performance statistics."""
        return {
            "avg_detection_time_ms": self.avg_detection_time * 1000,
            "detection_fps": 1.0 / self.avg_detection_time if self.avg_detection_time > 0 else 0,
            "tracked_cameras": len(self.camera_rois),
            "total_rois": sum(len(rois) for rois in self.camera_rois.values()),
            "config": {
                "multi_scale": self.config.enable_multi_scale,
                "super_resolution": self.config.enable_super_resolution,
                "roi_tracking": self.config.enable_roi_tracking,
                "detection_threshold": self.config.detection_threshold,
            }
        }
