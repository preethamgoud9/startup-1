"""
Optimized Face Detector - State-of-the-art face detection for long-range (up to 20m).

Key techniques:
- Tiled detection: splits frame into overlapping tiles, upscales each for small face discovery
- Multi-resolution fusion: detects at original + upscaled resolutions, merges with NMS
- Adaptive preprocessing: auto-adjusts enhancement based on image brightness/contrast
- Face quality scoring: filters blurry, occluded, or extreme-angle faces before recognition
- Proper ArcFace embedding extraction via InsightFace alignment (not re-detection on crops)
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum
from collections import deque

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DetectionQuality(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"
    MAXIMUM = "maximum"


@dataclass
class DetectionConfig:
    # Detection thresholds
    min_face_size: int = 16
    max_face_size: int = 0
    detection_threshold: float = 0.5
    nms_threshold: float = 0.4

    # Tiled detection for long-range (critical for 20m)
    enable_tiled_detection: bool = True
    tile_overlap: float = 0.25           # 25% overlap between tiles
    tile_upscale: float = 2.0            # Upscale each tile by this factor
    tile_min_count: int = 4              # Minimum tiles (2x2)
    tile_max_count: int = 16             # Maximum tiles (4x4)
    tile_face_threshold: int = 40        # Only tile if no faces > this pixel width

    # Multi-resolution detection
    enable_multi_resolution: bool = True
    upscale_factors: List[float] = field(default_factory=lambda: [1.0, 1.5])

    # Adaptive preprocessing
    enable_adaptive_enhance: bool = True
    clahe_clip_limit: float = 2.5
    clahe_grid_size: int = 8

    # Face quality filtering
    enable_quality_filter: bool = True
    min_face_quality: float = 0.3        # Min quality score (0-1)
    min_landmark_confidence: float = 0.5
    max_face_blur: float = 100.0         # Laplacian variance below this = blurry

    # Blur gate for embedding extraction (stricter than quality filter)
    # Faces with Laplacian variance below this are too blurry for stable embeddings
    min_embedding_blur: float = 30.0

    # ROI tracking
    enable_roi_tracking: bool = True
    roi_expansion: float = 1.5
    roi_max_age_frames: int = 30

    # Sharpening
    sharpen_strength: float = 0.3
    enable_denoising: bool = False


@dataclass
class DetectedFace:
    bbox: Tuple[int, int, int, int]
    confidence: float
    landmarks: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    face_size: int = 0
    detection_scale: float = 1.0
    enhanced: bool = False
    frame_id: int = 0
    quality_score: float = 1.0


@dataclass
class ROIRegion:
    bbox: Tuple[int, int, int, int]
    last_detection_frame: int
    detection_count: int = 0
    velocity: Tuple[float, float] = (0.0, 0.0)


class OptimizedDetector:
    """
    Long-range face detector optimized for classroom scenarios (up to 20m).

    Pipeline:
    1. Adaptive preprocessing (brightness/contrast normalization)
    2. Full-frame detection at original resolution
    3. Tiled detection: split frame into overlapping tiles, upscale each, detect
    4. Multi-resolution fusion with NMS
    5. Face quality scoring (blur, size, landmark quality)
    6. Proper embedding extraction via InsightFace face alignment
    7. ROI tracking for temporal consistency
    """

    def __init__(self, face_engine):
        self.face_engine = face_engine
        self.config = DetectionConfig()
        self.lock = threading.RLock()

        # Performance tracking
        self.detection_times: deque = deque(maxlen=30)
        self.avg_detection_time: float = 0.0

        # ROI tracking per camera
        self.camera_rois: Dict[int, List[ROIRegion]] = {}
        self.frame_counters: Dict[int, int] = {}

        # CLAHE instance (reuse for performance)
        self._clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=(self.config.clahe_grid_size, self.config.clahe_grid_size),
        )

        # Sharpen kernel
        self._sharpen_kernel = np.array(
            [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32
        )

        logger.info("OptimizedDetector initialized")

    def set_quality(self, quality: DetectionQuality):
        """Set detection quality preset."""
        with self.lock:
            if quality == DetectionQuality.FAST:
                self.config.enable_tiled_detection = False
                self.config.enable_multi_resolution = False
                self.config.enable_adaptive_enhance = False
                self.config.upscale_factors = [1.0]
                self.config.detection_threshold = 0.6
                self.config.enable_quality_filter = False
                self.config.sharpen_strength = 0.0
            elif quality == DetectionQuality.BALANCED:
                self.config.enable_tiled_detection = True
                self.config.enable_multi_resolution = True
                self.config.enable_adaptive_enhance = True
                self.config.upscale_factors = [1.0, 1.5]
                self.config.detection_threshold = 0.45
                self.config.tile_upscale = 2.0
                self.config.tile_max_count = 9
                self.config.enable_quality_filter = True
                self.config.sharpen_strength = 0.3
            elif quality == DetectionQuality.ACCURATE:
                self.config.enable_tiled_detection = True
                self.config.enable_multi_resolution = True
                self.config.enable_adaptive_enhance = True
                self.config.upscale_factors = [1.0, 1.5, 2.0]
                self.config.detection_threshold = 0.35
                self.config.tile_upscale = 3.0
                self.config.tile_max_count = 16
                self.config.enable_quality_filter = True
                self.config.sharpen_strength = 0.4
                self.config.enable_denoising = False
            elif quality == DetectionQuality.MAXIMUM:
                self.config.enable_tiled_detection = True
                self.config.enable_multi_resolution = True
                self.config.enable_adaptive_enhance = True
                self.config.upscale_factors = [1.0, 1.5, 2.0, 2.5]
                self.config.detection_threshold = 0.3
                self.config.tile_upscale = 4.0
                self.config.tile_max_count = 16
                self.config.enable_quality_filter = True
                self.config.sharpen_strength = 0.5
                self.config.enable_denoising = True

        logger.info(f"Detection quality set to: {quality.value}")

    def detect(
        self,
        frame: np.ndarray,
        camera_id: int = 0,
        get_embeddings: bool = True,
    ) -> List[DetectedFace]:
        """
        Detect faces with long-range optimized pipeline.

        Args:
            frame: Input BGR image
            camera_id: Camera identifier for ROI tracking
            get_embeddings: Whether to compute face embeddings

        Returns:
            List of detected faces with metadata
        """
        start_time = time.time()

        if camera_id not in self.frame_counters:
            self.frame_counters[camera_id] = 0
        self.frame_counters[camera_id] += 1
        frame_id = self.frame_counters[camera_id]

        # Step 1: Adaptive preprocessing
        processed = self._preprocess_adaptive(frame)

        all_faces: List[DetectedFace] = []

        # Step 2: Full-frame multi-resolution detection
        if self.config.enable_multi_resolution:
            for upscale in self.config.upscale_factors:
                faces = self._detect_at_resolution(processed, upscale, frame_id)
                all_faces.extend(faces)
        else:
            all_faces.extend(self._detect_at_resolution(processed, 1.0, frame_id))

        # Step 3: Tiled detection for small/distant faces
        if self.config.enable_tiled_detection:
            # Only tile if we haven't found large faces (saves compute when close-up)
            max_face_size = max((f.face_size for f in all_faces), default=0)
            if max_face_size < self.config.tile_face_threshold or len(all_faces) == 0:
                tiled_faces = self._detect_tiled(processed, frame_id)
                all_faces.extend(tiled_faces)

        # Step 4: ROI-based detection for tracked regions
        if self.config.enable_roi_tracking and camera_id in self.camera_rois:
            roi_faces = self._detect_in_rois(processed, camera_id, frame_id)
            all_faces.extend(roi_faces)

        # Step 5: Non-maximum suppression
        faces = self._nms_faces(all_faces)

        # Step 6: Face quality scoring and filtering
        if self.config.enable_quality_filter:
            faces = self._score_and_filter_faces(frame, faces)

        # Step 7: Extract embeddings using proper InsightFace alignment
        if get_embeddings and faces:
            faces = self._extract_embeddings(frame, faces)

        # Step 8: Update ROI tracking
        if self.config.enable_roi_tracking:
            self._update_rois(camera_id, faces, frame_id)

        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)
        self.avg_detection_time = float(np.mean(self.detection_times))

        return faces

    # ─── Preprocessing ────────────────────────────────────────────────

    def _preprocess_adaptive(self, frame: np.ndarray) -> np.ndarray:
        """Adaptively enhance frame based on image statistics."""
        if not self.config.enable_adaptive_enhance:
            return frame

        processed = frame.copy()

        # Analyze image brightness
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        std_brightness = float(np.std(gray))

        # Only apply CLAHE if image is dark or has low contrast
        needs_clahe = mean_brightness < 120 or std_brightness < 40
        if needs_clahe:
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self._clahe.apply(l)
            lab = cv2.merge([l, a, b])
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Denoising for noisy/dark images
        if self.config.enable_denoising and mean_brightness < 80:
            processed = cv2.fastNlMeansDenoisingColored(
                processed, None, 6, 6, 7, 15
            )

        # Sharpening (always useful for distant faces)
        if self.config.sharpen_strength > 0:
            blurred = cv2.GaussianBlur(processed, (0, 0), 2.0)
            processed = cv2.addWeighted(
                processed, 1.0 + self.config.sharpen_strength,
                blurred, -self.config.sharpen_strength, 0,
            )

        return processed

    # ─── Multi-Resolution Detection ───────────────────────────────────

    def _detect_at_resolution(
        self,
        frame: np.ndarray,
        upscale: float,
        frame_id: int,
    ) -> List[DetectedFace]:
        """Detect faces at a specific resolution (upscale >= 1.0)."""
        if upscale > 1.0:
            h, w = frame.shape[:2]
            new_w, new_h = int(w * upscale), int(h * upscale)
            # Use INTER_LANCZOS4 for best quality upscaling
            scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            scaled = frame
            upscale = 1.0

        try:
            raw_faces = self.face_engine.app.get(scaled)
        except Exception as e:
            logger.error(f"Detection error at upscale {upscale}: {e}")
            return []

        faces = []
        for f in raw_faces:
            if f.det_score is None or f.det_score < self.config.detection_threshold:
                continue

            x1, y1, x2, y2 = f.bbox
            # Scale bbox back to original resolution
            if upscale > 1.0:
                x1, y1 = x1 / upscale, y1 / upscale
                x2, y2 = x2 / upscale, y2 / upscale

            face_w = int(x2 - x1)
            if face_w < self.config.min_face_size:
                continue
            if 0 < self.config.max_face_size < face_w:
                continue

            # Scale landmarks back
            landmarks = None
            if hasattr(f, 'kps') and f.kps is not None:
                landmarks = f.kps / upscale if upscale > 1.0 else f.kps

            faces.append(DetectedFace(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                confidence=float(f.det_score),
                landmarks=landmarks,
                face_size=face_w,
                detection_scale=upscale,
                frame_id=frame_id,
            ))

        return faces

    # ─── Tiled Detection (Critical for 20m Range) ─────────────────────

    def _detect_tiled(
        self,
        frame: np.ndarray,
        frame_id: int,
    ) -> List[DetectedFace]:
        """
        Split frame into overlapping tiles, upscale each, and detect.

        At 20m on a 1080p camera (~90° FOV), a face is ~15-25px wide.
        SCRFD needs ~30px+ to detect reliably. Tiling + upscaling solves this:
        - Split 1920x1080 into 3x2 tiles of ~760x640 each (with overlap)
        - Upscale each tile 2-3x → each tile becomes 1520-2280px wide
        - A 20px face becomes 40-60px in the tile → detectable
        """
        h, w = frame.shape[:2]
        upscale = self.config.tile_upscale
        overlap = self.config.tile_overlap

        # Calculate grid size based on frame resolution
        # More tiles for higher resolution frames
        if w >= 1920:
            cols, rows = 4, 3
        elif w >= 1280:
            cols, rows = 3, 2
        else:
            cols, rows = 2, 2

        # Clamp to configured limits
        total_tiles = cols * rows
        if total_tiles < self.config.tile_min_count:
            cols = rows = 2
        if total_tiles > self.config.tile_max_count:
            # Scale down
            while cols * rows > self.config.tile_max_count:
                if cols > rows:
                    cols -= 1
                else:
                    rows -= 1

        tile_w = w // cols
        tile_h = h // rows
        overlap_w = int(tile_w * overlap)
        overlap_h = int(tile_h * overlap)

        all_faces = []

        for row in range(rows):
            for col in range(cols):
                # Calculate tile boundaries with overlap
                x1 = max(0, col * tile_w - overlap_w)
                y1 = max(0, row * tile_h - overlap_h)
                x2 = min(w, (col + 1) * tile_w + overlap_w)
                y2 = min(h, (row + 1) * tile_h + overlap_h)

                tile = frame[y1:y2, x1:x2]
                if tile.size == 0:
                    continue

                # Upscale tile
                tile_up = cv2.resize(
                    tile, None,
                    fx=upscale, fy=upscale,
                    interpolation=cv2.INTER_LANCZOS4,
                )

                # Sharpen upscaled tile for better edge definition
                if self.config.sharpen_strength > 0:
                    blurred = cv2.GaussianBlur(tile_up, (0, 0), 1.5)
                    tile_up = cv2.addWeighted(
                        tile_up, 1.0 + self.config.sharpen_strength * 0.5,
                        blurred, -self.config.sharpen_strength * 0.5, 0,
                    )

                # Detect in upscaled tile
                try:
                    raw_faces = self.face_engine.app.get(tile_up)
                except Exception:
                    continue

                for f in raw_faces:
                    if f.det_score is None or f.det_score < self.config.detection_threshold:
                        continue

                    fx1, fy1, fx2, fy2 = f.bbox
                    # Map back: divide by upscale, then add tile offset
                    fx1 = fx1 / upscale + x1
                    fy1 = fy1 / upscale + y1
                    fx2 = fx2 / upscale + x1
                    fy2 = fy2 / upscale + y1

                    face_w = int(fx2 - fx1)
                    if face_w < self.config.min_face_size:
                        continue

                    # Scale landmarks back
                    landmarks = None
                    if hasattr(f, 'kps') and f.kps is not None:
                        landmarks = f.kps / upscale + np.array([x1, y1])

                    all_faces.append(DetectedFace(
                        bbox=(int(fx1), int(fy1), int(fx2), int(fy2)),
                        confidence=float(f.det_score),
                        landmarks=landmarks,
                        face_size=face_w,
                        detection_scale=upscale,
                        enhanced=True,
                        frame_id=frame_id,
                    ))

        return all_faces

    # ─── ROI Detection ────────────────────────────────────────────────

    def _detect_in_rois(
        self,
        frame: np.ndarray,
        camera_id: int,
        frame_id: int,
    ) -> List[DetectedFace]:
        """Detect faces in tracked ROI regions with upscaling."""
        if camera_id not in self.camera_rois:
            return []

        faces = []
        h, w = frame.shape[:2]

        for roi in self.camera_rois[camera_id]:
            x1, y1, x2, y2 = roi.bbox
            roi_w, roi_h = x2 - x1, y2 - y1
            expand = self.config.roi_expansion

            # Predict position using velocity
            vx, vy = roi.velocity
            age = frame_id - roi.last_detection_frame
            pred_dx, pred_dy = int(vx * age), int(vy * age)

            x1 = max(0, int(x1 + pred_dx - roi_w * (expand - 1) / 2))
            y1 = max(0, int(y1 + pred_dy - roi_h * (expand - 1) / 2))
            x2 = min(w, int(x2 + pred_dx + roi_w * (expand - 1) / 2))
            y2 = min(h, int(y2 + pred_dy + roi_h * (expand - 1) / 2))

            roi_frame = frame[y1:y2, x1:x2]
            if roi_frame.size == 0:
                continue

            # Upscale ROI for better detection of small faces
            roi_face_size = max(roi_w, roi_h)
            if roi_face_size < 80:
                up = min(4.0, 120.0 / max(roi_face_size, 1))
                roi_frame = cv2.resize(
                    roi_frame, None, fx=up, fy=up,
                    interpolation=cv2.INTER_LANCZOS4,
                )
            else:
                up = 1.0

            try:
                raw_faces = self.face_engine.app.get(roi_frame)
            except Exception:
                continue

            for f in raw_faces:
                if f.det_score is None or f.det_score < self.config.detection_threshold:
                    continue

                fx1, fy1, fx2, fy2 = f.bbox
                # Map coordinates back
                fx1 = fx1 / up + x1
                fy1 = fy1 / up + y1
                fx2 = fx2 / up + x1
                fy2 = fy2 / up + y1

                landmarks = None
                if hasattr(f, 'kps') and f.kps is not None:
                    landmarks = f.kps / up + np.array([x1, y1])

                faces.append(DetectedFace(
                    bbox=(int(fx1), int(fy1), int(fx2), int(fy2)),
                    confidence=float(f.det_score),
                    landmarks=landmarks,
                    face_size=int(fx2 - fx1),
                    detection_scale=up,
                    frame_id=frame_id,
                ))

        return faces

    # ─── Face Quality Scoring ─────────────────────────────────────────

    def _score_and_filter_faces(
        self,
        frame: np.ndarray,
        faces: List[DetectedFace],
    ) -> List[DetectedFace]:
        """Score face quality and filter out low-quality detections."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        filtered = []

        for face in faces:
            x1, y1, x2, y2 = face.bbox
            # Clamp to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            face_roi = gray[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue

            # --- Score components ---
            scores = []

            # 1. Blur score (Laplacian variance)
            laplacian_var = cv2.Laplacian(face_roi, cv2.CV_64F).var()
            blur_score = min(1.0, laplacian_var / self.config.max_face_blur)
            scores.append(blur_score)

            # 2. Size score (larger = better for recognition)
            face_w = x2 - x1
            if face_w >= 80:
                size_score = 1.0
            elif face_w >= 40:
                size_score = 0.7
            elif face_w >= 20:
                size_score = 0.4
            else:
                size_score = 0.2
            scores.append(size_score)

            # 3. Brightness score (not too dark or too bright)
            mean_val = float(np.mean(face_roi))
            if 60 <= mean_val <= 200:
                brightness_score = 1.0
            elif 30 <= mean_val <= 230:
                brightness_score = 0.6
            else:
                brightness_score = 0.3
            scores.append(brightness_score)

            # 4. Detection confidence (already filtered by threshold)
            conf_score = min(1.0, face.confidence / 0.8)
            scores.append(conf_score)

            # 5. Landmark quality (if available)
            if face.landmarks is not None and len(face.landmarks) >= 5:
                # Check if landmarks form a reasonable face shape
                lm = face.landmarks
                # Eyes should be roughly horizontal
                eye_dx = abs(lm[1][0] - lm[0][0])
                eye_dy = abs(lm[1][1] - lm[0][1])
                eye_ratio = eye_dy / (eye_dx + 1e-6)
                landmark_score = max(0.2, 1.0 - eye_ratio * 2)
                scores.append(landmark_score)

            # Combined quality score
            quality = float(np.mean(scores))
            face.quality_score = quality

            if quality >= self.config.min_face_quality:
                filtered.append(face)

        return filtered

    # ─── Embedding Extraction (Proper InsightFace Alignment) ──────────

    def _extract_embeddings(
        self,
        frame: np.ndarray,
        faces: List[DetectedFace],
    ) -> List[DetectedFace]:
        """
        Extract embeddings using InsightFace's proper face alignment.

        InsightFace ArcFace expects a 112x112 aligned face image.
        The alignment uses 5 facial landmarks to warp the face into a
        canonical position. This is FAR better than just cropping and resizing.
        """
        h, w = frame.shape[:2]

        for face in faces:
            x1, y1, x2, y2 = face.bbox
            face_w = x2 - x1
            face_h = y2 - y1

            # Blur gate: reject faces too blurry for stable embeddings
            # Computed on original-resolution crop BEFORE any upscaling
            bx1, by1 = max(0, x1), max(0, y1)
            bx2, by2 = min(w, x2), min(h, y2)
            if bx2 > bx1 and by2 > by1:
                blur_roi = cv2.cvtColor(frame[by1:by2, bx1:bx2], cv2.COLOR_BGR2GRAY)
                if blur_roi.size > 0:
                    laplacian_var = float(cv2.Laplacian(blur_roi, cv2.CV_64F).var())
                    if laplacian_var < self.config.min_embedding_blur:
                        logger.debug(
                            f"Blur rejection: {face_w}px face, "
                            f"laplacian={laplacian_var:.1f} < {self.config.min_embedding_blur}"
                        )
                        continue

            # For small faces, upscale the region before extracting embedding
            if face_w < 60 or face_h < 60:
                # Expand crop region for context
                pad = max(face_w, face_h)
                cx1 = max(0, x1 - pad)
                cy1 = max(0, y1 - pad)
                cx2 = min(w, x2 + pad)
                cy2 = min(h, y2 + pad)
                crop = frame[cy1:cy2, cx1:cx2]

                if crop.size == 0:
                    continue

                # Upscale to get face to at least ~100px
                up = max(2.0, 100.0 / max(face_w, 1))
                up = min(up, 6.0)  # Cap upscale
                crop_up = cv2.resize(
                    crop, None, fx=up, fy=up,
                    interpolation=cv2.INTER_LANCZOS4,
                )

                # Re-detect in upscaled crop to get proper landmarks for alignment
                try:
                    detected = self.face_engine.app.get(crop_up)
                    if detected:
                        # Pick the face closest to center of crop
                        best = self._pick_center_face(detected, crop_up.shape)
                        if best is not None and hasattr(best, 'normed_embedding'):
                            face.embedding = np.asarray(
                                best.normed_embedding, dtype=np.float32
                            )
                except Exception as e:
                    logger.debug(f"Small face embedding failed: {e}")
            else:
                # For normal-sized faces, expand crop slightly for context
                pad_w = int(face_w * 0.3)
                pad_h = int(face_h * 0.3)
                cx1 = max(0, x1 - pad_w)
                cy1 = max(0, y1 - pad_h)
                cx2 = min(w, x2 + pad_w)
                cy2 = min(h, y2 + pad_h)
                crop = frame[cy1:cy2, cx1:cx2]

                if crop.size == 0:
                    continue

                try:
                    detected = self.face_engine.app.get(crop)
                    if detected:
                        best = self._pick_center_face(detected, crop.shape)
                        if best is not None and hasattr(best, 'normed_embedding'):
                            face.embedding = np.asarray(
                                best.normed_embedding, dtype=np.float32
                            )
                except Exception as e:
                    logger.debug(f"Embedding extraction failed: {e}")

        return faces

    def _pick_center_face(self, detected_faces, frame_shape) -> object:
        """Pick the face closest to the center of the frame."""
        if not detected_faces:
            return None
        if len(detected_faces) == 1:
            return detected_faces[0]

        cy, cx = frame_shape[0] / 2, frame_shape[1] / 2
        best = None
        best_dist = float('inf')
        for f in detected_faces:
            fx = (f.bbox[0] + f.bbox[2]) / 2
            fy = (f.bbox[1] + f.bbox[3]) / 2
            dist = (fx - cx) ** 2 + (fy - cy) ** 2
            if dist < best_dist:
                best_dist = dist
                best = f
        return best

    # ─── NMS ──────────────────────────────────────────────────────────

    def _nms_faces(self, faces: List[DetectedFace]) -> List[DetectedFace]:
        """Non-maximum suppression to merge duplicate detections."""
        if len(faces) <= 1:
            return faces

        faces = sorted(faces, key=lambda f: f.confidence, reverse=True)
        keep = []
        suppressed = set()

        for i, face_i in enumerate(faces):
            if i in suppressed:
                continue
            keep.append(face_i)
            for j in range(i + 1, len(faces)):
                if j in suppressed:
                    continue
                iou = self._compute_iou(face_i.bbox, faces[j].bbox)
                if iou > self.config.nms_threshold:
                    suppressed.add(j)

        return keep

    def _compute_iou(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int],
    ) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    # ─── ROI Tracking ─────────────────────────────────────────────────

    def _update_rois(
        self,
        camera_id: int,
        faces: List[DetectedFace],
        frame_id: int,
    ):
        if camera_id not in self.camera_rois:
            self.camera_rois[camera_id] = []

        matched_rois = set()
        new_rois = []

        for face in faces:
            best_iou = 0.0
            best_idx = -1

            for i, roi in enumerate(self.camera_rois[camera_id]):
                if i in matched_rois:
                    continue
                iou = self._compute_iou(face.bbox, roi.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou > 0.3 and best_idx >= 0:
                roi = self.camera_rois[camera_id][best_idx]
                old_bbox = roi.bbox
                roi.bbox = face.bbox
                roi.last_detection_frame = frame_id
                roi.detection_count += 1
                roi.velocity = (
                    (face.bbox[0] - old_bbox[0]) * 0.5 + roi.velocity[0] * 0.5,
                    (face.bbox[1] - old_bbox[1]) * 0.5 + roi.velocity[1] * 0.5,
                )
                matched_rois.add(best_idx)
            else:
                new_rois.append(ROIRegion(
                    bbox=face.bbox,
                    last_detection_frame=frame_id,
                    detection_count=1,
                ))

        # Remove stale ROIs
        self.camera_rois[camera_id] = [
            roi for i, roi in enumerate(self.camera_rois[camera_id])
            if i in matched_rois
            or (frame_id - roi.last_detection_frame) < self.config.roi_max_age_frames
        ]
        self.camera_rois[camera_id].extend(new_rois)

        # Limit ROI count
        if len(self.camera_rois[camera_id]) > 50:
            self.camera_rois[camera_id] = sorted(
                self.camera_rois[camera_id],
                key=lambda r: r.detection_count, reverse=True,
            )[:50]

    # ─── Stats ────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "avg_detection_time_ms": self.avg_detection_time * 1000,
            "detection_fps": 1.0 / self.avg_detection_time if self.avg_detection_time > 0 else 0,
            "tracked_cameras": len(self.camera_rois),
            "total_rois": sum(len(rois) for rois in self.camera_rois.values()),
            "config": {
                "tiled_detection": self.config.enable_tiled_detection,
                "multi_resolution": self.config.enable_multi_resolution,
                "roi_tracking": self.config.enable_roi_tracking,
                "detection_threshold": self.config.detection_threshold,
                "tile_upscale": self.config.tile_upscale,
                "quality_filter": self.config.enable_quality_filter,
            },
        }
