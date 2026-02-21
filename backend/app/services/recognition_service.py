import base64
import logging
import threading
import time
from datetime import datetime
from typing import Optional, Union

import cv2
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class RecognitionService:
    def __init__(self):
        self.camera_thread: Optional[threading.Thread] = None
        self.running = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_results: list[dict] = []
        self.frame_lock = threading.Lock()
        self.frame_count = 0

    def start_camera(self, source: Optional[Union[int, str]] = None):
        if self.running:
            logger.warning("Camera already running")
            return

        # Determine source based on settings or override
        if source is not None:
            resolved_source = source
        elif settings.camera.source_type == "rtsp" and settings.camera.rtsp_url:
            resolved_source = settings.camera.rtsp_url
        else:
            resolved_source = settings.camera.usb_device_id

        log_source = self._describe_source(resolved_source)
        logger.info(f"Attempting to connect to: {log_source}")

        # For RTSP, use specific backend and options for better compatibility
        if isinstance(resolved_source, str) and resolved_source.startswith("rtsp"):
            self.cap = cv2.VideoCapture(resolved_source, cv2.CAP_FFMPEG)
            # Set buffer size to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            self.cap = cv2.VideoCapture(resolved_source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera: {log_source}")

        self.running = True
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()
        logger.info(f"Camera started: {log_source}")

    def stop_camera(self):
        if not self.running:
            return

        self.running = False
        if self.camera_thread:
            self.camera_thread.join(timeout=5.0)

        if self.cap:
            self.cap.release()
            self.cap = None

        logger.info("Camera stopped")

    def _camera_loop(self):
        while self.running:
            if self.cap is None:
                break

            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            self.frame_count += 1
            if self.frame_count % settings.camera.frame_skip != 0:
                continue

            with self.frame_lock:
                self.latest_frame = frame.copy()

            time.sleep(1.0 / settings.camera.fps_limit if settings.camera.fps_limit > 0 else 0.01)

    def _describe_source(self, source: Union[int, str]) -> str:
        if isinstance(source, str):
            # Mask credentials in RTSP URL for logging
            if "@" in source:
                parts = source.split("@")
                return f"rtsp://***@{parts[-1]}"
            return source
        return f"usb device {source}"

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def process_frame(self, face_engine) -> list[dict]:
        frame = self.get_latest_frame()
        if frame is None:
            return []

        try:
            faces = face_engine.detect_faces(frame)
            if not faces:
                with self.frame_lock:
                    self.latest_results = []
                return []

            # Collect embeddings and quality scores for batch recognition
            valid_faces = []
            embeddings_list = []
            quality_list = []
            for face in faces:
                embedding = getattr(face, "normed_embedding", None)
                if embedding is None:
                    continue
                valid_faces.append(face)
                embeddings_list.append(np.asarray(embedding, dtype=np.float32))
                quality_list.append(float(getattr(face, "det_score", 1.0)))

            if not embeddings_list:
                with self.frame_lock:
                    self.latest_results = []
                return []

            # Batch recognition — single matmul for all faces
            embeddings_batch = np.stack(embeddings_list, axis=0)
            quality_batch = np.array(quality_list, dtype=np.float32)
            matches = face_engine.recognize_batch(embeddings_batch, quality_batch)

            results = []
            now = datetime.now().isoformat()
            for face, (student_id, confidence) in zip(valid_faces, matches):
                bbox = face.bbox.astype(int).tolist()
                result = {
                    "student_id": student_id,
                    "confidence": float(confidence),
                    "is_known": student_id is not None,
                    "bbox": bbox,
                    "timestamp": now,
                }

                if student_id and student_id in face_engine.gallery_metadata:
                    metadata = face_engine.gallery_metadata[student_id]
                    result["name"] = metadata.get("name")
                    result["class"] = metadata.get("class")

                results.append(result)

            with self.frame_lock:
                self.latest_results = results

            return results

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return []

    def get_annotated_frame(self) -> Optional[str]:
        frame = self.get_latest_frame()
        if frame is None:
            return None

        with self.frame_lock:
            results = self.latest_results.copy()

        for result in results:
            bbox = result.get("bbox")
            if not bbox:
                continue

            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if result["is_known"] else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label_parts = []
            if result.get("name"):
                label_parts.append(result["name"])
            elif result["is_known"] and result.get("student_id"):
                label_parts.append(result["student_id"])
            else:
                label_parts.append("Unknown")

            label_parts.append(f"{result['confidence']:.2f}")
            label = " - ".join(label_parts)

            cv2.putText(
                frame,
                label,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{frame_base64}"


recognition_service = RecognitionService()
