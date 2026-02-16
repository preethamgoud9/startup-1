import base64
import logging
import uuid
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class EnrollmentSession:
    def __init__(self, student_id: str, name: str, class_name: str):
        self.session_id = str(uuid.uuid4())
        self.student_id = student_id
        self.name = name
        self.class_name = class_name
        self.embeddings: list[np.ndarray] = []
        self.images_captured = 0
        self.required_images = settings.enrollment.required_images
        self.poses = settings.enrollment.poses.copy()
        self.current_pose_index = 0
        self.created_at = datetime.now()

    def get_next_pose(self) -> Optional[str]:
        if self.images_captured >= self.required_images:
            return None
        
        pose_cycle_length = len(self.poses)
        return self.poses[self.images_captured % pose_cycle_length]

    def is_complete(self) -> bool:
        return self.images_captured >= self.required_images

    def add_embedding(self, embedding: np.ndarray):
        self.embeddings.append(embedding)
        self.images_captured += 1


class EnrollmentService:
    def __init__(self):
        self.active_sessions: dict[str, EnrollmentSession] = {}

    def start_session(self, student_id: str, name: str, class_name: str) -> EnrollmentSession:
        for session_id, session in list(self.active_sessions.items()):
            if session.student_id == student_id:
                del self.active_sessions[session_id]
                logger.info(f"Removed existing session for student {student_id}")

        session = EnrollmentSession(student_id, name, class_name)
        self.active_sessions[session.session_id] = session
        logger.info(f"Started enrollment session {session.session_id} for student {student_id}")
        return session

    def get_session(self, session_id: str) -> Optional[EnrollmentSession]:
        return self.active_sessions.get(session_id)

    def capture_image(
        self,
        session_id: str,
        image_data: str,
        face_engine,
    ) -> tuple[bool, str, Optional[EnrollmentSession]]:
        session = self.get_session(session_id)
        if session is None:
            return False, "Session not found", None

        if session.is_complete():
            return False, "Session already completed", session

        try:
            image = self._decode_base64_image(image_data)
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            return False, f"Invalid image data: {str(e)}", session

        try:
            embedding, face = face_engine.get_embedding(image)
            if embedding is None or face is None:
                return False, "No face detected or face quality too low", session

            if face.det_score < settings.enrollment.min_face_quality:
                return False, f"Face quality too low: {face.det_score:.2f}", session

            session.add_embedding(embedding)
            logger.info(
                f"Captured image {session.images_captured}/{session.required_images} "
                f"for session {session_id}"
            )

            if session.is_complete():
                face_engine.add_student(
                    session.student_id,
                    session.name,
                    session.class_name,
                    session.embeddings,
                )
                del self.active_sessions[session_id]
                return True, "Enrollment completed successfully", session

            return True, f"Image captured successfully", session

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return False, f"Error processing image: {str(e)}", session

    def quick_enroll(
        self,
        student_id: str,
        name: str,
        class_name: str,
        image_data: str,
        face_engine,
    ) -> tuple[bool, str]:
        """
        Quick enrollment with a single image.
        Returns (success: bool, message: str)
        """
        try:
            image = self._decode_base64_image(image_data)
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            return False, f"Invalid image data: {str(e)}"

        try:
            embedding, face = face_engine.get_embedding(image)
            if embedding is None or face is None:
                return False, "No face detected in image"

            if face.det_score < settings.enrollment.min_face_quality:
                return False, f"Face quality too low: {face.det_score:.2f}. Please use better lighting or get closer to camera."

            # Add student with single embedding
            face_engine.add_student(
                student_id,
                name,
                class_name,
                [embedding],  # Single embedding in a list
                metadata={"quick_enrolled": True}
            )
            
            logger.info(f"Quick enrolled student {student_id} ({name})")
            return True, "Quick enrollment successful! Note: For best accuracy, consider re-enrolling with full 15-image process."

        except Exception as e:
            logger.error(f"Error during quick enrollment: {e}")
            return False, f"Error processing enrollment: {str(e)}"

    def upload_enroll(
        self,
        student_id: str,
        name: str,
        class_name: str,
        files: list,
        face_engine,
    ) -> tuple[bool, str, int, int]:
        """
        Enroll student from uploaded image files.
        Returns (success, message, processed_count, failed_count).
        """
        embeddings = []
        failed = 0

        for file in files:
            try:
                import asyncio
                # Read file bytes (UploadFile is async, but we can read sync in thread)
                contents = file.file.read()
                if not contents:
                    failed += 1
                    continue

                img_array = np.frombuffer(contents, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img is None:
                    logger.warning(f"Could not decode file: {file.filename}")
                    failed += 1
                    continue

                embedding, face = face_engine.get_embedding(img)
                if embedding is None or face is None:
                    logger.warning(f"No face detected in: {file.filename}")
                    failed += 1
                    continue

                if face.det_score < settings.enrollment.min_face_quality:
                    logger.warning(
                        f"Low quality face in {file.filename}: {face.det_score:.2f}"
                    )
                    failed += 1
                    continue

                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {e}")
                failed += 1

        if not embeddings:
            return False, "No valid faces found in uploaded images.", 0, failed

        try:
            face_engine.add_student(
                student_id,
                name,
                class_name,
                embeddings,
                metadata={"upload_enrolled": True, "source_images": len(embeddings)},
            )

            msg = (
                f"Upload enrollment successful! "
                f"{len(embeddings)} images processed"
            )
            if failed > 0:
                msg += f", {failed} failed"
            msg += "."

            logger.info(
                f"Upload enrolled student {student_id} ({name}) "
                f"with {len(embeddings)} images ({failed} failed)"
            )
            return True, msg, len(embeddings), failed
        except Exception as e:
            logger.error(f"Error during upload enrollment: {e}")
            return False, f"Enrollment failed: {str(e)}", 0, failed

    def cancel_session(self, session_id: str) -> bool:
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Cancelled session {session_id}")
            return True
        return False

    def _decode_base64_image(self, image_data: str) -> np.ndarray:
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image")

        return img


enrollment_service = EnrollmentService()
