import logging
from pathlib import Path
from typing import Optional

import numpy as np
from insightface.app import FaceAnalysis

from app.core.config import settings

logger = logging.getLogger(__name__)


class FaceEngine:
    def __init__(self):
        self.app: Optional[FaceAnalysis] = None
        self.gallery_embeddings: Optional[np.ndarray] = None
        self.gallery_labels: Optional[np.ndarray] = None
        self.gallery_metadata: dict[str, dict] = {}
        self.initialized = False

    def initialize(self):
        if self.initialized:
            logger.warning("FaceEngine already initialized")
            return

        logger.info("Initializing FaceEngine...")
        
        providers = ["CPUExecutionProvider"]
        if settings.face_recognition.device == "gpu":
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

        self.app = FaceAnalysis(providers=providers)
        det_w, det_h = settings.face_recognition.det_size
        self.app.prepare(ctx_id=-1, det_size=(det_w, det_h))

        self.load_embeddings()
        self.initialized = True
        logger.info("FaceEngine initialized successfully")

    def load_embeddings(self):
        embeddings_dir = Path(settings.embeddings.storage_dir)
        gallery_path = embeddings_dir / settings.embeddings.gallery_file

        if not gallery_path.exists():
            logger.warning(f"Gallery file not found: {gallery_path}")
            self.gallery_embeddings = np.empty((0, settings.face_recognition.embedding_dim), dtype=np.float32)
            self.gallery_labels = np.empty((0,), dtype=object)
            return

        try:
            data = np.load(str(gallery_path), allow_pickle=True)
            self.gallery_embeddings = data["embeddings"]
            self.gallery_labels = data["labels"]
            
            if "metadata" in data:
                self.gallery_metadata = data["metadata"].item()
            
            logger.info(f"Loaded {len(self.gallery_embeddings)} embeddings from gallery")
        except Exception as e:
            logger.error(f"Failed to load gallery: {e}")
            self.gallery_embeddings = np.empty((0, settings.face_recognition.embedding_dim), dtype=np.float32)
            self.gallery_labels = np.empty((0,), dtype=object)

    def save_embeddings(self):
        embeddings_dir = Path(settings.embeddings.storage_dir)
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        gallery_path = embeddings_dir / settings.embeddings.gallery_file

        try:
            np.savez_compressed(
                str(gallery_path),
                embeddings=self.gallery_embeddings,
                labels=self.gallery_labels,
                metadata=self.gallery_metadata,
            )
            logger.info(f"Saved {len(self.gallery_embeddings)} embeddings to gallery")
        except Exception as e:
            logger.error(f"Failed to save gallery: {e}")
            raise

    def detect_faces(self, image: np.ndarray):
        if not self.initialized or self.app is None:
            raise RuntimeError("FaceEngine not initialized")
        
        faces = self.app.get(image)
        return [
            f for f in faces
            if f.det_score is not None and f.det_score >= settings.face_recognition.min_detection_score
        ]

    def get_embedding(self, image: np.ndarray):
        faces = self.detect_faces(image)
        if not faces:
            return None, None
        
        face = self._largest_face(faces)
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            return None, None
        
        return np.asarray(embedding, dtype=np.float32), face

    def recognize(self, embedding: np.ndarray) -> tuple[Optional[str], float]:
        if self.gallery_embeddings is None or len(self.gallery_embeddings) == 0:
            return None, 0.0

        similarities = np.dot(self.gallery_embeddings, embedding)
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= settings.face_recognition.recognition_threshold:
            return str(self.gallery_labels[best_idx]), best_score
        
        return None, best_score

    def add_student(self, student_id: str, name: str, class_name: str, embeddings: list[np.ndarray]):
        if not embeddings:
            raise ValueError("No embeddings provided")

        mean_embedding = np.mean(np.stack(embeddings, axis=0), axis=0)
        mean_embedding = mean_embedding / (np.linalg.norm(mean_embedding) + 1e-12)
        mean_embedding = mean_embedding.astype(np.float32)

        if self.gallery_embeddings is None or len(self.gallery_embeddings) == 0:
            self.gallery_embeddings = mean_embedding.reshape(1, -1)
            self.gallery_labels = np.array([student_id], dtype=object)
        else:
            existing_idx = np.where(self.gallery_labels == student_id)[0]
            if len(existing_idx) > 0:
                self.gallery_embeddings[existing_idx[0]] = mean_embedding
            else:
                self.gallery_embeddings = np.vstack([self.gallery_embeddings, mean_embedding])
                self.gallery_labels = np.append(self.gallery_labels, student_id)

        self.gallery_metadata[student_id] = {
            "name": name,
            "class": class_name,
        }

        self.save_embeddings()
        logger.info(f"Added student {student_id} ({name}) to gallery")

    def _largest_face(self, faces):
        if not faces:
            return None
        areas = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox
            areas.append(float(max(0.0, x2 - x1) * max(0.0, y2 - y1)))
        return faces[int(np.argmax(areas))]

    def cleanup(self):
        logger.info("Cleaning up FaceEngine...")
        self.app = None
        self.initialized = False
