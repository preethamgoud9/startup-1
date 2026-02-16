"""
Face Engine - State-of-the-art recognition with ResNet100 backbone (antelopev2).

Upgrades over previous version:
- Model: antelopev2 (glintr100 ResNet100) vs buffalo_l (w600k_r50 ResNet50)
  - 2x larger backbone trained on cleaner/larger dataset (GLint360K)
  - Significantly better on hard cases: low quality, profile views, long-range
- Two-stage matching: centroid pre-filter → exemplar fusion → margin decision
- Margin-based rejection: ambiguous matches (close 1st/2nd) are rejected
- Pre-computed centroids: O(n_students) pre-filter before O(n_embeddings) exemplar search
- Gallery versioning: detects model mismatch and warns about re-enrollment
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from insightface.app import FaceAnalysis

from app.core.config import settings

logger = logging.getLogger(__name__)

# --- Model configuration ---
PRIMARY_MODEL = "antelopev2"  # glintr100 ResNet100 — state-of-the-art
FALLBACK_MODEL = "buffalo_l"  # w600k_r50 ResNet50 — decent fallback

# --- Recognition thresholds (tuned per model) ---
# antelopev2/glintr100 is more discriminative so thresholds can be slightly lower
MODEL_THRESHOLDS = {
    "antelopev2": {"high": 0.36, "medium": 0.42, "low": 0.50},
    "buffalo_l": {"high": 0.42, "medium": 0.48, "low": 0.55},
}

# Minimum margin between best and second-best student score.
# Rejects ambiguous matches where two students score similarly.
MIN_MARGIN = 0.05

# Score fusion weights: centroid vs exemplar
CENTROID_WEIGHT = 0.30
EXEMPLAR_WEIGHT = 0.70

# Maximum embeddings per student in gallery
MAX_EMBEDDINGS_PER_STUDENT = 10

# Augmentation noise (slightly lower for R100 — embeddings are more precise)
AUGMENT_EMBEDDING_NOISE_STD = 0.008


class FaceEngine:
    def __init__(self):
        self.app: Optional[FaceAnalysis] = None
        self.model_name: str = ""
        # Gallery storage
        self.gallery_embeddings: Optional[np.ndarray] = None
        self.gallery_labels: Optional[np.ndarray] = None
        self.gallery_metadata: dict[str, dict] = {}
        # Pre-computed index structures
        self._student_embedding_counts: dict[str, int] = {}
        self._student_centroids: dict[str, np.ndarray] = {}
        self._unique_student_ids: list[str] = []
        # Per-student label masks (precomputed for fast lookup)
        self._student_masks: dict[str, np.ndarray] = {}
        self.initialized = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self):
        if self.initialized:
            logger.warning("FaceEngine already initialized")
            return

        logger.info("Initializing FaceEngine...")

        providers = ["CPUExecutionProvider"]
        if settings.face_recognition.device == "gpu":
            providers = [
                "CUDAExecutionProvider",
                "CoreMLExecutionProvider",
                "CPUExecutionProvider",
            ]

        # Try primary model (R100), fall back to R50
        for model_name in [PRIMARY_MODEL, FALLBACK_MODEL]:
            try:
                logger.info(f"Loading model pack: {model_name}")
                self.app = FaceAnalysis(name=model_name, providers=providers)
                det_w, det_h = settings.face_recognition.det_size
                self.app.prepare(ctx_id=-1, det_size=(det_w, det_h))
                self.model_name = model_name
                logger.info(f"Successfully loaded model pack: {model_name}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                if model_name == FALLBACK_MODEL:
                    raise RuntimeError(
                        f"Could not load any face recognition model. "
                        f"Tried {PRIMARY_MODEL} and {FALLBACK_MODEL}. Last error: {e}"
                    )

        self.load_embeddings()
        self.initialized = True

        label = self.model_name
        if self.model_name == "antelopev2":
            label += " (glintr100 ResNet100)"
        elif self.model_name == "buffalo_l":
            label += " (w600k_r50 ResNet50)"
        logger.info(f"FaceEngine ready — recognition model: {label}")

    # ------------------------------------------------------------------
    # Gallery I/O
    # ------------------------------------------------------------------

    def load_embeddings(self):
        embeddings_dir = Path(settings.embeddings.storage_dir)
        gallery_path = embeddings_dir / settings.embeddings.gallery_file

        if not gallery_path.exists():
            logger.warning(f"Gallery file not found: {gallery_path}")
            self._init_empty_gallery()
            return

        try:
            data = np.load(str(gallery_path), allow_pickle=True)
            self.gallery_embeddings = data["embeddings"]
            self.gallery_labels = data["labels"]

            if "metadata" in data:
                self.gallery_metadata = data["metadata"].item()

            # Check for model mismatch
            if "model_name" in data:
                gallery_model = str(data["model_name"])
                if gallery_model and gallery_model != self.model_name:
                    logger.warning(
                        f"⚠ Gallery was created with '{gallery_model}' but current model is "
                        f"'{self.model_name}'. Embeddings are INCOMPATIBLE — "
                        f"all students must be re-enrolled for accurate recognition."
                    )

            self._rebuild_index()

            logger.info(
                f"Loaded gallery: {len(self.gallery_embeddings)} embeddings, "
                f"{len(self._unique_student_ids)} students"
            )
        except Exception as e:
            logger.error(f"Failed to load gallery: {e}")
            self._init_empty_gallery()

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
                model_name=self.model_name,
            )
            logger.info(f"Saved {len(self.gallery_embeddings)} embeddings to gallery")
        except Exception as e:
            logger.error(f"Failed to save gallery: {e}")
            raise

    def _init_empty_gallery(self):
        dim = settings.face_recognition.embedding_dim
        self.gallery_embeddings = np.empty((0, dim), dtype=np.float32)
        self.gallery_labels = np.empty((0,), dtype=object)
        self._student_embedding_counts = {}
        self._student_centroids = {}
        self._unique_student_ids = []
        self._student_masks = {}

    def _rebuild_index(self):
        """Rebuild per-student counts, centroids, masks, and unique ID list."""
        self._student_embedding_counts = {}
        self._student_centroids = {}
        self._student_masks = {}

        if self.gallery_labels is None or len(self.gallery_labels) == 0:
            self._unique_student_ids = []
            return

        # Count embeddings per student
        for label in self.gallery_labels:
            sid = str(label)
            self._student_embedding_counts[sid] = (
                self._student_embedding_counts.get(sid, 0) + 1
            )

        self._unique_student_ids = list(self._student_embedding_counts.keys())

        # Pre-compute per-student masks and centroids
        str_labels = np.array([str(l) for l in self.gallery_labels])
        for sid in self._unique_student_ids:
            mask = str_labels == sid
            self._student_masks[sid] = mask

            embs = self.gallery_embeddings[mask]
            centroid = np.mean(embs, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            self._student_centroids[sid] = centroid

    # ------------------------------------------------------------------
    # Detection & embedding extraction
    # ------------------------------------------------------------------

    def detect_faces(self, image: np.ndarray):
        if not self.initialized or self.app is None:
            raise RuntimeError("FaceEngine not initialized")

        faces = self.app.get(image)
        return [
            f
            for f in faces
            if f.det_score is not None
            and f.det_score >= settings.face_recognition.min_detection_score
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

    # ------------------------------------------------------------------
    # Recognition (two-stage: centroid pre-filter → exemplar fusion)
    # ------------------------------------------------------------------

    def recognize(
        self, embedding: np.ndarray, quality_score: float = 1.0
    ) -> tuple[Optional[str], float]:
        """
        Two-stage face recognition with margin-based scoring.

        Pipeline:
        1. Centroid pre-filter — fast dot-product against per-student centroids
           to narrow candidates (O(n_students) instead of O(n_embeddings)).
        2. Exemplar matching — top-K similarity against individual gallery entries
           for each candidate student.
        3. Score fusion — weighted combination of centroid and exemplar scores.
        4. Margin check — reject if best and second-best are too close (ambiguous).

        Args:
            embedding: L2-normalized face embedding from the recognition model.
            quality_score: Face quality score (0-1) from detector.

        Returns:
            (student_id, confidence) or (None, best_similarity).
        """
        if self.gallery_embeddings is None or len(self.gallery_embeddings) == 0:
            return None, 0.0

        if len(self._unique_student_ids) == 0:
            return None, 0.0

        # --- Threshold selection ---
        thresholds = MODEL_THRESHOLDS.get(
            self.model_name, MODEL_THRESHOLDS[FALLBACK_MODEL]
        )
        if quality_score >= 0.7:
            threshold = thresholds["high"]
        elif quality_score >= 0.4:
            threshold = thresholds["medium"]
        else:
            threshold = thresholds["low"]

        # Config override (use the stricter of the two)
        threshold = max(threshold, settings.face_recognition.recognition_threshold)

        # --- Stage 1: Centroid pre-filter ---
        centroid_scores: dict[str, float] = {}
        for sid, centroid in self._student_centroids.items():
            centroid_scores[sid] = float(np.dot(centroid, embedding))

        # Relaxed threshold for pre-filter (don't miss potential matches)
        prefilter_threshold = threshold - 0.12
        candidates = [
            sid
            for sid, score in centroid_scores.items()
            if score >= prefilter_threshold
        ]

        if not candidates:
            best_centroid = max(centroid_scores.values()) if centroid_scores else 0.0
            return None, best_centroid

        # --- Stage 2: Exemplar matching on candidates ---
        all_similarities = np.dot(self.gallery_embeddings, embedding)

        student_fused_scores: dict[str, float] = {}
        for sid in candidates:
            mask = self._student_masks[sid]
            student_sims = all_similarities[mask]

            # Top-K exemplar score (mean of top 3)
            k = min(3, len(student_sims))
            top_k_sims = np.partition(student_sims, -k)[-k:]
            exemplar_score = float(np.mean(top_k_sims))

            # Fuse centroid and exemplar scores
            fused = (
                CENTROID_WEIGHT * centroid_scores[sid]
                + EXEMPLAR_WEIGHT * exemplar_score
            )
            student_fused_scores[sid] = fused

        # --- Stage 3: Margin-based decision ---
        sorted_students = sorted(
            student_fused_scores.items(), key=lambda x: x[1], reverse=True
        )
        best_sid, best_score = sorted_students[0]

        # Check absolute threshold
        if best_score < threshold:
            return None, best_score

        # Check margin (does the best match clearly stand out?)
        if len(sorted_students) > 1:
            second_score = sorted_students[1][1]
            margin = best_score - second_score

            # Adaptive margin: stricter for low quality faces
            required_margin = MIN_MARGIN
            if quality_score < 0.4:
                required_margin = MIN_MARGIN * 1.5

            if margin < required_margin:
                logger.debug(
                    f"Ambiguous match rejected: {best_sid}={best_score:.3f} vs "
                    f"{sorted_students[1][0]}={second_score:.3f} "
                    f"(margin={margin:.3f} < {required_margin:.3f})"
                )
                return None, best_score

        # --- Stage 4: Z-score validation (gallery-adaptive) ---
        # Checks that the best match is a statistical outlier relative to all
        # candidate scores. Catches false positives where absolute threshold
        # passes but the match isn't distinctive for this gallery composition.
        if len(sorted_students) >= 3:
            all_scores = np.array([s for _, s in sorted_students])
            mu = float(np.mean(all_scores))
            sigma = float(np.std(all_scores))
            z_score = (best_score - mu) / (sigma + 0.01)

            min_z = 2.0 if quality_score < 0.5 else 1.5

            if z_score < min_z:
                logger.debug(
                    f"Z-score rejection: {best_sid} z={z_score:.2f} < {min_z} "
                    f"(score={best_score:.3f}, mu={mu:.3f}, sigma={sigma:.3f})"
                )
                return None, best_score

        return best_sid, best_score

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------

    def add_student(
        self,
        student_id: str,
        name: str,
        class_name: str,
        embeddings: list[np.ndarray],
        metadata: Optional[dict] = None,
    ):
        """
        Add student with multi-embedding gallery + augmentation.

        Stores diverse original embeddings plus augmented variants to capture
        natural variation in pose, lighting, and expression.
        """
        if not embeddings:
            raise ValueError("No embeddings provided")

        # Normalize all input embeddings
        normalized = []
        for emb in embeddings:
            emb = np.asarray(emb, dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            normalized.append(emb)

        # Select diverse embeddings (farthest-point sampling)
        selected = self._select_diverse_embeddings(
            normalized, max_count=MAX_EMBEDDINGS_PER_STUDENT // 2
        )

        # Generate augmented embeddings
        augmented = self._augment_embeddings(selected)

        # Combine and cap
        all_embeddings = selected + augmented
        if len(all_embeddings) > MAX_EMBEDDINGS_PER_STUDENT:
            all_embeddings = all_embeddings[:MAX_EMBEDDINGS_PER_STUDENT]

        # Remove existing entries for this student
        if self.gallery_labels is not None and len(self.gallery_labels) > 0:
            mask = np.array([str(l) != student_id for l in self.gallery_labels])
            self.gallery_embeddings = self.gallery_embeddings[mask]
            self.gallery_labels = self.gallery_labels[mask]

        # Add new embeddings
        new_embeddings = np.stack(all_embeddings, axis=0).astype(np.float32)
        new_labels = np.array([student_id] * len(all_embeddings), dtype=object)

        if self.gallery_embeddings is None or len(self.gallery_embeddings) == 0:
            self.gallery_embeddings = new_embeddings
            self.gallery_labels = new_labels
        else:
            self.gallery_embeddings = np.vstack(
                [self.gallery_embeddings, new_embeddings]
            )
            self.gallery_labels = np.append(self.gallery_labels, new_labels)

        # Update metadata
        student_metadata = {
            "name": name,
            "class": class_name,
            "embedding_count": len(all_embeddings),
        }
        if metadata:
            student_metadata.update(metadata)
        self.gallery_metadata[student_id] = student_metadata

        # Rebuild index structures (counts, centroids, masks)
        self._rebuild_index()

        self.save_embeddings()
        logger.info(
            f"Added student {student_id} ({name}) with "
            f"{len(all_embeddings)} embeddings ({len(selected)} original + "
            f"{len(augmented)} augmented)"
        )

    def _select_diverse_embeddings(
        self, embeddings: list[np.ndarray], max_count: int
    ) -> list[np.ndarray]:
        """
        Greedy farthest-point sampling to select the most diverse embeddings.
        Maximizes coverage of the student's appearance variation.
        """
        if len(embeddings) <= max_count:
            return embeddings

        selected = [embeddings[0]]
        remaining = list(range(1, len(embeddings)))

        while len(selected) < max_count and remaining:
            best_idx = -1
            best_min_dist = -1.0

            for i in remaining:
                min_dist = min(
                    1.0 - float(np.dot(embeddings[i], sel)) for sel in selected
                )
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = i

            if best_idx >= 0:
                selected.append(embeddings[best_idx])
                remaining.remove(best_idx)

        return selected

    def _augment_embeddings(
        self, embeddings: list[np.ndarray]
    ) -> list[np.ndarray]:
        """
        Generate augmented embeddings by adding small noise in embedding space.
        Fills gaps in the embedding space to handle unseen pose/lighting variations.
        """
        augmented = []
        for emb in embeddings:
            noise = (
                np.random.randn(*emb.shape).astype(np.float32)
                * AUGMENT_EMBEDDING_NOISE_STD
            )
            aug = emb + noise
            aug = aug / (np.linalg.norm(aug) + 1e-12)
            augmented.append(aug)
        return augmented

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_student_name(self, student_id: str) -> Optional[str]:
        meta = self.gallery_metadata.get(student_id)
        if meta:
            return meta.get("name")
        return None

    def get_gallery_stats(self) -> dict:
        total = (
            len(self.gallery_embeddings)
            if self.gallery_embeddings is not None
            else 0
        )
        return {
            "total_embeddings": total,
            "total_students": len(self._student_embedding_counts),
            "embeddings_per_student": dict(self._student_embedding_counts),
            "model": self.model_name,
        }

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
