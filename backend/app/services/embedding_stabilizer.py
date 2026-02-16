"""
Embedding Stabilizer — Temporal aggregation for long-range face recognition.

At 10-20m, faces are 15-40px wide. Single-frame embeddings from heavily
interpolated 112x112 ArcFace crops have high variance. This module tracks
faces across frames and produces stabilized embeddings via quality-weighted
exponential moving average (EMA).

Math:
    alpha_t = alpha_base * q_t                              (quality-adaptive rate)
    E_t = normalize(alpha_t * e_t + (1 - alpha_t) * E_{t-1})   (EMA update)

    Final recognition embedding (quality-weighted average of buffer):
    E_final = normalize(sum(q_i * e_i) / sum(q_i))

Recognition is gated on:
    1. Track maturity: >= min_frames consistent embeddings accumulated
    2. Quality budget: sum of quality scores >= min_quality_sum
    3. Consistency: cos_sim(e_t, E_{t-1}) >= min_consistency_sim for each accepted frame
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StabilizerConfig:
    # EMA learning rate: final alpha = alpha_base * quality_score
    # High quality (q=1.0): alpha=0.3 → 30% new, 70% history
    # Low quality (q=0.3): alpha=0.09 → 9% new, 91% history
    alpha_base: float = 0.3

    # Consistency: cosine similarity between current and stabilized embedding
    # Real faces at same distance: >0.80 between consecutive frames
    # False detections (textures): <0.40 between frames
    min_consistency_sim: float = 0.70

    # Maturity gate
    min_frames_for_recognition: int = 3
    min_quality_sum: float = 1.5

    # Track lifecycle
    max_track_age_seconds: float = 5.0
    max_tracks_per_camera: int = 30
    iou_match_threshold: float = 0.25  # Lower than typical (0.5) because small faces shift more

    # Embedding buffer for final quality-weighted average
    max_buffer_size: int = 10

    enabled: bool = True


@dataclass
class FaceTrack:
    """Temporal state for a single tracked face across frames."""
    track_id: int
    camera_id: int
    bbox: Tuple[int, int, int, int]

    stabilized_embedding: Optional[np.ndarray] = None
    embedding_buffer: List[Tuple[np.ndarray, float]] = field(default_factory=list)

    consistent_frame_count: int = 0
    total_quality_sum: float = 0.0

    created_at: float = 0.0
    last_seen: float = 0.0

    recognized_id: Optional[str] = None
    recognized_confidence: float = 0.0

    def is_mature(self, config: StabilizerConfig) -> bool:
        return (
            self.consistent_frame_count >= config.min_frames_for_recognition
            and self.total_quality_sum >= config.min_quality_sum
            and self.stabilized_embedding is not None
        )


class EmbeddingStabilizer:
    """
    Per-camera per-face temporal embedding stabilizer.

    Sits between OptimizedDetector output and FaceEngine.recognize() input.
    Tracks faces via IoU matching, applies quality-weighted EMA, and gates
    recognition on track maturity.
    """

    def __init__(self, config: Optional[StabilizerConfig] = None):
        self.config = config or StabilizerConfig()
        self._tracks: Dict[int, Dict[int, FaceTrack]] = {}  # camera_id -> {track_id -> track}
        self._next_track_id = 0
        self._lock = threading.Lock()

    def process(
        self,
        camera_id: int,
        detected_faces: list,
    ) -> List[dict]:
        """
        Process detected faces through temporal stabilization.

        Args:
            camera_id: Camera identifier.
            detected_faces: List of DetectedFace from OptimizedDetector.detect().

        Returns:
            List of dicts with keys:
                face, stabilized_embedding, track_id, is_mature, consistent_frames
        """
        if not self.config.enabled:
            return [
                {
                    "face": face,
                    "stabilized_embedding": face.embedding,
                    "track_id": -1,
                    "is_mature": face.embedding is not None,
                    "consistent_frames": 1,
                }
                for face in detected_faces
            ]

        now = time.time()

        with self._lock:
            if camera_id not in self._tracks:
                self._tracks[camera_id] = {}

            camera_tracks = self._tracks[camera_id]

            # Expire stale tracks
            expired = [
                tid
                for tid, track in camera_tracks.items()
                if now - track.last_seen > self.config.max_track_age_seconds
            ]
            for tid in expired:
                del camera_tracks[tid]

            # Match faces to existing tracks via IoU
            matched_pairs = self._match_faces_to_tracks(detected_faces, camera_tracks)

            results = []

            for face_idx, track_id in matched_pairs:
                face = detected_faces[face_idx]

                if face.embedding is None:
                    results.append(
                        {
                            "face": face,
                            "stabilized_embedding": None,
                            "track_id": track_id if track_id is not None else -1,
                            "is_mature": False,
                            "consistent_frames": 0,
                        }
                    )
                    continue

                # Get or create track
                if track_id is not None and track_id in camera_tracks:
                    track = camera_tracks[track_id]
                else:
                    track = self._create_track(camera_id, face, now)
                    camera_tracks[track.track_id] = track

                # Update with new embedding
                self._update_track(track, face, now)

                mature = track.is_mature(self.config)
                stab_emb = None
                if mature:
                    stab_emb = self._get_recognition_embedding(track)

                results.append(
                    {
                        "face": face,
                        "stabilized_embedding": stab_emb,
                        "track_id": track.track_id,
                        "is_mature": mature,
                        "consistent_frames": track.consistent_frame_count,
                    }
                )

            # Enforce max tracks per camera
            if len(camera_tracks) > self.config.max_tracks_per_camera:
                sorted_tracks = sorted(
                    camera_tracks.items(), key=lambda x: x[1].last_seen
                )
                for tid, _ in sorted_tracks[
                    : len(camera_tracks) - self.config.max_tracks_per_camera
                ]:
                    del camera_tracks[tid]

            return results

    # ------------------------------------------------------------------
    # Track matching
    # ------------------------------------------------------------------

    def _match_faces_to_tracks(
        self,
        faces: list,
        tracks: Dict[int, FaceTrack],
    ) -> List[Tuple[int, Optional[int]]]:
        """Greedy IoU matching: faces → existing tracks."""
        if not tracks:
            return [(i, None) for i in range(len(faces))]

        track_ids = list(tracks.keys())
        track_bboxes = [tracks[tid].bbox for tid in track_ids]

        # Collect all valid IoU pairs
        iou_pairs: List[Tuple[float, int, int]] = []
        for fi, face in enumerate(faces):
            for ti, tid in enumerate(track_ids):
                iou = self._compute_iou(face.bbox, track_bboxes[ti])
                if iou >= self.config.iou_match_threshold:
                    iou_pairs.append((iou, fi, tid))

        # Greedy assignment by descending IoU
        iou_pairs.sort(reverse=True)
        used_faces: set = set()
        used_tracks: set = set()
        matched: List[Tuple[int, Optional[int]]] = []

        for _, fi, tid in iou_pairs:
            if fi in used_faces or tid in used_tracks:
                continue
            matched.append((fi, tid))
            used_faces.add(fi)
            used_tracks.add(tid)

        # Unmatched faces → new tracks
        for fi in range(len(faces)):
            if fi not in used_faces:
                matched.append((fi, None))

        return matched

    # ------------------------------------------------------------------
    # Track lifecycle
    # ------------------------------------------------------------------

    def _create_track(
        self, camera_id: int, face, now: float
    ) -> FaceTrack:
        self._next_track_id += 1
        return FaceTrack(
            track_id=self._next_track_id,
            camera_id=camera_id,
            bbox=face.bbox,
            created_at=now,
            last_seen=now,
        )

    def _update_track(self, track: FaceTrack, face, now: float):
        """
        Update track with new frame's embedding.

        EMA: E_t = normalize(alpha * q * e_t + (1 - alpha * q) * E_{t-1})
        Consistency: reject if cos_sim(e_t, E_{t-1}) < threshold
        """
        embedding = np.asarray(face.embedding, dtype=np.float32)
        quality = face.quality_score

        track.bbox = face.bbox
        track.last_seen = now

        if track.stabilized_embedding is None:
            # First embedding — initialize track
            track.stabilized_embedding = embedding.copy()
            track.consistent_frame_count = 1
            track.total_quality_sum = quality
            track.embedding_buffer.append((embedding.copy(), quality))
            return

        # Consistency check
        sim = float(np.dot(embedding, track.stabilized_embedding))

        if sim < self.config.min_consistency_sim:
            # Inconsistent — decrement but don't destroy history.
            # Could be: false detection, extreme pose change, or new person in bbox.
            track.consistent_frame_count = max(
                0, track.consistent_frame_count - 1
            )
            logger.debug(
                f"Track {track.track_id}: inconsistent sim={sim:.3f} "
                f"< {self.config.min_consistency_sim}, frames={track.consistent_frame_count}"
            )
            return

        # EMA update: quality-weighted learning rate
        alpha = self.config.alpha_base * quality
        updated = alpha * embedding + (1.0 - alpha) * track.stabilized_embedding
        norm = np.linalg.norm(updated)
        if norm > 1e-8:
            track.stabilized_embedding = (updated / norm).astype(np.float32)

        track.consistent_frame_count += 1
        track.total_quality_sum += quality

        # Buffer for final weighted-average recognition
        track.embedding_buffer.append((embedding.copy(), quality))
        if len(track.embedding_buffer) > self.config.max_buffer_size:
            track.embedding_buffer.pop(0)

    def _get_recognition_embedding(self, track: FaceTrack) -> np.ndarray:
        """
        Quality-weighted average of buffered embeddings.

        Better than EMA for final recognition because it gives equal
        consideration to all good frames rather than exponentially decaying.

        E_final = normalize(sum(q_i * e_i) / sum(q_i))
        """
        if not track.embedding_buffer:
            return track.stabilized_embedding

        embeddings = np.array([e for e, _ in track.embedding_buffer])
        qualities = np.array([q for _, q in track.embedding_buffer])

        weights = qualities / (qualities.sum() + 1e-12)
        weighted_avg = np.sum(embeddings * weights[:, np.newaxis], axis=0)

        norm = np.linalg.norm(weighted_avg)
        if norm > 1e-8:
            return (weighted_avg / norm).astype(np.float32)

        return track.stabilized_embedding

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_iou(
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int],
    ) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
        area2 = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def clear_camera(self, camera_id: int):
        with self._lock:
            if camera_id in self._tracks:
                del self._tracks[camera_id]

    def clear_all(self):
        with self._lock:
            self._tracks.clear()

    def get_stats(self) -> dict:
        with self._lock:
            total = sum(len(t) for t in self._tracks.values())
            mature = sum(
                sum(1 for tr in ct.values() if tr.is_mature(self.config))
                for ct in self._tracks.values()
            )
            return {
                "total_tracks": total,
                "mature_tracks": mature,
                "cameras_tracked": len(self._tracks),
                "enabled": self.config.enabled,
            }
