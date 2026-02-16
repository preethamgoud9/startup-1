from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    name: str = "Face Recognition Attendance System"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True


class FaceRecognitionConfig(BaseSettings):
    detector: str = "SCRFD"
    embedding_model: str = "ArcFace"
    embedding_dim: int = 512
    det_size: list[int] = Field(default_factory=lambda: [640, 640])
    min_detection_score: float = 0.6
    recognition_threshold: float = 0.4
    device: str = "cpu"
    # Embedding stabilizer (temporal aggregation for long-range accuracy)
    stabilizer_enabled: bool = True
    stabilizer_min_frames: int = 3
    stabilizer_alpha: float = 0.3
    stabilizer_min_consistency: float = 0.70


class CameraConfig(BaseSettings):
    source_type: str = "usb"  # "usb" or "rtsp"
    usb_device_id: int = 0
    rtsp_url: str = ""
    fps_limit: int = 2
    frame_skip: int = 1


class EnrollmentConfig(BaseSettings):
    required_images: int = 15
    poses: list[str] = Field(default_factory=lambda: ["center", "left", "right", "up", "down"])
    min_face_quality: float = 0.7


class AttendanceConfig(BaseSettings):
    data_dir: str = "data/attendance"
    export_dir: str = "data/attendance/exports"


class EmbeddingsConfig(BaseSettings):
    storage_dir: str = "data/embeddings"
    gallery_file: str = "gallery.npz"


class SecurityConfig(BaseSettings):
    secret_key: str = "your-secret-key-change-this-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24  # 24 hours
    admin_username: str = "admin"
    admin_password: str = "admin123"  # Default password


class Settings(BaseSettings):
    app: AppConfig = Field(default_factory=AppConfig)
    face_recognition: FaceRecognitionConfig = Field(default_factory=FaceRecognitionConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    enrollment: EnrollmentConfig = Field(default_factory=EnrollmentConfig)
    attendance: AttendanceConfig = Field(default_factory=AttendanceConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Settings":
        with open(path, "r") as f:
            data: dict[str, Any] = yaml.safe_load(f)
        
        return cls(
            app=AppConfig(**data.get("app", {})),
            face_recognition=FaceRecognitionConfig(**data.get("face_recognition", {})),
            camera=CameraConfig(**data.get("camera", {})),
            enrollment=EnrollmentConfig(**data.get("enrollment", {})),
            attendance=AttendanceConfig(**data.get("attendance", {})),
            embeddings=EmbeddingsConfig(**data.get("embeddings", {})),
            security=SecurityConfig(**data.get("security", {})),
        )


def get_settings() -> Settings:
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    if config_path.exists():
        return Settings.from_yaml(config_path)
    return Settings()


settings = get_settings()
