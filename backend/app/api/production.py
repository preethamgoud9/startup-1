"""
Production API - Endpoints for multi-camera production deployment.
Supports up to 32 cameras with GPU acceleration.
"""

import base64
import logging
import time
from typing import Annotated, Optional, List

import cv2
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.api.auth import get_current_user
from app.services.production_manager import production_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/production", tags=["production"])


# Request/Response Models
class CameraAddRequest(BaseModel):
    name: str
    stream_url: str
    enabled: bool = True
    fps_limit: int = 5
    resolution_scale: float = 1.0
    detection_enabled: bool = True
    recording_enabled: bool = False


class CameraUpdateRequest(BaseModel):
    name: Optional[str] = None
    stream_url: Optional[str] = None
    enabled: Optional[bool] = None
    fps_limit: Optional[int] = None
    resolution_scale: Optional[float] = None
    detection_enabled: Optional[bool] = None
    recording_enabled: Optional[bool] = None


class ProductionConfigUpdate(BaseModel):
    max_cameras: Optional[int] = None
    worker_threads: Optional[int] = None
    frame_buffer_size: Optional[int] = None
    detection_interval_ms: Optional[int] = None
    enable_recording: Optional[bool] = None
    recording_path: Optional[str] = None
    enable_alerts: Optional[bool] = None
    alert_cooldown_seconds: Optional[int] = None


class ProcessingModeRequest(BaseModel):
    mode: str  # cpu, cuda, opencl
    device_id: int = 0


class DetectionQualityRequest(BaseModel):
    quality: str  # fast, balanced, accurate, maximum


class CameraPriorityRequest(BaseModel):
    priority: str  # critical, high, normal, low, background


# System endpoints
@router.get("/status")
async def get_system_status(
    _user: Annotated[dict, Depends(get_current_user)],
):
    """Get overall production system status."""
    return production_manager.get_system_stats()


@router.get("/config")
async def get_production_config(
    _user: Annotated[dict, Depends(get_current_user)],
):
    """Get production configuration."""
    return production_manager.get_config()


@router.post("/config")
async def update_production_config(
    data: ProductionConfigUpdate,
    _user: Annotated[dict, Depends(get_current_user)],
):
    """Update production configuration."""
    return production_manager.update_config(data.model_dump(exclude_none=True))


@router.get("/gpu")
async def get_gpu_info(
    _user: Annotated[dict, Depends(get_current_user)],
):
    """Get GPU availability and current processing mode."""
    return production_manager.get_gpu_info()


@router.post("/gpu")
async def set_processing_mode(
    data: ProcessingModeRequest,
    _user: Annotated[dict, Depends(get_current_user)],
):
    """Set processing mode (CPU/CUDA/OpenCL)."""
    result = production_manager.set_processing_mode(data.mode, data.device_id)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/detection-qualities")
async def get_detection_qualities(
    _user: Annotated[dict, Depends(get_current_user)],
):
    """Get available detection quality presets."""
    return production_manager.get_detection_qualities()


@router.post("/detection-quality")
async def set_detection_quality(
    data: DetectionQualityRequest,
    _user: Annotated[dict, Depends(get_current_user)],
):
    """Set detection quality preset (fast/balanced/accurate/maximum)."""
    result = production_manager.set_detection_quality(data.quality)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/cameras/{camera_id}/detections")
async def get_camera_detections(
    camera_id: int,
    _user: Annotated[dict, Depends(get_current_user)],
):
    """Get latest face detections for a camera."""
    return production_manager.get_camera_detections(camera_id)


@router.get("/cameras/{camera_id}/recognitions")
async def get_camera_recognitions(
    camera_id: int,
    _user: Annotated[dict, Depends(get_current_user)],
):
    """Get latest face recognition results for a camera (with student IDs)."""
    return production_manager.get_camera_recognitions(camera_id)


@router.get("/attendance-feed")
async def get_attendance_feed(
    _user: Annotated[dict, Depends(get_current_user)],
):
    """Get recent auto-attendance events across all cameras."""
    return production_manager.get_recent_attendance_feed()


@router.get("/cameras/{camera_id}/annotated-frame")
async def get_annotated_frame(
    camera_id: int,
    _user: Annotated[dict, Depends(get_current_user)],
):
    """Get camera frame with recognition bounding boxes drawn."""
    frame = production_manager.get_frame(camera_id)
    if frame is None:
        raise HTTPException(status_code=404, detail="No frame available")

    frame = frame.copy()
    recognitions = production_manager.get_camera_recognitions(camera_id)

    for result in recognitions:
        bbox = result.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0) if result.get("is_known") else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = result.get("name") or result.get("student_id") or "Unknown"
        cv2.putText(
            frame,
            f"{label} {result['confidence']:.2f}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    return {
        "camera_id": camera_id,
        "frame": f"data:image/jpeg;base64,{frame_base64}",
        "recognitions": recognitions,
    }


@router.post("/cameras/from-cctv-setup")
async def add_camera_from_cctv_setup(
    data: CameraAddRequest,
    _user: Annotated[dict, Depends(get_current_user)],
):
    """Add camera using verified CCTV setup connection result."""
    result = production_manager.add_camera(data.model_dump())
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])

    # Auto-start if system is running
    if production_manager.running:
        production_manager.start_camera(result["camera_id"])

    return result


@router.post("/start")
async def start_production(
    _user: Annotated[dict, Depends(get_current_user)],
):
    """Start all enabled cameras."""
    result = production_manager.start_all()
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/stop")
async def stop_production(
    _user: Annotated[dict, Depends(get_current_user)],
):
    """Stop all cameras."""
    return production_manager.stop_all()


# Camera management endpoints
@router.get("/cameras")
async def get_all_cameras(
    _user: Annotated[dict, Depends(get_current_user)],
):
    """Get all cameras with their states."""
    return production_manager.get_all_cameras()


@router.post("/cameras")
async def add_camera(
    data: CameraAddRequest,
    _user: Annotated[dict, Depends(get_current_user)],
):
    """Add a new camera."""
    result = production_manager.add_camera(data.model_dump())
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/cameras/{camera_id}")
async def get_camera(
    camera_id: int,
    _user: Annotated[dict, Depends(get_current_user)],
):
    """Get a specific camera's state."""
    camera = production_manager.get_camera(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    return camera


@router.put("/cameras/{camera_id}")
async def update_camera(
    camera_id: int,
    data: CameraUpdateRequest,
    _user: Annotated[dict, Depends(get_current_user)],
):
    """Update camera configuration."""
    result = production_manager.update_camera(camera_id, data.model_dump(exclude_none=True))
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.delete("/cameras/{camera_id}")
async def remove_camera(
    camera_id: int,
    _user: Annotated[dict, Depends(get_current_user)],
):
    """Remove a camera."""
    result = production_manager.remove_camera(camera_id)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.post("/cameras/{camera_id}/start")
async def start_camera(
    camera_id: int,
    _user: Annotated[dict, Depends(get_current_user)],
):
    """Start a specific camera."""
    result = production_manager.start_camera(camera_id)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/cameras/{camera_id}/stop")
async def stop_camera(
    camera_id: int,
    _user: Annotated[dict, Depends(get_current_user)],
):
    """Stop a specific camera."""
    result = production_manager.stop_camera(camera_id)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/cameras/{camera_id}/frame")
async def get_camera_frame(
    camera_id: int,
    _user: Annotated[dict, Depends(get_current_user)],
):
    """Get the latest frame from a camera as base64 JPEG."""
    frame = production_manager.get_frame(camera_id)
    if frame is None:
        raise HTTPException(status_code=404, detail="No frame available")

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "camera_id": camera_id,
        "frame": f"data:image/jpeg;base64,{frame_base64}",
    }


@router.get("/grid")
async def get_camera_grid(
    _user: Annotated[dict, Depends(get_current_user)],
    columns: int = 4,
    max_cameras: int = 16,
):
    """Get a grid view of all camera frames."""
    cameras = production_manager.get_all_cameras()
    frames = []

    for camera in cameras[:max_cameras]:
        camera_id = camera["id"]
        frame = production_manager.get_frame(camera_id)
        
        frame_data = {
            "camera_id": camera_id,
            "name": camera["name"],
            "status": camera["status"],
            "fps": camera["fps"],
            "health": camera.get("health", "red"),
            "reconnect_attempts": camera.get("reconnect_attempts", 0),
            "recognitions": production_manager.get_camera_recognitions(camera_id),
            "frame": None,
        }

        if frame is not None:
            # Resize for grid view
            height, width = frame.shape[:2]
            grid_width = 320
            grid_height = int(height * (grid_width / width))
            resized = cv2.resize(frame, (grid_width, grid_height))
            
            _, buffer = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 60])
            frame_data["frame"] = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

        frames.append(frame_data)

    return {
        "columns": columns,
        "cameras": frames,
        "total": len(cameras),
        "timestamp": time.time(),  # For frontend sync
    }
