import logging
from fastapi import APIRouter, HTTPException, Depends
from app.core.settings_manager import load_dynamic_settings, save_dynamic_settings, CameraSettingsUpdate
from app.api.auth import get_current_user
from typing import Annotated
from urllib.parse import quote
import cv2

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/settings", tags=["settings"])

@router.get("/camera")
async def get_camera_settings(_user: Annotated[dict, Depends(get_current_user)]):
    return load_dynamic_settings()

def _sanitize_rtsp_credentials(rtsp_url: str) -> str:
    if not rtsp_url or not rtsp_url.lower().startswith("rtsp://"):
        return rtsp_url

    _prefix, remainder = rtsp_url.split("rtsp://", 1)
    if "@" not in remainder:
        return rtsp_url

    creds, rest = remainder.rsplit("@", 1)
    if ":" not in creds:
        return rtsp_url

    username, password = creds.split(":", 1)
    encoded_username = quote(username, safe="")
    encoded_password = quote(password, safe="")
    return f"rtsp://{encoded_username}:{encoded_password}@{rest}"


@router.post("/camera")
async def update_camera_settings(
    data: CameraSettingsUpdate,
    _user: Annotated[dict, Depends(get_current_user)]
):
    # Validate source
    source = data.rtsp_url if data.source_type == "rtsp" else data.usb_device_id
    if isinstance(source, str):
        source = _sanitize_rtsp_credentials(source)
        data.rtsp_url = source

    # Configure OpenCV for RTSP with longer timeout
    logger.info(f"Attempting to connect to camera source: {source}")
    logger.info(f"Source type: {data.source_type}")
    
    # Try with default backend first (auto-detect)
    cap = cv2.VideoCapture(source)
    
    logger.info(f"VideoCapture created, isOpened: {cap.isOpened()}")
    
    # Try to read a frame to confirm connection works
    if not cap.isOpened():
        logger.error(f"Failed to open camera source: {source}")
        cap.release()
        raise HTTPException(
            status_code=400, 
            detail=f"Unable to connect to camera source: {source}"
        )
    
    # Attempt to grab a frame to verify stream is working
    logger.info("Camera opened, attempting to read frame...")
    ret, frame = cap.read()
    if frame is not None:
        logger.info(f"Frame read result: ret={ret}, frame shape={frame.shape}")
    else:
        logger.info(f"Frame read result: ret={ret}, frame is None")
    cap.release()
    
    if not ret:
        logger.error(f"Connected but unable to read frames from: {source}")
        raise HTTPException(
            status_code=400,
            detail=f"Connected but unable to read frames from: {source}"
        )
    
    logger.info(f"Successfully connected and read frame from: {source}")
    
    save_dynamic_settings(data.model_dump())
    return {"status": "success", "message": "Camera configuration updated"}
