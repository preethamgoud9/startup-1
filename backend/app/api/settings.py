import logging
from typing import Annotated

import cv2
from fastapi import APIRouter, HTTPException, Depends

from app.api.auth import get_current_user
from app.core.settings_manager import (
    CameraSettingsUpdate,
    load_dynamic_settings,
    save_dynamic_settings,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("/camera")
async def get_camera_settings(_user: Annotated[dict, Depends(get_current_user)]):
    return load_dynamic_settings()


@router.post("/camera")
async def update_camera_settings(
    data: CameraSettingsUpdate,
    _user: Annotated[dict, Depends(get_current_user)]
):
    source_type = data.source_type

    if source_type == "rtsp":
        rtsp_url = data.rtsp_url
        if not rtsp_url:
            raise HTTPException(
                status_code=400,
                detail="RTSP URL is required when source type is 'rtsp'",
            )

        # Mask credentials for logging
        log_url = rtsp_url
        if "@" in rtsp_url:
            parts = rtsp_url.split("@")
            log_url = f"rtsp://***@{parts[-1]}"

        logger.info(f"Validating RTSP stream: {log_url}")
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            logger.error(f"Failed to connect to RTSP stream: {log_url}")
            cap.release()
            raise HTTPException(
                status_code=400,
                detail="Unable to connect to RTSP stream. Check URL and credentials.",
            )

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            logger.error(f"RTSP stream {log_url} connected but returned no frames")
            raise HTTPException(
                status_code=400,
                detail="RTSP stream connected but did not return frames. Check stream availability.",
            )

        logger.info(f"RTSP stream {log_url} validated successfully")

    else:
        usb_id = data.usb_device_id
        logger.info(f"Validating USB camera interface {usb_id}")
        cap = cv2.VideoCapture(usb_id)

        if not cap.isOpened():
            logger.error(f"Failed to open USB camera: {usb_id}")
            cap.release()
            raise HTTPException(
                status_code=400,
                detail=f"Unable to connect to USB camera index {usb_id}",
            )

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            logger.error(f"USB camera {usb_id} opened but returned no frames")
            raise HTTPException(
                status_code=400,
                detail=f"USB camera {usb_id} opened but did not return frames",
            )

        logger.info(f"USB camera {usb_id} validated successfully")

    save_dynamic_settings(data.model_dump())
    return {"status": "success", "message": "Camera configuration updated"}
