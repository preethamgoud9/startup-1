"""
CCTV Connection API - Endpoints for easy CCTV setup and management.
Supports multiple protocols (RTSP, HTTP, RTMP), DDNS resolution, and auto-detection.
"""

import logging
from typing import Annotated, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.api.auth import get_current_user
from app.services.cctv_helper import cctv_helper, ConnectionResult
from app.core.settings_manager import save_dynamic_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cctv", tags=["cctv"])


class CCTVConnectionRequest(BaseModel):
    brand: str
    ip: str  # Can be IP address or DDNS hostname
    username: str
    password: str
    port: int = 554
    channel: int = 1
    protocol: str = "rtsp"  # rtsp, rtsp_tcp, http, https, rtmp
    use_tcp: bool = False  # Force TCP transport for RTSP


class CCTVTestRequest(BaseModel):
    stream_url: str
    protocol: str = "rtsp"
    use_tcp: bool = False


class DDNSValidateRequest(BaseModel):
    hostname: str
    port: int = 554


class CCTVConnectionResponse(BaseModel):
    success: bool
    message: str
    stream_url: Optional[str] = None
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None
    fps: Optional[float] = None
    codec: Optional[str] = None
    protocol: Optional[str] = None
    resolved_ip: Optional[str] = None
    troubleshooting_tips: Optional[list[str]] = None


class DDNSValidationResponse(BaseModel):
    hostname: str
    is_ddns: bool
    provider: Optional[str] = None
    resolved_ip: Optional[str] = None
    port_open: bool
    error: Optional[str] = None


class BrandInfo(BaseModel):
    name: str
    description: str
    default_port: int
    default_username: str
    notes: Optional[str] = ""
    has_http: bool = False
    has_ddns: bool = False


@router.get("/protocols")
async def get_supported_protocols():
    """Get list of supported streaming protocols."""
    return cctv_helper.get_supported_protocols()


@router.get("/brands")
async def get_supported_brands() -> dict[str, BrandInfo]:
    """Get list of supported CCTV brands with their default settings."""
    return cctv_helper.get_supported_brands()


@router.post("/validate-ddns", response_model=DDNSValidationResponse)
async def validate_ddns(
    data: DDNSValidateRequest,
    _user: Annotated[dict, Depends(get_current_user)],
):
    """Validate DDNS hostname resolution and port connectivity."""
    result = cctv_helper.validate_ddns_connectivity(data.hostname, data.port)
    return DDNSValidationResponse(**result)


@router.post("/test-url", response_model=CCTVConnectionResponse)
async def test_stream_url(
    data: CCTVTestRequest,
    _user: Annotated[dict, Depends(get_current_user)],
):
    """Test a specific stream URL (RTSP, HTTP, RTMP)."""
    result = cctv_helper.test_connection(
        data.stream_url,
        protocol=data.protocol,
        use_tcp=data.use_tcp,
    )
    
    response = CCTVConnectionResponse(
        success=result.success,
        message=result.message,
        stream_url=result.url if result.success else None,
        frame_width=result.frame_width,
        frame_height=result.frame_height,
        fps=result.fps,
        codec=result.codec,
        protocol=result.protocol,
        resolved_ip=result.resolved_ip,
    )
    
    if not result.success:
        if "credentials" in result.message.lower() or "authentication" in result.message.lower():
            response.troubleshooting_tips = cctv_helper.get_troubleshooting_tips("authentication_failed")
        elif "no frames" in result.message.lower():
            response.troubleshooting_tips = cctv_helper.get_troubleshooting_tips("no_frames")
        elif "timeout" in result.message.lower():
            response.troubleshooting_tips = cctv_helper.get_troubleshooting_tips("timeout")
        else:
            response.troubleshooting_tips = cctv_helper.get_troubleshooting_tips("connection_refused")
    
    return response


@router.post("/auto-connect", response_model=CCTVConnectionResponse)
async def auto_connect_cctv(
    data: CCTVConnectionRequest,
    _user: Annotated[dict, Depends(get_current_user)],
):
    """
    Automatically detect and connect to CCTV.
    Tries multiple URL patterns and protocols for the specified brand.
    Supports DDNS hostnames and automatic protocol fallback.
    """
    logger.info(f"Auto-connecting to {data.brand} camera at {data.ip}:{data.port} (protocol: {data.protocol})")
    
    result = cctv_helper.auto_detect_url(
        brand=data.brand,
        host=data.ip,
        username=data.username,
        password=data.password,
        port=data.port,
        channel=data.channel,
        protocol=data.protocol,
        use_tcp=data.use_tcp,
    )
    
    response = CCTVConnectionResponse(
        success=result.success,
        message=result.message,
        stream_url=result.url if result.success else None,
        frame_width=result.frame_width,
        frame_height=result.frame_height,
        fps=result.fps,
        codec=result.codec,
        protocol=result.protocol,
        resolved_ip=result.resolved_ip,
    )
    
    if not result.success:
        response.troubleshooting_tips = cctv_helper.get_troubleshooting_tips("general")
    
    return response


@router.post("/connect-and-save", response_model=CCTVConnectionResponse)
async def connect_and_save_cctv(
    data: CCTVConnectionRequest,
    _user: Annotated[dict, Depends(get_current_user)],
):
    """
    Auto-detect CCTV connection and save settings if successful.
    This is the main endpoint for easy CCTV setup.
    Supports DDNS, multiple protocols, and automatic fallback.
    """
    logger.info(f"Connecting and saving {data.brand} camera at {data.ip}:{data.port} (protocol: {data.protocol})")
    
    result = cctv_helper.auto_detect_url(
        brand=data.brand,
        host=data.ip,
        username=data.username,
        password=data.password,
        port=data.port,
        channel=data.channel,
        protocol=data.protocol,
        use_tcp=data.use_tcp,
    )
    
    if result.success:
        # Save the working configuration
        save_dynamic_settings({
            "source_type": "rtsp",
            "usb_device_id": 0,
            "rtsp_url": result.url,
        })
        logger.info("CCTV settings saved successfully")
    
    response = CCTVConnectionResponse(
        success=result.success,
        message=result.message + (" Settings saved!" if result.success else ""),
        stream_url=result.url if result.success else None,
        frame_width=result.frame_width,
        frame_height=result.frame_height,
        fps=result.fps,
        codec=result.codec,
        protocol=result.protocol,
        resolved_ip=result.resolved_ip,
    )
    
    if not result.success:
        response.troubleshooting_tips = cctv_helper.get_troubleshooting_tips("general")
    
    return response


@router.get("/scan-network")
async def scan_network_for_cameras(
    _user: Annotated[dict, Depends(get_current_user)],
):
    """
    Scan local network for potential CCTV devices.
    Returns list of IPs with open RTSP ports.
    """
    try:
        devices = cctv_helper.scan_network_for_cameras()
        return {
            "success": True,
            "devices": [{"ip": d.ip, "port": d.port} for d in devices],
            "message": f"Found {len(devices)} potential camera(s)" if devices else "No cameras found on network",
        }
    except Exception as e:
        logger.error(f"Network scan failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/troubleshooting/{error_type}")
async def get_troubleshooting_tips(error_type: str):
    """Get troubleshooting tips for a specific error type."""
    tips = cctv_helper.get_troubleshooting_tips(error_type)
    return {"error_type": error_type, "tips": tips}
