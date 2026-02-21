"""
Remote Access API - Endpoints for Tailscale-based remote camera management.
Provides status checks, network info, and connectivity testing for remote cameras.
"""

import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.auth import get_current_user
from app.services.tailscale_helper import tailscale_helper

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/remote", tags=["remote"])


# --- Request/Response models ---

class TCPTestRequest(BaseModel):
    host: str
    port: int = 554


class RTSPTestRequest(BaseModel):
    url: str
    timeout: int = 10


class TailscaleStatusResponse(BaseModel):
    installed: bool
    running: bool
    version: Optional[str] = None
    tailnet_name: Optional[str] = None
    self_ip: Optional[str] = None
    hostname: Optional[str] = None
    backend_state: Optional[str] = None
    error: Optional[str] = None


class PeerResponse(BaseModel):
    hostname: str
    ip: str
    os: Optional[str] = None
    online: bool = False
    is_exit_node: bool = False
    is_subnet_router: bool = False
    subnet_routes: list[str] = []
    tags: list[str] = []
    tailscale_ips: list[str] = []


class NetworkInfoResponse(BaseModel):
    peers: list[PeerResponse] = []
    subnet_routes: list[str] = []
    online_count: int = 0
    total_count: int = 0


class TCPTestResponse(BaseModel):
    success: bool
    host: str
    port: int
    latency_ms: Optional[float] = None
    error: Optional[str] = None


class RTSPTestResponse(BaseModel):
    success: bool
    url: str
    message: str
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None
    fps: Optional[float] = None
    codec: Optional[str] = None


# --- Endpoints ---

@router.get("/status", response_model=TailscaleStatusResponse)
async def get_tailscale_status(
    _user: Annotated[dict, Depends(get_current_user)],
):
    """Check Tailscale installation and connection status."""
    status = tailscale_helper.get_status()
    return TailscaleStatusResponse(
        installed=status.installed,
        running=status.running,
        version=status.version,
        tailnet_name=status.tailnet_name,
        self_ip=status.self_ip,
        hostname=status.hostname,
        backend_state=status.backend_state,
        error=status.error,
    )


@router.get("/network", response_model=NetworkInfoResponse)
async def get_network_info(
    _user: Annotated[dict, Depends(get_current_user)],
):
    """Get Tailscale network peers and subnet routes."""
    info = tailscale_helper.get_network_info()
    return NetworkInfoResponse(
        peers=[
            PeerResponse(
                hostname=p.hostname,
                ip=p.ip,
                os=p.os,
                online=p.online,
                is_exit_node=p.is_exit_node,
                is_subnet_router=p.is_subnet_router,
                subnet_routes=p.subnet_routes,
                tags=p.tags,
                tailscale_ips=p.tailscale_ips,
            )
            for p in info.peers
        ],
        subnet_routes=info.subnet_routes,
        online_count=info.online_count,
        total_count=info.total_count,
    )


@router.post("/test-connection", response_model=TCPTestResponse)
async def test_remote_connection(
    data: TCPTestRequest,
    _user: Annotated[dict, Depends(get_current_user)],
):
    """Test TCP connectivity to a remote IP:port over Tailscale."""
    result = tailscale_helper.test_tcp_connection(data.host, data.port)
    return TCPTestResponse(
        success=result.success,
        host=result.host,
        port=result.port,
        latency_ms=result.latency_ms,
        error=result.error,
    )


@router.post("/test-rtsp", response_model=RTSPTestResponse)
async def test_remote_rtsp(
    data: RTSPTestRequest,
    _user: Annotated[dict, Depends(get_current_user)],
):
    """Test RTSP stream over Tailscale tunnel."""
    result = tailscale_helper.test_rtsp_stream(data.url, data.timeout)
    return RTSPTestResponse(
        success=result.success,
        url=result.url,
        message=result.message,
        frame_width=result.frame_width,
        frame_height=result.frame_height,
        fps=result.fps,
        codec=result.codec,
    )
