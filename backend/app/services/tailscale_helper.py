"""
Tailscale Helper - CLI wrapper for Tailscale status, network info, and connectivity testing.
Used by the Remote Access feature to manage surveillance cameras over Tailscale VPN.
"""

import json
import logging
import platform
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2

logger = logging.getLogger(__name__)


@dataclass
class TailscaleStatus:
    installed: bool = False
    running: bool = False
    version: Optional[str] = None
    tailnet_name: Optional[str] = None
    self_ip: Optional[str] = None
    hostname: Optional[str] = None
    backend_state: Optional[str] = None
    error: Optional[str] = None


@dataclass
class TailscalePeer:
    hostname: str
    ip: str
    os: Optional[str] = None
    online: bool = False
    is_exit_node: bool = False
    is_subnet_router: bool = False
    subnet_routes: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    tailscale_ips: list[str] = field(default_factory=list)


@dataclass
class NetworkInfo:
    peers: list[TailscalePeer] = field(default_factory=list)
    subnet_routes: list[str] = field(default_factory=list)
    online_count: int = 0
    total_count: int = 0


@dataclass
class TCPTestResult:
    success: bool
    host: str
    port: int
    latency_ms: Optional[float] = None
    error: Optional[str] = None


@dataclass
class RTSPTestResult:
    success: bool
    url: str
    message: str
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None
    fps: Optional[float] = None
    codec: Optional[str] = None


class TailscaleHelper:
    """Wrapper around the Tailscale CLI for status checks and network info."""

    def __init__(self):
        self._binary_path: Optional[str] = None

    def get_tailscale_binary(self) -> Optional[str]:
        """Find the Tailscale CLI binary, cross-platform."""
        if self._binary_path:
            return self._binary_path

        system = platform.system()

        if system == "Darwin":
            # macOS: check app bundle first, then PATH
            mac_paths = [
                "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
                "/usr/local/bin/tailscale",
            ]
            for p in mac_paths:
                if Path(p).exists():
                    self._binary_path = p
                    return p

        # All platforms: check PATH
        found = shutil.which("tailscale")
        if found:
            self._binary_path = found
            return found

        if system == "Windows":
            # Windows: check common install paths
            win_paths = [
                Path.home() / "AppData" / "Local" / "Tailscale" / "tailscale.exe",
                Path("C:/Program Files/Tailscale/tailscale.exe"),
                Path("C:/Program Files (x86)/Tailscale/tailscale.exe"),
            ]
            for p in win_paths:
                if p.exists():
                    self._binary_path = str(p)
                    return str(p)

        return None

    def _run_cli(self, args: list[str], timeout: int = 10) -> tuple[bool, str]:
        """Run a tailscale CLI command and return (success, output)."""
        binary = self.get_tailscale_binary()
        if not binary:
            return False, "Tailscale binary not found"

        try:
            result = subprocess.run(
                [binary] + args,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                return True, result.stdout
            return False, result.stderr or result.stdout or f"Exit code {result.returncode}"
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except FileNotFoundError:
            self._binary_path = None
            return False, "Tailscale binary not found"
        except Exception as e:
            return False, str(e)

    def get_status(self) -> TailscaleStatus:
        """Get Tailscale installation and connection status."""
        status = TailscaleStatus()

        binary = self.get_tailscale_binary()
        if not binary:
            status.error = "Tailscale is not installed"
            return status

        status.installed = True

        ok, output = self._run_cli(["status", "--json"])
        if not ok:
            status.error = f"Tailscale not running: {output}"
            return status

        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            status.error = "Failed to parse Tailscale status"
            return status

        status.running = data.get("BackendState") == "Running"
        status.backend_state = data.get("BackendState")
        status.version = data.get("Version")

        # Extract self node info
        self_node = data.get("Self", {})
        if self_node:
            ts_ips = self_node.get("TailscaleIPs", [])
            if ts_ips:
                status.self_ip = ts_ips[0]
            status.hostname = self_node.get("HostName")

        # Tailnet name from CurrentTailnet or MagicDNSSuffix
        current_tailnet = data.get("CurrentTailnet", {})
        if current_tailnet:
            status.tailnet_name = current_tailnet.get("Name") or current_tailnet.get("MagicDNSSuffix")
        if not status.tailnet_name:
            status.tailnet_name = data.get("MagicDNSSuffix")

        return status

    def get_network_info(self) -> NetworkInfo:
        """Get peers and subnet routes from Tailscale."""
        info = NetworkInfo()

        ok, output = self._run_cli(["status", "--json"])
        if not ok:
            return info

        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return info

        peer_map = data.get("Peer") or {}
        all_subnet_routes = set()

        for _key, peer_data in peer_map.items():
            ts_ips = peer_data.get("TailscaleIPs", [])
            primary_ip = ts_ips[0] if ts_ips else ""
            online = peer_data.get("Online", False)

            # Parse subnet routes from AllowedIPs (non-Tailscale IPs)
            allowed_ips = peer_data.get("AllowedIPs", [])
            subnets = [
                ip for ip in allowed_ips
                if not ip.startswith("100.") and not ip.startswith("fd7a:")
            ]

            peer = TailscalePeer(
                hostname=peer_data.get("HostName", "unknown"),
                ip=primary_ip,
                os=peer_data.get("OS", ""),
                online=online,
                is_exit_node=peer_data.get("ExitNode", False),
                is_subnet_router=len(subnets) > 0,
                subnet_routes=subnets,
                tags=peer_data.get("Tags", []) or [],
                tailscale_ips=ts_ips,
            )
            info.peers.append(peer)

            if online:
                info.online_count += 1

            all_subnet_routes.update(subnets)

        info.total_count = len(info.peers)
        info.subnet_routes = sorted(all_subnet_routes)

        return info

    def test_tcp_connection(self, host: str, port: int, timeout: int = 5) -> TCPTestResult:
        """Test TCP connectivity to a remote host:port with latency measurement."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)

            start = time.monotonic()
            result = sock.connect_ex((host, port))
            elapsed = (time.monotonic() - start) * 1000  # ms

            sock.close()

            if result == 0:
                return TCPTestResult(
                    success=True,
                    host=host,
                    port=port,
                    latency_ms=round(elapsed, 1),
                )
            else:
                return TCPTestResult(
                    success=False,
                    host=host,
                    port=port,
                    error=f"Connection refused (port {port} closed)",
                )
        except socket.timeout:
            return TCPTestResult(
                success=False,
                host=host,
                port=port,
                error=f"Connection timed out after {timeout}s",
            )
        except socket.gaierror:
            return TCPTestResult(
                success=False,
                host=host,
                port=port,
                error=f"Could not resolve hostname: {host}",
            )
        except Exception as e:
            return TCPTestResult(
                success=False,
                host=host,
                port=port,
                error=str(e),
            )

    def test_rtsp_stream(self, url: str, timeout: int = 10) -> RTSPTestResult:
        """Test RTSP stream connectivity using OpenCV (same pattern as cctv_helper)."""
        logger.info(f"Testing RTSP stream: {self._mask_url(url)}")

        try:
            import os
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|rtsp_flags;prefer_tcp"

            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout * 1000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout * 1000)

            if not cap.isOpened():
                cap.release()
                return RTSPTestResult(
                    success=False,
                    url=url,
                    message="Failed to open stream. Check URL, IP, and credentials.",
                )

            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                return RTSPTestResult(
                    success=False,
                    url=url,
                    message="Stream opened but no frames received. Camera may be offline.",
                )

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

            cap.release()

            return RTSPTestResult(
                success=True,
                url=url,
                message=f"Stream OK! {width}x{height} @ {fps:.1f} FPS",
                frame_width=width,
                frame_height=height,
                fps=fps if fps > 0 else None,
                codec=codec if codec.strip() else None,
            )

        except Exception as e:
            logger.error(f"RTSP test failed: {e}")
            return RTSPTestResult(
                success=False,
                url=url,
                message=f"RTSP error: {str(e)}",
            )

    def _mask_url(self, url: str) -> str:
        """Mask credentials in URL for logging."""
        if "@" in url:
            parts = url.split("@")
            return f"rtsp://***:***@{parts[-1]}"
        return url


# Singleton
tailscale_helper = TailscaleHelper()
