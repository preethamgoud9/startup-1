"""
CCTV Connection Helper - Production-level RTSP/ONVIF connection management.
Handles auto-discovery, brand-specific URL generation, DDNS resolution,
multiple protocols, and connection diagnostics.
"""

import logging
import re
import socket
import ssl
import subprocess
import threading
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2

logger = logging.getLogger(__name__)

# Supported streaming protocols
PROTOCOLS = {
    "rtsp": {
        "name": "RTSP",
        "description": "Real Time Streaming Protocol - Most common for IP cameras",
        "default_port": 554,
        "prefix": "rtsp://",
    },
    "rtsp_tcp": {
        "name": "RTSP over TCP",
        "description": "RTSP with TCP transport - More reliable over unstable networks",
        "default_port": 554,
        "prefix": "rtsp://",
        "transport": "tcp",
    },
    "http": {
        "name": "HTTP MJPEG",
        "description": "HTTP Motion JPEG stream - Works through firewalls",
        "default_port": 80,
        "prefix": "http://",
    },
    "https": {
        "name": "HTTPS MJPEG",
        "description": "Secure HTTP Motion JPEG stream",
        "default_port": 443,
        "prefix": "https://",
    },
    "rtmp": {
        "name": "RTMP",
        "description": "Real Time Messaging Protocol - For some newer cameras",
        "default_port": 1935,
        "prefix": "rtmp://",
    },
}

# Common DDNS providers and their patterns
DDNS_PROVIDERS = {
    "hikvision": ["hik-connect.com", "hikvisionddns.com"],
    "dahua": ["dahuaddns.com", "quickddns.com"],
    "no-ip": ["ddns.net", "no-ip.org", "no-ip.biz"],
    "dyndns": ["dyndns.org", "dyndns.com"],
    "generic": ["mynetav.net", "myfoscam.org", "dvrdns.org"],
}

# Common CCTV brands and their URL patterns for different protocols
CAMERA_BRANDS = {
    "hikvision": {
        "name": "Hikvision",
        "default_port": 554,
        "default_username": "admin",
        "rtsp_patterns": [
            "/Streaming/Channels/{channel}01",
            "/Streaming/Channels/{channel}02",
            "/ISAPI/Streaming/Channels/{channel}01",
            "/h264/ch{channel}/main/av_stream",
            "/h264/ch{channel}/sub/av_stream",
        ],
        "http_patterns": [
            "/ISAPI/Streaming/channels/{channel}01/httpPreview",
            "/Streaming/Channels/{channel}01/httppreview",
        ],
        "ddns_domains": ["hik-connect.com", "hikvisionddns.com"],
        "description": "Hikvision DVR/NVR/IP Cameras",
        "notes": "Default password format: admin + device verification code",
    },
    "dahua": {
        "name": "Dahua",
        "default_port": 554,
        "default_username": "admin",
        "rtsp_patterns": [
            "/cam/realmonitor?channel={channel}&subtype=0",
            "/cam/realmonitor?channel={channel}&subtype=1",
            "/live",
        ],
        "http_patterns": [
            "/cgi-bin/mjpg/video.cgi?channel={channel}&subtype=1",
        ],
        "ddns_domains": ["dahuaddns.com", "quickddns.com"],
        "description": "Dahua DVR/NVR/IP Cameras",
        "notes": "Also used by CP Plus, Amcrest, and other OEM brands",
    },
    "axis": {
        "name": "Axis",
        "default_port": 554,
        "default_username": "root",
        "rtsp_patterns": [
            "/axis-media/media.amp",
            "/axis-media/media.amp?videocodec=h264",
            "/mpeg4/{channel}/media.amp",
        ],
        "http_patterns": [
            "/mjpg/video.mjpg",
            "/axis-cgi/mjpg/video.cgi",
        ],
        "description": "Axis IP Cameras",
    },
    "uniview": {
        "name": "Uniview",
        "default_port": 554,
        "default_username": "admin",
        "rtsp_patterns": [
            "/media/video{channel}",
            "/unicast/c{channel}/s0/live",
            "/video{channel}",
        ],
        "http_patterns": [
            "/cgi-bin/snapshot.cgi?channel={channel}",
        ],
        "description": "Uniview NVR/IP Cameras",
    },
    "cp_plus": {
        "name": "CP Plus",
        "default_port": 554,
        "default_username": "admin",
        "rtsp_patterns": [
            "/cam/realmonitor?channel={channel}&subtype=0",
            "/cam/realmonitor?channel={channel}&subtype=1",
            "/live/ch{channel}",
        ],
        "http_patterns": [
            "/cgi-bin/mjpg/video.cgi?channel={channel}",
        ],
        "description": "CP Plus DVR/NVR (uses Dahua protocol)",
    },
    "hanwha": {
        "name": "Hanwha (Samsung)",
        "default_port": 554,
        "default_username": "admin",
        "rtsp_patterns": [
            "/profile{channel}/media.smp",
            "/onvif/profile{channel}/media.smp",
            "/rtsp/profile{channel}",
        ],
        "http_patterns": [
            "/video/mjpg.cgi",
        ],
        "description": "Hanwha Techwin / Samsung IP Cameras",
    },
    "reolink": {
        "name": "Reolink",
        "default_port": 554,
        "default_username": "admin",
        "rtsp_patterns": [
            "/h264Preview_01_main",
            "/h264Preview_01_sub",
            "//Preview_01_main",
        ],
        "http_patterns": [
            "/cgi-bin/api.cgi?cmd=Snap&channel={channel}",
        ],
        "description": "Reolink IP Cameras and NVRs",
    },
    "foscam": {
        "name": "Foscam",
        "default_port": 88,
        "default_username": "admin",
        "rtsp_patterns": [
            "/videoMain",
            "/videoSub",
            "/11",
            "/12",
        ],
        "http_patterns": [
            "/cgi-bin/CGIStream.cgi?cmd=GetMJStream",
            "/videostream.cgi",
        ],
        "description": "Foscam IP Cameras",
    },
    "generic_onvif": {
        "name": "Generic ONVIF",
        "default_port": 554,
        "default_username": "admin",
        "rtsp_patterns": [
            "/onvif1",
            "/stream1",
            "/video1",
            "/h264",
            "/live.sdp",
            "/media/video1",
            "/cam/realmonitor?channel=1&subtype=0",
            "/Streaming/Channels/101",
        ],
        "http_patterns": [
            "/video.mjpg",
            "/mjpeg.cgi",
            "/snap.jpg",
        ],
        "description": "Generic ONVIF-compatible cameras",
    },
}


@dataclass
class ConnectionResult:
    success: bool
    url: str
    message: str
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None
    fps: Optional[float] = None
    codec: Optional[str] = None
    protocol: Optional[str] = None
    resolved_ip: Optional[str] = None


@dataclass
class DiscoveredDevice:
    ip: str
    port: int
    brand_hint: Optional[str] = None
    is_ddns: bool = False
    ddns_provider: Optional[str] = None


@dataclass
class DDNSResolution:
    success: bool
    hostname: str
    resolved_ip: Optional[str] = None
    provider: Optional[str] = None
    error: Optional[str] = None


class CCTVHelper:
    """Production-level CCTV connection helper with protocol and DDNS support."""

    def __init__(self):
        self.connection_timeout = 10  # seconds

    def get_supported_protocols(self) -> dict:
        """Return list of supported streaming protocols."""
        return PROTOCOLS

    def get_supported_brands(self) -> dict:
        """Return list of supported camera brands with their details."""
        return {
            brand_id: {
                "name": info["name"],
                "description": info["description"],
                "default_port": info["default_port"],
                "default_username": info["default_username"],
                "notes": info.get("notes", ""),
                "has_http": "http_patterns" in info,
                "has_ddns": "ddns_domains" in info,
            }
            for brand_id, info in CAMERA_BRANDS.items()
        }

    def is_ddns_hostname(self, hostname: str) -> tuple[bool, Optional[str]]:
        """Check if hostname is a DDNS address and identify the provider."""
        hostname_lower = hostname.lower()
        
        # Check if it's an IP address (not DDNS)
        try:
            socket.inet_aton(hostname)
            return False, None
        except socket.error:
            pass
        
        # Check against known DDNS providers
        for provider, domains in DDNS_PROVIDERS.items():
            for domain in domains:
                if domain in hostname_lower:
                    return True, provider
        
        # Check brand-specific DDNS
        for brand_id, brand_info in CAMERA_BRANDS.items():
            if "ddns_domains" in brand_info:
                for domain in brand_info["ddns_domains"]:
                    if domain in hostname_lower:
                        return True, brand_id
        
        # If it contains dots but isn't an IP, assume it might be a hostname
        if "." in hostname and not hostname.replace(".", "").isdigit():
            return True, "unknown"
        
        return False, None

    def resolve_ddns(self, hostname: str) -> DDNSResolution:
        """Resolve DDNS hostname to IP address."""
        is_ddns, provider = self.is_ddns_hostname(hostname)
        
        if not is_ddns:
            # It's already an IP
            return DDNSResolution(
                success=True,
                hostname=hostname,
                resolved_ip=hostname,
                provider=None,
            )
        
        try:
            logger.info(f"Resolving DDNS hostname: {hostname}")
            resolved_ip = socket.gethostbyname(hostname)
            logger.info(f"Resolved {hostname} to {resolved_ip}")
            
            return DDNSResolution(
                success=True,
                hostname=hostname,
                resolved_ip=resolved_ip,
                provider=provider,
            )
        except socket.gaierror as e:
            logger.error(f"Failed to resolve DDNS hostname {hostname}: {e}")
            return DDNSResolution(
                success=False,
                hostname=hostname,
                error=f"Could not resolve hostname: {str(e)}",
                provider=provider,
            )

    def validate_ddns_connectivity(self, hostname: str, port: int = 554) -> dict:
        """Validate DDNS hostname can be resolved and port is reachable."""
        result = {
            "hostname": hostname,
            "is_ddns": False,
            "provider": None,
            "resolved_ip": None,
            "port_open": False,
            "error": None,
        }
        
        # Check if DDNS
        is_ddns, provider = self.is_ddns_hostname(hostname)
        result["is_ddns"] = is_ddns
        result["provider"] = provider
        
        # Resolve hostname
        resolution = self.resolve_ddns(hostname)
        if not resolution.success:
            result["error"] = resolution.error
            return result
        
        result["resolved_ip"] = resolution.resolved_ip
        
        # Check port connectivity
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock_result = sock.connect_ex((resolution.resolved_ip, port))
            sock.close()
            result["port_open"] = sock_result == 0
            if not result["port_open"]:
                result["error"] = f"Port {port} is not reachable"
        except Exception as e:
            result["error"] = f"Connection test failed: {str(e)}"
        
        return result

    def generate_stream_url(
        self,
        brand: str,
        host: str,
        username: str,
        password: str,
        port: int = 554,
        channel: int = 1,
        protocol: str = "rtsp",
        pattern_index: int = 0,
        stream_type: str = "main",
    ) -> str:
        """Generate stream URL for a specific brand and protocol."""
        if brand not in CAMERA_BRANDS:
            brand = "generic_onvif"

        brand_info = CAMERA_BRANDS[brand]
        protocol_info = PROTOCOLS.get(protocol, PROTOCOLS["rtsp"])
        
        # Select patterns based on protocol
        if protocol in ["http", "https"]:
            patterns = brand_info.get("http_patterns", brand_info.get("rtsp_patterns", []))
        else:
            patterns = brand_info.get("rtsp_patterns", [])
        
        if not patterns:
            patterns = ["/stream1"]
        
        pattern = patterns[min(pattern_index, len(patterns) - 1)]

        # Replace channel placeholder
        path = pattern.format(channel=channel)

        # URL encode credentials for special characters
        encoded_username = urllib.parse.quote(username, safe='')
        encoded_password = urllib.parse.quote(password, safe='')

        # Build URL with credentials
        prefix = protocol_info["prefix"]
        if username and password:
            return f"{prefix}{encoded_username}:{encoded_password}@{host}:{port}{path}"
        return f"{prefix}{host}:{port}{path}"

    def generate_rtsp_url(
        self,
        brand: str,
        ip: str,
        username: str,
        password: str,
        port: int = 554,
        channel: int = 1,
        pattern_index: int = 0,
    ) -> str:
        """Generate RTSP URL for a specific brand (legacy compatibility)."""
        return self.generate_stream_url(
            brand=brand,
            host=ip,
            username=username,
            password=password,
            port=port,
            channel=channel,
            protocol="rtsp",
            pattern_index=pattern_index,
        )

    def test_connection(
        self,
        stream_url: str,
        timeout: int = 10,
        protocol: str = "rtsp",
        use_tcp: bool = False,
    ) -> ConnectionResult:
        """Test stream connection and return detailed diagnostics."""
        logger.info(f"Testing connection to: {self._mask_url(stream_url)}")

        # Detect protocol from URL
        detected_protocol = "rtsp"
        if stream_url.startswith("http://"):
            detected_protocol = "http"
        elif stream_url.startswith("https://"):
            detected_protocol = "https"
        elif stream_url.startswith("rtmp://"):
            detected_protocol = "rtmp"

        try:
            # Force TCP transport for RTSP (more reliable than UDP)
            import os
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|rtsp_flags;prefer_tcp"
            
            # Configure capture based on protocol
            if detected_protocol in ["http", "https"]:
                cap = cv2.VideoCapture(stream_url)
            else:
                # Use FFmpeg backend for RTSP/RTMP
                cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Increased buffer
                
                # Force TCP transport if requested (more reliable)
                if use_tcp or protocol == "rtsp_tcp":
                    # Set RTSP transport to TCP via environment or FFmpeg options
                    import os
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

            # Set timeout via OpenCV (in milliseconds)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout * 1000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout * 1000)

            if not cap.isOpened():
                cap.release()
                return ConnectionResult(
                    success=False,
                    url=stream_url,
                    message="Failed to open stream. Check IP, port, and credentials.",
                    protocol=detected_protocol,
                )

            # Try to read a frame
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                return ConnectionResult(
                    success=False,
                    url=stream_url,
                    message="Stream opened but no frames received. Camera may be offline or stream path incorrect.",
                    protocol=detected_protocol,
                )

            # Get stream properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

            cap.release()

            return ConnectionResult(
                success=True,
                url=stream_url,
                message=f"Connected successfully! Resolution: {width}x{height}, FPS: {fps:.1f}",
                frame_width=width,
                frame_height=height,
                fps=fps if fps > 0 else None,
                codec=codec if codec.strip() else None,
                protocol=detected_protocol,
            )

        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return ConnectionResult(
                success=False,
                url=stream_url,
                message=f"Connection error: {str(e)}",
                protocol=detected_protocol,
            )

    def auto_detect_url(
        self,
        brand: str,
        host: str,
        username: str,
        password: str,
        port: int = 554,
        channel: int = 1,
        protocol: str = "rtsp",
        use_tcp: bool = False,
    ) -> ConnectionResult:
        """Try all URL patterns for a brand and return the first working one."""
        if brand not in CAMERA_BRANDS:
            brand = "generic_onvif"

        brand_info = CAMERA_BRANDS[brand]
        
        # Resolve DDNS if needed
        resolution = self.resolve_ddns(host)
        if not resolution.success:
            return ConnectionResult(
                success=False,
                url="",
                message=f"Failed to resolve hostname: {resolution.error}",
            )
        
        resolved_host = resolution.resolved_ip
        
        # Select patterns based on protocol
        if protocol in ["http", "https"]:
            patterns = brand_info.get("http_patterns", [])
            if not patterns:
                patterns = brand_info.get("rtsp_patterns", [])
        else:
            patterns = brand_info.get("rtsp_patterns", [])

        logger.info(f"Auto-detecting URL for {brand_info['name']} at {host}:{port} (resolved: {resolved_host})")

        # Try each pattern
        for i, pattern in enumerate(patterns):
            url = self.generate_stream_url(
                brand=brand,
                host=resolved_host,
                username=username,
                password=password,
                port=port,
                channel=channel,
                protocol=protocol,
                pattern_index=i,
            )

            result = self.test_connection(url, timeout=5, protocol=protocol, use_tcp=use_tcp)
            if result.success:
                logger.info(f"Found working URL pattern: {pattern}")
                result.resolved_ip = resolved_host
                return result

        # If RTSP fails, try HTTP as fallback
        if protocol == "rtsp" and "http_patterns" in brand_info:
            logger.info("RTSP patterns failed, trying HTTP MJPEG...")
            http_result = self.auto_detect_url(
                brand=brand,
                host=host,
                username=username,
                password=password,
                port=80,
                channel=channel,
                protocol="http",
            )
            if http_result.success:
                return http_result

        # If brand-specific patterns fail, try generic patterns
        if brand != "generic_onvif":
            logger.info("Brand-specific patterns failed, trying generic ONVIF patterns...")
            return self.auto_detect_url(
                brand="generic_onvif",
                host=host,
                username=username,
                password=password,
                port=port,
                channel=channel,
                protocol=protocol,
                use_tcp=use_tcp,
            )

        # Try TCP transport as last resort for RTSP
        if protocol == "rtsp" and not use_tcp:
            logger.info("UDP transport failed, trying TCP transport...")
            return self.auto_detect_url(
                brand=brand,
                host=host,
                username=username,
                password=password,
                port=port,
                channel=channel,
                protocol="rtsp",
                use_tcp=True,
            )

        return ConnectionResult(
            success=False,
            url="",
            message=f"Could not find working stream URL. Verify host ({host}), port ({port}), and credentials.",
            resolved_ip=resolved_host,
        )

    def scan_network_for_cameras(
        self,
        subnet: str = None,
        ports: list[int] = None,
    ) -> list[DiscoveredDevice]:
        """
        Scan local network for potential CCTV devices.
        This is a basic port scan - for production, consider ONVIF discovery.
        """
        if ports is None:
            ports = [554, 8554, 8080, 80]

        # Get local IP to determine subnet
        if subnet is None:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
                subnet = ".".join(local_ip.split(".")[:-1])
            except Exception:
                subnet = "192.168.1"

        discovered = []
        logger.info(f"Scanning network {subnet}.0/24 for cameras...")

        def check_host(ip: str, port: int) -> Optional[DiscoveredDevice]:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                result = sock.connect_ex((ip, port))
                sock.close()
                if result == 0:
                    return DiscoveredDevice(ip=ip, port=port)
            except Exception:
                pass
            return None

        # Scan common IP ranges with thread pool
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = []
            for i in range(1, 255):
                ip = f"{subnet}.{i}"
                for port in ports:
                    futures.append(executor.submit(check_host, ip, port))

            for future in as_completed(futures):
                result = future.result()
                if result:
                    discovered.append(result)
                    logger.info(f"Found device: {result.ip}:{result.port}")

        return discovered

    def _mask_url(self, url: str) -> str:
        """Mask credentials in URL for logging."""
        if "@" in url:
            parts = url.split("@")
            return f"rtsp://***:***@{parts[-1]}"
        return url

    def get_troubleshooting_tips(self, error_type: str) -> list[str]:
        """Return troubleshooting tips based on error type."""
        tips = {
            "connection_refused": [
                "Verify the camera/DVR IP address is correct",
                "Check if RTSP is enabled on the device (usually port 554)",
                "Ensure no firewall is blocking the connection",
                "Try accessing the device's web interface first",
            ],
            "authentication_failed": [
                "Double-check username and password",
                "Some devices use 'admin' as default username",
                "Password may be case-sensitive",
                "Check if the account has RTSP access permissions",
            ],
            "no_frames": [
                "The stream path may be incorrect for your device model",
                "Try a different channel number",
                "Use sub-stream (lower quality) if main stream fails",
                "Check if the camera is recording or in standby mode",
            ],
            "timeout": [
                "Network latency may be too high",
                "Camera may be overloaded with connections",
                "Try reducing video quality/resolution in camera settings",
                "Check network bandwidth between server and camera",
            ],
            "general": [
                "Ensure camera firmware is up to date",
                "Try connecting via VLC player first to verify stream works",
                "Check camera's web interface for correct RTSP URL",
                "Contact camera manufacturer for RTSP URL format",
            ],
        }
        return tips.get(error_type, tips["general"])


# Singleton instance
cctv_helper = CCTVHelper()
