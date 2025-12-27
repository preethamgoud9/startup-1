# How to Connect to CCTV Camera

This guide explains how to connect your Face Recognition Attendance System to an Ethernet-based CCTV/IP camera.

## Prerequisites
- CCTV/IP camera connected to your network (Ethernet or WiFi)
- Camera's IP address
- Camera's username and password
- Camera's RTSP stream URL format

---

## Step 1: Find Your Camera's IP Address

You can find your camera's IP address by:
- Checking your router's connected devices list
- Using the camera manufacturer's software/app
- Checking the camera's display/settings (if available)

Example IP: `192.168.1.64`

---

## Step 2: Get the RTSP URL Format

RTSP (Real-Time Streaming Protocol) URL format varies by manufacturer.

### Common RTSP URL Formats by Brand:

**Hikvision:**
```
rtsp://username:password@CAMERA_IP:554/Streaming/Channels/101
```

**Dahua:**
```
rtsp://username:password@CAMERA_IP:554/cam/realmonitor?channel=1&subtype=0
```

**Axis:**
```
rtsp://username:password@CAMERA_IP/axis-media/media.amp
```

**Foscam:**
```
rtsp://username:password@CAMERA_IP:88/videoMain
```

**Amcrest:**
```
rtsp://username:password@CAMERA_IP:554/cam/realmonitor?channel=1&subtype=0
```

**Generic/Other:**
```
rtsp://username:password@CAMERA_IP:554/stream1
```

**Note:** Check your camera's manual or manufacturer website for the exact RTSP URL format.

---

## Step 3: Configure the System

1. Open the configuration file:
   ```bash
   cd /Users/daivik2/Desktop/face_recognition
   nano backend/config.yaml
   ```

2. Update the `camera` section with your CCTV camera's RTSP URL:
   ```yaml
   camera:
     rtsp_url: "rtsp://admin:password123@192.168.1.64:554/Streaming/Channels/101"
     usb_device_id: 0
     fps_limit: 2
     frame_skip: 1
   ```

3. Replace the following:
   - `admin` → Your camera's username
   - `password123` → Your camera's password
   - `192.168.1.64` → Your camera's IP address
   - `/Streaming/Channels/101` → Your camera's stream path

---

## Step 4: Test the Connection

1. Test the RTSP URL using VLC Media Player (optional but recommended):
   - Open VLC
   - Go to: File → Open Network
   - Paste your RTSP URL
   - If the video plays, the URL is correct

2. Start the application:
   ```bash
   ./start.sh
   ```

3. Navigate to "Live Recognition" and click "Start Camera"

4. You should see your CCTV camera feed

---

## Troubleshooting

**Camera not connecting?**
- Verify the IP address is correct
- Check username and password
- Ensure camera is on the same network
- Try different stream paths (e.g., `/stream1`, `/stream2`)
- Check if camera's RTSP port is 554 (default) or different

**Video is laggy?**
- Adjust `fps_limit` in config (lower = less lag)
- Use lower resolution stream if available (e.g., `/Streaming/Channels/102` for Hikvision sub-stream)

**Connection timeout?**
- Check firewall settings
- Ensure RTSP port (554) is not blocked
- Verify camera supports RTSP (some only support HTTP/ONVIF)

---

## Camera Configuration Examples

### Example 1: Hikvision Camera
```yaml
camera:
  rtsp_url: "rtsp://admin:Hik12345@192.168.1.64:554/Streaming/Channels/101"
  usb_device_id: 0
  fps_limit: 2
  frame_skip: 1
```

### Example 2: Dahua Camera
```yaml
camera:
  rtsp_url: "rtsp://admin:Dahua123@192.168.1.65:554/cam/realmonitor?channel=1&subtype=0"
  usb_device_id: 0
  fps_limit: 2
  frame_skip: 1
```

### Example 3: Generic IP Camera
```yaml
camera:
  rtsp_url: "rtsp://admin:camera123@192.168.1.66:554/stream1"
  usb_device_id: 0
  fps_limit: 2
  frame_skip: 1
```

---

## Switching Between Cameras

To switch back to different camera sources:

**Mac Built-in Webcam:**
```yaml
camera:
  rtsp_url: ""
  usb_device_id: 0
```

**External USB Camera:**
```yaml
camera:
  rtsp_url: ""
  usb_device_id: 1  # or 2, 3, etc.
```

**Mobile Phone (IP Webcam app):**
```yaml
camera:
  rtsp_url: "http://PHONE_IP:8080/videofeed"
  usb_device_id: 0
```

**CCTV Camera:**
```yaml
camera:
  rtsp_url: "rtsp://username:password@CAMERA_IP:554/stream"
  usb_device_id: 0
```

After changing the configuration, restart the application:
```bash
./start.sh
```

---

## Network Access

The CCTV camera feed will be accessible from any device on your network:
- **Local access:** `http://localhost:5173`
- **Network access:** `http://YOUR_MAC_IP:5173`

All connected devices will see the same CCTV camera feed with face recognition results.

---

## Security Notes

- **Never expose RTSP credentials** in public repositories
- Use strong passwords for your cameras
- Consider using a separate VLAN for cameras
- Keep camera firmware updated
- Disable unused camera features/ports
