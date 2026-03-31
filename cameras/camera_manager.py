import cv2
import threading
import time
from typing import Dict, Any
from urllib.parse import urlparse


# Common RTSP stream paths to try when a bare IP is given (no path after port)
RTSP_PROBE_PATHS = [
    "/Streaming/Channels/101",               # Hikvision main stream
    "/Streaming/Channels/102",               # Hikvision sub stream
    "/Streaming/Channels/1",                 # Hikvision alt
    "/cam/realmonitor?channel=1&subtype=0",  # Dahua main
    "/cam/realmonitor?channel=1&subtype=1",  # Dahua sub
    "/h264/ch1/main/av_stream",              # Generic Hikvision
    "/live/ch00_0",                          # Generic
    "/stream1",                              # Generic
    "/axis-media/media.amp",                 # Axis
    "/onvif-media/media.amp",               # ONVIF
    "/MediaInput/h264",                      # Honeywell
    "",                                      # bare — try as-is last
]


def probe_rtsp_url(url: str) -> str:
    """
    If the RTSP URL has no path (just host:port), try common stream paths
    and return the first one that opens successfully.
    Returns the original URL if none work or if it already has a path.
    """
    parsed = urlparse(url)
    # If there's already a meaningful path, don't probe
    if parsed.path and parsed.path not in ("", "/"):
        print(f"[CameraProbe] URL already has path: {parsed.path}")
        return url

    base = url.rstrip("/")
    print(f"[CameraProbe] No stream path found, probing {len(RTSP_PROBE_PATHS)} common paths...")

    for path in RTSP_PROBE_PATHS:
        candidate = base + path
        print(f"[CameraProbe] Trying: ...{path or '(bare)'}")
        cap = cv2.VideoCapture(candidate, cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Give it a moment to negotiate
        opened = cap.isOpened()
        if opened:
            # Also try to grab a frame to confirm it's truly alive
            ret = cap.grab()
            cap.release()
            if ret:
                print(f"[CameraProbe] ✓ Found working path: {path}")
                return candidate
        else:
            cap.release()

    print(f"[CameraProbe] ✗ No working path found for {url}, using as-is")
    return url


def _open_capture(source):
    """
    Open a VideoCapture using the default backend.
    For local cameras, forces MJPEG FourCC for lower latency.
    """
    cap = cv2.VideoCapture(source, cv2.CAP_ANY)
    if cap.isOpened():
        # Request MJPEG from local cameras (USB/integrated)
        if isinstance(source, int):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class CameraHandler:
    def __init__(self, camera_id, source):
        self.camera_id = camera_id
        self.source = source
        self.cap = _open_capture(source)
        self.frame = None
        self.frame_id = 0
        self.running = True
        self.paused = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        fails = 0
        while self.running:
            if getattr(self, "paused", False):
                if getattr(self, "cap", None) and self.cap.isOpened():
                    self.cap.release()
                time.sleep(0.5)
                continue

            # Re-open if capture was released (pause→resume or reconnect)
            if not getattr(self, "cap", None) or not self.cap.isOpened():
                if not getattr(self, "paused", False):
                    self.cap = _open_capture(self.source)
                    if self.cap.isOpened():
                        fails = 0

            if not getattr(self, "cap", None) or not self.cap.isOpened():
                time.sleep(1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                fails += 1
                if fails > 30:
                    # Reconnect after sustained failures
                    if getattr(self, "cap", None):
                        self.cap.release()
                    time.sleep(1)
                    if not getattr(self, "paused", False):
                        self.cap = _open_capture(self.source)
                        if self.cap.isOpened():
                            fails = 0
                    fails = 0
                else:
                    time.sleep(0.05)
                continue

            fails = 0
            with self.lock:
                self.frame = frame
                self.frame_id += 1

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def get_frame_with_id(self):
        with self.lock:
            return (self.frame.copy(), self.frame_id) if self.frame is not None else (None, 0)

    def stop(self):
        self.running = False
        if getattr(self, "cap", None) and self.cap.isOpened():
            self.cap.release()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False


class CameraManager:
    def __init__(self):
        self.cameras: Dict[str, Any] = {}

    def add_camera(self, camera_id, source):
        if camera_id not in self.cameras:
            # For RTSP URLs without a stream path, auto-discover the correct one
            if isinstance(source, str) and source.lower().startswith("rtsp://"):
                source = probe_rtsp_url(source)
            handler = CameraHandler(camera_id, source)
            self.cameras[camera_id] = handler
            return True
        return False

    def remove_camera(self, camera_id):
        if camera_id in self.cameras:
            self.cameras[camera_id].stop()
            self.cameras.pop(camera_id, None)
            return True
        return False

    def toggle_camera(self, camera_id):
        if camera_id in self.cameras:
            handler = self.cameras[camera_id]
            if handler.paused:
                handler.resume()
            else:
                handler.pause()
            return True, not handler.paused
        return False, False

    def get_camera_frame(self, camera_id):
        if camera_id in self.cameras and not self.cameras[camera_id].paused:
            return self.cameras[camera_id].get_frame()
        return None

    def get_camera_frame_with_id(self, camera_id):
        if camera_id in self.cameras and not self.cameras[camera_id].paused:
            return self.cameras[camera_id].get_frame_with_id()
        return None, 0

    def get_active_cameras(self):
        return [cid for cid, handler in self.cameras.items() if not handler.paused]

    def get_all_cameras_info(self):
        return [
            {"id": cid, "status": "paused" if handler.paused else "active", "source": handler.source}
            for cid, handler in self.cameras.items()
        ]
