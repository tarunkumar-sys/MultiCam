import os
import time
import subprocess
import threading
import logging
from typing import Dict, Any, Optional
from config.settings import settings

logger = logging.getLogger(__name__)

class RecordingService:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.camera_writers: Dict[str, Any] = {}
        self.recording_threads: Dict[str, threading.Thread] = {}
        self.recording_stop_events: Dict[str, threading.Event] = {}
        self.writer_lock = threading.Lock()

    def start_recording(self, camera_id: str, w: int, h: int):
        with self.writer_lock:
            if camera_id in self.camera_writers:
                return True
                
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        local_path = os.path.join(settings.RECORDINGS_DIR, f"rec_{camera_id}_{timestamp}.mp4")
        
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", "10",
            "-i", "-", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "ultrafast", "-crf", "28", local_path
        ]
        
        try:
            process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            db_id = self.db_manager.start_recording(camera_id, local_path)
            stop_event = threading.Event()
            
            writer_data = {
                "process": process,
                "db_id": db_id,
                "start_time": time.time(),
                "file_path": local_path,
                "w": w, "h": h
            }
            
            with self.writer_lock:
                self.camera_writers[camera_id] = writer_data
                self.recording_stop_events[camera_id] = stop_event
                
            thread = threading.Thread(target=self._writer_loop, args=(camera_id, stop_event, process), daemon=True)
            thread.start()
            self.recording_threads[camera_id] = thread
            
            logger.info(f"✓ RecordingService: Started recording for {camera_id}")
            return True
        except Exception as e:
            logger.error(f"RecordingService Start Error: {e}")
            return False

    def stop_recording(self, camera_id: str):
        with self.writer_lock:
            writer_data = self.camera_writers.pop(camera_id, None)
            stop_event = self.recording_stop_events.pop(camera_id, None)
            
        if writer_data:
            if stop_event: stop_event.set()
            if camera_id in self.recording_threads:
                self.recording_threads[camera_id].join(timeout=5)
                del self.recording_threads[camera_id]
                
            try:
                writer_data["process"].stdin.close()
                writer_data["process"].wait(timeout=10)
            except Exception:
                writer_data["process"].kill()
                
            self.db_manager.end_recording(writer_data["db_id"])
            logger.info(f"! RecordingService: Stopped recording for {camera_id}")
            return True
        return False

    def _writer_loop(self, camera_id, stop_event, process):
        """Internal loop to feed frames to FFmpeg."""
        from services.vision import camera_results, results_lock
        while not stop_event.is_set():
            frame = None
            with results_lock:
                if camera_id in camera_results:
                    frame = camera_results[camera_id].get("rendered_frame")
            
            if frame is not None:
                try:
                    process.stdin.write(frame.tobytes())
                except Exception:
                    break
            time.sleep(0.1) # 10 FPS
