import time
import cv2
import numpy as np
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List
from core.tracker import ObjectTracker
from config.settings import settings
from services.notify import notification_service

logger = logging.getLogger(__name__)

# Global lock for camera results (to be used by routes/streams)
camera_results: Dict[str, Any] = {}
results_lock = threading.Lock()

class VisionService:
    def __init__(self, detector, camera_manager):
        from core.hw_manager import hw_manager
        self.detector = detector
        self.camera_manager = camera_manager
        self.batch_size = 4 if hw_manager.is_gpu_available() else 2
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="VisionWorker")
        self.trackers: Dict[str, ObjectTracker] = {}
        self.last_frame_ids: Dict[str, int] = {}
        self.occupancy_last_count: Dict[str, int] = {}

    def start(self):
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()
        logger.info("✓ VisionService: Orchestrator loop started.")

    def stop(self):
        self.running = False
        self.executor.shutdown(wait=True)

    def _run(self):
        while self.running:
            active_cams = self.camera_manager.get_active_cameras()
            if not active_cams:
                time.sleep(0.5)
                continue

            to_process = []
            for cam_id in active_cams:
                frame, frame_id = self.camera_manager.get_camera_frame_with_id(cam_id)
                if frame is not None and frame_id != self.last_frame_ids.get(cam_id, -1):
                    to_process.append((cam_id, frame, frame_id))
                    self.last_frame_ids[cam_id] = frame_id
                if len(to_process) >= self.batch_size * 2: break

            if not to_process:
                time.sleep(0.01)
                continue

            for i in range(0, len(to_process), self.batch_size):
                chunk = to_process[i:i + self.batch_size]
                chunk_cams = [c[0] for c in chunk]
                chunk_frames = [c[1] for c in chunk]
                chunk_ids = [c[2] for c in chunk]
                
                try:
                    batch_results = self.detector.detect_batch(chunk_frames)
                    for j, results in enumerate(batch_results):
                        self.executor.submit(
                            self.post_process_camera,
                            chunk_cams[j], chunk_frames[j], chunk_ids[j], results
                        )
                except Exception as e:
                    logger.error(f"VisionService Batch Error: {e}")

    def post_process_camera(self, camera_id, frame, frame_id, detections):
        try:
            h, w = frame.shape[:2]
            if camera_id not in self.trackers:
                self.trackers[camera_id] = ObjectTracker(max_age=30, n_init=1)
            
            tracker = self.trackers[camera_id]
            tracks = tracker.update(detections, frame)
            tracks = sorted(tracks, key=lambda x: x["bbox"][3])
            
            record_frame = frame.copy()
            processed = []
            
            for i, t in enumerate(tracks):
                bx1, by1, bx2, by2 = [int(v) for v in t["bbox"]]
                tid = t["id"]
                hue = (tid * 137) % 180
                color = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0])
                
                # Occlusion-aware rendering
                mask = np.zeros((h, w), dtype=np.uint8)
                for j in range(i + 1, len(tracks)):
                    fbx1, fby1, fbx2, fby2 = [int(v) for v in tracks[j]["bbox"]]
                    cv2.rectangle(mask, (fbx1, fby1), (fbx2, fby2), 255, -1)
                
                box_img = np.zeros_like(record_frame)
                cv2.rectangle(box_img, (bx1, by1), (bx2, by2), color, 2)
                cv2.putText(box_img, f"#{tid}", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                record_frame[mask == 0] = cv2.addWeighted(record_frame, 1.0, box_img, 1.0, 0)[mask == 0]

                processed.append({"id": tid, "bbox": [bx1, by1, bx2, by2], "name": "Unknown"})
            
            # Prepare person cards for sidebar (Face Crops)
            import base64
            person_cards = []
            identities = []
            for t in processed:
                tid = t["id"]
                name = t["name"]
                bx1, by1, bx2, by2 = t["bbox"]
                identities.append(name)
                
                # Extract face/person crop
                crop = frame[max(0, by1):min(h, by2), max(0, bx1):min(w, bx2)]
                if crop.size > 0:
                    _, b64_buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    b64_str = base64.b64encode(b64_buf).decode('utf-8')
                    person_cards.append({
                        "id": tid,
                        "name": name,
                        "face_crop": b64_str
                    })

            # Alert Logic (Simulated for now, can be expanded with area-rules)
            alert_active = False
            curr_count = len(tracks)
            if curr_count > self.occupancy_last_count.get(camera_id, 0):
                # alert_active = True # Trigger flash on new detection
                pass
            self.occupancy_last_count[camera_id] = curr_count

            with results_lock:
                camera_results[camera_id] = {
                    "rendered_frame": record_frame,
                    "raw_frame": frame,
                    "frame_id": frame_id,
                    "count": curr_count,
                    "identities": identities,
                    "person_cards": person_cards,
                    "alert_active": alert_active,
                    "timestamp": time.time()
                }
            
        except Exception as e:
            logger.error(f"Post-process error for {camera_id}: {e}")
