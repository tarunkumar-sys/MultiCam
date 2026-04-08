import cv2
import numpy as np
import os
import shutil
import torch
from fastapi import FastAPI, Request, File, UploadFile, Form, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
from database.sqlite_manager import SqliteManager
from utils.detector import PersonDetector
from utils.tracker import ObjectTracker
from utils.recognizer import FaceRecognizer
from cameras.camera_manager import CameraManager
import threading
import time
from typing import Dict, Any, Optional, Set
from datetime import datetime, timedelta
import pytz
from concurrent.futures import ThreadPoolExecutor
import queue
import asyncio
import logging
import json
import base64
import random
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set IST timezone
IST = pytz.timezone('Asia/Kolkata')

def get_ist_time():
    """Get current time in IST."""
    return datetime.now(IST)

def format_12h(dt):
    """Format datetime to 12-hour AM/PM string (e.g. 05:30:15 PM)."""
    if dt is None: return "N/A"
    # Convert to IST if needed
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt).astimezone(IST)
    else:
        dt = dt.astimezone(IST)
    return dt.strftime("%I:%M:%S %p")

# Security setup
security = HTTPBasic(auto_error=False)

# Simple admin credentials (in production, use database)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "deiadmin@789"

# Session storage (in production, use proper session management)
authenticated_sessions: set = set()

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify admin credentials."""
    if credentials:
        is_correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
        is_correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
        if is_correct_username and is_correct_password:
            return credentials.username
    return None

def require_auth(request: Request):
    """Check if user is authenticated via session cookie."""
    session_token = request.cookies.get("session")
    if session_token and session_token in authenticated_sessions:
        return True
    return False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize_rtsp_url(url: str) -> str:
    """Percent-encode special characters in the password portion of an RTSP URL.
    Handles passwords containing multiple '@' signs by using rfind to locate the
    last '@' as the user:pass / host boundary.
    """
    if not isinstance(url, str):
        return url
    url = url.strip()
    if not url.startswith("rtsp://"):
        return url

    # Everything after rtsp://
    rest = url[7:]
    last_at = rest.rfind("@")
    if last_at == -1:
        return url  # No auth in URL

    auth_part = rest[:last_at]       # e.g. "test:dei@12@12"
    host_part = rest[last_at + 1:]   # e.g. "10.7.16.48:554"

    colon = auth_part.find(":")
    if colon == -1:
        return url  # No password, nothing to encode

    user = auth_part[:colon]
    pwd  = auth_part[colon + 1:]     # e.g. "dei@12@12"

    # Encode only '@' in the password — FFmpeg requires this
    safe_pwd = pwd.replace("@", "%40")

    return f"rtsp://{user}:{safe_pwd}@{host_part}"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

os.makedirs("snapshots", exist_ok=True)
os.makedirs("dataset", exist_ok=True)
os.makedirs("recordings", exist_ok=True)

# Local storage paths
SNAPSHOTS_DIR = "snapshots"
DATASET_DIR = "dataset"
RECORDINGS_DIR = "recordings"
LOCAL_RECORDINGS_DIR = "recordings"  # alias used throughout recording logic

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Reload all saved cameras from the database on startup."""
    # Store the running event loop so worker threads can broadcast SSE events
    notification_manager.set_loop(asyncio.get_event_loop())
    print("[Startup] Loading persistent cameras from database...")
    cameras = db_manager.get_cameras()
    for cam_id, source in cameras:
        # Handle webcam IDs stored as strings
        parsed_source = source
        if str(source).isdigit():
            parsed_source = int(source)
        
        if camera_manager.add_camera(cam_id, parsed_source):
             threading.Thread(target=process_camera, args=(cam_id,), daemon=True).start()
             print(f"[Startup] Restored camera: {cam_id}")
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/snapshots", StaticFiles(directory="snapshots"), name="snapshots")
app.mount("/dataset", StaticFiles(directory="dataset"), name="dataset")
app.mount("/recordings", StaticFiles(directory="recordings"), name="recordings")

# Configure Jinja2 templates with cache disabled to avoid unhashable type error
templates = Jinja2Templates(directory="templates")
templates.env.cache_size = 0

# Initialize Database Manager (SQLite)
try:
    from database.sqlite_manager import SqliteManager
    db_manager = SqliteManager()
    logger.info("✓ Connected to SQLite (Local)")
except Exception as e:
    logger.critical(f"✗ Failed to connect to SQLite: {e}")
    # Force exit if DB is unreachable
    import sys
    sys.exit(1)

class GlobalReIDManager:
    """Manages cross-camera person re-identification using face encodings."""
    def __init__(self, db_manager):
        self.db = db_manager
        self.lock = threading.Lock()
        self.identities = [] # List of {id, encoding}
        self._load_identities()
        
    def _load_identities(self):
        with self.lock:
            try:
                data = self.db.get_recent_active_targets(hours=24)
                for item in data:
                    # SQLite stores as BLOB (bytes), MongoDB used list
                    encoding = item["encoding"]
                    if isinstance(encoding, bytes):
                        encoding = np.frombuffer(encoding, dtype=np.float32)
                    else:
                        encoding = np.array(encoding, dtype=np.float32)
                        
                    self.identities.append({
                        "id": item["global_id"],
                        "encoding": encoding
                    })
                logger.info(f"✓ Global Re-ID: Loaded {len(self.identities)} active identities.")
            except Exception as e:
                logger.error(f"✗ Global Re-ID Load Error: {e}")

    def match(self, encoding, threshold=0.65):
        """Find matching global ID for an encoding using cosine similarity."""
        if encoding is None: return None
        with self.lock:
            best_id = None
            best_sim = threshold
            for item in self.identities:
                # cosine similarity
                na = np.linalg.norm(encoding)
                nb = np.linalg.norm(item["encoding"])
                if na == 0 or nb == 0: continue
                sim = float(np.dot(encoding, item["encoding"]) / (na * nb))
                if sim > best_sim:
                    best_sim = sim
                    best_id = item["id"]
            return best_id

    def register_new(self, encoding, thumbnail_binary=None):
        """Register a new unknown person in the global registry."""
        with self.lock:
            # Generate a slightly random/unique ID to avoid collisions
            import random
            new_id = f"U-{random.randint(1000, 9999)}"
            while any(i["id"] == new_id for i in self.identities):
                new_id = f"U-{random.randint(1000, 9999)}"
                
            self.identities.append({"id": new_id, "encoding": encoding})
            self.db.upsert_global_unknown(new_id, encoding, thumbnail_binary)
            return new_id

detector = PersonDetector()
recognizer = FaceRecognizer()
camera_manager = CameraManager()
recognizer.load_known_faces(db_manager)
reid_manager = GlobalReIDManager(db_manager)

# Global ID mapping: (camera_id, track_id) -> global_id
global_reid_assignments: Dict[tuple, str] = {}
reid_lock = threading.Lock()
# Daily re-log set: {(camera_id, global_id, date_str)} — ensures each person
# gets a journey entry once per day even if their track_id doesn't change
reid_daily_logged: set = set()

class NotificationManager:
    """Manages real-time event broadcasting to multiple web clients via SSE."""
    def __init__(self):
        self.clients = []
        self.lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Store the running event loop so worker threads can schedule onto it."""
        self._loop = loop

    async def subscribe(self):
        """Add a new client queue for SSE."""
        q = asyncio.Queue()
        with self.lock:
            self.clients.append(q)
        return q

    def unsubscribe(self, q):
        """Remove a client queue."""
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)

    def broadcast(self, data: dict):
        """Push a message to all connected clients (safe to call from any thread)."""
        msg = f"data: {json.dumps(data)}\n\n"
        with self.lock:
            loop = self._loop
            clients = list(self.clients)
        if loop is None or not loop.is_running():
            return
        for q in clients:
            try:
                loop.call_soon_threadsafe(q.put_nowait, msg)
            except Exception:
                pass

notification_manager = NotificationManager()

# ---------------------------------------------------------------------------
# Background task: Clean old logs (older than 8 hours)
# ---------------------------------------------------------------------------
def storage_optimization_task():
    """Periodically clean old recordings and snapshots to save disk space."""
    while True:
        try:
            # Run cleanup every hour
            time.sleep(3600)
            
            # Retention Policy (Optimized for 10GB Storage)
            SNAPSHOT_RETENTION_HOURS = 24
            RECORDING_RETENTION_DAYS = 2
            
            # 1. Clean DB and get paths to delete
            paths_to_delete = db_manager.cleanup_old_data(
                snapshot_hours=SNAPSHOT_RETENTION_HOURS, 
                recording_days=RECORDING_RETENTION_DAYS
            )
            
            # 2. Perform file deletion
            local_deleted = 0
            
            
            for path in paths_to_delete:
                if not path: continue
                try:
                    if os.path.exists(path):
                        os.remove(path)
                        local_deleted += 1
                except Exception: pass
            
            if local_deleted:
                logger.info(f"✓ Storage Cleaned: {local_deleted} local files removed.")
                
        except Exception as e:
            logger.error(f"✗ Storage optimization error: {e}")

# Start combined cleanup thread
cleanup_thread = threading.Thread(target=storage_optimization_task, daemon=True)
cleanup_thread.start()

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

# Per-camera: latest tracks for video overlay
camera_results: Dict[str, Any] = {}
results_lock = threading.Lock()  # Single shared lock for camera_results

# Per-camera: recognized persons info
camera_recognized_persons: Dict[str, Dict[int, str]] = {}
recognized_lock = threading.Lock()

# Recording state
camera_writers: Dict[str, Any] = {}
writer_lock = threading.Lock()
occupancy_last_count: Dict[str, int] = {}
occupancy_last_track_ids: Dict[str, Set[int]] = {}
alert_cooldowns: Dict[str, float] = {}  # {camera_id: last_alert_time}
ALERT_COOLDOWN_SECONDS = 30 # Don't log same intrusion for 30s

# Recording frame writer threads
recording_threads: Dict[str, Any] = {}
recording_stop_events: Dict[str, threading.Event] = {}

# Resource management
recognition_executor = ThreadPoolExecutor(max_workers=4)
transfer_queue = queue.Queue(maxsize=100)
recognition_cooldowns: Dict[tuple, float] = {}  # (camera_id, track_id) -> last_process_time
cooldown_lock = threading.Lock()

import atexit
def _cleanup_executor():
    try:
        recognition_executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass
atexit.register(_cleanup_executor)

def transfer_worker():
    """Background worker to process filesystem tasks sequentially."""
    print("[TransferWorker] Started")
    while True:
        try:
            # item can be (local_path, remote_dir, callback) OR (bytes_data, local_path, callback)
            item = transfer_queue.get()
            if item is None: break # Sentinel
            
            data, destination, callback = item
            
            if isinstance(data, (bytes, bytearray)):
                # Case 1: Binary data stream
                success = _perform_direct_stream(data, destination)
            else:
                # Case 2: Local file path processing
                success = _perform_actual_process(data, destination)
                
            if callback:
                callback(success)
            
            transfer_queue.task_done()
        except Exception as e:
            print(f"[TransferWorker] Error: {e}")
            time.sleep(1)

def _perform_direct_stream(data: bytes, local_path: str) -> bool:
    """Stream binary data directly to a local file."""
    try:
        parent = os.path.dirname(local_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(local_path, 'wb') as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"[Local Save] Error: {e}")
        return False

def _perform_actual_process(src_path: str, dest_dir: str) -> bool:
    """Copy files to local storage."""
    try:
        import shutil
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(src_path, dest_dir)
        return True
    except Exception:
        return False

# Start transfer worker
threading.Thread(target=transfer_worker, daemon=True).start()

def stream_bytes_to_local(data: bytes, local_path: str, callback=None) -> bool:
    """Queue binary data for direct saving to local disk."""
    try:
        transfer_queue.put((data, local_path, callback), block=False)
        return True
    except queue.Full:
        return False

# Keep for compatibility where files are still used (if any)
def save_to_local(local_path: str, destination_dir: str, callback=None) -> bool:
    """Push local file processing task to the background queue."""
    try:
        transfer_queue.put((local_path, destination_dir, callback), block=False)
        return True
    except queue.Full:
        return False


def recording_writer_thread(camera_id: str, stop_event: threading.Event,
                            process: subprocess.Popen):
    """Background thread to write rendered frames to FFmpeg stdin at 2 FPS."""
    print(f"[Recording:{camera_id}] Writer thread started")
    FRAME_INTERVAL = 0.5  # 2 FPS

    while not stop_event.is_set():
        try:
            if process.poll() is not None:
                print(f"[Recording:{camera_id}] FFmpeg exited (code {process.returncode})")
                break

            with results_lock:
                data = camera_results.get(camera_id, {})
                frame = data.get("rendered_frame")

            if frame is not None:
                try:
                    process.stdin.write(frame.tobytes())
                    process.stdin.flush()
                except (IOError, BrokenPipeError) as e:
                    print(f"[Recording:{camera_id}] Pipe broken: {e}")
                    break

            stop_event.wait(timeout=FRAME_INTERVAL)

        except Exception as e:
            print(f"[Recording:{camera_id}] Writer error: {e}")
            stop_event.wait(timeout=1)

    print(f"[Recording:{camera_id}] Writer thread stopped")

# Active search mission — set by /api/start_search, cleared by /api/stop_search
# {person_id, name, encoding, found_track_ids: set, running: bool}
active_search: Dict[str, Any] = {}
active_search_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Passive camera processing — ONLY detection + tracking, NO recognition
# ---------------------------------------------------------------------------

def process_camera(camera_id: str):
    """Background thread per camera: detection + tracking + face recognition.
    Process exactly 2 FPS for high accuracy with reduced system load.
    """
    print(f"[Camera:{camera_id}] Processing thread started (2 FPS mode)")
    
    # Wait for camera to be ready
    warmup_frames = 0
    while warmup_frames < 5:
        frame, _ = camera_manager.get_camera_frame_with_id(camera_id)
        if frame is not None:
            warmup_frames += 1
        time.sleep(0.1)
    print(f"[Camera:{camera_id}] Camera ready - Processing at 2 FPS")
    
    # Check if recording should be started automatically
    import subprocess
    import os
    db_setting = db_manager.get_camera_recording_setting(camera_id)
    if db_setting == 1:
        with writer_lock:
            if camera_id not in camera_writers:
                try:
                    # Dimensions should be known from dummy get_camera_frame_with_id
                    h, w = frame.shape[:2]
                    ist_now = get_ist_time()
                    timestamp = ist_now.strftime("%Y%m%d_%H%M%S")
                    filename = f"rec_{camera_id}_{timestamp}.mp4"
                    local_path = f"{LOCAL_RECORDINGS_DIR}/{filename}"
                    os.makedirs(LOCAL_RECORDINGS_DIR, exist_ok=True)
                    
                    ffmpeg_cmd = [
                        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
                        "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", "20",
                        "-i", "-", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-crf", "28",
                        "-tune", "zerolatency", local_path
                    ]
                    
                    p_ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    db_id = db_manager.start_recording(camera_id, local_path)
                    stop_event = threading.Event()
                    # Set camera_writers BEFORE starting thread to avoid race condition
                    camera_writers[camera_id] = {
                        "process": p_ffmpeg,
                        "db_id": db_id,
                        "start_time": ist_now,
                        "file_path": local_path,
                        "camera_id": camera_id,
                        "w": w, "h": h
                    }
                    r_thread = threading.Thread(target=recording_writer_thread, args=(camera_id, stop_event, p_ffmpeg), daemon=True)
                    r_thread.start()
                    recording_threads[camera_id] = r_thread
                    recording_stop_events[camera_id] = stop_event
                    print(f"[Recording:{camera_id}] Auto-started (FFmpeg)")
                except Exception as err:
                    logger.error(f"Failed to auto-start FFmpeg for {camera_id}: {err}")
    
    # Tracker: max_age=8 (4s at 2FPS), low IoU threshold for fast movers
    tracker: ObjectTracker = ObjectTracker(max_age=8, n_init=1, iou_threshold=0.15)
    last_frame_id: int = -1
    frame_count: int = 0
    
    # Process exactly 2 frames per second
    FRAME_INTERVAL: float = 0.5  # 500ms = 2 FPS
    
    # Recognition cache: track_id -> (name, confidence, frame_number)
    RECOGNITION_CACHE_FRAMES: int = 6  # Cache valid for 6 frames (~3s at 2 FPS)
    recognition_cache: Dict[Any, tuple] = {}
    
    # Track IDs currently in frame (to prevent double counting)
    current_frame_track_ids: set = set()
    
    # Face encoding cache for deduplication: track_id -> encoding
    face_encoding_cache: Dict[int, np.ndarray] = {}
    # Track merge map: old_id -> new_id (for deduplication)
    track_merge_map: Dict[int, int] = {}
    last_process_time: float = 0

    while True:
        # Wait for next 2 FPS interval
        current_time = time.time()
        elapsed = current_time - last_process_time
        if elapsed < FRAME_INTERVAL:
            time.sleep(FRAME_INTERVAL - elapsed)
        
        frame, frame_id = camera_manager.get_camera_frame_with_id(camera_id)
        if frame is None:
            continue
            
        # Get latest frame (may skip some camera frames to maintain 2 FPS)
        last_frame_id = frame_id
        frame_count += 1
        last_process_time = time.time()

        try:
            h, w = frame.shape[:2]
            
            # Run detection on EVERY frame (2 FPS) for high accuracy
            detections = detector.detect(frame)
            
            # Run recognition on EVERY frame (2 FPS) - no skip
            
            # Update tracker
            tracks = tracker.update(detections, frame)
            
            # Build current frame track IDs for anti-double-counting
            new_track_ids = set(t["id"] for t in tracks)
            
            # Log count on every frame at 2 FPS
            if len(new_track_ids) != len(current_frame_track_ids):
                print(f"[Camera:{camera_id}] Persons: {len(tracks)}")
            current_frame_track_ids = new_track_ids

            # 1. Non-Maximum Suppression (Overlapping Box Kill) on raw tracks
            final_tracks = []
            tracks = sorted(tracks, key=lambda x: x["id"])
            for i, t1 in enumerate(tracks):
                keep = True
                for j, t2 in enumerate(final_tracks):
                    box1, box2 = t1["bbox"], t2["bbox"]
                    ix1, iy1 = max(box1[0], box2[0]), max(box1[1], box2[1])
                    ix2, iy2 = min(box1[2], box2[2]), min(box1[3], box2[3])
                    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                    inter = iw * ih
                    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
                    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
                    union = area1 + area2 - inter
                    iou = inter / union if union > 0 else 0
                    if iou > 0.7:
                        keep = False
                        break
                if keep:
                    final_tracks.append(t1)
            tracks = final_tracks

            # 2. Build processed tracks with cached recognition
            processed = []
            for t in tracks:
                tid = t["id"]
                bbox = t["bbox"]
                
                # Check recognition cache
                name, conf = "Unknown", 0.0
                if tid in recognition_cache:
                    cached_name, cached_conf, cached_frame = recognition_cache[tid]
                    if (frame_count - cached_frame) < RECOGNITION_CACHE_FRAMES:
                        name, conf = cached_name, cached_conf

                processed.append({
                    "id": tid,
                    "bbox": bbox,
                    "name": name,
                    "confidence": conf,
                    "stable": True
                })

            # 3. Submit for Face Recognition (Worker Thread)
            for t in processed:
                tid = t["id"]
                # Skip if cache is still fresh
                if tid in recognition_cache and (frame_count - recognition_cache[tid][2]) < (RECOGNITION_CACHE_FRAMES // 2):
                    continue
                
                # Cooldown: 4s if already identified, 1s if unknown (retry faster)
                now = time.time()
                with cooldown_lock:
                    last_time = recognition_cooldowns.get((camera_id, tid), 0)
                    cooldown = 4.0 if t["name"] != "Unknown" else 1.0
                    if now - last_time < cooldown:
                        continue
                    recognition_cooldowns[(camera_id, tid)] = now

                bx1, by1, bx2, by2 = [int(v) for v in t["bbox"]]
                # Pass full body bbox — recognizer uses MTCNN internally to find tight face
                body_box = [bx1, by1, bx2, by2]

                try:
                    recognition_executor.submit(
                        self_recognition_worker,
                        frame.copy(), body_box, tid, recognition_cache, frame_count,
                        face_encoding_cache, track_merge_map, camera_id
                    )
                except RuntimeError: break

            # Render at full rate - every frame gets overlay
            record_frame = frame.copy()
            people_count = len(processed)
            
            # Generate distinct colors for each person ID
            def get_person_color(pid):
                # Use HSV color space for distinct colors
                hue = (pid * 137) % 180
                hsv = np.uint8([[[hue, 255, 255]]])
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
                return tuple(int(c) for c in rgb)

            alert_active = False
            final_processed_with_crops = []

            for t in processed:
                bx1, by1, bx2, by2 = [int(v) for v in t["bbox"]]
                name = str(t["name"])
                conf = float(t["confidence"])
                tid = int(t["id"])

                if name != "Unknown":
                    body_color = (0, 255, 0)  # Green for recognized
                    label = f"{name}"
                else:
                    base_tid = tid
                    while base_tid in track_merge_map:
                        base_tid = track_merge_map[base_tid]
                    body_color = get_person_color(base_tid)
                    label = f"#{base_tid}"
                
                # Draw Box
                cv2.rectangle(record_frame, (bx1, by1), (bx2, by2), body_color, 2)
                cv2.putText(record_frame, label, (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, body_color, 2)

                # Extract face crop for sidebar UI
                cropped_face = None
                try:
                    ch = by2 - by1
                    fy2 = min(h-1, by1 + int(0.45 * ch))
                    if bx2 > bx1 and fy2 > by1:
                        face_img = frame[by1:fy2, bx1:bx2]
                        if face_img.size > 0:
                            face_img = cv2.resize(face_img, (100, 120))
                            _, buffer = cv2.imencode('.jpg', face_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            cropped_face = base64.b64encode(buffer).decode('utf-8')
                except Exception: pass

                final_processed_with_crops.append({
                    "id": tid,
                    "bbox": [bx1, by1, bx2, by2],
                    "name": name,
                    "confidence": conf,
                    "face_crop": cropped_face
                })
            
            processed = final_processed_with_crops

            # SMART LOGGING: Only log if the set of people in the frame actually changed
            try:
                current_ids = set(t["id"] for t in processed)
                last_ids = occupancy_last_track_ids.get(camera_id, set())

                if current_ids != last_ids:
                    occupancy_last_track_ids[camera_id] = current_ids
                    people_count = len(current_ids)
                    
                    db_manager.log_occupancy(camera_id, people_count)
                    
                    # Save snapshot with bounding boxes ONLY when count changes
                    if people_count > 0:
                        # Use IST timestamp
                        now_ist = get_ist_time()
                        timestamp = now_ist.strftime("%Y%m%d_%H%M%S")
                        local_snapshot_path = f"{SNAPSHOTS_DIR}/{camera_id}/snapshot_{timestamp}.jpg"
                        
                        # Save bbox data as JSON and capture encodings
                        import json
                        snapshot_processed = []
                        current_encodings = []
                        person_crops = [] # Store cropped faces as bytes
                        
                        for t in processed:
                            tid = t["id"]
                            bx1, by1, bx2, by2 = [int(v) for v in t["bbox"]]
                            snapshot_processed.append({
                                "id": tid,
                                "bbox": t["bbox"],
                                "name": t["name"]
                            })
                            
                            # Extract face/body crop for "extract faces of all persons"
                            cw, ch = bx2-bx1, by2-by1
                            # Face approx: top 40% of body
                            fbx1, fby1 = max(0, bx1), max(0, by1)
                            fbx2, fby2 = min(w-1, bx2), min(h-1, by1 + int(ch*0.45))
                            crop = frame[fby1:fby2, fbx1:fbx2]
                            if crop.size > 0:
                                _, cbuf = cv2.imencode('.jpg', crop)
                                person_crops.append(cbuf.tobytes())

                            # Get encoding from cache if available
                            if tid in face_encoding_cache:
                                current_encodings.append(face_encoding_cache[tid])

                        bbox_data = snapshot_processed
                        
                        # Encode to JPEG with compression (quality 60 = ~70% smaller)
                        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 60, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
                        _, buffer = cv2.imencode('.jpg', record_frame, encode_params)
                        img_bytes = buffer.tobytes()
                        
                        # Save directly to local storage
                        def on_snapshot_complete(success, _cam=camera_id, _count=people_count, _path=local_snapshot_path, _bbox=bbox_data, _encs=current_encodings, _ts=now_ist):
                            if success:
                                # Log snapshots to SQLite database with IST timestamp
                                db_manager.log_detection_snapshot(
                                    _cam, _count, _path,
                                    _bbox, face_encodings=_encs,
                                    timestamp=_ts
                                )
                                print(f"[Camera:{_cam}] Detection Change: {_count} Snapshot logged at {format_12h(_ts)}.")
                        
                        stream_bytes_to_local(img_bytes, local_snapshot_path, callback=on_snapshot_complete)
            except Exception as e:
                print(f"[Camera:{camera_id}] Count/Snapshot error: {e}")
            
            # Display count - only currently detected persons
            count_text = f"Persons: {people_count}"
            # Final State Sync
            with results_lock:
                camera_results[camera_id] = {
                    "rendered_frame": record_frame, 
                    "frame_id": frame_id, 
                    "tracks": processed,
                    "alert_active": alert_active,
                    "timestamp": time.time()
                }
            
            # Store recognized persons for API and update last seen
            registered_detected = []
            with recognized_lock:
                recognized_dict = {}
                for t in processed:
                    if t["name"] != "Unknown":
                        recognized_dict[t["id"]] = t["name"]
                        registered_detected.append(t["name"])
                        # Update last seen in database
                        try:
                            db_manager.update_person_last_seen(t["name"], camera_id)
                        except Exception as e:
                            print(f"[Camera:{camera_id}] Error updating last seen: {e}")
                camera_recognized_persons[camera_id] = recognized_dict

            # Save registered person snapshot if any registered person detected
            if registered_detected:
                try:
                    ist_now = get_ist_time()
                    timestamp = ist_now.strftime("%Y%m%d_%H%M%S")
                    local_snapshot_path = f"{SNAPSHOTS_DIR}/{camera_id}/registered_{timestamp}.jpg"
                    
                    # Prepare bbox data as object
                    bbox_data = [{
                        "id": t["id"],
                        "bbox": t["bbox"],
                        "name": t["name"]
                    } for t in processed if t["name"] != "Unknown"]
                    
                    # Encode to JPEG with compression
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 60, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
                    _, buffer = cv2.imencode('.jpg', record_frame, encode_params)
                    img_bytes = buffer.tobytes()
                    
                    def on_reg_snapshot_complete(success, _cam=camera_id, _detected=list(registered_detected), _path=local_snapshot_path, _bbox=bbox_data, _ts=ist_now):
                        if success:
                            db_manager.log_detection_snapshot(
                                camera_id=_cam,
                                person_count=len(_detected),
                                snapshot_path=_path,
                                bbox_data=_bbox,
                                face_encodings=None,
                                timestamp=_ts
                            )
                            print(f"[Camera:{_cam}] Registered person snapshot streamed: {_detected}")
                    
                    stream_bytes_to_local(img_bytes, local_snapshot_path, callback=on_reg_snapshot_complete)
                except Exception as e:
                    print(f"[Camera:{camera_id}] Registered person snapshot error: {e}")

            # Auto-split recording every 2.5 hours (9000 seconds)
            with writer_lock:
                writer_data = camera_writers.get(camera_id)
                if writer_data and "process" in writer_data:
                    ist_now = get_ist_time()
                    recording_duration = (ist_now - writer_data["start_time"]).total_seconds()
                    if recording_duration > 9000:  # 2.5 hours
                        try:
                            writer_data["process"].stdin.close()
                            writer_data["process"].wait(timeout=10)
                            db_manager.end_recording(writer_data["db_id"])
                            print(f"[Recording] Auto-split {camera_id} after {recording_duration/3600:.1f} hours")
                            
                            new_timestamp = ist_now.strftime("%Y%m%d_%H%M%S")
                            new_filename = f"rec_{camera_id}_{new_timestamp}.mp4"
                            new_local_path = f"{RECORDINGS_DIR}/{new_filename}"
                            
                            os.makedirs(RECORDINGS_DIR, exist_ok=True)
                            ffmpeg_cmd = [
                                "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
                                "-s", f"{writer_data['w']}x{writer_data['h']}", "-pix_fmt", "bgr24", "-r", "20",
                                "-i", "-", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-crf", "28",
                                "-tune", "zerolatency", new_local_path
                            ]
                            
                            p_ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            new_db_id = db_manager.start_recording(camera_id, new_local_path)
                            camera_writers[camera_id] = {
                                "process": p_ffmpeg,
                                "db_id": new_db_id,
                                "start_time": ist_now,
                                "file_path": new_local_path,
                                "camera_id": camera_id,
                                "w": writer_data["w"],
                                "h": writer_data["h"]
                            }
                            print(f"[Recording] Started new segment {camera_id} direct to {new_local_path}")
                        except Exception as e:
                            print(f"[Camera:{camera_id}] Error auto-splitting recording: {e}")
            
            # No frame rate limiting - run as fast as possible for smooth video

        except Exception as e:
            print(f"[Camera:{camera_id}] Error: {e}")
            import traceback; traceback.print_exc()


def self_recognition_worker(frame, face_box, track_id, recognition_cache, frame_count, face_encoding_cache, track_merge_map, camera_id):
    """Background task for periodic biometric verification with global Re-ID."""
    try:
        name, conf, face_encoding = recognizer.recognize_with_encoding(frame, face_box)
        
        # 1. Update local caches for track deduplication
        if face_encoding is not None:
            face_encoding_cache[track_id] = face_encoding
            
            # Check for duplicate tracks in this camera
            for other_id, other_encoding in face_encoding_cache.items():
                if other_id != track_id:
                    distance = np.linalg.norm(face_encoding - other_encoding)
                    if distance < 0.6:
                        if track_id < other_id:
                            track_merge_map[other_id] = track_id
                        else:
                            track_merge_map[track_id] = other_id
                        break
        
        # 2. Update recognition cache if registered person
        if name != "Unknown" and conf >= 0.65:
            recognition_cache[track_id] = (name, conf, frame_count)
            
        # 3. GLOBAL RE-ID & JOURNEY LOGGING
        global_id = None
        
        # Case A: Person is recognized as Registered
        if name != "Unknown" and conf >= 0.65:
            global_id = name
        
        # Case B: Person is Unknown - attempt Global Re-ID
        elif face_encoding is not None:
            # Check if already matched in this session
            with reid_lock:
                global_id = global_reid_assignments.get((camera_id, track_id))
            
            if not global_id:
                # Attempt to match against global registry
                matched_id = reid_manager.match(face_encoding)
                
                if matched_id:
                    global_id = matched_id
                else:
                    # Register as a new global unknown
                    # Use a thumbnail for the journey record
                    try:
                        fx1, fy1, fx2, fy2 = face_box
                        crop = frame[max(0, fy1):fy2, max(0, fx1):fx2]
                        if crop.size > 0:
                            _, buf = cv2.imencode('.jpg', crop)
                            thumbnail = buf.tobytes()
                        else:
                            thumbnail = None
                    except: thumbnail = None
                    
                    global_id = reid_manager.register_new(face_encoding, thumbnail)
        
        # Update global mapping and log sighting
        if global_id:
            with reid_lock:
                old_gid = global_reid_assignments.get((camera_id, track_id))
                is_new_link = (old_gid != global_id)
                if is_new_link:
                    global_reid_assignments[(camera_id, track_id)] = global_id

                # Log journey once per day per (camera, global_id) — keeps 24h total accurate
                now_ist = get_ist_time()
                day_key = (camera_id, str(global_id), now_ist.strftime("%Y%m%d"))
                should_log = day_key not in reid_daily_logged

                if should_log:
                    reid_daily_logged.add(day_key)

            if is_new_link or should_log:
                    now_ist = get_ist_time()
                    ts_str = now_ist.strftime("%Y%m%d_%H%M%S")
                    sighting_path = f"snapshots/{camera_id}/journey_{global_id}_{ts_str}.jpg"
                    os.makedirs(os.path.dirname(sighting_path), exist_ok=True)
                    
                    try:
                        _, full_buf = cv2.imencode('.jpg', frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 60, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
                        with open(sighting_path, 'wb') as f:
                            f.write(full_buf.tobytes())
                    except Exception as e:
                        print(f"[Worker] Snapshot save failed: {e}")
                        sighting_path = None
                    
                    db_manager.log_journey_event(
                        global_id=global_id,
                        camera_id=camera_id,
                        snapshot_path=sighting_path,
                        timestamp=now_ist
                    )
                    
                    # Broadcast ONLY for registered/known persons
                    try:
                        is_registered = "U-" not in str(global_id)
                        if is_registered and is_new_link:
                            thumb_url = f"https://ui-avatars.com/api/?name={str(global_id)}&background=e8192c&color=fff"
                            notification_manager.broadcast({
                                "type": "registered_person",
                                "camera": camera_id,
                                "target": str(global_id),
                                "thumbnail": thumb_url,
                                "time": now_ist.strftime("%I:%M %p")
                            })
                    except Exception: pass
                    
                    if is_new_link:
                        print(f"[Global Re-ID] Linked {camera_id}:{track_id} -> {global_id}")

    except Exception as e:
        print(f"[Worker Error] {e}")


# ---------------------------------------------------------------------------
# Search & Forensics Utilities
# ---------------------------------------------------------------------------

def scan_video_for_person(
    video_path: str,
    target_encoding: np.ndarray,
    sample_interval: int = 10
) -> list:
    """
    Scan a video file frame-by-frame looking for a target face.
    """
    results = []
    if not os.path.exists(video_path):
        return results

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return results

    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    frame_count = 0
    current_segment = None
    last_match_frame = -1
    min_segment_gap = int(fps * 2)   # 2 seconds gap = new segment
    DISTANCE_THRESHOLD = 0.65  # Using same threshold as recognizer for consistency

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % sample_interval == 0:
            try:
                # Use FaceRecognizer for consistent results
                # recognizer.recognize needs a body_box.
                # Here we just detect faces in the whole frame.
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                with recognizer.ai_lock:
                    boxes, probs = recognizer.mtcnn.detect(frame_rgb)

                match_found = False
                best_sim = 0.0

                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        fx1, fy1, fx2, fy2 = [int(b) for b in box]
                        fx1, fy1 = max(0, fx1), max(0, fy1)
                        fx2 = min(frame.shape[1], fx2)
                        fy2 = min(frame.shape[0], fy2)

                        if (fx2 - fx1) < 30 or (fy2 - fy1) < 30:
                            continue

                        face_crop = frame_rgb[fy1:fy2, fx1:fx2]
                        if face_crop.size == 0:
                            continue

                        # Generate embedding using recognizer's internal helper
                        embedding = recognizer._embed(face_crop)
                        if embedding is None:
                            continue
                            
                        # Cosine similarity (recognizer uses matrix @ vector)
                        sim = float(np.dot(target_encoding, embedding))
                        
                        if sim >= 0.65:
                            match_found = True
                            if sim > best_sim:
                                best_sim = sim

                if match_found:
                    timestamp_sec = frame_count / fps
                    minutes = int(timestamp_sec // 60)
                    seconds = int(timestamp_sec % 60)
                    ts_str = f"{minutes}:{seconds:02d}"

                    # Start new segment or extend existing one
                    if current_segment is None or (frame_count - last_match_frame) > min_segment_gap:
                        if current_segment is not None:
                            results.append(current_segment)
                        current_segment = {
                            "start_seconds": timestamp_sec,
                            "start_timestamp": ts_str,
                            "end_seconds": timestamp_sec,
                            "end_timestamp": ts_str,
                            "confidence": best_sim,
                            "start_frame": frame_count,
                            "end_frame": frame_count,
                        }
                    else:
                        current_segment["end_seconds"] = timestamp_sec
                        current_segment["end_timestamp"] = ts_str
                        current_segment["end_frame"] = frame_count
                        if best_sim > current_segment["confidence"]:
                            current_segment["confidence"] = best_sim

                    last_match_frame = frame_count

            except Exception as e:
                print(f"Scan error at frame {frame_count}: {e}")

        frame_count += 1

    if current_segment is not None:
        results.append(current_segment)

    cap.release()
    return results


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Check authentication
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "index.html", {})

@app.get("/journey", response_class=HTMLResponse)
async def journey_page(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "journey.html", {})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {})

@app.post("/api/login")
async def api_login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Handle login form submission."""
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        import uuid
        session_token = str(uuid.uuid4())
        authenticated_sessions.add(session_token)
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie(key="session", value=session_token, httponly=True)
        return response
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/logout")
async def logout(request: Request):
    """Logout and clear session."""
    session_token = request.cookies.get("session")
    if session_token and session_token in authenticated_sessions:
        authenticated_sessions.discard(session_token)
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie("session")
    return response

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "dashboard.html", {})

@app.get("/api/dashboard_metrics")
async def dashboard_metrics(request: Request):
    if not require_auth(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Calculate metrics
    active_cameras = len(camera_manager.cameras)
    registered_persons = len(db_manager.get_registered_persons())
    total_recordings = len(db_manager.get_recorded_videos())
    
    try:
        recent_detections = db_manager.get_detections()
        recent_detections = recent_detections[-20:] if recent_detections else []
        recent_detections.reverse() # newest first
    except Exception as e:
        print(f"Error fetching detections: {e}")
        recent_detections = []
        
    return {
        "active_cameras": active_cameras,
        "registered_persons": registered_persons,
        "total_recordings": total_recordings,
        "recent_detections": recent_detections
    }

@app.get("/api/server_time")
async def get_server_time():
    """Return the current server time in IST for frontend clock sync."""
    now = get_ist_time()
    return {
        "iso": now.isoformat(),
        "timestamp_ms": int(now.timestamp() * 1000),
        "display": now.strftime("%d %b %Y, %I:%M:%S %p"),
        "timezone": "Asia/Kolkata (IST)"
    }

@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "search.html", {"nav": "search"})

@app.get("/api/search")
async def api_search(
    name: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
):
    """Search detection history by name and/or date range."""
    results = db_manager.search_detections(name, start_time, end_time)
    res = []
    for r in results:
        # r = (id, name, camera_id, timestamp, image_path, person_name)
        res.append({
            "id": r[0],
            "person_name": r[1] or "Unknown",
            "camera_id": r[2],
            "timestamp": r[3].isoformat() if hasattr(r[3], 'isoformat') else str(r[3]),
            "image_path": r[4],
            "face_path": r[4]
        })
    return res

@app.post("/api/search_by_image")
async def search_by_image(file: UploadFile = File(...)):
    """Upload a face — finds all detections of the matching registered person."""
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    encoding = recognizer.get_encoding(image)
    if encoding is None:
        return []

    best_person_name = None
    best_sim = 0.65

    # Match using recognizer's internal matrix for speed
    if recognizer._enc_matrix is not None:
        sims = recognizer._enc_matrix @ encoding
        idx = int(np.argmax(sims))
        if sims[idx] >= best_sim:
            best_person_name = recognizer._enc_names[idx]

    if not best_person_name:
        return []

    results = db_manager.search_detections(name=best_person_name)
    return [
        {
            "id": r[0],
            "person_name": r[1] or "Unknown",
            "camera_id": r[2],
            "timestamp": r[3].isoformat() if hasattr(r[3], 'isoformat') else str(r[3]),
            "image_path": r[4],
            "face_path": r[4]
        }
        for r in results
    ]

@app.post("/api/search_video_by_name")
async def search_video_by_name(request: Request):
    """Scan selected video recordings to find a registered person by name."""
    data = await request.json()
    name = data.get("name")
    video_ids = data.get("video_ids", [])

    if not name or not video_ids:
        return {"status": "error", "message": "Name and video IDs required"}

    # Look up registered person
    persons = db_manager.get_registered_persons()
    target = next((p for p in persons if p[1].lower() == name.lower()), None)
    if target is None:
        return {"status": "error", "message": f"Person '{name}' not found"}

    target_encoding = np.frombuffer(target[3], dtype=np.float32)

    all_results = []
    for vid_id in video_ids:
        rec = db_manager.get_recording(vid_id)
        if rec and os.path.exists(rec[4]):
            segments = scan_video_for_person(rec[4], target_encoding)
            for segment in segments:
                all_results.append({
                    **segment,
                    "video_id": vid_id,
                    "video_name": os.path.basename(rec[4]),
                    "video_path": rec[4],
                    "camera_id": rec[1],
                    "person_name": name,
                })

    return {
        "status": "success",
        "results": all_results,
        "total_segments": len(all_results),
        "videos_searched": len(video_ids),
    }

@app.post("/api/search_video_by_image")
async def search_video_by_image(
    file: UploadFile = File(...),
    video_ids: str = Form(...) 
):
    """Scan selected video recordings to find a face from an uploaded image."""
    video_ids_list = json.loads(video_ids)
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    target_encoding = recognizer.get_encoding(image)
    if target_encoding is None:
        return {"status": "error", "message": "No face detected in photo"}

    all_results = []
    for vid_id in video_ids_list:
        rec = db_manager.get_recording(vid_id)
        if rec and os.path.exists(rec[4]):
            segments = scan_video_for_person(rec[4], target_encoding)
            for segment in segments:
                all_results.append({
                    **segment,
                    "video_id": vid_id,
                    "video_name": os.path.basename(rec[4]),
                    "video_path": rec[4],
                    "camera_id": rec[1],
                    "person_name": "Target Face",
                })

    return {
        "status": "success",
        "results": all_results,
        "total_segments": len(all_results),
        "videos_searched": len(video_ids_list),
    }

@app.get("/api/recordings")
async def api_recordings_list():
    """List available video recordings."""
    results = db_manager.get_recorded_videos()
    return [
        {
            "id": r[0],
            "camera_id": r[1],
            "start_time": r[2].isoformat() if hasattr(r[2], 'isoformat') else str(r[2]),
            "end_time": r[3].isoformat() if hasattr(r[3], 'isoformat') else str(r[3]),
            "file_path": r[4],
        }
        for r in results
    ]

@app.post("/clear_history")
async def clear_history_api():
    """Wipe all detections from database."""
    db_manager.delete_all_detections()
    return {"status": "success"}

@app.get("/recordings_page", response_class=HTMLResponse)
async def recordings_page(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "recordings.html", {})

@app.get("/detection_logs", response_class=HTMLResponse)
async def detection_logs_page(request: Request, camera_id: Optional[str] = None):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "detection_logs.html", {"camera_id": camera_id})

@app.get("/registered_detections", response_class=HTMLResponse)
async def registered_detections_page(request: Request, person_name: Optional[str] = None):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "registered_detections.html", {"person_name": person_name})

@app.get("/people", response_class=HTMLResponse)
async def people_page(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "people.html", {})

@app.get("/api/recent_alerts")
async def get_recent_alerts_api(request: Request, limit: int = 10):
    if not require_auth(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    alerts = db_manager.get_recent_alerts(limit=limit)
    return [{
        "id": a["id"],
        "camera_id": a["camera_id"],
        "person_id": a["person_id"],
        "snapshot_path": a.get("snapshot_path"),
        "timestamp": format_12h(a["timestamp"]),
        "type": a["type"]
    } for a in alerts]

# --- Spatial Tracking / Re-ID APIs ---

@app.get("/api/active_targets")
async def get_active_targets(request: Request, hours: int = 24):
    """Retrieve unique people seen recently across all cameras."""
    if not require_auth(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    targets = db_manager.get_recent_active_targets(hours=hours)
    res = []
    for t in targets:
        res.append({
            "id": t["global_id"],
            "type": t.get("type", "unknown"),
            "first_seen": format_12h(t["first_seen"]),
            "last_seen": format_12h(t["last_seen"]),
            "last_camera": t.get("last_camera", "Unknown"),
            "has_thumbnail": "thumbnail" in t
        })
    return res

@app.get("/api/target_journey/{global_id}")
async def get_target_journey(request: Request, global_id: str):
    """Get the chronological path of a specific person."""
    if not require_auth(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    journey = db_manager.get_target_journey(global_id)
    res = []
    for point in journey:
        res.append({
            "camera_id": point["camera_id"],
            "timestamp": format_12h(point["timestamp"]),
            "date": point["timestamp"].strftime("%d %b"),
            "snapshot_path": point.get("snapshot_path")
        })
    return res

@app.get("/api/target_thumbnail/{global_id}")
async def get_target_thumbnail(global_id: str):
    """Return the binary thumbnail image for an unknown person."""
    # Note: thumbnail is public for easier image tag usage
    target = db_manager.get_global_identity_by_id(global_id)
    if not target or "thumbnail" not in target:
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    
    from fastapi.responses import Response
    return Response(content=target["thumbnail"], media_type="image/jpeg")

@app.get("/api/notifications/stream")
async def notification_stream(request: Request):
    """SSE endpoint for real-time alerts."""
    q = await notification_manager.subscribe()
    
    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                data = await q.get()
                yield data
        except asyncio.CancelledError:
            pass
        finally:
            notification_manager.unsubscribe(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/cameras", response_class=HTMLResponse)
async def cameras_page(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "cameras.html", {})

@app.get("/add_camera", response_class=HTMLResponse)
async def add_camera_page(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "add_camera.html", {})

@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    if not require_auth(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(request, "analytics.html", {})
@app.post("/register")
async def register_person(name: str = Form(...), file: UploadFile = File(...)):
    # Read bytes into memory
    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"status": "error", "message": "Invalid image file."}

    encoding = recognizer.get_encoding(image)
    if encoding is not None:
        # Save directly to local storage
        local_path = f"{DATASET_DIR}/{name}/{file.filename}"
        
        def on_reg_complete(success):
            if success:
                db_manager.register_person(name, local_path, encoding.tobytes())
                recognizer.load_known_faces(db_manager)
                print(f"[Register] {name} saved to local storage.")

        if stream_bytes_to_local(content, local_path, callback=on_reg_complete):
            return {"status": "success", "message": f"{name} registration queued for local saving."}
        else:
            return {"status": "error", "message": "Storage queue full."}
            
    return {"status": "error", "message": "No face detected in the image."}


@app.post("/api/add_camera")
async def add_camera(request: Request, camera_id: str = Form(None), camera_type: str = Form(None), source: str = Form(None)):
    # Support both form and JSON payload from UI
    if camera_id is None or camera_type is None or source is None:
        try:
            payload = await request.json()
            camera_id = camera_id or payload.get("camera_id")
            camera_type = camera_type or payload.get("camera_type")
            source = source or payload.get("source")
        except Exception:
            pass

    if not camera_id or not source:
        return {"status": "error", "message": "camera_id and source are required"}

    parsed = source
    if camera_type == "webcam":
        try:
            parsed = int(source)
        except ValueError:
            pass
    elif camera_type == "rtsp":
        parsed = sanitize_rtsp_url(source)
    elif camera_type == "droidcam":
        if not source.startswith("http"):
            parsed = f"http://{source}:4747/video" if ":" not in source else f"http://{source}/video"
    elif camera_type == "ipwebcam":
        if not source.startswith("http"):
            parsed = f"http://{source}:8080/video" if ":" not in source else f"http://{source}/video"
    elif camera_type == "mjpeg":
        # Direct MJPEG HTTP stream — pass as-is
        parsed = source.strip()

    if camera_manager.add_camera(camera_id, parsed):
        db_manager.add_camera_to_db(camera_id, parsed)
        threading.Thread(target=process_camera, args=(camera_id,), daemon=True).start()
        return {"status": "success"}
    return {"status": "error", "message": "Camera already exists or could not connect."}


@app.delete("/api/remove_camera/{camera_id}")
async def delete_camera(camera_id: str):
    print(f"[Delete Camera] Attempting to remove: {camera_id}")
    print(f"[Delete Camera] Active cameras: {camera_manager.get_active_cameras()}")
    
    # Stop any recording first
    with writer_lock:
        if camera_id in camera_writers:
            writer_data = camera_writers.pop(camera_id)
            # Stop the recording thread
            if camera_id in recording_stop_events:
                recording_stop_events[camera_id].set()
                if camera_id in recording_threads:
                    recording_threads[camera_id].join(timeout=5)
                    del recording_threads[camera_id]
                del recording_stop_events[camera_id]
            # Close FFmpeg process
            if "process" in writer_data:
                try:
                    writer_data["process"].stdin.close()
                    writer_data["process"].wait(timeout=10)
                except Exception:
                    writer_data["process"].kill()
            db_manager.end_recording(writer_data["db_id"])
            print(f"[Delete Camera] Stopped recording for {camera_id}")
    
    # Remove from camera manager
    cam_removed = camera_manager.remove_camera(camera_id)
    print(f"[Delete Camera] Camera manager removal result: {cam_removed}")
    
    # Remove from database (always try this even if camera not active)
    db_manager.remove_camera_from_db(camera_id)
    print(f"[Delete Camera] Removed from database")
    
    # Clean up results
    camera_results.pop(camera_id, None)
    camera_recognized_persons.pop(camera_id, None)
    occupancy_last_count.pop(camera_id, None)
    
    return {"status": "success", "message": f"Camera {camera_id} removed"}


@app.get("/api/cameras")
async def api_cameras():
    """Get all active cameras with their source info."""
    cameras = []
    for cam_id in camera_manager.get_active_cameras():
        # Get camera source from database
        cam_info = {"id": cam_id, "source": "Unknown"}
        try:
            db_cams = db_manager.get_cameras()
            for db_cam in db_cams:
                if db_cam[0] == cam_id:
                    cam_info["source"] = db_cam[1] if len(db_cam) > 1 else "Local"
                    break
        except:
            pass
        cameras.append(cam_info)
    return cameras

@app.get("/api/recognized/{camera_id}")
async def api_recognized_persons(camera_id: str):
    """Get recognized persons for a specific camera."""
    with recognized_lock:
        persons = camera_recognized_persons.get(camera_id, {})
        return [{"track_id": tid, "name": name} for tid, name in persons.items()]

@app.get("/api/occupancy")
async def api_occupancy(camera_id: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None):
    """Get occupancy data - either current counts or historical."""
    # If no time range specified, return current live counts from camera_results
    if not start_time and not end_time:
        results = []
        for cam_id in camera_manager.get_active_cameras():
            if camera_id and cam_id != camera_id:
                continue
            with results_lock:
                data = camera_results.get(cam_id, {})
                tracks = data.get("tracks", []) or []
                count = len(tracks)
            results.append({
                "id": cam_id,
                "camera_id": cam_id,
                "timestamp": int(time.time()),
                "count": count
            })
        return results
    
    # Historical data query
    rows = db_manager.search_occupancy(camera_id, start_time, end_time)
    return [{"id": r[0], "camera_id": r[1], "timestamp": r[2], "count": r[3]} for r in rows]

@app.get("/api/camera_daily_stats")
async def api_camera_daily_stats():
    """
    Returns today's person count stats per camera split into two 12-hour windows (IST):
      - am: 12:00 AM → 12:00 PM  (morning half)
      - pm: 12:00 PM → 12:00 AM  (evening half)
      - total: am + pm
    """
    stats = db_manager.get_camera_daily_person_stats()
    # Also include cameras currently active but with no detections yet
    for cam_id in camera_manager.get_active_cameras():
        if cam_id not in stats:
            stats[cam_id] = {"am": 0, "pm": 0, "total": 0}
    return stats

# ---------------------------------------------------------------------------
# Recording API
# ---------------------------------------------------------------------------
@app.post("/api/toggle_recording")
async def toggle_recording(camera_id: str = Form(...)):
    with writer_lock:
        is_recording = camera_id in camera_writers
        writer_data = camera_writers.pop(camera_id, None) if is_recording else None

    if is_recording and writer_data:
        # Stop
        if camera_id in recording_stop_events:
            recording_stop_events[camera_id].set()
            if camera_id in recording_threads:
                recording_threads[camera_id].join(timeout=5)
                del recording_threads[camera_id]
            del recording_stop_events[camera_id]
        proc = writer_data.get("process")
        if proc:
            try:
                proc.stdin.close()
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
        db_manager.end_recording(writer_data["db_id"])
        print(f"[Recording:{camera_id}] Stopped")
        return {"status": "success", "recording": False}

    # Start — get frame dims outside any lock
    with results_lock:
        data = camera_results.get(camera_id, {})
        frame = data.get("rendered_frame")
    if frame is None:
        return {"status": "error", "message": "Camera offline or warming up"}

    h, w = frame.shape[:2]
    ist_now = get_ist_time()
    timestamp = ist_now.strftime("%Y%m%d_%H%M%S")
    local_path = f"{LOCAL_RECORDINGS_DIR}/rec_{camera_id}_{timestamp}.mp4"
    os.makedirs(LOCAL_RECORDINGS_DIR, exist_ok=True)

    ffmpeg_cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", "2",
        "-i", "-", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "ultrafast", "-crf", "28", "-tune", "zerolatency",
        local_path
    ]
    try:
        p_ffmpeg = subprocess.Popen(
            ffmpeg_cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        db_id = db_manager.start_recording(camera_id, local_path)
        stop_event = threading.Event()
        thread = threading.Thread(
            target=recording_writer_thread, args=(camera_id, stop_event, p_ffmpeg), daemon=True)
        thread.start()
        with writer_lock:
            camera_writers[camera_id] = {
                "process": p_ffmpeg, "db_id": db_id,
                "start_time": ist_now, "file_path": local_path,
                "camera_id": camera_id, "w": w, "h": h
            }
        recording_threads[camera_id] = thread
        recording_stop_events[camera_id] = stop_event
        print(f"[Recording:{camera_id}] Started → {local_path}")
        return {"status": "success", "recording": True}
    except Exception as e:
        print(f"[Recording:{camera_id}] Start failure: {e}")
        return {"status": "error", "message": f"FFmpeg error: {e}"}

@app.get("/api/recording_status")
async def get_recording_status():
    with writer_lock:
        return {"active_recordings": list(camera_writers.keys())}

@app.get("/api/video_timeline/{record_id}")
async def video_timeline(record_id: str):
    """Get all detection timestamps relative to the start of a video recording."""
    rec = db_manager.get_recording(record_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found")
        
    camera_id = rec[1]
    start_time = rec[2]
    end_time = rec[3]
    
    # Optional logic: if video is ongoing, fetch till now
    from datetime import datetime
    import pytz
    if not end_time:
        end_time = datetime.now(pytz.timezone('Asia/Kolkata'))
        
    # Get all snapshots for this camera in this timeframe
    # Limit to 500 to prevent huge payloads, but capture all major events
    snapshots = db_manager.get_detection_snapshots(camera_id, start_time, end_time, limit=500)
    
    events = []
    # Note: start_time might be a naive or aware datetime depending on DB. 
    # Usually it's UTC or IST natively saved. We just compute total_seconds.
    for snap in snapshots:
        snap_time = snap[2] # 2 is timestamp
        
        # Ensure timezone info doesn't break subtraction
        try:
            if start_time.tzinfo is None and snap_time.tzinfo is not None:
                snap_time = snap_time.replace(tzinfo=None)
            elif start_time.tzinfo is not None and snap_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=None)
                
            offset = (snap_time - start_time).total_seconds()
            
            # Snapshots have exact timestamp. Ensure offset is realistic
            if offset >= 0:
                events.append({
                    "offset_sec": round(offset, 2),
                    "person_count": snap[3],
                    "timestamp": snap_time.isoformat()
                })
        except Exception:
            pass
            
    # Sort chronologically by offset
    events.sort(key=lambda x: x["offset_sec"])
    return {
        "status": "success",
        "start_time": start_time.isoformat() if hasattr(start_time, 'isoformat') else str(start_time),
        "events": events
    }


# ---------------------------------------------------------------------------
# Active Search API
# ---------------------------------------------------------------------------

@app.post("/api/start_search")
async def start_search(name: str = Form(...)):
    """Start an active face-search mission for the given person."""
    persons = db_manager.get_registered_persons()
    target = next((p for p in persons if p[1].lower() == name.lower()), None)
    if target is None:
        return {"status": "error", "message": f"'{name}' is not registered."}

    encoding = np.frombuffer(target[3], dtype=np.float32)
    with active_search_lock:
        active_search.clear()
        active_search.update({
            "running": True,
            "person_id": target[0],
            "name": target[1],
            "encoding": encoding,
            "found_track_ids": set()
        })
    print(f"[ActiveSearch] Mission started for: {target[1]}")
    return {
        "status": "success",
        "message": f"Searching for {target[1]}",
        "name": target[1],
        "image_path": target[2]  # registered photo from dataset/
    }


@app.post("/api/stop_search")
async def stop_search():
    """Stop the active search mission."""
    with active_search_lock:
        active_search.clear()
    print("[ActiveSearch] Mission stopped.")
    return {"status": "success"}


@app.get("/api/active_search")
async def get_active_search():
    """Return current active search target (if any)."""
    with active_search_lock:
        name = active_search.get("name")
    return {"active": name is not None, "name": name}


# ---------------------------------------------------------------------------
# History Search API
# ---------------------------------------------------------------------------

@app.get("/api/search")
async def api_search(name: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None):
    results = db_manager.search_detections(name, start_time, end_time)
    return [{"id": r[0], "person_name": r[5] or "Unknown", "camera_id": r[2], "timestamp": format_12h(r[3]), "image_path": r[4]} for r in results]


@app.get("/api/registered_detections")
async def api_registered_detections(name: Optional[str] = None):
    """Get logs for registered detections (for the "Detected" button)."""
    logs = db_manager.get_registered_detections(name)
    
    # Get camera IP for logs
    cameras = {c[0]: c[1] for c in db_manager.get_cameras()}
    
    # Get reference photos for registered persons
    persons_db = db_manager.get_registered_persons()
    person_images = {p[1]: p[2] for p in persons_db}
    
    formatted = []
    for l in logs:
        cam_id = l.get("camera_id")
        cam_source = cameras.get(cam_id, "Unknown")
        cam_ip = "Unknown"
        if cam_source and cam_source != "Unknown":
            cam_ip = cam_source
            if "@" in cam_source:
                 cam_ip = cam_source.split("@")[-1].split(":")[0].split("/")[0]
                 
        pname = l.get("person_name", "Unknown")
        pimage = person_images.get(pname)
                 
        formatted.append({
            "id": str(l.get("_id", l.get("id"))),
            "person_name": pname,
            "image_path": pimage,
            "camera_id": cam_id,
            "camera_ip": cam_ip,
            "timestamp": format_12h(l["timestamp"]),
        })
    return formatted

@app.post("/api/set_tracking_area/{camera_id}")
async def set_tracking_area(camera_id: str, area: dict):
    """Set tracking area for a camera."""
    db_manager.set_camera_tracking_area(camera_id, area)
    return {"status": "success"}

@app.get("/api/get_tracking_area/{camera_id}")
async def get_tracking_area(camera_id: str):
    """Get tracking area for a camera."""
    area = db_manager.get_camera_tracking_area(camera_id)
    return area or {}


@app.get("/api/search_detections")
async def api_search_detections(name: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None):
    """Search detection history with filters."""
    # Convert string times to datetime
    start = datetime.fromisoformat(start_time).replace(tzinfo=IST) if start_time else None
    end = datetime.fromisoformat(end_time).replace(tzinfo=IST) if end_time else None
    
    snapshots = db_manager.get_detection_snapshots(start_time=start, end_time=end)
    
    # Get all cameras to map names/IPs
    all_cams = {c[0]: c[1] for c in db_manager.get_cameras()}
    
    def extract_ip(url):
        if not url: return "Local"
        if not isinstance(url, str): return "Local"
        if "rtsp://" not in url: return url
        try:
            host_part = url.split("@")[-1].split("/")[0]
            return host_part.split(":")[0]
        except: return url

    formatted = []
    for s in snapshots:
        # filter by name if needed (bbox_data contains names)
        bbox = json.loads(s[5]) if s[5] else []
        names = [p.get("name") or f"#{p.get('id')}" for p in bbox]
        
        if name and name.lower() not in [n.lower() for n in names]:
            continue
            
        cam_id = s[1]
        cam_source = all_cams.get(cam_id, "N/A")
        
        formatted.append({
            "id": s[0],
            "camera_name": cam_id,
            "camera_id": cam_id,
            "camera_ip": extract_ip(cam_source),
            "timestamp": s[2].isoformat() if isinstance(s[2], datetime) else s[2],
            "person_count": s[3],
            "image_path": s[4],
            "person_names": names,
            "person_crops": s[6] if len(s) > 6 else []
        })
        
    return formatted

@app.post("/api/search_by_image")
async def search_by_image(
    file: UploadFile = File(...), 
    start_time: Optional[str] = Form(None), 
    end_time: Optional[str] = Form(None)
):
    """Historical similarity search across all snapshots within a time range."""
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"error": "Invalid image format"}
        
    target_encoding = recognizer.get_encoding(image)
    if target_encoding is None:
        return {"error": "No face detected in uploaded image"}

    # Parse time strings to datetime objects
    start_dt = datetime.fromisoformat(start_time).replace(tzinfo=IST) if start_time else None
    end_dt = datetime.fromisoformat(end_time).replace(tzinfo=IST) if end_time else None

    # Search snapshots in DB using vector similarity
    matches = db_manager.search_snapshots_by_similarity(target_encoding, start_dt, end_dt)
    
    results = []
    for m in matches:
        person_name = "Detected Person"
        if m.get("bbox_data"):
            registered = [p.get("name") for p in m["bbox_data"] if p.get("name") and p.get("name") != "Unknown"]
            if registered:
                person_name = ", ".join(registered)

        results.append({
            "id": str(m["_id"]),
            "timestamp": m["timestamp"].isoformat() if hasattr(m["timestamp"], "isoformat") else m["timestamp"],
            "camera_id": m["camera_id"],
            "image_path": m["snapshot_path"],
            "person_name": person_name
        })
    
    return results


@app.post("/clear_history")
async def clear_history():
    try:
        db_manager.delete_all_detections()
    except Exception as e:
        print(f"DB clear error: {e}")

    snaps_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
    deleted: int = 0
    if os.path.isdir(snaps_dir):
        for entry in os.listdir(snaps_dir):
            entry_path = os.path.join(snaps_dir, entry)
            try:
                if os.path.isfile(entry_path):
                    os.remove(entry_path)
                    deleted += 1
                elif os.path.isdir(entry_path):
                    # Per-camera subdirectory — remove all files inside
                    for fname in os.listdir(entry_path):
                        fpath = os.path.join(entry_path, fname)
                        if os.path.isfile(fpath):
                            os.remove(fpath)
                            deleted += 1
            except Exception:
                pass

    print(f"Cleared {deleted} snapshots.")
    return {"status": "success", "message": f"Cleared {deleted} records"}

@app.get("/api/recordings")
async def api_recordings(camera_id: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None):
    results = db_manager.search_recordings(camera_id, start_time, end_time)
    return [{
        "id": r[0], 
        "camera_id": r[1], 
        "start_time": format_12h(r[2]), 
        "end_time": format_12h(r[3]) if r[3] else None, 
        "file_path": r[4], 
        "has_registered_person": r[5], 
        "registered_person_times": [format_12h(ts) for ts in (r[6] if len(r) > 6 else [])]
    } for r in results]

@app.delete("/api/recordings/{record_id}")
async def delete_recording(record_id: str):
    rec = db_manager.get_recording(record_id)
    if rec:
        file_path = rec[4]
        # Delete local file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass
        db_manager.delete_recording(record_id)
    return {"status": "success"}

# ---------------------------------------------------------------------------
# Camera Recording Settings API
# ---------------------------------------------------------------------------

@app.get("/api/camera_settings/{camera_id}")
async def get_camera_settings(camera_id: str):
    """Get recording settings for a camera."""
    db_setting = db_manager.get_camera_recording_setting(camera_id)
    # Also check if actually recording
    with writer_lock:
        actually_recording = camera_id in camera_writers
    return {"camera_id": camera_id, "recording_enabled": bool(db_setting), "actually_recording": actually_recording}

@app.post("/api/camera_settings/{camera_id}")
async def set_camera_settings(camera_id: str, enabled: bool = Form(...)):
    """Set recording settings for a camera and start/stop actual recording."""
    db_manager.set_camera_recording(camera_id, enabled)

    if enabled:
        with writer_lock:
            already = camera_id in camera_writers
        if already:
            return {"status": "success", "camera_id": camera_id, "recording_enabled": True}

        # Grab frame dimensions OUTSIDE writer_lock to avoid deadlock
        with results_lock:
            data = camera_results.get(camera_id, {})
            frame = data.get("rendered_frame")
        if frame is None:
            return {"status": "error", "message": "Camera not streaming yet — try again in a moment"}
        h, w = frame.shape[:2]

        ist_now = get_ist_time()
        timestamp = ist_now.strftime("%Y%m%d_%H%M%S")
        filename = f"rec_{camera_id}_{timestamp}.mp4"
        local_path = f"{LOCAL_RECORDINGS_DIR}/{filename}"
        os.makedirs(LOCAL_RECORDINGS_DIR, exist_ok=True)

        ffmpeg_cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", "2",
            "-i", "-", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "ultrafast", "-crf", "28", "-tune", "zerolatency",
            local_path
        ]
        try:
            p_ffmpeg = subprocess.Popen(
                ffmpeg_cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            db_id = db_manager.start_recording(camera_id, local_path)
            stop_event = threading.Event()
            # Set camera_writers BEFORE starting thread to avoid race condition
            with writer_lock:
                camera_writers[camera_id] = {
                    "process": p_ffmpeg, "db_id": db_id,
                    "start_time": ist_now, "file_path": local_path,
                    "camera_id": camera_id, "w": w, "h": h
                }
            thread = threading.Thread(
                target=recording_writer_thread, args=(camera_id, stop_event, p_ffmpeg), daemon=True)
            thread.start()
            recording_threads[camera_id] = thread
            recording_stop_events[camera_id] = stop_event
            print(f"[Recording:{camera_id}] Started → {local_path}")
        except Exception as e:
            logger.error(f"Failed to start FFmpeg: {e}")
            return {"status": "error", "message": str(e)}

    else:
        with writer_lock:
            writer_data = camera_writers.pop(camera_id, None)
        if writer_data:
            # Signal thread to stop
            if camera_id in recording_stop_events:
                recording_stop_events[camera_id].set()
                if camera_id in recording_threads:
                    recording_threads[camera_id].join(timeout=5)
                    del recording_threads[camera_id]
                del recording_stop_events[camera_id]
            # Close FFmpeg
            proc = writer_data.get("process")
            if proc:
                try:
                    proc.stdin.close()
                    proc.wait(timeout=10)
                except Exception:
                    proc.kill()
            db_manager.end_recording(writer_data["db_id"])
            print(f"[Recording:{camera_id}] Stopped")

    return {"status": "success", "camera_id": camera_id, "recording_enabled": enabled}

# ---------------------------------------------------------------------------
# Detection Snapshots API
# ---------------------------------------------------------------------------

@app.get("/api/detection_snapshots")
async def get_detection_snapshots(
    camera_id: Optional[str] = None,
    limit: int = 20,
    skip: int = 0
):
    """Get detection snapshots with pagination."""
    snapshots = db_manager.get_detection_snapshots(
        camera_id=camera_id, limit=limit, skip=skip)
    total = db_manager.count_detection_snapshots(camera_id=camera_id)
    return {
        "items": [
            {
                "id": s[0],
                "camera_id": s[1],
                "timestamp": s[2].isoformat() if hasattr(s[2], 'isoformat') else s[2],
                "person_count": s[3],
                "snapshot_path": s[4],
                "bbox_data": s[5]
            }
            for s in snapshots
        ],
        "total": total,
        "skip": skip,
        "limit": limit
    }

@app.get("/api/snapshot/{snapshot_id}")
async def get_snapshot(snapshot_id: str):
    """Get a specific snapshot with bounding box data."""
    snapshot = db_manager.get_snapshot(snapshot_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return {
        "id": snapshot[0],
        "camera_id": snapshot[1],
        "timestamp": snapshot[2],
        "person_count": snapshot[3],
        "snapshot_path": snapshot[4],
        "bbox_data": snapshot[5]
    }

# ---------------------------------------------------------------------------
# Remote Image Proxy API
# ---------------------------------------------------------------------------

@app.get("/api/snapshot_image")
async def get_snapshot_image(path: str):
    """Serve images from local filesystem."""
    try:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                content = f.read()
            from fastapi.responses import Response
            return Response(content=content, media_type="image/jpeg")
        else:
            raise HTTPException(status_code=404, detail="Image not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/recording_video")
async def get_recording_video(path: str, request: Request):
    """Serve video directly from local filesystem."""
    try:
        import os
        from fastapi.responses import FileResponse
        if os.path.exists(path):
            return FileResponse(
                path, 
                media_type="video/mp4",
                filename=os.path.basename(path),
                headers={"Accept-Ranges": "bytes"}
            )
        else:
            raise HTTPException(status_code=404, detail="Video not found")
    except Exception as e:
        logger.error(f"Error streaming video {path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# Video Person Search API
# ---------------------------------------------------------------------------

import json
from fastapi import BackgroundTasks

# Store video search progress
video_search_progress: Dict[str, Any] = {}
video_search_lock = threading.Lock()

@app.get("/api/persons")
async def api_persons():
    """Get all registered persons with last seen info."""
    persons = db_manager.get_persons_with_last_seen()
    return persons


@app.get("/api/registered_persons")
async def api_registered_persons():
    """Alias for /api/persons for frontend compatibility."""
    persons = db_manager.get_persons_with_last_seen()
    return persons


@app.get("/api/analytics/hourly")
async def api_analytics_hourly(camera_id: Optional[str] = None):
    """Get max person count for each hour of last 24h."""
    analytics_data = db_manager.get_hourly_analytics(camera_id)
    
    # Map back to full 24h list
    hour_map = {int(r["_id"]): r for r in analytics_data}
    data = []
    now = get_ist_time()
    for i in range(24):
        check_time = now - timedelta(hours=(23-i))
        h = check_time.hour
        h_data = hour_map.get(h, {"max_count": 0, "camera_ids": []})
        data.append({
            "hour": h,
            "label": check_time.strftime("%I %p"), # "04 PM"
            "count": h_data["max_count"],
            "camera_id": h_data["camera_ids"][0] if h_data["camera_ids"] else (camera_id or "")
        })
    return data

@app.get("/api/analytics/daily")
async def api_analytics_daily(camera_id: Optional[str] = None, days: int = 7):
    """Get max person count for each day of last N days."""
    analytics_data = db_manager.get_daily_analytics(camera_id, days=days)
    
    # Map to days list
    day_map = {f"{r['_id']['year']}-{r['_id']['month']:02d}-{r['_id']['day']:02d}": r["max_count"] for r in analytics_data}
    data = []
    now = get_ist_time()
    for i in range(days):
        check_date = now - timedelta(days=(days-1-i))
        key = f"{check_date.year}-{check_date.month:02d}-{check_date.day:02d}"
        data.append({
            "date": key,
            "label": check_date.strftime("%d %b"),
            "count": day_map.get(key, 0)
        })
    return data


@app.post("/api/register_person")
async def api_register_person(name: str = Form(...), file: UploadFile = File(...)):
    """Register a person via API (direct stream)."""
    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"status": "error", "message": "Invalid image file."}

    encoding = recognizer.get_encoding(image)
    if encoding is not None:
        local_path = f"{DATASET_DIR}/{name}/{file.filename}"
        
        def on_api_reg_complete(success):
            if success:
                db_manager.register_person(name, local_path, encoding.tobytes())
                recognizer.load_known_faces(db_manager)
        
        if stream_bytes_to_local(content, local_path, callback=on_api_reg_complete):
            return {"status": "success", "message": f"{name} registration queued for local saving."}
        else:
            return {"status": "error", "message": "Storage queue full."}
            
    return {"status": "error", "message": "No face detected in the image."}


@app.delete("/api/delete_person/{person_id}")
async def api_delete_person(person_id: int):
    """Delete a registered person from the database."""
    try:
        # Get person info to delete files
        persons = db_manager.get_registered_persons()
        person = next((p for p in persons if str(p[0]) == str(person_id)), None)
        
        if person:
            # Delete local files
            image_path = person[2]
            if image_path:
                try:
                    import shutil
                    d = os.path.dirname(image_path)
                    if d and os.path.exists(d):
                        shutil.rmtree(d)
                except Exception as e:
                    print(f"[Delete Person] Error deleting files: {e}")
            
            # Delete from DB
            db_manager.delete_person_from_db(person_id)
            recognizer.load_known_faces(db_manager)
            return {"status": "success"}
        
        return {"status": "error", "message": "Person not found"}
    except Exception as e:
        print(f"[Delete Person] Error: {e}")
        return {"status": "error", "message": str(e)}


def scan_video_for_person(video_path: str, target_encoding: np.ndarray, sample_interval: int = 10) -> list:
    """
    Scan a video file for ALL occurrences of a person with the target face encoding.
    Detects every face in each frame and matches against the target person.
    Groups continuous appearances into flagged segments with start/end timestamps.
    Returns list of detection segments where the person appears.
    """
    results = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[VideoScan] ERROR: Could not open video {video_path}")
        return results
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Track continuous appearances
    current_segment = None
    last_match_frame = -1
    min_segment_gap = int(fps * 2)  # 2 seconds gap to create new segment
    
    # Lower threshold for better detection (same as live recognition)
    DISTANCE_THRESHOLD = 1.15
    
    print(f"[VideoScan] Starting scan of {video_path}")
    print(f"[VideoScan] Total frames: {total_frames}, FPS: {fps}, Sample interval: {sample_interval}")
    
    matches_found = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every Nth frame for efficiency
        if frame_count % sample_interval == 0:
            try:
                # Detect ALL faces in frame using full frame (not just body crop)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with recognizer.ai_lock:
                    boxes, probs = recognizer.mtcnn.detect(frame_rgb)
                
                match_found = False
                best_confidence = 0.0
                best_distance = 999.0
                
                if boxes is not None and len(boxes) > 0:
                    # Check EACH face in the frame against target
                    for i, box in enumerate(boxes):
                        fx1, fy1, fx2, fy2 = [int(b) for b in box]
                        
                        # Ensure valid box
                        fx1, fy1 = max(0, fx1), max(0, fy1)
                        fx2, fy2 = min(frame.shape[1], fx2), min(frame.shape[0], fy2)
                        
                        fw, fh = fx2 - fx1, fy2 - fy1
                        if fw < 30 or fh < 30:  # Skip very small faces
                            continue
                        
                        face_crop = frame_rgb[fy1:fy2, fx1:fx2]
                        
                        if face_crop.size > 0:
                            face_resized = cv2.resize(face_crop, (160, 160))
                            face_tensor = torch.tensor(np.transpose(face_resized, (2, 0, 1))).float().unsqueeze(0).to(recognizer.device)
                            face_tensor = (face_tensor - 127.5) / 128.0
                            
                            with recognizer.ai_lock:
                                with torch.no_grad():
                                    embedding = recognizer.resnet(face_tensor).cpu().numpy()[0]
                            
                            # Compare with target
                            distance = float(np.linalg.norm(target_encoding - embedding))
                            confidence = 1 - (distance / 2.0)
                            
                            if distance < DISTANCE_THRESHOLD:  # Match found
                                match_found = True
                                matches_found += 1
                                if confidence > best_confidence:
                                    best_confidence = confidence
                                    best_distance = distance
                                if frame_count % 100 == 0:  # Log every 100th match frame
                                    print(f"[VideoScan] Match at frame {frame_count}, dist: {distance:.3f}, conf: {confidence:.2f}")
                
                # Handle segment tracking
                if match_found:
                    timestamp_sec = frame_count / fps
                    
                    if current_segment is None or (frame_count - last_match_frame) > min_segment_gap:
                        # Start new segment
                        if current_segment is not None:
                            results.append(current_segment)
                        current_segment = {
                            "start_seconds": timestamp_sec,
                            "start_timestamp": f"{int(timestamp_sec // 60)}:{int(timestamp_sec % 60):02d}",
                            "end_seconds": timestamp_sec,
                            "end_timestamp": f"{int(timestamp_sec // 60)}:{int(timestamp_sec % 60):02d}",
                            "confidence": best_confidence,
                            "start_frame": frame_count,
                            "end_frame": frame_count
                        }
                        print(f"[VideoScan] New segment started at {current_segment['start_timestamp']}")
                    else:
                        # Extend current segment
                        current_segment["end_seconds"] = timestamp_sec
                        current_segment["end_timestamp"] = f"{int(timestamp_sec // 60)}:{int(timestamp_sec % 60):02d}"
                        current_segment["end_frame"] = frame_count
                        if best_confidence > current_segment["confidence"]:
                            current_segment["confidence"] = best_confidence
                    
                    last_match_frame = frame_count
                    
            except Exception as e:
                print(f"[VideoScan] Error processing frame {frame_count}: {e}")
                import traceback
                traceback.print_exc()
        
        frame_count += 1
        
        # Progress update every 500 frames
        if frame_count % 500 == 0 and total_frames > 0:
            progress = (frame_count / total_frames) * 100
            print(f"[VideoScan] Progress: {progress:.1f}% ({frame_count}/{total_frames})")
    
    # Don't forget the last segment
    if current_segment is not None:
        results.append(current_segment)
    
    cap.release()
    print(f"[VideoScan] Scan complete. Found {len(results)} segments, {matches_found} total matches")
    return results


@app.post("/api/search_video_by_name")
async def search_video_by_name(request: Request):
    """Search for a person by name across selected videos."""
    data = await request.json()
    name = data.get("name")
    video_ids = data.get("video_ids", [])
    
    if not name or not video_ids:
        return {"status": "error", "message": "Name and video IDs required"}
    
    # Get person's encoding
    persons = db_manager.get_registered_persons()
    target = next((p for p in persons if p[1].lower() == name.lower()), None)
    if target is None:
        return {"status": "error", "message": f"Person '{name}' not found"}
    
    target_encoding = np.frombuffer(target[3], dtype=np.float32)
    
    # Search each video
    all_results = []
    total_segments = 0
    for vid_id in video_ids:
        rec = db_manager.get_recording(vid_id)
        if rec and os.path.exists(rec[4]):
            segments = scan_video_for_person(rec[4], target_encoding)
            total_segments += len(segments)
            for segment in segments:
                all_results.append({
                    **segment,
                    "video_id": vid_id,
                    "video_name": os.path.basename(rec[4]),
                    "video_path": rec[4],
                    "camera_id": rec[1],
                    "person_name": name
                })
    
    # Sort by start time
    all_results.sort(key=lambda x: x["start_seconds"])
    
    return {
        "status": "success", 
        "results": all_results,
        "total_segments": total_segments,
        "videos_searched": len(video_ids)
    }


@app.post("/api/search_video_by_image")
async def search_video_by_image(file: UploadFile = File(...), video_ids: str = Form(...)):
    """Search for a person using an uploaded image across selected videos."""
    video_ids_list = json.loads(video_ids)
    
    if not video_ids_list:
        return {"status": "error", "message": "Video IDs required"}
    
    # Get encoding from uploaded image
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    target_encoding = recognizer.get_encoding(image)
    if target_encoding is None:
        return {"status": "error", "message": "No face detected in uploaded image"}
    
    # Search each video
    all_results = []
    total_segments = 0
    for vid_id in video_ids_list:
        rec = db_manager.get_recording(vid_id)
        if rec and os.path.exists(rec[4]):
            segments = scan_video_for_person(rec[4], target_encoding)
            total_segments += len(segments)
            for segment in segments:
                all_results.append({
                    **segment,
                    "video_id": vid_id,
                    "video_name": os.path.basename(rec[4]),
                    "video_path": rec[4],
                    "camera_id": rec[1],
                    "person_name": "Unknown (from image)"
                })
    
    # Sort by start time
    all_results.sort(key=lambda x: x["start_seconds"])
    
    return {
        "status": "success", 
        "results": all_results,
        "total_segments": total_segments,
        "videos_searched": len(video_ids_list)
    }


@app.post("/api/upload_video_and_search")
async def upload_video_and_search(
    video: UploadFile = File(...),
    person_image: Optional[UploadFile] = File(None),
    person_name: Optional[str] = Form(None),
):
    """
    Upload a video file + either a person image or registered person name.
    Scans the uploaded video and returns timeline segments where the person appears.
    """
    # 1. Save uploaded video to a temp path
    import tempfile
    suffix = os.path.splitext(video.filename)[1] or ".mp4"
    tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=suffix,
                                            dir=RECORDINGS_DIR)
    try:
        content = await video.read()
        tmp_video.write(content)
        tmp_video.flush()
        tmp_video.close()
        video_path = tmp_video.name

        # 2. Get target encoding
        target_encoding = None
        display_name = "Unknown"

        if person_name:
            persons = db_manager.get_registered_persons()
            target = next((p for p in persons if p[1].lower() == person_name.lower()), None)
            if target is None:
                return {"status": "error", "message": f"Person '{person_name}' not registered"}
            target_encoding = np.frombuffer(target[3], dtype=np.float32)
            display_name = target[1]

        elif person_image:
            img_bytes = await person_image.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return {"status": "error", "message": "Invalid image file"}
            target_encoding = recognizer.get_encoding(img)
            if target_encoding is None:
                return {"status": "error", "message": "No face detected in uploaded image"}
            display_name = "Uploaded Person"
        else:
            return {"status": "error", "message": "Provide person_name or person_image"}

        # 3. Scan video
        segments = scan_video_for_person(video_path, target_encoding)

        # 4. Register the uploaded video in DB so it can be played back
        ist_now = get_ist_time()
        db_id = db_manager.start_recording("uploaded", video_path)
        db_manager.end_recording(db_id)

        results = []
        for seg in segments:
            results.append({
                **seg,
                "video_id": db_id,
                "video_name": video.filename,
                "video_path": video_path,
                "camera_id": "uploaded",
                "person_name": display_name,
            })

        return {
            "status": "success",
            "results": results,
            "total_segments": len(results),
            "video_id": db_id,
            "video_path": video_path,
            "person_name": display_name,
        }
    except Exception as e:
        logger.error(f"[UploadVideoSearch] {e}")
        return {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------
# Video streaming
# ---------------------------------------------------------------------------

async def gen_frames(camera_id: str):
    """Generate MJPEG stream at 2 FPS matching processing rate."""
    import cv2
    import time
    import asyncio
    
    last_sent_id = -1
    last_send_time = 0
    FRAME_INTERVAL = 0.5  # 2 FPS to match processing
    
    while True:
        with results_lock:
            data = camera_results.get(camera_id, {})
            frame = data.get("rendered_frame")
            frame_id = data.get("frame_id", -1)
        
        # Skip if no frame
        if frame is None:
            await asyncio.sleep(0.05)
            continue
        
        # Rate limit to 2 FPS
        current_time = time.time()
        if current_time - last_send_time < FRAME_INTERVAL:
            await asyncio.sleep(0.05)
            continue
        
        # Send latest frame even if not new (maintains 2 FPS stream)
        last_sent_id = frame_id
        last_send_time = current_time

        # Resize for streaming
        h, w = frame.shape[:2]
        target_w = 1280
        if w > target_w:
            scale = target_w / w
            target_h = int(h * scale)
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # JPEG encoding
        encode_params = [
            cv2.IMWRITE_JPEG_QUALITY, 75,
            cv2.IMWRITE_JPEG_OPTIMIZE, 0,
        ]
        
        ret, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n"
               b"\r\n" + frame_bytes + b"\r\n")


@app.get("/video_feed/{camera_id}")
async def video_feed(camera_id: str):
    return StreamingResponse(gen_frames(camera_id), media_type="multipart/x-mixed-replace; boundary=frame")



@app.get("/api/capture_frame/{camera_id}")
async def capture_frame(camera_id: str):
    """Return the latest frame as a static JPEG for the pause feature."""
    with results_lock:
        data = camera_results.get(camera_id, {})
        frame = data.get("rendered_frame")
    if frame is None:
        raise HTTPException(status_code=404, detail="No frame available")
    ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ret:
        raise HTTPException(status_code=500, detail="Encode failed")
    from fastapi.responses import Response
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/api/live_results/{camera_id}")
async def get_live_results(camera_id: str):
    """Get the current tracking results and face crops for a camera."""
    with results_lock:
        data = camera_results.get(camera_id, {})
        persons = data.get("tracks", []) or []
    
    return [
        {
            "id": p["id"],
            "name": p["name"],
            "face_crop": p.get("face_crop")
        } 
        for p in persons
    ]


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
