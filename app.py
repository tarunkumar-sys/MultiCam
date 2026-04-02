import cv2
import numpy as np
import json
import os
import shutil
import hashlib
import secrets
from fastapi import FastAPI, Request, File, UploadFile, Form, Response, Depends, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from database.db_manager import DatabaseManager
from utils.detector import PersonDetector
from utils.tracker import ObjectTracker
from utils.recognizer import FaceRecognizer
from cameras.camera_manager import CameraManager
import threading
import time
import urllib.parse
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
load_dotenv()

from utils.alert_manager import AlertManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize_rtsp_url(url: str) -> str:
    if not isinstance(url, str) or not url.startswith("rtsp://"):
        return url
    url_str = str(url)
    last_at = url_str.rfind("@")
    if last_at == -1:
        return url_str
    auth_part = url_str[7:last_at]
    if ":" in auth_part:
        user, pwd = auth_part.split(":", 1)
        safe_pwd = urllib.parse.quote(pwd)
        return f"rtsp://{user}:{safe_pwd}{url_str[last_at:]}"
    return url_str

def calculate_iou(box1, box2):
    """box as [x1, y1, x2, y2]"""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return interArea / float(box1Area + box2Area - interArea + 1e-6)

def get_person_color(pid: int):
    import colorsys
    # Use golden angle for distinct hues
    hue = (pid * 137.508) % 360
    r, g, b = colorsys.hsv_to_rgb(hue / 360.0, 0.9, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))  # BGR for OpenCV

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

os.makedirs("snapshots", exist_ok=True)
os.makedirs("dataset", exist_ok=True)
os.makedirs("recordings", exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/snapshots", StaticFiles(directory="snapshots"), name="snapshots")
app.mount("/dataset", StaticFiles(directory="dataset"), name="dataset")
app.mount("/recordings", StaticFiles(directory="recordings"), name="recordings")
templates = Jinja2Templates(directory="templates")

db_manager = DatabaseManager()
detector = PersonDetector()
recognizer = FaceRecognizer()
camera_manager = CameraManager()
recognizer.load_known_faces(db_manager)
alert_manager = AlertManager()

# Session store (simple in-memory)
sessions: Dict[str, str] = {}

# ---------------------------------------------------------------------------
# Authentication Middleware
# ---------------------------------------------------------------------------

PUBLIC_PATHS = {"/login", "/api/login", "/static", "/favicon.ico"}

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        # Allow public paths
        if path == "/login" or path == "/api/login" or path.startswith("/static"):
            return await call_next(request)
        # Check session cookie
        session_id = request.cookies.get("session_id")
        if not session_id or session_id not in sessions:
            if path.startswith("/api/") or path.startswith("/video_feed/"):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return RedirectResponse("/login", status_code=302)
        return await call_next(request)

app.add_middleware(AuthMiddleware)

# ---------------------------------------------------------------------------
# Background: Log cleanup every 30 minutes
# ---------------------------------------------------------------------------

def log_cleanup_worker():
    while True:
        try:
            deleted = db_manager.cleanup_old_logs()
            if deleted:
                print(f"[LogCleanup] Deleted {deleted} old log entries")
        except Exception as e:
            print(f"[LogCleanup] Error: {e}")
        time.sleep(1800)  # 30 minutes

threading.Thread(target=log_cleanup_worker, daemon=True).start()

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

camera_results: Dict[str, Any] = {}
results_lock = threading.Lock()

camera_writers: Dict[str, Any] = {}
writer_lock = threading.Lock()
occupancy_last_count: Dict[str, int] = {}

active_search: Dict[str, Any] = {}
active_search_lock = threading.Lock()

# camera_id -> list of face embeddings for deduplication
saved_encodings: Dict[str, List[np.ndarray]] = {}
saved_encodings_lock = threading.Lock()

# camera_id -> ROI (either [x,y,w,h] or list of [x,y] points)
all_rois_raw = db_manager.get_all_camera_rois()
camera_rois: Dict[str, Any] = {}
for cid, r_json in all_rois_raw.items():
    try:
        camera_rois[cid] = json.loads(r_json)
    except:
        camera_rois[cid] = None
rois_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Face crop helper
# ---------------------------------------------------------------------------

def extract_face_crop(frame, bbox, padding=0.3):
    """Extract a face crop from the upper portion of a person bbox with padding."""
    h, w = frame.shape[:2]
    bx1, by1, bx2, by2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    bw = bx2 - bx1
    bh = by2 - by1
    
    # Face region: upper 45% of body bbox, centered horizontally
    fx1 = bx1 + int(0.1 * bw)
    fx2 = bx2 - int(0.1 * bw)
    fy1 = by1
    fy2 = by1 + int(0.45 * bh)
    
    # Add padding
    pw = int((fx2 - fx1) * padding)
    ph = int((fy2 - fy1) * padding)
    fx1 = max(0, fx1 - pw)
    fy1 = max(0, fy1 - ph)
    fx2 = min(w, fx2 + pw)
    fy2 = min(h, fy2 + ph)
    
    if fx2 - fx1 < 20 or fy2 - fy1 < 20:
        return None
    
    return frame[fy1:fy2, fx1:fx2]

# ---------------------------------------------------------------------------
# Camera processing
# ---------------------------------------------------------------------------

def process_camera(camera_id: str):
    print(f"[process_camera] Thread started for: {camera_id}")
    tracker: Any = ObjectTracker()
    last_frame_id: int = -1
    
    FRAME_INTERVAL = 0.5  # 2 FPS processing
    RECOGNITION_INTERVAL = 2.0  # Recognition every 2 seconds if person present
    
    last_process_time = 0
    last_recognition_time = 0
    
    current_tracks = []
    recognition_cache: Dict[Any, tuple] = {}
    seen_track_ids = set()
    
    # Cache for tracking recognition results
    RECOGNITION_CACHE_TTL = 30  # Seconds
    
    while True:
        frame, frame_id = camera_manager.get_camera_frame_with_id(camera_id)
        if frame is None or frame_id == last_frame_id:
            time.sleep(0.01)
            continue
        last_frame_id = frame_id
        
        now = time.time()
        h, w = frame.shape[:2]

        # 1. Higher-level Logic: Detection & Tracking at 2 FPS
        if now - last_process_time >= FRAME_INTERVAL:
            last_process_time = now
            
            # Apply ROI if defined
            current_roi = None
            with rois_lock:
                current_roi = camera_rois.get(camera_id)
            
            detect_frame = frame.copy()
            if current_roi:
                mask = np.zeros((h, w), dtype=np.uint8)
                if isinstance(current_roi, list) and len(current_roi) > 0:
                    if isinstance(current_roi[0], list): # Polygon points
                        # Map normalized [0..1] points to absolute pixels
                        abs_pts = []
                        for p in current_roi:
                            if len(p) >= 2:
                                abs_pts.append([int(p[0] * w), int(p[1] * h)])
                        if abs_pts:
                            pts = np.array(abs_pts, np.int32)
                            cv2.fillPoly(mask, [pts], 255)
                    elif len(current_roi) == 4: # [x, y, w, h]
                        # Rect could also be absolute or relative. Let's assume relative if < 1
                        rx, ry, rw, rh = current_roi
                        if rx <= 1 and ry <= 1 and rw <= 1 and rh <= 1:
                            rx, ry, rw, rh = int(rx*w), int(ry*h), int(rw*w), int(rh*h)
                        cv2.rectangle(mask, (rx, ry), (rx+rw, ry+rh), 255, -1)
                
                detect_frame = cv2.bitwise_and(frame, frame, mask=mask)
            
            # Run detection
            detections = detector.detect(detect_frame)
            
            # Run tracking
            raw_tracks = tracker.update(detections, frame)
            
            # 2. Double Box Elimination (Manual NMS after tracking)
            # Sort by ID (lowest first)
            sorted_tracks = sorted(raw_tracks, key=lambda x: int(x["id"]) if str(x["id"]).isdigit() else 9999)
            filtered_tracks = []
            
            for t1 in sorted_tracks:
                is_duplicate = False
                b1 = t1["bbox"]
                for t2 in filtered_tracks:
                    b2 = t2["bbox"]
                    if calculate_iou(b1, b2) > 0.7:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    filtered_tracks.append(t1)
            
            current_tracks = filtered_tracks
            
            # Maintain total count seen
            for t in current_tracks:
                if t["id"] not in seen_track_ids:
                    seen_track_ids.add(t["id"])

            # 3. Recognition Logic at slightly slower rate
            if now - last_recognition_time >= RECOGNITION_INTERVAL:
                last_recognition_time = now
                
                for t in current_tracks:
                    tid = t["id"]
                    
                    # Check cache
                    cached = recognition_cache.get(tid)
                    if cached:
                        cached_name, cached_conf, cached_time = cached
                        if now - cached_time < RECOGNITION_CACHE_TTL and cached_name != "Unknown":
                            continue
                    
                    # Try recognition
                    bbox = t["bbox"]
                    face_box = [
                        bbox[0] + int(0.15 * (bbox[2]-bbox[0])),
                        bbox[1],
                        bbox[2] - int(0.15 * (bbox[2]-bbox[0])),
                        bbox[1] + int(0.45 * (bbox[3]-bbox[1]))
                    ]
                    
                    # Ensure face box is within frame
                    face_box = [max(0, face_box[0]), max(0, face_box[1]), min(w, face_box[2]), min(h, face_box[3])]
                    
                    if (face_box[2] - face_box[0]) > 20 and (face_box[3] - face_box[1]) > 20:
                        try:
                            # Use threaded worker for recognition to avoid blocking
                            threading.Thread(
                                target=recognition_worker,
                                args=(frame.copy(), face_box, tid, camera_id, recognition_cache),
                                daemon=True
                            ).start()
                        except Exception:
                            pass

        # 4. Rendering & Streaming (FULL RATE)
        record_frame = frame.copy()
        people_count = 0
        
        display_tracks = []
        for t in current_tracks:
            tid = t["id"]
            bbox = t["bbox"]
            
            # Fetch recognition result from cache if exists
            name, conf = "Unknown", 0.0
            cached = recognition_cache.get(tid)
            if cached:
                name, conf, _ = cached
            
            bx1, by1, bx2, by2 = [int(v) for v in bbox]
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(w, bx2), min(h, by2)
            
            color = get_person_color(int(tid) if str(tid).isdigit() else 999)
            if name != "Unknown":
                label = f"{name} ({conf:.2f})"
            else:
                label = f"Person #{tid}"
            
            people_count += 1
            cv2.rectangle(record_frame, (bx1, by1), (bx2, by2), color, 2)
            cv2.putText(record_frame, label, (bx1, max(20, by1 - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            display_tracks.append({
                "id": tid, "bbox": [bx1, by1, bx2, by2],
                "name": name, "confidence": conf
            })

            # Auto-log UNKNOWN persons if not already logged for this track
            if name == "Unknown" and tid not in getattr(process_camera, '_logged_tracks', set()):
                if not hasattr(process_camera, '_logged_tracks'):
                    process_camera._logged_tracks = set()
                process_camera._logged_tracks.add(tid)
                
                # Save crop
                face_crop = extract_face_crop(frame, bbox)
                if face_crop is not None and face_crop.size > 0:
                    # Deduplication check
                    new_encoding = recognizer.get_encoding(face_crop)
                    is_duplicate = False
                    if new_encoding is not None:
                        with saved_encodings_lock:
                            if camera_id not in saved_encodings:
                                saved_encodings[camera_id] = []
                            for old_enc in saved_encodings[camera_id]:
                                dist = np.linalg.norm(new_encoding - old_enc)
                                if dist < 0.7:
                                    is_duplicate = True
                                    break
                            if not is_duplicate:
                                saved_encodings[camera_id].append(new_encoding)
                    
                    if not is_duplicate:
                        ts = int(time.time())
                        full_path = f"snapshots/full_unknown_{camera_id}_{tid}_{ts}.jpg"
                        face_path = f"snapshots/face_unknown_{camera_id}_{tid}_{ts}.jpg"
                        
                        cv2.imwrite(full_path, frame)
                        cv2.imwrite(face_path, face_crop)
                        
                        # Store paths as JSON
                        snapshot_json = json.dumps({"full": full_path, "face": face_path})
                        db_manager.log_detection_event("Unknown", camera_id, snapshot_json, is_known=False)
        
        # Overlay count
        cv2.putText(record_frame, f"Occupancy: {people_count} | Total Seen: {len(seen_track_ids)}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Update occupancy log if changed
        if occupancy_last_count.get(camera_id) != people_count:
            occupancy_last_count[camera_id] = people_count
            db_manager.log_occupancy(camera_id, people_count)

        # Update Shared Results for Streaming
        with results_lock:
            camera_results[camera_id] = {
                "rendered_frame": record_frame, 
                "frame_id": frame_id, 
                "tracks": display_tracks
            }

        # Handle Recording
        with writer_lock:
            writer_data = camera_writers.get(camera_id)
            if writer_data and writer_data.get("writer"):
                writer_data["writer"].write(record_frame)


def recognition_worker(frame, face_bbox, track_id, camera_id, recognition_cache):
    # Only perform recognition if active search is running OR for regular identification
    name, confidence = recognizer.recognize(frame, face_bbox)
    
    if name != "Unknown" and confidence > 0.35:
        # Update cache
        recognition_cache[track_id] = (name, confidence, time.time())
        
        # Log to DB and emit alert if first time for this track
        persons = db_manager.get_registered_persons()
        person_data = next((p for p in persons if p[1].lower() == name.lower()), None)
        
        if person_data:
            person_id = person_data[0]
            # Save face crop
            face_crop = extract_face_crop(frame, [face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3]])
            if face_crop is not None and face_crop.size > 0:
                # Deduplication check
                new_encoding = recognizer.get_encoding(face_crop)
                is_duplicate = False
                if new_encoding is not None:
                    with saved_encodings_lock:
                        if camera_id not in saved_encodings:
                            saved_encodings[camera_id] = []
                        for old_enc in saved_encodings[camera_id]:
                            dist = np.linalg.norm(new_encoding - old_enc)
                            if dist < 0.7:
                                is_duplicate = True
                                break
                        if not is_duplicate:
                            saved_encodings[camera_id].append(new_encoding)
                
                if not is_duplicate:
                    ts = int(time.time())
                    full_path = f"snapshots/full_detected_{camera_id}_{track_id}_{ts}.jpg"
                    face_path = f"snapshots/face_detected_{camera_id}_{track_id}_{ts}.jpg"
                    
                    cv2.imwrite(full_path, frame)
                    cv2.imwrite(face_path, face_crop)
                    
                    snapshot_json = json.dumps({"full": full_path, "face": face_path})
                    db_manager.log_detection(person_id, camera_id, snapshot_json)
                    db_manager.log_detection_event(name, camera_id, snapshot_json, is_known=True)
                    
                    # Fire alert
                    alert_manager.fire(
                        person_name=name,
                        camera_id=camera_id,
                        snapshot_path=snapshot_json,
                        confidence=confidence
                    )
    else:
        # Mark as Unknown in cache to prevent constant re-scanning if desired
        if track_id not in recognition_cache:
            recognition_cache[track_id] = ("Unknown", 0.0, time.time())


# ---------------------------------------------------------------------------
# Auth Routes
# ---------------------------------------------------------------------------

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html")

@app.post("/api/login")
async def api_login(request: Request, response: Response):
    form = await request.form()
    username = form.get("username", "")
    password = form.get("password", "")
    if db_manager.verify_login(username, password):
        session_id = secrets.token_hex(32)
        sessions[session_id] = username
        resp = JSONResponse({"status": "success"})
        resp.set_cookie("session_id", session_id, httponly=True, max_age=86400)
        return resp
    return JSONResponse({"status": "error", "message": "Invalid credentials"}, status_code=401)

@app.post("/api/logout")
async def api_logout(request: Request):
    session_id = request.cookies.get("session_id")
    if session_id:
        sessions.pop(session_id, None)
    resp = JSONResponse({"status": "success"})
    resp.delete_cookie("session_id")
    return resp

@app.post("/api/update_credentials")
async def api_update_credentials(request: Request):
    form = await request.form()
    new_user = form.get("new_username", "").strip()
    new_pass = form.get("new_password", "").strip()
    current_pass = form.get("current_password", "").strip()
    
    session_id = request.cookies.get("session_id")
    current_user = sessions.get(session_id, "")
    
    if not db_manager.verify_login(current_user, current_pass):
        return JSONResponse({"status": "error", "message": "Current password is incorrect"}, status_code=400)
    
    if not new_user or not new_pass:
        return JSONResponse({"status": "error", "message": "Username and password required"}, status_code=400)
    
    db_manager.update_credentials(new_user, new_pass)
    # Update session
    sessions[session_id] = new_user
    return JSONResponse({"status": "success", "message": "Credentials updated"})


# ---------------------------------------------------------------------------
# Page Routes (all serve the SPA shell)
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    try:
        page = int(request.query_params.get("page", "1"))
        per_page = int(request.query_params.get("per_page", "4"))
    except Exception:
        page, per_page = 1, 4
    cameras_all = camera_manager.get_active_cameras()
    total = len(cameras_all)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    end = start + per_page
    cameras = cameras_all[start:end]
    return templates.TemplateResponse(request, "index.html", {"cameras": cameras, "page": page, "total_pages": total_pages, "per_page": per_page, "total": total})

@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request):
    return templates.TemplateResponse(request, "search.html")

@app.get("/recordings_page", response_class=HTMLResponse)
async def recordings_page(request: Request):
    return templates.TemplateResponse(request, "recordings.html")

@app.get("/people", response_class=HTMLResponse)
async def people_page(request: Request):
    return templates.TemplateResponse(request, "people.html")

@app.get("/cameras", response_class=HTMLResponse)
async def cameras_page(request: Request):
    return templates.TemplateResponse(request, "cameras.html")

@app.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request):
    return templates.TemplateResponse(request, "logs.html")


# ---------------------------------------------------------------------------
# Person Registration
# ---------------------------------------------------------------------------

@app.post("/register")
async def register_person(name: str = Form(...), file: UploadFile = File(...)):
    img_dir = f"dataset/{name}"
    os.makedirs(img_dir, exist_ok=True)
    file_path = f"{img_dir}/{file.filename}"
    with open(file_path, "wb") as buf:
        buf.write(await file.read())

    image = cv2.imread(file_path)
    encoding = recognizer.get_encoding(image)
    if encoding is not None:
        db_manager.register_person(name, file_path, encoding.tobytes())
        recognizer.load_known_faces(db_manager)
        return {"status": "success", "message": f"{name} registered."}
    return {"status": "error", "message": "No face detected in the image."}

@app.delete("/api/persons/{person_id}")
async def delete_person(person_id: int):
    if db_manager.delete_person(person_id):
        recognizer.load_known_faces(db_manager)
        return {"status": "success"}
    return {"status": "error", "message": "Person not found"}


# ---------------------------------------------------------------------------
# Camera Management
# ---------------------------------------------------------------------------

@app.post("/add_camera")
async def add_camera(camera_id: str = Form(...), camera_type: str = Form(...), source: str = Form(...)):
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

    if camera_manager.add_camera(camera_id, parsed):
        threading.Thread(target=process_camera, args=(camera_id,), daemon=True).start()
        return {"status": "success"}
    return {"status": "error", "message": "Camera already exists or could not connect."}

@app.post("/delete_camera")
async def delete_camera(camera_id: str = Form(...)):
    if camera_manager.remove_camera(camera_id):
        camera_results.pop(camera_id, None)
        return {"status": "success"}
    return {"status": "error"}

@app.get("/api/cameras")
async def api_cameras():
    return camera_manager.get_all_cameras_info()

@app.post("/api/cameras/toggle")
async def toggle_camera(camera_id: str = Form(...)):
    success, is_active = camera_manager.toggle_camera(camera_id)
    return {"status": "success" if success else "error", "is_active": is_active}

@app.get("/api/cameras_info")
async def api_cameras_info():
    return camera_manager.get_all_cameras_info()

@app.post("/toggle_camera")
async def toggle_camera(camera_id: str = Form(...)):
    success, is_active = camera_manager.toggle_camera(camera_id)
    if success:
        return {"status": "success", "active": is_active}
    return {"status": "error"}

@app.get("/api/occupancy")
async def api_occupancy(camera_id: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None):
    rows = db_manager.search_occupancy(camera_id, start_time, end_time)
    return [{"id": r[0], "camera_id": r[1], "timestamp": r[2], "count": r[3]} for r in rows]

@app.delete("/api/occupancy/{log_id}")
async def delete_occupancy(log_id: int):
    if db_manager.delete_occupancy_log(log_id):
        return {"status": "success"}
    return {"status": "error"}

@app.delete("/api/occupancy_all")
async def delete_occupancy_all(camera_id: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None):
    count = db_manager.delete_occupancy_logs_filtered(camera_id, start_time, end_time)
    return {"status": "success", "deleted": count}


# ---------------------------------------------------------------------------
# ─── Camera Settings API ───────────────────────────────────────────

@app.get("/api/cameras/{camera_id}/roi")
async def get_camera_roi(camera_id: str):
    with rois_lock:
        roi = camera_rois.get(camera_id)
    return {"roi": roi}

@app.post("/api/cameras/{camera_id}/roi")
async def set_camera_roi(camera_id: str, request: Request):
    try:
        data = await request.json()
        roi = data.get("roi") # Expected: [x,y,w,h] or [[x,y], [x,y], ...]
        db_manager.set_camera_roi(camera_id, json.dumps(roi))
        with rois_lock:
            camera_rois[camera_id] = roi
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": f"Malformed ROI: {e}"}

@app.post("/api/cameras/{camera_id}/clear_roi")
async def clear_camera_roi(camera_id: str):
    db_manager.set_camera_roi(camera_id, "null")
    with rois_lock:
        camera_rois[camera_id] = None
    return {"status": "success"}

# ─── Recording API ────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@app.post("/api/toggle_recording")
async def toggle_recording(camera_id: str = Form(...)):
    with writer_lock:
        if camera_id in camera_writers:
            writer_data = camera_writers.pop(camera_id)
            writer_data["writer"].release()
            db_manager.end_recording(writer_data["db_id"])
            return {"status": "success", "recording": False}
        else:
            with results_lock:
                data = camera_results.get(camera_id, {})
                frame = data.get("rendered_frame")
            if frame is None:
                return {"status": "error", "message": "Camera offline or warming up"}
                
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'VP80')
            timestamp = int(time.time())
            filename = f"rec_{camera_id}_{timestamp}.webm"
            file_path = f"recordings/{filename}"
            writer = cv2.VideoWriter(file_path, fourcc, 20.0, (w, h))
            
            db_id = db_manager.start_recording(camera_id, file_path)
            camera_writers[camera_id] = {"writer": writer, "db_id": db_id}
            return {"status": "success", "recording": True}

@app.get("/api/recording_status")
async def get_recording_status():
    with writer_lock:
        return {"active_recordings": list(camera_writers.keys())}


# ---------------------------------------------------------------------------
# Active Search API
# ---------------------------------------------------------------------------

@app.post("/api/start_search")
async def start_search(name: str = Form(...)):
    persons = db_manager.get_registered_persons()
    target = next((p for p in persons if p[1].lower() == name.lower()), None)
    if target is None:
        return {"status": "error", "message": f"'{name}' is not registered."}
    encoding = np.frombuffer(target[3], dtype=np.float32)
    with active_search_lock:
        active_search.clear()
        active_search.update({
            "running": True, "person_id": target[0],
            "name": target[1], "encoding": encoding,
            "found_track_ids": set()
        })
    return {"status": "success", "message": f"Searching for {target[1]}", "name": target[1], "image_path": target[2]}

@app.post("/api/stop_search")
async def stop_search():
    with active_search_lock:
        active_search.clear()
    return {"status": "success"}

@app.get("/api/active_search")
async def get_active_search():
    with active_search_lock:
        name = active_search.get("name")
    return {"active": name is not None, "name": name}


# ---------------------------------------------------------------------------
# History Search API
# ---------------------------------------------------------------------------

@app.get("/api/search")
async def api_search(name: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None):
    results = db_manager.search_detections(name, start_time, end_time)
    res = []
    for r in results:
        path = r[4]
        try:
            paths = json.loads(path)
            full = paths.get("full")
            face = paths.get("face")
        except:
            full = path
            face = path
        res.append({
            "id": r[0], "person_name": r[5] or "Unknown", 
            "camera_id": r[2], "timestamp": r[3], 
            "image_path": full, "face_path": face
        })
    return res

@app.post("/api/search_by_image")
async def search_by_image(file: UploadFile = File(...)):
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    encoding = recognizer.get_encoding(image)
    if encoding is None:
        return []
    best_person_id = None
    min_dist = 1.0
    for p in db_manager.get_registered_persons():
        if p[3] is not None:
            db_enc = np.frombuffer(p[3], dtype=np.float32)
            dist = float(np.linalg.norm(db_enc - encoding))
            if dist < min_dist:
                min_dist = dist
                best_person_id = p[0]
    if best_person_id is None:
        return []
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''SELECT d.*, rp.name FROM detections d
                          LEFT JOIN registered_persons rp ON d.person_id = rp.id
                          WHERE d.person_id = ? ORDER BY d.timestamp DESC''', (best_person_id,))
        results = cursor.fetchall()
    return [{"id": r[0], "person_name": r[5] or "Unknown", "camera_id": r[2], "timestamp": r[3], "image_path": r[4]} for r in results]

@app.post("/clear_history")
async def clear_history():
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM detections")
            conn.commit()
    except Exception as e:
        print(f"DB clear error: {e}")
    snaps_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
    deleted: int = 0
    if os.path.isdir(snaps_dir):
        for fname in os.listdir(snaps_dir):
            try:
                os.remove(os.path.join(snaps_dir, fname))
                deleted = int(deleted) + 1
            except Exception:
                pass
    return {"status": "success", "message": f"Cleared {deleted} records"}

@app.get("/api/recordings")
async def api_recordings(camera_id: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None):
    results = db_manager.search_recordings(camera_id, start_time, end_time)
    return [{"id": r[0], "camera_id": r[1], "start_time": r[2], "end_time": r[3], "file_path": r[4]} for r in results]

@app.delete("/api/recordings/{record_id}")
async def delete_recording(record_id: int):
    rec = db_manager.get_recording(record_id)
    if rec:
        try:
            os.remove(rec[4])
        except Exception:
            pass
        db_manager.delete_recording(record_id)
    return {"status": "success"}


# ---------------------------------------------------------------------------
# Detection Logs API (NEW)
# ---------------------------------------------------------------------------

@app.get("/api/logs")
async def api_logs(limit: int = 200):
    rows = db_manager.get_detection_logs(limit)
    res = []
    for r in rows:
        path = r[4]
        try:
            paths = json.loads(path)
            full = paths.get("full")
            face = paths.get("face")
        except:
            full = path
            face = path
        res.append({
            "id": r[0], "person_name": r[1], "camera_id": r[2],
            "timestamp": r[3], "snapshot_path": full,
            "face_path": face, "is_known": bool(r[5])
        })
    return res

@app.delete("/api/logs")
async def clear_logs():
    db_manager.clear_all_logs()
    return {"status": "success"}


# ---------------------------------------------------------------------------
# Alerts API (simplified)
# ---------------------------------------------------------------------------

@app.get("/api/alerts")
async def api_alerts():
    return alert_manager.get_alerts()

@app.get("/api/alerts/status")
async def api_alert_status():
    return alert_manager.status

@app.post("/api/alerts/read")
async def mark_alerts_read():
    alert_manager.mark_read()
    return {"status": "success"}

@app.get("/api/alerts/unread_count")
async def alerts_unread_count():
    return {"count": alert_manager.get_unread_count()}

# ---------------------------------------------------------------------------
# Person Alerts Config
# ---------------------------------------------------------------------------

@app.get("/api/person_alerts")
async def get_person_alerts():
    rows = db_manager.get_all_person_alerts()
    return [{"id": r[0], "name": r[1], "image_path": r[2], "alert_enabled": bool(r[3])} for r in rows]

@app.post("/api/person_alerts/{person_id}")
async def toggle_person_alert(person_id: int, request: Request):
    data = await request.json()
    enabled = data.get("enabled", False)
    db_manager.set_person_alert(person_id, enabled)
    return {"status": "success"}


# ---------------------------------------------------------------------------
# Persons API
# ---------------------------------------------------------------------------

@app.get("/api/persons")
async def api_persons():
    persons = db_manager.get_registered_persons()
    return [{"id": p[0], "name": p[1], "image_path": p[2]} for p in persons]


# ---------------------------------------------------------------------------
# Video Person Search API
# ---------------------------------------------------------------------------

import json
from fastapi import BackgroundTasks
import torch

video_search_progress: Dict[str, Any] = {}
video_search_lock = threading.Lock()

def scan_video_for_person(video_path: str, target_encoding: np.ndarray, sample_interval: int = 10) -> list:
    results = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return results
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_segment = None
    last_match_frame = -1
    min_segment_gap = int(fps * 2)
    DISTANCE_THRESHOLD = 1.15
    matches_found = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sample_interval == 0:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with recognizer.ai_lock:
                    boxes, probs = recognizer.mtcnn.detect(frame_rgb)
                match_found = False
                best_confidence = 0.0
                if boxes is not None and len(boxes) > 0:
                    for i, box in enumerate(boxes):
                        fx1, fy1, fx2, fy2 = [int(b) for b in box]
                        fx1, fy1 = max(0, fx1), max(0, fy1)
                        fx2, fy2 = min(frame.shape[1], fx2), min(frame.shape[0], fy2)
                        fw, fh = fx2 - fx1, fy2 - fy1
                        if fw < 30 or fh < 30:
                            continue
                        face_crop = frame_rgb[fy1:fy2, fx1:fx2]
                        if face_crop.size > 0:
                            face_resized = cv2.resize(face_crop, (160, 160))
                            face_tensor = torch.tensor(np.transpose(face_resized, (2, 0, 1))).float().unsqueeze(0).to(recognizer.device)
                            face_tensor = (face_tensor - 127.5) / 128.0
                            with recognizer.ai_lock:
                                with torch.no_grad():
                                    embedding = recognizer.resnet(face_tensor).cpu().numpy()[0]
                            distance = float(np.linalg.norm(target_encoding - embedding))
                            confidence = 1 - (distance / 2.0)
                            if distance < DISTANCE_THRESHOLD:
                                match_found = True
                                matches_found += 1
                                if confidence > best_confidence:
                                    best_confidence = confidence
                if match_found:
                    timestamp_sec = frame_count / fps
                    if current_segment is None or (frame_count - last_match_frame) > min_segment_gap:
                        if current_segment is not None:
                            results.append(current_segment)
                        current_segment = {
                            "start_seconds": timestamp_sec,
                            "start_timestamp": f"{int(timestamp_sec // 60)}:{int(timestamp_sec % 60):02d}",
                            "end_seconds": timestamp_sec,
                            "end_timestamp": f"{int(timestamp_sec // 60)}:{int(timestamp_sec % 60):02d}",
                            "confidence": best_confidence,
                            "start_frame": frame_count, "end_frame": frame_count
                        }
                    else:
                        current_segment["end_seconds"] = timestamp_sec
                        current_segment["end_timestamp"] = f"{int(timestamp_sec // 60)}:{int(timestamp_sec % 60):02d}"
                        current_segment["end_frame"] = frame_count
                        if best_confidence > current_segment["confidence"]:
                            current_segment["confidence"] = best_confidence
                    last_match_frame = frame_count
            except Exception as e:
                pass
        frame_count += 1
    if current_segment is not None:
        results.append(current_segment)
    cap.release()
    return results


@app.post("/api/search_video_by_name")
async def search_video_by_name(request: Request):
    data = await request.json()
    name = data.get("name")
    video_ids = data.get("video_ids", [])
    if not name or not video_ids:
        return {"status": "error", "message": "Name and video IDs required"}
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
                all_results.append({**segment, "video_id": vid_id, "video_name": os.path.basename(rec[4]),
                    "video_path": rec[4], "camera_id": rec[1], "person_name": name})
    all_results.sort(key=lambda x: x["start_seconds"])
    return {"status": "success", "results": all_results, "total_segments": len(all_results), "videos_searched": len(video_ids)}


@app.post("/api/search_video_by_image")
async def search_video_by_image(file: UploadFile = File(...), video_ids: str = Form(...)):
    video_ids_list = json.loads(video_ids)
    if not video_ids_list:
        return {"status": "error", "message": "Video IDs required"}
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    target_encoding = recognizer.get_encoding(image)
    if target_encoding is None:
        return {"status": "error", "message": "No face detected in uploaded image"}
    all_results = []
    for vid_id in video_ids_list:
        rec = db_manager.get_recording(vid_id)
        if rec and os.path.exists(rec[4]):
            segments = scan_video_for_person(rec[4], target_encoding)
            for segment in segments:
                all_results.append({**segment, "video_id": vid_id, "video_name": os.path.basename(rec[4]),
                    "video_path": rec[4], "camera_id": rec[1], "person_name": "Unknown (from image)"})
    all_results.sort(key=lambda x: x["start_seconds"])
    return {"status": "success", "results": all_results, "total_segments": len(all_results), "videos_searched": len(video_ids_list)}


# ---------------------------------------------------------------------------
# Video streaming
# ---------------------------------------------------------------------------

def gen_frames(camera_id: str):
    last_sent_id = -1
    while True:
        with results_lock:
            data = camera_results.get(camera_id, {})
            frame = data.get("rendered_frame")
            frame_id = data.get("frame_id", -1)
        if frame is None or frame_id == last_sent_id:
            time.sleep(0.01)
            continue
        last_sent_id = frame_id
        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"

@app.get("/video_feed/{camera_id}")
async def video_feed(camera_id: str):
    return StreamingResponse(gen_frames(camera_id), media_type="multipart/x-mixed-replace; boundary=frame")


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    return templates.TemplateResponse(request, "analytics.html")

@app.get("/api/analytics/summary")
async def api_analytics_summary(camera_id: Optional[str] = None, hours: int = 24):
    return db_manager.analytics_summary(camera_id, hours)

@app.get("/api/analytics/occupancy_trend")
async def api_analytics_trend(camera_id: Optional[str] = None, hours: int = 24):
    return db_manager.analytics_occupancy_trend(camera_id, hours)

@app.get("/api/analytics/heatmap")
async def api_analytics_heatmap(camera_id: Optional[str] = None):
    return db_manager.analytics_heatmap(camera_id)

@app.get("/api/analytics/top_persons")
async def api_analytics_top(camera_id: Optional[str] = None, hours: int = 24, limit: int = 10):
    return db_manager.analytics_top_persons(camera_id, hours, limit)

@app.get("/api/analytics/per_camera")
async def api_analytics_per_camera(hours: int = 24):
    return db_manager.analytics_per_camera(hours)

@app.get("/api/analytics/identity_breakdown")
async def api_analytics_identity(camera_id: Optional[str] = None, hours: int = 24):
    return db_manager.analytics_identity_breakdown(camera_id, hours)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)