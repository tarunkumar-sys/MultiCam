# AI Vigilance — Smart Multi-Camera Surveillance System

A production-ready, real-time surveillance platform built for multi-camera environments. Tracks, identifies, and logs individuals across live feeds and recorded video using a fully local AI stack — no cloud dependency.

---

## Features

**Live Surveillance**
- Multi-camera live MJPEG streaming with per-camera bounding box overlays
- Live head count HUD on every feed (current persons in frame)
- 24-hour unique person count per camera (updates every 30 seconds)
- Pause/resume any feed with a frozen frame snapshot
- Fullscreen mode with live count and pause controls

**Person Detection & Tracking**
- YOLOv8n person detection at 2 FPS per camera
- Custom IoU + center-distance tracker with 4-second occlusion tolerance
- Non-maximum suppression to eliminate duplicate boxes
- Per-person colour-coded bounding boxes with ID labels

**Face Recognition**
- MTCNN detects the precise face region inside each person bounding box
- InceptionResnetV1 (FaceNet, vggface2) generates 512-d embeddings
- Cosine similarity matching — robust to lighting and pose variation
- Multi-photo averaging: register multiple photos per person for higher accuracy
- Threshold: cosine similarity ≥ 0.65 (tuned for vggface2)
- Recognised persons shown in green with name label on live feed

**Global Re-ID & Journey Tracking**
- Cross-camera person re-identification using face embeddings
- Unknown persons assigned unique IDs (U-XXXX) and tracked across cameras
- Journey timeline: chronological log of every camera a person appeared on
- Daily unique person count derived from journey events

**Video Recording**
- Toggle recording per camera from the Cameras page
- FFmpeg H.264 encoding at 2 FPS (matches processing rate), CRF 28
- Auto-split recordings every 2.5 hours
- Recordings stored locally in `/recordings/`

**Video Person Search**
- Search saved recordings for a specific person by name or uploaded photo
- Upload any external video and search it for a registered person
- Results shown as timeline segments with start/end timestamps and confidence
- Click any result to jump directly to that moment in the video player

**Detection Logs**
- Paginated snapshot history (20/50/100 per page)
- Filter by camera
- View snapshot image with bounding box data

**Analytics**
- Hourly and daily person count charts
- Per-camera AM/PM breakdown

**People Registry**
- Register persons with a face photo
- View last-seen time and camera per person
- Delete persons and their dataset files

**Identity Logs**
- Full history of every time a registered person was detected
- Filter by person name

---

## AI Stack

| Component | Model | Purpose |
|---|---|---|
| Person Detection | YOLOv8n (COCO class 0) | Bounding boxes on persons in frame |
| Face Detection | MTCNN | Tight face crop inside person box |
| Face Embedding | InceptionResnetV1 (vggface2) | 512-d biometric vector |
| Tracking | Custom IoU + distance tracker | Persistent IDs across frames |
| Matching | Cosine similarity ≥ 0.65 | Identity verification |

---

## Backend Stack

- **FastAPI + Uvicorn** — async Python backend, MJPEG streaming, SSE notifications
- **OpenCV** — RTSP/webcam capture with UDP low-latency flags
- **FFmpeg** — H.264 video recording from raw frames
- **SQLite3** — local database for all detections, persons, recordings, journeys
- **PyTorch** — FaceNet inference, GPU-accelerated when CUDA available
- **Jinja2** — server-side HTML templates

---

## Hardware Requirements

| Setup | Recommended GPU | Cameras |
|---|---|---|
| Development / testing | CPU only (i7/i9) | 1–2 at 2 FPS |
| Small deployment | RTX 3060 12GB | 2–3 at 15–20 FPS |
| Standard deployment | RTX 3070 / 4070 | 3–5 at 30 FPS |
| Large deployment | RTX 3090 / 4080 | 6+ at 30 FPS |

> The app runs at **2 FPS** by default to balance accuracy and CPU load. To increase FPS, reduce `FRAME_INTERVAL` in `app.py` and ensure a CUDA-capable GPU is available.

---

## Setup

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

### Linux / Headless VM

```bash
sudo apt-get install -y libgl1 libglib2.0-0 ffmpeg

python3 -m venv .venv && source .venv/bin/activate
# CPU-only PyTorch (skip ~2GB CUDA download)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python app.py
```

### Access

```
http://localhost:8000
http://<server-ip>:8000   (from another machine on the network)
```

Default credentials: `admin` / `deiadmin@789`

---

## Supported Camera Sources

| Type | Example Source |
|---|---|
| Local webcam | `0` |
| RTSP IP camera | `rtsp://user:pass@192.168.1.100:554/stream` |
| IP Webcam app (Android) | `192.168.1.100:8080` |
| DroidCam | `192.168.1.100` |
| MJPEG stream | `http://192.168.1.100/video.mjpg` |

---

## Repository Structure

```
app.py                  Main FastAPI app, camera processing threads, all API routes
cameras/
  camera_manager.py     RTSP/webcam capture with auto-reconnect
database/
  sqlite_manager.py     All DB tables, queries, and migrations
utils/
  detector.py           YOLOv8 person detection
  recognizer.py         MTCNN + FaceNet face recognition pipeline
  tracker.py            Custom IoU/distance multi-object tracker
templates/              Jinja2 HTML pages
static/                 CSS and JS
dataset/                Registered person face photos
snapshots/              Detection snapshot images (auto-cleaned after 24h)
recordings/             MP4 video recordings (auto-cleaned after 2 days)
```

---

## API Reference (key endpoints)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/cameras` | List active cameras |
| POST | `/api/add_camera` | Add a new camera |
| DELETE | `/api/remove_camera/{id}` | Remove a camera |
| GET | `/api/occupancy` | Live person counts per camera |
| GET | `/api/camera_daily_stats` | 24h unique person count per camera |
| POST | `/api/register_person` | Register a person with face photo |
| DELETE | `/api/delete_person/{id}` | Delete a registered person |
| GET | `/api/detection_snapshots` | Paginated snapshot history |
| GET | `/api/recordings` | List video recordings |
| POST | `/api/toggle_recording` | Start/stop recording for a camera |
| POST | `/api/search_video_by_name` | Search recordings for a person by name |
| POST | `/api/search_video_by_image` | Search recordings using a face photo |
| POST | `/api/upload_video_and_search` | Upload video and search for a person |
| GET | `/api/target_journey/{id}` | Journey timeline for a global ID |
| GET | `/api/notifications/stream` | SSE stream for real-time alerts |
