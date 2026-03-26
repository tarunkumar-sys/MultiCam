# AI VIGILANCE — Intelligent Multi-Camera Surveillance System

> Real-time AI-powered surveillance with face recognition, person tracking, and active search across multiple camera feeds.

---

## What Is AI Vigilance?

AI Vigilance is a self-hosted, web-based surveillance platform that combines computer vision and deep learning to monitor multiple camera streams simultaneously. It detects and tracks people in real time, recognizes registered faces, and lets operators launch an **Active Search Mission** — instantly scanning all live feeds to locate a specific person and logging every sighting with a timestamped snapshot.

Built for security teams, research labs, and smart building operators who need more than passive recording.

---

## Key Features

- **Multi-camera live dashboard** — webcams, RTSP IP cameras, DroidCam, IP Webcam (Android)
- **Real-time person detection** — YOLOv8n for fast, accurate body detection
- **Multi-person tracking** — DeepSORT keeps consistent IDs across frames
- **Face recognition** — FaceNet (InceptionResnetV1 + MTCNN) for identity matching
- **Active Search Mission** — launch a live hunt for any registered person across all cameras
- **Detection history** — searchable log with snapshot images, camera ID, and timestamps
- **Search by photo** — upload an image to find matching detections in history
- **Video recordings** — browse and manage saved MP4 recordings per camera
- **One-click person registration** — upload a photo to enroll a new face in seconds

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   FastAPI Backend                    │
│                                                     │
│  CameraManager ──► CameraHandler threads            │
│       │                  │ raw frames               │
│       ▼                  ▼                          │
│  process_camera()  ◄── frame queue                  │
│       │                                             │
│  PersonDetector (YOLOv8n)                           │
│       │ bounding boxes                              │
│  ObjectTracker (DeepSORT)                           │
│       │ track IDs                                   │
│  FaceRecognizer (MTCNN + FaceNet)                   │
│       │ identity + confidence                       │
│  DatabaseManager (SQLite)                           │
│       │                                             │
│  StreamingResponse ──► MJPEG to browser             │
└─────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Library |
|---|---|
| Web framework | FastAPI + Uvicorn |
| Person detection | Ultralytics YOLOv8n |
| Multi-object tracking | DeepSORT Realtime |
| Face detection | MTCNN (facenet-pytorch) |
| Face recognition | InceptionResnetV1 / FaceNet (facenet-pytorch) |
| Video capture | OpenCV |
| Database | SQLite |
| Frontend | Vanilla JS + Jinja2 templates |

---

## Requirements

- Python 3.10 or 3.11
- A CUDA-capable GPU is recommended but CPU-only works
- Cameras: USB webcam, RTSP stream, Android phone (DroidCam or IP Webcam app)

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/AI-VIGILANCE.git
cd AI-VIGILANCE

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt
```

The YOLOv8n model (`yolov8n.pt`) and FaceNet weights download automatically on first run.

---

## Running the App

```bash
python app.py
```

Then open your browser at: **http://localhost:8000**

> The app starts with a local webcam (index 0) and an optional RTSP camera pre-configured in `app.py`. Edit the `__main__` block to change defaults, or add cameras live from the dashboard.

---

## Adding Cameras

From the dashboard sidebar, select a camera type and fill in the source:

| Type | Source format |
|---|---|
| Local Webcam | `0` (or `1`, `2` for additional webcams) |
| RTSP | `rtsp://user:password@192.168.1.100:554` |
| IP Webcam (Android) | `192.168.1.100` (port 8080 auto-appended) |
| DroidCam | `192.168.1.100` (port 4747 auto-appended) |

> For RTSP passwords containing special characters like `@`, the app automatically percent-encodes them.

---

## Registering a Person

1. Go to the **Dashboard**
2. In the sidebar under **Register Person**, enter the person's full name
3. Upload a clear, front-facing photo
4. Click **Register**

The face encoding is extracted and stored in the database immediately.

---

## Active Search Mission

1. Go to **Search & History**
2. Type the registered person's name
3. Click **Start Search**

The system will scan every active camera feed in real time. When the person is detected, a snapshot is saved and logged. The mission runs until you click **Stop Search**.

---

## Project Structure

```
AI-VIGILANCE/
├── app.py                  # FastAPI app, routes, camera processing loop
├── cameras/
│   └── camera_manager.py   # Camera capture threads
├── database/
│   └── db_manager.py       # SQLite operations
├── utils/
│   ├── detector.py         # YOLOv8 person detection
│   ├── tracker.py          # DeepSORT tracking
│   └── recognizer.py       # MTCNN + FaceNet recognition
├── templates/              # Jinja2 HTML templates
├── static/                 # CSS + JS
├── snapshots/              # Auto-saved detection images
├── dataset/                # Registered person photos
├── recordings/             # Saved video files
└── requirements.txt
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Live dashboard |
| GET | `/search` | Search & history page |
| GET | `/recordings_page` | Video recordings browser |
| POST | `/register` | Register a new person |
| POST | `/add_camera` | Add a camera |
| POST | `/delete_camera` | Remove a camera |
| GET | `/video_feed/{camera_id}` | MJPEG live stream |
| POST | `/api/start_search` | Start active search mission |
| POST | `/api/stop_search` | Stop active search |
| GET | `/api/active_search` | Current search target |
| GET | `/api/search` | Query detection history |
| POST | `/api/search_by_image` | Find person by uploaded photo |
| GET | `/api/recordings` | List video recordings |
| DELETE | `/api/recordings/{id}` | Delete a recording |
| POST | `/clear_history` | Wipe all detections + snapshots |

---

## Troubleshooting

**SSL certificate error on first run**
The app patches SSL verification automatically for the FaceNet model download. If it still fails, run:
```bash
pip install --upgrade certifi
```

**RTSP stream not connecting**
- Confirm the camera IP and credentials are correct
- Try the URL directly in VLC first
- Ensure UDP port 554 is not blocked by firewall

**Low FPS / high CPU**
- Detection runs every 5 frames, face recognition every 10 — adjust `frame_count % N` in `app.py`
- Use a GPU: install the CUDA version of PyTorch before installing other deps

**Face not recognized**
- Register with a well-lit, front-facing photo
- The recognition threshold is `distance < 1.0` — lower it (e.g. `0.8`) for stricter matching in `recognizer.py`

---

## License

MIT License — free to use, modify, and distribute.
