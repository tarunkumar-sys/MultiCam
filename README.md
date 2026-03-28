<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Orbitron&size=32&duration=3000&pause=1000&color=00FFCC&center=true&vCenter=true&width=600&lines=MultiCam+AI+Surveillance;Real-Time+Face+Recognition;DBSCAN+Unknown+Clustering;Multi-Camera+Live+Dashboard" alt="MultiCam" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=for-the-badge&logo=yolo&logoColor=white)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-00FFCC?style=for-the-badge)](LICENSE)

<br/>

> **MultiCam** is a self-hosted, real-time AI surveillance platform.  
> It detects, tracks, and recognises people across multiple camera feeds simultaneously —  
> and clusters unrecognised faces automatically so no one slips through unnoticed.

<br/>

</div>

---

## ✨ Features

| | Feature | Description |
|---|---|---|
| 🎥 | **Multi-Camera Dashboard** | Webcams, RTSP, DroidCam, IP Webcam — all live |
| 🧠 | **YOLOv8 Person Detection** | Fast, accurate body detection at 60 FPS |
| 🔁 | **DeepSORT Tracking** | Persistent IDs across frames and cameras |
| 👤 | **FaceNet Recognition** | InceptionResnetV1 + MTCNN identity matching |
| 🔍 | **Active Search Mission** | Hunt for a registered person across all live feeds |
| 🧩 | **DBSCAN Auto-Clustering** | Unknown faces grouped by appearance — no one is just "Unknown" |
| 📋 | **Register After the Fact** | Name and enrol clustered unknowns directly from the UI |
| 🗂️ | **Detection History** | Searchable log with snapshots, camera ID, timestamps |
| 📷 | **Search by Photo** | Upload any image to find matching detections |
| 🎬 | **Video Recordings** | Browse and manage saved MP4s per camera |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     FastAPI Backend                      │
│                                                          │
│  CameraManager ──► CameraHandler threads (non-blocking)  │
│        │                    │ raw frames                 │
│        ▼                    ▼                            │
│   process_camera()  ◄── frame queue                      │
│        │                                                 │
│   PersonDetector  (YOLOv8n)      every 5 frames          │
│        │                                                 │
│   ObjectTracker   (DeepSORT)     persistent IDs          │
│        │                                                 │
│   FaceRecognizer  (MTCNN + FaceNet)                      │
│        ├── known face → name + confidence                │
│        └── unknown   → UnknownFaceClusterer (DBSCAN)     │
│                              │                           │
│   DatabaseManager (SQLite)   │ sightings persisted       │
│        │                                                 │
│   StreamingResponse ──► MJPEG → browser                  │
└──────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | FastAPI + Uvicorn |
| Person Detection | Ultralytics YOLOv8n |
| Multi-Object Tracking | DeepSORT Realtime |
| Face Detection | MTCNN (facenet-pytorch) |
| Face Recognition | InceptionResnetV1 / FaceNet |
| Unknown Clustering | DBSCAN (scikit-learn) |
| Video Capture | OpenCV |
| Database | SQLite |
| Frontend | Vanilla JS + Jinja2 |

---

## ⚡ Quick Start

### 1 — Clone the repo

```bash
git clone https://github.com/tarunkumar-sys/MultiCam.git
cd MultiCam
```

### 2 — Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> 💡 For GPU acceleration install the CUDA build of PyTorch **before** running the above:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 4 — Run the app

```bash
python app.py
```
user: test
pasword: dei@12@12
ip:10.7.16.48
port: 554

Open your browser at **http://localhost:8000**

> Model weights (YOLOv8n + FaceNet) download automatically on first run.

---

## 📷 Adding Cameras

From the dashboard sidebar → **Add Camera**:

| Type | Source format |
|---|---|
| Local Webcam | `0` &nbsp;(or `1`, `2` for additional webcams) |
| RTSP | `rtsp://user:password@192.168.1.100:554` |
| IP Webcam (Android) | `192.168.1.100` &nbsp;(port 8080 auto-appended) |
| DroidCam | `192.168.1.100` &nbsp;(port 4747 auto-appended) |

> Passwords containing special characters like `@` are automatically percent-encoded.

---

## 👤 Registering a Person

1. Go to the **Dashboard**
2. In the sidebar under **Register Person**, enter the full name
3. Upload a clear, front-facing photo
4. Click **Register**

The face encoding is extracted and stored immediately.

---

## 🔴 Active Search Mission

1. Go to **Search & History**
2. Type the registered person's name
3. Click **Start Search**

MultiCam scans every active camera in real time. When the person is spotted, a snapshot is saved and logged. Click **Stop Search** to end the mission.

---

## 🧩 Unknown Person Clustering

Faces that don't match any registered person are **not** discarded as "Unknown".  
Instead, MultiCam:

1. Extracts a 512-d FaceNet embedding for each unrecognised face
2. Assigns it to the nearest DBSCAN cluster (or creates a new one)
3. Labels clusters **Unknown Person A**, **B**, **C** …
4. Logs every sighting with camera ID and timestamp
5. Re-runs DBSCAN every 60 seconds to merge drifted clusters

On the **Search & History** page, scroll to **Unknown Persons (Auto-Clustered)** to see:

```
👤 Unknown Person A
   📍 Cam1  🕒 09:12:34
   📍 Cam3  🕒 09:45:01
   [ Enter name ]  [ Register ]
```

Click **Register** to promote the cluster directly into the known persons database — no re-upload needed.

---

## 🗂️ Project Structure

```
MultiCam/
├── app.py                  # FastAPI app, routes, camera processing loop
├── cameras/
│   └── camera_manager.py   # Non-blocking camera capture threads
├── database/
│   └── db_manager.py       # SQLite — persons, detections, clusters, recordings
├── utils/
│   ├── detector.py         # YOLOv8 person detection
│   ├── tracker.py          # DeepSORT tracking
│   ├── recognizer.py       # MTCNN + FaceNet recognition
│   └── clustering.py       # DBSCAN unknown face clustering
├── templates/              # Jinja2 HTML templates
├── static/                 # CSS + JS
├── requirements.txt
└── .gitignore
```

---

## 🌐 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Live dashboard |
| `GET` | `/search` | Search & history page |
| `GET` | `/recordings_page` | Video recordings browser |
| `POST` | `/register` | Register a new person |
| `POST` | `/add_camera` | Add a camera |
| `POST` | `/delete_camera` | Remove a camera |
| `GET` | `/video_feed/{camera_id}` | MJPEG live stream |
| `POST` | `/api/start_search` | Start active search mission |
| `POST` | `/api/stop_search` | Stop active search |
| `GET` | `/api/active_search` | Current search target |
| `GET` | `/api/search` | Query detection history |
| `POST` | `/api/search_by_image` | Find person by uploaded photo |
| `GET` | `/api/unknown_clusters` | List DBSCAN clusters |
| `POST` | `/api/register_unknown` | Promote cluster to registered person |
| `DELETE` | `/api/unknown_clusters/{id}` | Dismiss a cluster |
| `GET` | `/api/recordings` | List video recordings |
| `DELETE` | `/api/recordings/{id}` | Delete a recording |
| `POST` | `/clear_history` | Wipe all detections + snapshots |

---

## 🔧 Troubleshooting

**RTSP stream not connecting**
- Verify the URL in VLC first
- Ensure UDP port 554 is not blocked by your firewall
- Cameras are added non-blocking — the server starts immediately even if the stream is unreachable

**Face not recognised**
- Register with a well-lit, front-facing photo
- Default threshold is `distance < 1.0` — lower to `0.8` in `utils/recognizer.py` for stricter matching

**Low FPS / high CPU**
- Detection runs every 5 frames, face recognition every 10 — tune `frame_count % N` in `app.py`
- Use a GPU: install the CUDA PyTorch build before other deps

**scikit-learn not found**
```bash
pip install scikit-learn
```

---

## 📄 License

MIT — free to use, modify, and distribute.

---

<div align="center">

Made with 🧠 + ☕ &nbsp;|&nbsp; <a href="https://github.com/tarunkumar-sys/MultiCam">github.com/tarunkumar-sys/MultiCam</a>

</div>
