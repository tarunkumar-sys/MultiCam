# 🛡️ VigiLance: Industrial AI Surveillance System

![Build Status](https://img.shields.io/badge/Build-Stable-success?style=for-the-badge&logo=github)
![Python Version](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-v0.95+-009688?style=for-the-badge&logo=fastapi)
![AI Engine](https://img.shields.io/badge/AI-YOLOv8--OpenVINO-red?style=for-the-badge&logo=pypy)
![Hardware](https://img.shields.io/badge/Hardware-Agnostic-orange?style=for-the-badge)

VigiLance is a high-performance, modular, and industrial-grade surveillance platform. It leverages state-of-the-art AI for real-time person detection, multi-node tracking, and distributed stream management. Designed for reliability and scale, it decouples media ingestion from AI processing.

---

## 🏗️ System Architecture

VigiLance follows a **Distributed Microservice Pattern** to ensure high availability and low-latency processing:

### 1. 🎞️ Media Broker (`media/`)
A standalone high-performance streaming server that ingests raw RTSP/HTTP feeds and redistributes them as optimized proxies. This decouples the AI system from network-level camera instability.

### 2. 🧠 AI Engine (`core/`)
The analytical core of the system.
- **Dynamic Hardware Acceleration**: Automatically detects and utilizes **NVIDIA CUDA**, **Intel OpenVINO**, or **ONNX Runtime** for maximum FPS on any hardware.
- **Hungarian Tracking**: Global ID assignment with occlusion tolerance and depth sorting.

### 3. 🌐 Application Server (`app.py`)
The central orchestrator managing the database, business logic, and the web-based dashboard.

---

## 📁 Project Structure

```text
d:\VigiLance\
├── app.py              # Central Orchestrator & Bootstrapper
├── core/               # AI & Analytical DNA (Detector, Tracker, HW Manager)
├── gate/               # Input/Output Layer (Camera Management, Stream Client)
├── services/           # Business Logic (Vision, Record, Notify, Search)
├── routes/             # Modular API Endpoints (Node, Live, Archive)
├── media/              # Dedicated Streaming Microservice
├── config/             # Centralized Settings & IST Timezone
├── database/           # SQLite3 Local Forensics Management
└── models/             # Pydantic Schemas & Data Validation
```

---

## 🚀 Getting Started

### 📦 Prerequisites
- Python 3.10+
- FFmpeg (for video recording)
- (Optional) Docker & Docker Compose

### 🔧 Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/your-org/vigilance.git
cd vigilance
```

#### 2. Create and Activate Virtual Environment
**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux / MacOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🏃 Running the System

### Option A: Industrial Mode (Recommended)
VigiLance is designed to run as a distributed system using Docker Compose. This starts both the AI Engine and the Media Broker automatically.

```bash
docker-compose up --build
```

### Option B: Developer Mode (Single Command)
If running locally without Docker, simply start the main application. It will automatically detect and boot the Media Broker service in the background:

```bash
python app.py
```

### 🌍 Access & Demo
Once the system is initialized, the dashboard can be accessed via:
- **Local Access**: `http://localhost:8000`
- **Network Access**: `http://<your-ip>:8000`

*Note: Default credentials (if login is enabled) are managed by the System Administrator.*

---

## 🌟 Key Industrial Features

- **Occlusion-Aware Rendering**: Advanced depth-sorting to maintain visual clarity in crowded regions.
- **Smart Forensics**: Search through history by person name or appearance timestamps.
- **Adaptive Recording**: 10 FPS libx264 recording with automatic disk cleanup.
- **Real-time HUD**: High-accuracy occupancy counters and global identity tracking.
- **Zero Cloud Dependency**: 100% local processing for maximum privacy and security.

---

## 🛡️ License
Distributed under the MIT License. See `LICENSE` for more information.
