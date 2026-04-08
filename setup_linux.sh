#!/bin/bash
# Setup script for AI Vigilance on Linux VM (headless)

set -e

echo "[1/6] Installing system dependencies (SQLite3)..."
sudo apt-get update -qq
sudo apt-get install -y \
    python3 python3-pip python3-venv \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg libavcodec-dev libavformat-dev libswscale-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    sqlite3

echo "[2/6] Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "[3/6] Upgrading pip..."
pip install --upgrade pip

echo "[4/6] Installing Python dependencies..."
# Install CPU-only torch first to avoid pulling CUDA (~2GB)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

echo "[5/6] Creating workspace directories..."
mkdir -p snapshots dataset recordings database

echo "[6/6] Done. Run the app with:"
echo "  source .venv/bin/activate && python app.py"
