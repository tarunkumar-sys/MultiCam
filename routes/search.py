from fastapi import APIRouter, Request, Query, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional
import logging
import os
import json
import cv2
import numpy as np
from datetime import datetime

router = APIRouter(prefix="/api", tags=["Search"])
logger = logging.getLogger(__name__)

@router.get("/search")
async def search_detections(
    request: Request,
    name: Optional[str] = Query(None),
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None)
):
    """Database forensic search for detections."""
    db = request.app.state.db_manager
    
    # Parse times if provided
    start_dt = datetime.fromisoformat(start_time) if start_time else None
    end_dt = datetime.fromisoformat(end_time) if end_time else None
    
    # Note: Using search_snapshots logic or registered_detections depending on name
    # For now, let's use a combined approach or fallback to snapshots
    try:
        # This is a simplified search for the forensic dashboard
        with db._get_connection() as conn:
            query = "SELECT id, person_name, camera_id, timestamp, '' as image_path FROM registered_detections WHERE 1=1"
            params = []
            if name:
                query += " AND person_name LIKE ?"
                params.append(f"%{name}%")
            if start_dt:
                query += " AND timestamp >= ?"
                params.append(start_dt.isoformat())
            if end_dt:
                query += " AND timestamp <= ?"
                params.append(end_dt.isoformat())
            
            query += " ORDER BY timestamp DESC LIMIT 100"
            rows = conn.execute(query, params).fetchall()
            
            return [{
                "id": str(r["id"]),
                "person_name": r["person_name"],
                "camera_id": r["camera_id"],
                "timestamp": r["timestamp"],
                "image_path": r["image_path"]
            } for r in rows]
    except Exception as e:
        logger.error(f"Search Error: {e}")
        return []

@router.get("/snapshot_image")
async def get_snapshot_image(path: str):
    """Serves a snapshot image from disk."""
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Image not found")

@router.post("/clear_history")
async def clear_history(request: Request):
    """Wipes all forensic logs."""
    db = request.app.state.db_manager
    db.delete_all_detections()
    return {"status": "success"}

@router.post("/search_by_image")
async def search_by_image(request: Request, file: UploadFile = File(...)):
    """Find person occurrences by visual similarity."""
    db = request.app.state.db_manager
    recognizer = request.app.state.recognizer
    
    try:
        # 1. Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
            
        # 2. Extract encoding
        encoding = recognizer.get_encoding(img)
        if encoding is None:
            return [] # No face found
            
            # 3. Search DB
            results = db.search_snapshots_by_similarity(encoding)
            
            # Map results to frontend format
            return [{
                "id": r["_id"],
                "person_name": "Match Found",
                "camera_id": r["camera_id"],
                "timestamp": r["timestamp"].isoformat() if hasattr(r["timestamp"], 'isoformat') else r["timestamp"],
                "image_path": r["snapshot_path"]
            } for r in results]
        else:
            return []
    except Exception as e:
        logger.error(f"Image Search Error: {e}")
        return []

@router.post("/search_video_by_name")
async def search_video_by_name(request: Request, data: dict):
    """Stub for deep video scan by name."""
    return {"total_segments": 0, "results": []}

@router.post("/search_video_by_image")
async def search_video_by_image(request: Request, file: UploadFile = File(...), video_ids: str = Form(...)):
    """Stub for deep video scan by image."""
    return {"total_segments": 0, "results": []}

@router.get("/snapshots")
async def get_snapshots(
    request: Request,
    camera_id: str = None,
    limit: int = 20,
    skip: int = 0
):
    db = request.app.state.db_manager
    snapshots = db.get_snapshots(camera_id=camera_id, limit=limit, skip=skip)
    total = db.count_detection_snapshots(camera_id=camera_id)
    
    # snapshots returns [id, cam, ts, count, path, bbox, crops]
    items = [{
        "id": s[0],
        "camera_id": s[1],
        "timestamp": s[2].isoformat() if hasattr(s[2], "isoformat") else str(s[2]),
        "person_count": s[3],
        "snapshot_path": s[4],
        "bbox_data": s[5],
        "person_crops": s[6] if len(s) > 6 else []
    } for s in snapshots]
    
    return {"items": items, "total": total}
