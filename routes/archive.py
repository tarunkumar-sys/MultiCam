from fastapi import APIRouter, Request, HTTPException, Form
from fastapi.responses import FileResponse
from typing import Optional
import os
import logging

router = APIRouter(prefix="/api/recordings", tags=["Recordings"])
logger = logging.getLogger(__name__)

@router.get("")
async def list_recordings(request: Request, camera_id: Optional[str] = None):
    db_manager = request.app.state.db_manager
    results = db_manager.search_recordings(camera_id)
    return [{
        "id": r[0], 
        "camera_id": r[1], 
        "start_time": r[2], 
        "end_time": r[3], 
        "file_path": r[4]
    } for r in results]

@router.post("/toggle/{camera_id}")
async def toggle_recording(camera_id: str, request: Request):
    recording_service = request.app.state.recording_service
    vision_service = request.app.state.vision_service
    
    # Check if currently recording
    if camera_id in recording_service.camera_writers:
        recording_service.stop_recording(camera_id)
        return {"status": "success", "recording": False}
    else:
        # Get dimensions from vision service
        from services.vision import camera_results, results_lock
        with results_lock:
            data = camera_results.get(camera_id, {})
            frame = data.get("rendered_frame")
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Camera not streaming")
            
        h, w = frame.shape[:2]
        if recording_service.start_recording(camera_id, w, h):
            return {"status": "success", "recording": True}
        raise HTTPException(status_code=500, detail="Failed to start recording")

@router.get("/video")
async def get_video(path: str):
    if os.path.exists(path):
        return FileResponse(path, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Video file not found")

@router.delete("/{record_id}")
async def delete_recording(record_id: str, request: Request):
    db_manager = request.app.state.db_manager
    rec = db_manager.get_recording(record_id)
    if rec:
        if os.path.exists(rec[4]):
            os.remove(rec[4])
        db_manager.delete_recording(record_id)
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Recording not found")
