from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from models.schemas import CameraCreate
# CameraManager is accessed via request.app.state
import logging

router = APIRouter(prefix="/api/cameras", tags=["Cameras"])
logger = logging.getLogger(__name__)

# Note: Dependency injection for camera_manager would be better in main.py
# For now we'll assume it's passed or accessible.

@router.post("")
async def add_camera(camera: CameraCreate, request: Request):
    camera_manager = request.app.state.camera_manager
    stream_client = request.app.state.stream_client
    
    # 1. Register with Dedicated Streaming Server
    if stream_client.register_camera(camera.id, camera.source):
        # 2. Add to Main AI System using the proxied stream
        proxy_url = stream_client.get_stream_url(camera.id)
        success = camera_manager.add_camera(camera.id, proxy_url, camera.name)
        if success:
            return {"status": "success", "message": f"Camera {camera.id} proxied via streaming server"}
    
    raise HTTPException(status_code=400, detail="Failed to register or connect to camera")

@router.get("")
async def list_cameras(request: Request):
    camera_manager = request.app.state.camera_manager
    return camera_manager.get_cameras_status()

@router.put("/{camera_id}")
async def update_camera(camera_id: str, camera: CameraCreate, request: Request):
    camera_manager = request.app.state.camera_manager
    if camera_manager.remove_camera(camera_id):
        # Re-add with new settings
        if camera_manager.add_camera(camera.id, camera.source, camera.name):
            return {"status": "success"}
    raise HTTPException(status_code=404, detail="Camera not found")

@router.delete("/{camera_id}")
async def delete_camera(camera_id: str, request: Request):
    camera_manager = request.app.state.camera_manager
    if camera_manager.remove_camera(camera_id):
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Camera not found")
