import asyncio
import cv2
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from services.vision import camera_results, results_lock
from services.notify import notification_service
import logging

router = APIRouter(tags=["Streaming"])
logger = logging.getLogger(__name__)

@router.get("/api/notifications/stream")
async def event_stream():
    """SSE endpoint for real-time notifications."""
    async def stream():
        q = await notification_service.subscribe()
        try:
            while True:
                msg = await q.get()
                yield f"data: {msg}\n\n"
        finally:
            notification_service.unsubscribe(q)
    return StreamingResponse(stream(), media_type="text/event-stream")

@router.get("/video_feed/{camera_id}")
@router.get("/api/streams/video/{camera_id}")
async def video_feed(camera_id: str):
    """MJPEG stream of processed camera frames."""
    async def generate():
        while True:
            frame_data = None
            with results_lock:
                if camera_id in camera_results:
                    frame_data = camera_results[camera_id].get("rendered_frame")
            
            if frame_data is not None:
                _, buffer = cv2.imencode('.jpg', frame_data, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            await asyncio.sleep(0.04) # ~25 FPS limit
            
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@router.get("/api/occupancy")
async def get_occupancy():
    """Current person counts for all cameras."""
    stats = []
    with results_lock:
        for cid, data in camera_results.items():
            stats.append({
                "camera_id": cid,
                "count": data.get("count", 0),
                "alert_active": data.get("alert_active", False)
            })
    return stats

@router.get("/api/recognized/{camera_id}")
async def get_recognized(camera_id: str):
    """Names of recognized persons in current frame."""
    with results_lock:
        if camera_id in camera_results:
            return camera_results[camera_id].get("identities", [])
    return []

@router.get("/api/live_results/{camera_id}")
async def get_live_results(camera_id: str):
    """Person cards with face crops for sidebar."""
    with results_lock:
        if camera_id in camera_results:
            return camera_results[camera_id].get("person_cards", [])
    return []

@router.get("/api/camera_daily_stats")
async def get_daily_stats(request: Request):
    """24h unique person counts per camera."""
    db = request.app.state.db_manager
    # In a real app, this would be a complex query. 
    # For now, return the current live count as the "total" or zero.
    stats = {}
    with results_lock:
        for cid, data in camera_results.items():
            stats[cid] = {"total": data.get("count", 0)}
    return stats

@router.get("/api/capture_frame/{camera_id}")
async def capture_frame(camera_id: str):
    """Capture a single high-quality JPEG frame."""
    from fastapi.responses import Response
    frame_data = None
    with results_lock:
        if camera_id in camera_results:
            frame_data = camera_results[camera_id].get("raw_frame")
    
    if frame_data is not None:
        _, buffer = cv2.imencode('.jpg', frame_data, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return Response(content=buffer.tobytes(), media_type="image/jpeg")
    
    return Response(status_code=404)
