from fastapi import APIRouter, Request, Query
from typing import Optional
import logging

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])
logger = logging.getLogger(__name__)

@router.get("/hourly")
async def get_hourly_analytics(request: Request, camera_id: Optional[str] = Query(None)):
    """Returns hourly person count patterns for the last 24h."""
    db = request.app.state.db_manager
    data = db.get_hourly_analytics(camera_id)
    
    # Format for Chart.js in analytics.html
    # Expected: [{label: "10 AM", count: 5}, ...]
    formatted = []
    for item in data:
        hour = item["_id"]
        ampm = "AM" if hour < 12 else "PM"
        display_hour = hour % 12
        if display_hour == 0: display_hour = 12
        formatted.append({
            "label": f"{display_hour} {ampm}",
            "count": item["max_count"],
            "camera_ids": item.get("camera_ids", [])
        })
    return formatted

@router.get("/daily")
async def get_daily_analytics(request: Request, camera_id: Optional[str] = Query(None), days: int = 7):
    """Returns daily person count patterns for the last N days."""
    db = request.app.state.db_manager
    data = db.get_daily_analytics(camera_id, days)
    
    # Format for Chart.js
    # Expected: [{date: "2023-01-01", count: 50}, ...]
    formatted = []
    for item in data:
        formatted.append({
            "date": item["_id"],
            "count": item["count"]
        })
    return formatted
@router.get("/active_targets")
async def get_active_targets(request: Request):
    """Returns recently seen entities for journey selection."""
    db = request.app.state.db_manager
    targets = db.get_recent_active_targets(hours=24)
    return [{
        "id": t["global_id"],
        "last_seen": t["last_seen"].strftime("%H:%M:%S") if t["last_seen"] else "N/A",
        "last_camera": t["last_camera"]
    } for t in targets]

@router.get("/target_journey/{global_id}")
async def get_target_journey(global_id: str, request: Request):
    """Returns chronological path for a specific entity."""
    db = request.app.state.db_manager
    journey = db.get_target_journey(global_id)
    # Convert to frontend format
    return [{
        "timestamp": step["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
        "camera_id": step["camera_id"],
        "snapshot_path": step["snapshot_path"]
    } for step in journey]
