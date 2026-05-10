from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class CameraCreate(BaseModel):
    id: str
    source: str
    name: str

class CameraResponse(BaseModel):
    id: str
    source: str
    name: str
    is_active: bool

class DetectionSnapshot(BaseModel):
    camera_id: str
    person_count: int
    timestamp: str
    snapshot_path: str
    bbox_data: List[Dict[str, Any]]
