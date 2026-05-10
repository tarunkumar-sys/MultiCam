from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
import logging
from config.settings import settings

router = APIRouter(prefix="/api/identity", tags=["Identity Management"])
logger = logging.getLogger(__name__)

@router.get("/profiles")
async def list_profiles(request: Request):
    db = request.app.state.db_manager
    persons = db.get_persons_with_last_seen()
    return persons

@router.post("/register")
async def register_identity(
    request: Request,
    name: str = Form(...),
    file: UploadFile = File(...)
):
    db = request.app.state.db_manager
    recognizer = request.app.state.recognizer
    
    try:
        # 1. Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
            
        # 2. Extract encoding
        encoding = recognizer.get_encoding(img)
        if encoding is None:
            raise HTTPException(status_code=400, detail="No face detected in registration image")
            
        # 3. Save reference image
        ref_dir = os.path.join("static", "profiles")
        os.makedirs(ref_dir, exist_ok=True)
        img_path = os.path.join(ref_dir, f"{name.replace(' ', '_')}.jpg")
        cv2.imwrite(img_path, img)
        
        # 4. Save to DB
        db.register_person(name, img_path, encoding)
        
        # 5. Reload recognizer encodings
        recognizer.load_known_faces(db)
        
        return {"status": "success", "name": name}
        
    except Exception as e:
        logger.error(f"Registration Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/profiles/{person_id}")
async def delete_profile(person_id: str, request: Request):
    db = request.app.state.db_manager
    recognizer = request.app.state.recognizer
    
    db.delete_person_from_db(person_id)
    recognizer.load_known_faces(db)
    
    return {"status": "success"}

@router.get("/audit")
async def get_audit_logs(request: Request, name: str = None):
    db = request.app.state.db_manager
    logs = db.get_registered_detections(name=name)
    # Convert to frontend format
    return [{
        "id": str(i),
        "person_name": l[0],
        "camera_id": l[1],
        "timestamp": l[2].strftime("%Y-%m-%d %H:%M:%S") if hasattr(l[2], "strftime") else str(l[2]),
        "image_path": None, # Logic to find specific snapshot can be added if needed
        "camera_ip": "192.168.1.1XX" # Placeholder
    } for i, l in enumerate(logs)]
