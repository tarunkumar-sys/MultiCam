import os
import sys
import logging
import asyncio
import subprocess
import time
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.responses import RedirectResponse

from config.settings import settings
from database.sqlite_manager import SqliteManager
from core.detector import PersonDetector
from core.recognizer import FaceRecognizer
from gate.camera_manager import CameraManager
from gate.stream_orchestrator_client import StreamOrchestratorClient
from services import VisionService, RecordingService, notification_service
from routes import camera_router, stream_router, recording_router, analytics_router, search_router, identity_router

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VigiLance")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Logic
    try:
        import requests
        requests.get("http://localhost:8001/docs", timeout=1)
    except Exception:
        logger.info("📡 Media Broker not detected. Auto-starting...")
        subprocess.Popen([sys.executable, "media/server.py"], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL,
                         creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
        time.sleep(2) # Give it a moment to boot

    app.state.db_manager = SqliteManager()
    app.state.detector = PersonDetector()
    app.state.recognizer = FaceRecognizer()
    app.state.recognizer.load_known_faces(app.state.db_manager)
    
    app.state.stream_client = StreamOrchestratorClient(base_url="http://localhost:8001")
    app.state.camera_manager = CameraManager()
    
    notification_service.set_loop(asyncio.get_event_loop())
    app.state.recording_service = RecordingService(app.state.db_manager)
    app.state.vision_service = VisionService(app.state.detector, app.state.camera_manager)
    
    app.state.vision_service.start()
    logger.info("🚀 VigiLance System Ready")
    
    yield
    
    # Shutdown Logic
    app.state.vision_service.stop()
    app.state.camera_manager.stop_all()
    logger.info("🛑 VigiLance System Offline")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Industry-standard AI Surveillance System",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(SessionMiddleware, secret_key="vigilance_secret_key_123")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/snapshots", StaticFiles(directory=settings.SNAPSHOTS_DIR), name="snapshots")
app.mount("/recordings", StaticFiles(directory=settings.RECORDINGS_DIR), name="recordings")

templates = Jinja2Templates(directory="templates")

app.include_router(camera_router)
app.include_router(stream_router)
app.include_router(recording_router)
app.include_router(analytics_router)
app.include_router(search_router)
app.include_router(identity_router)

@app.get("/")
async def login_page(request: Request):
    if request.session.get("logged_in"):
        is_partial = request.query_params.get("partial") == "true"
        url = "/live?partial=true" if is_partial else "/live"
        return RedirectResponse(url=url, status_code=303)
    return templates.TemplateResponse(request=request, name="login.html")

@app.post("/api/login")
async def api_login(request: Request):
    form = await request.form()
    username = form.get("username")
    password = form.get("password")
    
    # Simple hardcoded auth for demonstration
    if username == "admin" and password == "admin":
        request.session["logged_in"] = True
        return {"status": "success"}
    else:
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/live")
async def live_streams(request: Request):
    if not request.session.get("logged_in"):
        return RedirectResponse(url="/", status_code=303)
    active_cams = app.state.camera_manager.get_cameras_status()
    is_partial = request.query_params.get("partial") == "true"
    base = "null.html" if is_partial else "base.html"
    return templates.TemplateResponse(request=request, name="index.html", context={"cameras": active_cams, "project_name": settings.PROJECT_NAME, "base_template": base})

@app.get("/dashboard")
async def dashboard(request: Request):
    if not request.session.get("logged_in"):
        return RedirectResponse(url="/", status_code=303)
    is_partial = request.query_params.get("partial") == "true"
    base = "null.html" if is_partial else "base.html"
    return templates.TemplateResponse(request=request, name="dashboard.html", context={"project_name": settings.PROJECT_NAME, "base_template": base})

@app.get("/api/server_time")
async def server_time():
    now = settings.get_ist_time()
    return {"timestamp_ms": int(now.timestamp() * 1000)}

@app.get("/cameras")
async def cameras_page(request: Request):
    is_partial = request.query_params.get("partial") == "true"
    base = "null.html" if is_partial else "base.html"
    return templates.TemplateResponse(request=request, name="cameras.html", context={"project_name": settings.PROJECT_NAME, "base_template": base})

@app.get("/add_camera")
async def add_camera_page(request: Request):
    is_partial = request.query_params.get("partial") == "true"
    base = "null.html" if is_partial else "base.html"
    return templates.TemplateResponse(request=request, name="add_camera.html", context={"project_name": settings.PROJECT_NAME, "base_template": base})

@app.get("/recordings_page")
async def recordings_page(request: Request):
    is_partial = request.query_params.get("partial") == "true"
    base = "null.html" if is_partial else "base.html"
    return templates.TemplateResponse(request=request, name="recordings.html", context={"project_name": settings.PROJECT_NAME, "base_template": base})

@app.get("/detection_logs")
async def detection_logs_page(request: Request):
    is_partial = request.query_params.get("partial") == "true"
    base = "null.html" if is_partial else "base.html"
    return templates.TemplateResponse(request=request, name="detection_logs.html", context={"project_name": settings.PROJECT_NAME, "base_template": base})

@app.get("/people")
async def people_page(request: Request):
    is_partial = request.query_params.get("partial") == "true"
    base = "null.html" if is_partial else "base.html"
    return templates.TemplateResponse(request=request, name="people.html", context={"project_name": settings.PROJECT_NAME, "base_template": base})

@app.get("/search")
async def search_page(request: Request):
    is_partial = request.query_params.get("partial") == "true"
    base = "null.html" if is_partial else "base.html"
    return templates.TemplateResponse(request=request, name="search.html", context={"project_name": settings.PROJECT_NAME, "base_template": base})

@app.get("/journey")
async def journey_page(request: Request):
    is_partial = request.query_params.get("partial") == "true"
    base = "null.html" if is_partial else "base.html"
    return templates.TemplateResponse(request=request, name="journey.html", context={"project_name": settings.PROJECT_NAME, "base_template": base})

@app.get("/analytics")
async def analytics_page(request: Request):
    is_partial = request.query_params.get("partial") == "true"
    base = "null.html" if is_partial else "base.html"
    return templates.TemplateResponse(request=request, name="analytics.html", context={"project_name": settings.PROJECT_NAME, "base_template": base})

@app.get("/registered_detections")
async def registered_detections_page(request: Request):
    is_partial = request.query_params.get("partial") == "true"
    base = "null.html" if is_partial else "base.html"
    return templates.TemplateResponse(request=request, name="registered_detections.html", context={"project_name": settings.PROJECT_NAME, "base_template": base})

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/", status_code=303)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
