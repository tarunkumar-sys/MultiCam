import requests
import logging

logger = logging.getLogger(__name__)

class StreamOrchestratorClient:
    """Client for the dedicated VigiLance Streaming Server."""
    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url

    def register_camera(self, camera_id: str, source: str):
        try:
            resp = requests.post(f"{self.base_url}/register/{camera_id}", params={"source": source}, timeout=5)
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"Failed to connect to Streaming Server: {e}")
            return False

    def get_stream_url(self, camera_id: str):
        return f"{self.base_url}/stream/{camera_id}"
