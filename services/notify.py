import asyncio
import json
import threading
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

class NotificationService:
    """Manages real-time event broadcasting to multiple web clients via SSE."""
    def __init__(self):
        self.clients: List[asyncio.Queue] = []
        self.lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    async def subscribe(self):
        q = asyncio.Queue()
        with self.lock:
            self.clients.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)

    def broadcast(self, data: dict):
        msg = f"data: {json.dumps(data)}\n\n"
        with self.lock:
            loop = self._loop
            clients = list(self.clients)
            
        if loop is None or not loop.is_running():
            return
            
        for q in clients:
            try:
                loop.call_soon_threadsafe(q.put_nowait, msg)
            except Exception:
                pass

notification_service = NotificationService()
