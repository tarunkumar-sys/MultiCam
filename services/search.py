import logging
import threading
import numpy as np
from typing import Dict, Any, Set

logger = logging.getLogger(__name__)

class SearchService:
    def __init__(self, db_manager, recognizer):
        self.db_manager = db_manager
        self.recognizer = recognizer
        self.active_search: Dict[str, Any] = {}
        self.search_lock = threading.Lock()

    def start_mission(self, person_name: str):
        persons = self.db_manager.get_registered_persons()
        target = next((p for p in persons if p[1].lower() == person_name.lower()), None)
        if not target:
            return False, f"Person {person_name} not registered"
            
        # target[3] is the encoding blob
        encoding = np.frombuffer(target[3], dtype=np.float32)
        with self.search_lock:
            self.active_search = {
                "running": True,
                "name": target[1],
                "encoding": encoding,
                "found_tracks": set()
            }
        return True, f"Search mission started for {target[1]}"

    def stop_mission(self):
        with self.search_lock:
            self.active_search = {}
        return True

    def get_status(self):
        with self.search_lock:
            return {
                "active": self.active_search.get("running", False),
                "target": self.active_search.get("name")
            }

search_service = None # To be initialized in main.py
