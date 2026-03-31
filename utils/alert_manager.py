"""
=============================================================================
AI VIGILANCE — SIMPLIFIED IN-APP ALERT SYSTEM
File: utils/alert_manager.py

Simple alert system — alerts ONLY when a REGISTERED person is detected.
No SMTP, no email. Just in-memory alert queue for the frontend to poll.
=============================================================================
"""

import threading
import time
import logging
from datetime import datetime
from typing import Optional, List, Dict

log = logging.getLogger("AlertManager")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s"
)


class AlertManager:
    """
    In-app alert manager for AI Vigilance.
    Fires alerts ONLY for registered persons.
    Alerts are stored in-memory and polled by the frontend.
    """

    def __init__(self):
        self.cooldown = 60  # seconds between alerts for same person+camera
        self.enabled = True
        self._cooldown_map: Dict[str, float] = {}
        self._alerts: List[Dict] = []  # Recent alerts (max 100)
        self._lock = threading.Lock()
        self._max_alerts = 100

    def fire(
        self,
        person_name: str,
        camera_id: str,
        snapshot_path: Optional[str] = None,
        confidence: float = 0.0,
    ) -> bool:
        """
        Queue an alert. Only fires for KNOWN/REGISTERED persons.
        Returns True if alert was created, False if suppressed.
        """
        if not self.enabled:
            return False

        # ONLY alert for registered persons — skip unknown
        is_unknown = (person_name.lower() in ("unknown", ""))
        if is_unknown:
            return False

        # Cooldown check
        cooldown_key = f"{person_name}::{camera_id}"
        now = time.time()
        with self._lock:
            last = self._cooldown_map.get(cooldown_key, 0)
            if now - last < self.cooldown:
                return False
            self._cooldown_map[cooldown_key] = now

            alert = {
                "id": len(self._alerts) + 1,
                "person_name": person_name,
                "camera_id": camera_id,
                "snapshot_path": snapshot_path,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "read": False,
            }
            self._alerts.insert(0, alert)

            # Keep only max alerts
            if len(self._alerts) > self._max_alerts:
                self._alerts = self._alerts[:self._max_alerts]

        log.info(f"Alert: {person_name} detected on {camera_id}")
        return True

    def get_alerts(self, limit: int = 50) -> List[Dict]:
        """Get recent alerts."""
        with self._lock:
            return self._alerts[:limit]

    def get_unread_count(self) -> int:
        """Get count of unread alerts."""
        with self._lock:
            return sum(1 for a in self._alerts if not a.get("read"))

    def mark_read(self, alert_id: int = None):
        """Mark alert(s) as read."""
        with self._lock:
            if alert_id:
                for a in self._alerts:
                    if a["id"] == alert_id:
                        a["read"] = True
                        break
            else:
                for a in self._alerts:
                    a["read"] = True

    def clear_alerts(self):
        """Clear all alerts."""
        with self._lock:
            self._alerts.clear()

    @property
    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "cooldown_secs": self.cooldown,
            "total_alerts": len(self._alerts),
            "unread": self.get_unread_count(),
        }
