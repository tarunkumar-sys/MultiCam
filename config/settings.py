import os
import pytz
from datetime import datetime

class Settings:
    PROJECT_NAME = "VigiLance AI"
    ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "deiadmin@789")
    
    # Timezone (Centralized)
    IST = pytz.timezone('Asia/Kolkata')
    
    # Paths
    SNAPSHOTS_DIR = "snapshots"
    DATASET_DIR = "dataset"
    RECORDINGS_DIR = "recordings"
    
    # Retention
    SNAPSHOT_RETENTION_HOURS = 24
    RECORDING_RETENTION_DAYS = 2
    
    # Alerting
    ALERT_COOLDOWN_SECONDS = 30

    @staticmethod
    def get_ist_time():
        return datetime.now(Settings.IST)

settings = Settings()

# Ensure directories
for d in [settings.SNAPSHOTS_DIR, settings.DATASET_DIR, settings.RECORDINGS_DIR]:
    os.makedirs(d, exist_ok=True)
