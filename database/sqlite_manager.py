"""
SQLite3 Database Manager for AI Vigilance.
Replaces MongoDB with SQLite3 for local-first storage.
"""
import sqlite3
import json
import os
import logging
from datetime import datetime, timedelta
import pytz
import numpy as np

# IST Timezone
IST = pytz.timezone('Asia/Kolkata')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SqliteManager:
    """SQLite3 database manager for surveillance system."""
    
    def __init__(self, db_path="db.sqlite3"):
        self.db_path = db_path
        try:
            self._init_db()
            logger.info(f"✓ Connected to SQLite: {db_path}")
        except Exception as e:
            logger.critical(f"✗ Failed to connect to SQLite: {e}")
            raise RuntimeError(f"SQLite connection failed: {e}")

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Cameras
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cameras (
                    camera_id TEXT PRIMARY KEY,
                    source TEXT,
                    updated_at DATETIME
                )
            ''')
            
            # 2. Camera Settings
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS camera_settings (
                    camera_id TEXT PRIMARY KEY,
                    recording_enabled INTEGER DEFAULT 0,
                    tracking_area TEXT
                )
            ''')
            
            # 3. Persons (Registered)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    image_path TEXT,
                    encoding BLOB,
                    last_seen DATETIME,
                    last_camera TEXT
                )
            ''')
            
            # 4. Registered Detections (History)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS registered_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_name TEXT,
                    camera_id TEXT,
                    timestamp DATETIME
                )
            ''')
            
            # 5. Detection Snapshots
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detection_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT,
                    person_count INTEGER,
                    snapshot_path TEXT,
                    bbox_data TEXT,
                    face_encodings TEXT,
                    person_crops TEXT,
                    timestamp DATETIME
                )
            ''')
            
            # 6. Occupancy Logs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS occupancy_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT,
                    timestamp DATETIME,
                    count INTEGER
                )
            ''')
            
            # 7. Video Recordings
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS video_recordings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT,
                    file_path TEXT,
                    start_time DATETIME,
                    end_time DATETIME,
                    has_registered_person INTEGER DEFAULT 0,
                    registered_person_times TEXT
                )
            ''')
            
            # 8. Alerts
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT,
                    person_id TEXT,
                    snapshot_path TEXT,
                    timestamp DATETIME,
                    type TEXT
                )
            ''')
            
            # 9. Global Identities (Re-ID)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS global_identities (
                    global_id TEXT PRIMARY KEY,
                    encoding BLOB,
                    first_seen DATETIME,
                    last_seen DATETIME,
                    last_camera TEXT,
                    type TEXT,
                    thumbnail BLOB
                )
            ''')
            
            # 10. Journeys
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS journeys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    global_id TEXT,
                    camera_id TEXT,
                    timestamp DATETIME,
                    snapshot_path TEXT,
                    type TEXT
                )
            ''')
            
            # Indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_snapshots_cam_time ON detection_snapshots (camera_id, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_reg_det_name_time ON registered_detections (person_name, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_video_cam_time ON video_recordings (camera_id, start_time)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_cam_time ON alerts (camera_id, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_journeys_id_time ON journeys (global_id, timestamp)')
            
            conn.commit()

    # --- Cameras ---
    def add_camera_to_db(self, camera_id, source):
        try:
            with self._get_connection() as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO cameras (camera_id, source, updated_at)
                    VALUES (?, ?, ?)
                ''', (camera_id, str(source), datetime.utcnow()))
                conn.commit()
        except Exception as e: logger.error(f"✗ Error adding camera: {e}")

    def remove_camera_from_db(self, camera_id):
        try:
            with self._get_connection() as conn:
                conn.execute('DELETE FROM cameras WHERE camera_id = ?', (camera_id,))
                conn.execute('DELETE FROM camera_settings WHERE camera_id = ?', (camera_id,))
                conn.commit()
        except Exception as e: logger.error(f"✗ Error removing camera: {e}")

    def update_camera_source(self, camera_id, new_source):
        try:
            with self._get_connection() as conn:
                conn.execute('''
                    UPDATE cameras SET source = ?, updated_at = ?
                    WHERE camera_id = ?
                ''', (str(new_source), datetime.utcnow(), camera_id))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"✗ Error updating camera source: {e}")
            return False

    def get_cameras(self):
        try:
            with self._get_connection() as conn:
                rows = conn.execute('SELECT camera_id, source FROM cameras').fetchall()
                return [[r["camera_id"], r["source"]] for r in rows]
        except Exception as e: return []

    # --- Settings ---
    def get_camera_recording_setting(self, camera_id):
        try:
            with self._get_connection() as conn:
                row = conn.execute('SELECT recording_enabled FROM camera_settings WHERE camera_id = ?', (camera_id,)).fetchone()
                return 1 if row and row["recording_enabled"] else 0
        except Exception: return 0

    def set_camera_recording(self, camera_id, enabled):
        try:
            with self._get_connection() as conn:
                conn.execute('''
                    INSERT INTO camera_settings (camera_id, recording_enabled)
                    VALUES (?, ?)
                    ON CONFLICT(camera_id) DO UPDATE SET recording_enabled = excluded.recording_enabled
                ''', (camera_id, 1 if enabled else 0))
                conn.commit()
        except Exception: pass

    def set_camera_tracking_area(self, camera_id, area):
        try:
            with self._get_connection() as conn:
                conn.execute('''
                    INSERT INTO camera_settings (camera_id, tracking_area)
                    VALUES (?, ?)
                    ON CONFLICT(camera_id) DO UPDATE SET tracking_area = excluded.tracking_area
                ''', (camera_id, json.dumps(area)))
                conn.commit()
        except Exception: pass

    def get_camera_tracking_area(self, camera_id):
        try:
            with self._get_connection() as conn:
                row = conn.execute('SELECT tracking_area FROM camera_settings WHERE camera_id = ?', (camera_id,)).fetchone()
                return json.loads(row["tracking_area"]) if row and row["tracking_area"] else None
        except Exception: return None

    # --- Persons ---
    def register_person(self, name, image_path, encoding):
        try:
            # encoding is likely a numpy array, store as bytes
            if hasattr(encoding, 'tobytes'):
                encoding_blob = encoding.tobytes()
            else:
                encoding_blob = encoding
                
            with self._get_connection() as conn:
                conn.execute('''
                    INSERT INTO persons (name, image_path, encoding)
                    VALUES (?, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET image_path = excluded.image_path, encoding = excluded.encoding
                ''', (name, image_path, encoding_blob))
                conn.commit()
            return name
        except Exception as e: 
            logger.error(f"Error registering person: {e}")
            return None

    def get_registered_persons(self):
        try:
            with self._get_connection() as conn:
                rows = conn.execute('SELECT id, name, image_path, encoding FROM persons').fetchall()
                return [[str(r["id"]), r["name"], r["image_path"], r["encoding"]] for r in rows]
        except Exception: return []

    def get_persons_with_last_seen(self):
        try:
            with self._get_connection() as conn:
                rows = conn.execute('SELECT id, name, image_path, last_seen, last_camera FROM persons').fetchall()
                return [{
                    "id": str(r["id"]), 
                    "name": r["name"], 
                    "image_path": r["image_path"], 
                    "last_seen": datetime.fromisoformat(r["last_seen"]) if r["last_seen"] else None, 
                    "last_camera": r["last_camera"]
                } for r in rows]
        except Exception: return []

    def get_detections(self, limit=20):
        """Alias for get_registered_detections for metrics."""
        return self.get_registered_detections(limit=limit)

    def update_person_last_seen(self, name, camera_id):
        try:
            now = datetime.now(IST).isoformat()
            with self._get_connection() as conn:
                conn.execute('UPDATE persons SET last_seen = ?, last_camera = ? WHERE name = ?', (now, camera_id, name))
                conn.execute('INSERT INTO registered_detections (person_name, camera_id, timestamp) VALUES (?, ?, ?)', (name, camera_id, now))
                conn.commit()
        except Exception: pass

    def search_detections(self, name=None, start_time=None, end_time=None):
        try:
            query = "SELECT * FROM registered_detections WHERE 1=1"
            params = []
            if name:
                query += " AND person_name = ?"
                params.append(name)
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY timestamp DESC"
            
            with self._get_connection() as conn:
                rows = conn.execute(query, params).fetchall()
                return [[r["id"], r["person_name"], r["camera_id"], 
                         datetime.fromisoformat(r["timestamp"]) if isinstance(r["timestamp"], str) else r["timestamp"],
                         None, r["person_name"]] for r in rows]
        except Exception: return []

    def get_registered_detections(self, name=None, limit=200):
        try:
            with self._get_connection() as conn:
                if name:
                    rows = conn.execute('SELECT person_name, camera_id, timestamp FROM registered_detections WHERE person_name = ? ORDER BY timestamp DESC LIMIT ?', (name, limit)).fetchall()
                else:
                    rows = conn.execute('SELECT person_name, camera_id, timestamp FROM registered_detections ORDER BY timestamp DESC LIMIT ?', (limit,)).fetchall()
                
                return [{
                    "person_name": r["person_name"],
                    "camera_id": r["camera_id"],
                    "timestamp": datetime.fromisoformat(r["timestamp"]) if isinstance(r["timestamp"], str) else r["timestamp"]
                } for r in rows]
        except Exception: return []

    # --- Snapshots ---
    def log_detection_snapshot(self, camera_id, person_count, snapshot_path, bbox_data, face_encodings=None, person_crops=None, timestamp=None):
        try:
            if timestamp is None:
                timestamp = datetime.now(IST)
            ts_iso = timestamp.isoformat() if hasattr(timestamp, 'isoformat') else timestamp
            
            # Serialize complex types safely
            bbox_str = bbox_data if isinstance(bbox_data, str) else json.dumps(bbox_data)
            
            if face_encodings:
                if isinstance(face_encodings, str):
                    face_enc_str = face_encodings
                else:
                    face_enc_str = json.dumps([e.tolist() if hasattr(e, 'tolist') else e for e in face_encodings])
            else:
                face_enc_str = None
                
            person_crops_str = person_crops if isinstance(person_crops, str) else (json.dumps(person_crops) if person_crops else None)
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO detection_snapshots (camera_id, person_count, snapshot_path, bbox_data, face_encodings, person_crops, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (camera_id, int(person_count), snapshot_path, bbox_str, face_enc_str, person_crops_str, ts_iso))
                conn.commit()
                return str(cursor.lastrowid)
        except Exception as e: 
            logger.error(f"✗ Error logging snapshot: {e}")
            return None

    def get_detection_snapshots(self, camera_id=None, start_time=None, end_time=None, limit=20, skip=0):
        try:
            query = "SELECT id, camera_id, timestamp, person_count, snapshot_path, bbox_data, person_crops FROM detection_snapshots WHERE 1=1"
            params = []
            if camera_id:
                query += " AND camera_id = ?"
                params.append(camera_id)
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat() if hasattr(start_time, 'isoformat') else start_time)
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat() if hasattr(end_time, 'isoformat') else end_time)
            
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, skip])
            
            with self._get_connection() as conn:
                rows = conn.execute(query, params).fetchall()
                return [
                    [
                        str(r["id"]), r["camera_id"], 
                        datetime.fromisoformat(r["timestamp"]) if isinstance(r["timestamp"], str) else r["timestamp"],
                        r["person_count"], 
                        r["snapshot_path"], json.loads(r["bbox_data"]) if r["bbox_data"] else [], 
                        json.loads(r["person_crops"]) if r["person_crops"] else []
                    ] 
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"Error getting snapshots: {e}")
            return []

    def count_detection_snapshots(self, camera_id=None):
        try:
            query = "SELECT COUNT(*) as cnt FROM detection_snapshots WHERE 1=1"
            params = []
            if camera_id:
                query += " AND camera_id = ?"
                params.append(camera_id)
            with self._get_connection() as conn:
                row = conn.execute(query, params).fetchone()
                return row["cnt"] if row else 0
        except Exception:
            return 0

    def get_snapshot(self, snapshot_id):
        try:
            with self._get_connection() as conn:
                r = conn.execute('SELECT * FROM detection_snapshots WHERE id = ?', (int(snapshot_id),)).fetchone()
                if r:
                    return [
                        str(r["id"]), r["camera_id"], 
                        datetime.fromisoformat(r["timestamp"]) if isinstance(r["timestamp"], str) else r["timestamp"],
                        r["person_count"], r["snapshot_path"], 
                        json.loads(r["bbox_data"]) if r["bbox_data"] else []
                    ]
                return None
        except Exception: return None

    def delete_all_detections(self):
        try:
            with self._get_connection() as conn:
                conn.execute('DELETE FROM detection_snapshots')
                conn.execute('DELETE FROM registered_detections')
                conn.execute('DELETE FROM occupancy_logs')
                conn.commit()
        except Exception: pass

    def log_occupancy(self, camera_id, count):
        try:
            now = datetime.now(IST).isoformat()
            with self._get_connection() as conn:
                conn.execute('INSERT INTO occupancy_logs (camera_id, timestamp, count) VALUES (?, ?, ?)', (camera_id, now, int(count)))
                conn.commit()
        except Exception: pass

    def search_occupancy(self, camera_id=None, start_time=None, end_time=None):
        try:
            query = "SELECT id, camera_id, timestamp, count FROM occupancy_logs WHERE 1=1"
            params = []
            if camera_id:
                query += " AND camera_id = ?"
                params.append(camera_id)
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat() if hasattr(start_time, 'isoformat') else start_time)
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat() if hasattr(end_time, 'isoformat') else end_time)
            
            query += " ORDER BY timestamp DESC"
            
            with self._get_connection() as conn:
                rows = conn.execute(query, params).fetchall()
                return [[str(r["id"]), r["camera_id"], r["timestamp"], r["count"]] for r in rows]
        except Exception: return []

    # --- Recordings ---
    def start_recording(self, camera_id, file_path):
        try:
            now = datetime.now(IST).isoformat()
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO video_recordings (camera_id, file_path, start_time)
                    VALUES (?, ?, ?)
                ''', (camera_id, file_path, now))
                conn.commit()
                return str(cursor.lastrowid)
        except Exception: return None

    def end_recording(self, record_id):
        try:
            now = datetime.now(IST).isoformat()
            with self._get_connection() as conn:
                conn.execute('UPDATE video_recordings SET end_time = ? WHERE id = ?', (now, int(record_id)))
                conn.commit()
        except Exception: pass

    def search_recordings(self, camera_id=None, start_time=None, end_time=None):
        try:
            query = "SELECT * FROM video_recordings WHERE 1=1"
            params = []
            if camera_id:
                query += " AND camera_id = ?"
                params.append(camera_id)
            if start_time:
                query += " AND start_time >= ?"
                params.append(start_time.isoformat() if hasattr(start_time, 'isoformat') else start_time)
            if end_time:
                query += " AND start_time <= ?"
                params.append(end_time.isoformat() if hasattr(end_time, 'isoformat') else end_time)
            
            query += " ORDER BY start_time DESC"
            
            with self._get_connection() as conn:
                rows = conn.execute(query, params).fetchall()
                return [[str(r["id"]), r["camera_id"], 
                         datetime.fromisoformat(r["start_time"]) if r["start_time"] else None,
                         datetime.fromisoformat(r["end_time"]) if r["end_time"] else None,
                         r["file_path"], bool(r["has_registered_person"]), 
                         json.loads(r["registered_person_times"]) if r["registered_person_times"] else []] for r in rows]
        except Exception: return []

    def get_recorded_videos(self):
        """Returns all recorded videos."""
        return self.search_recordings()

    def get_recording(self, record_id):
        try:
            with self._get_connection() as conn:
                r = conn.execute('SELECT * FROM video_recordings WHERE id = ?', (int(record_id),)).fetchone()
                return [str(r["id"]), r["camera_id"], r["start_time"], r["end_time"], r["file_path"]] if r else None
        except Exception: return None

    def delete_recording(self, record_id):
        try:
            with self._get_connection() as conn:
                conn.execute('DELETE FROM video_recordings WHERE id = ?', (int(record_id),))
                conn.commit()
        except Exception: pass

    # --- Analytics ---
    def get_hourly_analytics(self, camera_id=None):
        """SQLite version of hourly analytics. Group by hour part of timestamp."""
        try:
            # SQLite doesn't have a direct aggregate across 24h as easily as Mongo for grouping by IST hour
            # We'll fetch the last 24h of snapshots and do grouping in Python for accuracy with IST
            now = datetime.utcnow()
            start_time = now - timedelta(hours=24)
            
            with self._get_connection() as conn:
                query = "SELECT camera_id, person_count, timestamp FROM detection_snapshots WHERE timestamp >= ?"
                params = [start_time.isoformat()]
                if camera_id:
                    query += " AND camera_id = ?"
                    params.append(camera_id)
                
                rows = conn.execute(query, params).fetchall()
                
            # Grouping logic
            hourly_data = {} # {hour: {"max_count": val, "camera_ids": set}}
            for r in rows:
                dt = datetime.fromisoformat(r["timestamp"])
                # Convert to IST
                if dt.tzinfo is None:
                    dt = pytz.utc.localize(dt).astimezone(IST)
                else:
                    dt = dt.astimezone(IST)
                    
                h = dt.hour
                if h not in hourly_data:
                    hourly_data[h] = {"_id": h, "max_count": 0, "camera_ids": set()}
                
                hourly_data[h]["max_count"] = max(hourly_data[h]["max_count"], r["person_count"])
                hourly_data[h]["camera_ids"].add(r["camera_id"])
            
            result = list(hourly_data.values())
            for item in result:
                item["camera_ids"] = list(item["camera_ids"])
            
            return sorted(result, key=lambda x: x["_id"])
        except Exception as e:
            logger.error(f"Error in get_hourly_analytics: {e}")
            return []

    def get_daily_analytics(self, camera_id=None, days=7):
        try:
            now = datetime.utcnow()
            start_time = now - timedelta(days=days)
            
            with self._get_connection() as conn:
                query = "SELECT camera_id, person_count, timestamp FROM detection_snapshots WHERE timestamp >= ?"
                params = [start_time.isoformat()]
                if camera_id:
                    query += " AND camera_id = ?"
                    params.append(camera_id)
                
                rows = conn.execute(query, params).fetchall()
                
            daily_data = {} # {(y,m,d): max_count}
            for r in rows:
                dt = datetime.fromisoformat(r["timestamp"])
                if dt.tzinfo is None:
                    dt = pytz.utc.localize(dt).astimezone(IST)
                else:
                    dt = dt.astimezone(IST)
                
                key = (dt.year, dt.month, dt.day)
                daily_data[key] = max(daily_data.get(key, 0), r["person_count"])
                
            result = []
            for (y, m, d), count in daily_data.items():
                result.append({"_id": {"year": y, "month": m, "day": d}, "max_count": count})
                
            return sorted(result, key=lambda x: (x["_id"]["year"], x["_id"]["month"], x["_id"]["day"]))
        except Exception: return []

    def get_camera_daily_person_stats(self):
        """
        Returns unique person count seen today per camera.
        Counts distinct global_ids from journeys logged today.
        Also falls back to occupancy_logs max if journeys is empty.
        """
        try:
            now = datetime.now(IST)
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

            with self._get_connection() as conn:
                # Primary: distinct global_ids per camera from journeys today
                rows = conn.execute('''
                    SELECT camera_id, COUNT(DISTINCT global_id) as unique_count
                    FROM journeys
                    WHERE timestamp >= ?
                    GROUP BY camera_id
                ''', (today_start,)).fetchall()

                journey_stats = {r["camera_id"]: r["unique_count"] for r in rows}

                # Fallback: if no journeys yet, use max occupancy count today
                occ_rows = conn.execute('''
                    SELECT camera_id, MAX(count) as peak
                    FROM occupancy_logs
                    WHERE timestamp >= ?
                    GROUP BY camera_id
                ''', (today_start,)).fetchall()

            stats = {}
            all_cams = set(list(journey_stats.keys()) + [r["camera_id"] for r in occ_rows])
            occ_map = {r["camera_id"]: r["peak"] for r in occ_rows}

            for cam in all_cams:
                j = journey_stats.get(cam, 0)
                o = occ_map.get(cam, 0)
                # Use whichever is higher — journeys count unique people,
                # occupancy peak is a lower bound when face detection hasn't fired yet
                stats[cam] = {"total": max(j, o), "am": 0, "pm": 0}

            return stats
        except Exception as e:
            logger.error(f"get_camera_daily_person_stats error: {e}")
            return {}

    # --- Alerts ---
    def log_critical_alert(self, camera_id, person_id, snapshot_path):
        try:
            now = datetime.now(IST).isoformat()
            with self._get_connection() as conn:
                conn.execute('''
                    INSERT INTO alerts (camera_id, person_id, snapshot_path, timestamp, type)
                    VALUES (?, ?, ?, ?, ?)
                ''', (camera_id, person_id, snapshot_path, now, "CRITICAL"))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"✗ Error logging alert: {e}")
            return False

    def get_recent_alerts(self, limit=10):
        try:
            with self._get_connection() as conn:
                rows = conn.execute('SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?', (limit,)).fetchall()
                result = []
                for a in rows:
                    dt = a["timestamp"]
                    if isinstance(dt, str):
                        dt = datetime.fromisoformat(dt)
                    result.append({
                        "id": str(a["id"]),
                        "camera_id": a["camera_id"],
                        "person_id": a["person_id"],
                        "snapshot_path": a["snapshot_path"],
                        "timestamp": dt,
                        "type": a["type"]
                    })
                return result
        except Exception:
            return []

    # --- Storage Optimization ---
    def cleanup_old_data(self, snapshot_hours=24, recording_days=7):
        now = datetime.utcnow()
        deleted_files = []
        
        # 1. Old Snapshots
        snap_cutoff = (now - timedelta(hours=snapshot_hours)).isoformat()
        try:
            with self._get_connection() as conn:
                # Find files to delete
                rows = conn.execute('SELECT snapshot_path, person_crops FROM detection_snapshots WHERE timestamp < ?', (snap_cutoff,)).fetchall()
                for r in rows:
                    if r["snapshot_path"]: deleted_files.append(r["snapshot_path"])
                    if r["person_crops"]:
                        crops = json.loads(r["person_crops"])
                        deleted_files.extend(crops)
                
                conn.execute('DELETE FROM detection_snapshots WHERE timestamp < ?', (snap_cutoff,))
                conn.commit()
        except Exception: pass
            
        # 2. Old Recordings
        rec_cutoff = (now - timedelta(days=recording_days)).isoformat()
        try:
            with self._get_connection() as conn:
                rows = conn.execute('SELECT file_path FROM video_recordings WHERE start_time < ?', (rec_cutoff,)).fetchall()
                for r in rows:
                    if r["file_path"]: deleted_files.append(r["file_path"])
                
                conn.execute('DELETE FROM video_recordings WHERE start_time < ?', (rec_cutoff,))
                conn.commit()
        except Exception: pass
            
        return deleted_files

    # --- Global Re-ID & Journeys ---
    def get_all_global_identities(self):
        try:
            with self._get_connection() as conn:
                rows = conn.execute('SELECT * FROM global_identities ORDER BY last_seen DESC').fetchall()
                return [dict(r) for r in rows]
        except Exception: return []

    def upsert_global_unknown(self, global_id, encoding, thumbnail_binary=None):
        try:
            now = datetime.now(IST).isoformat()
            # encoding as blob
            if hasattr(encoding, 'tobytes'):
                encoding_blob = encoding.tobytes()
            else:
                encoding_blob = encoding
                
            with self._get_connection() as conn:
                # Manual upsert logic for compatibility
                conn.execute('''
                    INSERT INTO global_identities (global_id, encoding, first_seen, last_seen, type, thumbnail)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(global_id) DO UPDATE SET 
                        encoding = excluded.encoding, 
                        last_seen = excluded.last_seen,
                        thumbnail = CASE WHEN excluded.thumbnail IS NOT NULL THEN excluded.thumbnail ELSE thumbnail END
                ''', (global_id, encoding_blob, now, now, "unknown", thumbnail_binary))
                conn.commit()
        except Exception as e: logger.error(f"✗ Global ID Error: {e}")

    def log_journey_event(self, global_id, camera_id, snapshot_path=None, person_type="unknown", timestamp=None):
        try:
            now = timestamp if timestamp is not None else datetime.now(IST)
            if hasattr(now, 'isoformat'): now = now.isoformat()
            
            with self._get_connection() as conn:
                # 1. Update the identity's last seen info
                conn.execute('UPDATE global_identities SET last_seen = ?, last_camera = ?, type = ? WHERE global_id = ?', (now, camera_id, person_type, global_id))
                
                # 2. Add to journey history
                conn.execute('''
                    INSERT INTO journeys (global_id, camera_id, timestamp, snapshot_path, type)
                    VALUES (?, ?, ?, ?, ?)
                ''', (global_id, camera_id, now, snapshot_path, person_type))
                conn.commit()
        except Exception as e: logger.error(f"✗ Journey log error: {e}")

    def get_target_journey(self, global_id):
        try:
            with self._get_connection() as conn:
                rows = conn.execute('SELECT * FROM journeys WHERE global_id = ? ORDER BY timestamp DESC', (global_id,)).fetchall()
                res = []
                for r in rows:
                    item = dict(r)
                    item["timestamp"] = datetime.fromisoformat(r["timestamp"]) if r["timestamp"] else None
                    res.append(item)
                return res
        except Exception: return []

    def get_recent_active_targets(self, hours=24):
        try:
            since = (datetime.now(IST) - timedelta(hours=hours)).isoformat()
            with self._get_connection() as conn:
                rows = conn.execute('SELECT * FROM global_identities WHERE last_seen > ? ORDER BY last_seen DESC', (since,)).fetchall()
                res = []
                for r in rows:
                    item = dict(r)
                    item["first_seen"] = datetime.fromisoformat(r["first_seen"]) if r["first_seen"] else None
                    item["last_seen"] = datetime.fromisoformat(r["last_seen"]) if r["last_seen"] else None
                    res.append(item)
                return res
        except Exception: return []

    def get_global_identity_by_id(self, global_id):
        try:
            with self._get_connection() as conn:
                r = conn.execute('SELECT * FROM global_identities WHERE global_id = ?', (global_id,)).fetchone()
                return dict(r) if r else None
        except Exception: return None

    def delete_person_from_db(self, person_id):
        try:
            with self._get_connection() as conn:
                conn.execute('DELETE FROM persons WHERE id = ?', (int(person_id),))
                conn.commit()
        except Exception as e:
            logger.error(f"✗ Error deleting person: {e}")

    def search_snapshots_by_similarity(self, target_encoding, start_time=None, end_time=None):
        """Search detection snapshots for face encoding similarity.
        Loads stored encodings and computes L2 distance in Python (SQLite has no vector ops).
        Returns snapshots sorted by best match distance.
        """
        try:
            query = "SELECT id, camera_id, timestamp, snapshot_path, bbox_data, face_encodings FROM detection_snapshots WHERE face_encodings IS NOT NULL"
            params = []
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat() if hasattr(start_time, 'isoformat') else start_time)
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat() if hasattr(end_time, 'isoformat') else end_time)

            with self._get_connection() as conn:
                rows = conn.execute(query, params).fetchall()

            THRESHOLD = 1.15
            results = []
            for r in rows:
                try:
                    encodings = json.loads(r["face_encodings"])
                    best_dist = float('inf')
                    for enc in encodings:
                        enc_arr = np.array(enc, dtype=np.float32)
                        dist = float(np.linalg.norm(target_encoding - enc_arr))
                        if dist < best_dist:
                            best_dist = dist
                    if best_dist < THRESHOLD:
                        ts = r["timestamp"]
                        if isinstance(ts, str):
                            ts = datetime.fromisoformat(ts)
                        results.append({
                            "_id": str(r["id"]),
                            "camera_id": r["camera_id"],
                            "timestamp": ts,
                            "snapshot_path": r["snapshot_path"],
                            "bbox_data": json.loads(r["bbox_data"]) if r["bbox_data"] else [],
                            "distance": best_dist
                        })
                except Exception:
                    continue

            results.sort(key=lambda x: x["distance"])
            return results
        except Exception as e:
            logger.error(f"✗ search_snapshots_by_similarity error: {e}")
            return []

# Alias
DatabaseManager = SqliteManager
