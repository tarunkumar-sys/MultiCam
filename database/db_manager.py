import sqlite3
import os
import hashlib
import time

class DatabaseManager:
    def __init__(self, db_path='database/system.db'):
        self.db_path = db_path
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Table for registered persons
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS registered_persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    image_path TEXT,
                    encoding BLOB
                )
            ''')
            # Table for detections
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    camera_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    image_path TEXT,
                    FOREIGN KEY (person_id) REFERENCES registered_persons (id)
                )
            ''')
            # Table for video recordings
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS video_recordings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT,
                    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    end_time DATETIME,
                    file_path TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS occupancy_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    count INTEGER
                )
            ''')
            # ─── NEW: Detection logs (real-time, 24h auto-delete) ────────
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detection_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_name TEXT,
                    camera_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    snapshot_path TEXT,
                    is_known INTEGER DEFAULT 0
                )
            ''')
            # ─── NEW: App settings (credentials etc.) ────────────────────
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')
            # ─── NEW: Person alert config ────────────────────────────────
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS person_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER UNIQUE,
                    enabled INTEGER DEFAULT 1,
                    FOREIGN KEY (person_id) REFERENCES registered_persons (id)
                )
            ''')
            conn.commit()

        # Seed default admin credentials if not set
        self._seed_defaults()

    def _seed_defaults(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM app_settings WHERE key='admin_user'")
            if cursor.fetchone() is None:
                hashed = hashlib.sha256("admin123".encode()).hexdigest()
                cursor.execute("INSERT INTO app_settings (key, value) VALUES ('admin_user', 'admin')")
                cursor.execute("INSERT INTO app_settings (key, value) VALUES ('admin_pass', ?)", (hashed,))
                conn.commit()

    # ─── Auth ─────────────────────────────────────────────────────────
    def verify_login(self, username, password):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM app_settings WHERE key='admin_user'")
            row = cursor.fetchone()
            if not row or row[0] != username:
                return False
            cursor.execute("SELECT value FROM app_settings WHERE key='admin_pass'")
            row = cursor.fetchone()
            if not row:
                return False
            hashed = hashlib.sha256(password.encode()).hexdigest()
            return row[0] == hashed

    def update_credentials(self, new_user, new_pass):
        hashed = hashlib.sha256(new_pass.encode()).hexdigest()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO app_settings (key, value) VALUES ('admin_user', ?)", (new_user,))
            cursor.execute("INSERT OR REPLACE INTO app_settings (key, value) VALUES ('admin_pass', ?)", (hashed,))
            conn.commit()

    def get_setting(self, key):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM app_settings WHERE key=?", (key,))
            row = cursor.fetchone()
            return row[0] if row else None

    # ─── Detection Logs (24h auto-delete) ─────────────────────────────
    def log_detection_event(self, person_name, camera_id, snapshot_path, is_known=False):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO detection_logs (person_name, camera_id, snapshot_path, is_known) VALUES (?, ?, ?, ?)',
                (person_name, camera_id, snapshot_path, 1 if is_known else 0)
            )
            conn.commit()

    def get_detection_logs(self, limit=200):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, person_name, camera_id, timestamp, snapshot_path, is_known
                FROM detection_logs
                WHERE timestamp >= datetime('now', '-24 hours')
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            return cursor.fetchall()

    def cleanup_old_logs(self):
        """Delete logs older than 24 hours and their snapshot files."""
        import os
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Get old snapshot paths before deleting
            cursor.execute('''
                SELECT snapshot_path FROM detection_logs
                WHERE timestamp < datetime('now', '-24 hours')
            ''')
            old_snaps = cursor.fetchall()
            for (path,) in old_snaps:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass
            cursor.execute("DELETE FROM detection_logs WHERE timestamp < datetime('now', '-24 hours')")
            conn.commit()
            return cursor.rowcount

    # ─── Person Alerts ─────────────────────────────────────────────────
    def get_person_alert_status(self, person_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT enabled FROM person_alerts WHERE person_id=?", (person_id,))
            row = cursor.fetchone()
            return row[0] == 1 if row else False

    def set_person_alert(self, person_id, enabled):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO person_alerts (person_id, enabled) VALUES (?, ?)",
                (person_id, 1 if enabled else 0)
            )
            conn.commit()

    def get_all_person_alerts(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT rp.id, rp.name, rp.image_path, COALESCE(pa.enabled, 0) as alert_on
                FROM registered_persons rp
                LEFT JOIN person_alerts pa ON rp.id = pa.person_id
            ''')
            return cursor.fetchall()

    # ─── Existing methods ────────────────────────────────────────────
    def register_person(self, name, image_path, encoding):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO registered_persons (name, image_path, encoding) VALUES (?, ?, ?)',
                (name, image_path, encoding)
            )
            conn.commit()
            return cursor.lastrowid

    def log_detection(self, person_id, camera_id, image_path):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO detections (person_id, camera_id, image_path) VALUES (?, ?, ?)',
                (person_id, camera_id, image_path)
            )
            conn.commit()

    def update_detection_person(self, camera_id, image_path, person_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE detections SET person_id = ? WHERE camera_id = ? AND image_path = ?',
                (person_id, camera_id, image_path)
            )
            conn.commit()

    def start_recording(self, camera_id, file_path):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO video_recordings (camera_id, file_path, start_time) VALUES (?, ?, CURRENT_TIMESTAMP)',
                (camera_id, file_path)
            )
            conn.commit()
            return cursor.lastrowid

    def end_recording(self, record_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE video_recordings SET end_time = CURRENT_TIMESTAMP WHERE id = ?',
                (record_id,)
            )
            conn.commit()

    def search_recordings(self, camera_id=None, start_time=None, end_time=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = 'SELECT * FROM video_recordings WHERE 1=1'
            params = []
            if camera_id:
                query += " AND camera_id = ?"
                params.append(camera_id)
            if start_time:
                query += " AND start_time >= ?"
                params.append(start_time)
            if end_time:
                query += " AND start_time <= ?"
                params.append(end_time)
            query += " ORDER BY start_time DESC"
            cursor.execute(query, params)
            return cursor.fetchall()

    def get_recording(self, record_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM video_recordings WHERE id = ?', (record_id,))
            return cursor.fetchone()

    def delete_recording(self, record_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM video_recordings WHERE id = ?', (record_id,))
            conn.commit()

    def get_registered_persons(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM registered_persons')
            return cursor.fetchall()

    def delete_person(self, person_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM person_alerts WHERE person_id = ?', (person_id,))
            cursor.execute('DELETE FROM registered_persons WHERE id = ?', (person_id,))
            conn.commit()
            return cursor.rowcount > 0

    def search_detections(self, name=None, start_time=None, end_time=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = '''
                SELECT d.*, rp.name 
                FROM detections d 
                LEFT JOIN registered_persons rp ON d.person_id = rp.id
                WHERE 1=1
            '''
            params = []
            if name:
                query += " AND rp.name LIKE ?"
                params.append(f"%{name}%")
            if start_time:
                query += " AND d.timestamp >= ?"
                params.append(start_time)
            if end_time:
                query += " AND d.timestamp <= ?"
                params.append(end_time)
            query += " ORDER BY d.timestamp DESC"
            cursor.execute(query, params)
            return cursor.fetchall()

    def log_occupancy(self, camera_id, count):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO occupancy_log (camera_id, count) VALUES (?, ?)',
                (camera_id, int(count))
            )
            conn.commit()

    def search_occupancy(self, camera_id=None, start_time=None, end_time=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = 'SELECT id, camera_id, timestamp, count FROM occupancy_log WHERE 1=1'
            params = []
            if camera_id:
                query += " AND camera_id = ?"
                params.append(camera_id)
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            query += " ORDER BY timestamp DESC"
            cursor.execute(query, params)
            return cursor.fetchall()

    def delete_occupancy_log(self, log_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM occupancy_log WHERE id = ?", (log_id,))
            conn.commit()
            return cursor.rowcount > 0

    def delete_occupancy_logs_filtered(self, camera_id=None, start_time=None, end_time=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "DELETE FROM occupancy_log WHERE 1=1"
            params = []
            if camera_id:
                query += " AND camera_id = ?"
                params.append(camera_id)
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount

    # ─── Analytics ─────────────────────────────────────────────────────
    def analytics_summary(self, camera_id=None, hours=24):
        h = int(hours)
        cam_d = "AND camera_id = ?" if camera_id else ""
        cam_p = [camera_id] if camera_id else []

        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(f"""
                SELECT COUNT(*) FROM detections
                WHERE timestamp >= datetime('now', '-{h} hours') {cam_d}
            """, cam_p)
            total = cur.fetchone()[0]

            cur.execute(f"""
                SELECT COUNT(*) FROM detections
                WHERE date(timestamp) = date('now') {cam_d}
            """, cam_p)
            today = cur.fetchone()[0]

            cur.execute(f"""
                SELECT COUNT(DISTINCT person_id) FROM detections
                WHERE person_id IS NOT NULL
                AND timestamp >= datetime('now', '-{h} hours') {cam_d}
            """, cam_p)
            unique_persons = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM registered_persons")
            registered = cur.fetchone()[0]

            cur.execute(f"""
                SELECT COUNT(*) FROM detections
                WHERE person_id IS NULL
                AND timestamp >= datetime('now', '-{h} hours') {cam_d}
            """, cam_p)
            unknown = cur.fetchone()[0]

            occ_cam = "AND camera_id = ?" if camera_id else ""
            cur.execute(f"""
                SELECT MAX(count), timestamp FROM occupancy_log
                WHERE timestamp >= datetime('now', '-{h} hours') {occ_cam}
            """, cam_p)
            peak_row  = cur.fetchone()
            peak_cnt  = peak_row[0] or 0
            peak_time = peak_row[1] if peak_row else None

            cur.execute(f"""
                SELECT CAST(strftime('%H', timestamp) AS INTEGER) AS hr,
                       AVG(count) AS avg_c
                FROM occupancy_log
                WHERE timestamp >= datetime('now', '-7 days') {occ_cam}
                GROUP BY hr
                ORDER BY avg_c DESC
                LIMIT 1
            """, cam_p)
            busy = cur.fetchone()

        return {
            "total_detections":   total,
            "today_detections":   today,
            "unique_persons":     unique_persons,
            "registered_persons": registered,
            "unknown_detections": unknown,
            "peak_occupancy":     peak_cnt,
            "peak_time":          peak_time,
            "busiest_hour":       busy[0] if busy else None,
            "busiest_avg":        round(busy[1], 2) if busy else 0,
        }

    def analytics_occupancy_trend(self, camera_id=None, hours=24):
        h   = int(hours)
        cam = "AND camera_id = ?" if camera_id else ""
        prm = [camera_id] if camera_id else []
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(f"""
                SELECT
                    strftime('%Y-%m-%dT%H:00:00', timestamp) AS hour,
                    ROUND(AVG(count), 2)                     AS avg_count
                FROM occupancy_log
                WHERE timestamp >= datetime('now', '-{h} hours') {cam}
                GROUP BY hour
                ORDER BY hour ASC
            """, prm)
            return [{"hour": r[0], "avg_count": r[1]} for r in cur.fetchall()]

    def analytics_heatmap(self, camera_id=None):
        cam = "AND camera_id = ?" if camera_id else ""
        prm = [camera_id] if camera_id else []
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(f"""
                SELECT
                    CAST(strftime('%w', timestamp) AS INTEGER) AS dow,
                    CAST(strftime('%H', timestamp) AS INTEGER) AS hour,
                    ROUND(AVG(count), 2)                       AS avg_count
                FROM occupancy_log
                WHERE timestamp >= datetime('now', '-7 days') {cam}
                GROUP BY dow, hour
                ORDER BY dow, hour
            """, prm)
            return [{"dow": r[0], "hour": r[1], "avg_count": r[2]} for r in cur.fetchall()]

    def analytics_top_persons(self, camera_id=None, hours=24, limit=10):
        h   = int(hours)
        lim = int(limit)
        cam = "AND d.camera_id = ?" if camera_id else ""
        prm = [camera_id] if camera_id else []
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(f"""
                SELECT
                    rp.name,
                    COUNT(*) AS cnt,
                    (
                        SELECT d2.camera_id
                        FROM   detections d2
                        WHERE  d2.person_id = d.person_id
                        AND    d2.timestamp >= datetime('now', '-{h} hours')
                        GROUP  BY d2.camera_id
                        ORDER  BY COUNT(*) DESC
                        LIMIT  1
                    ) AS top_cam
                FROM detections d
                JOIN registered_persons rp ON rp.id = d.person_id
                WHERE d.person_id IS NOT NULL
                AND   d.timestamp >= datetime('now', '-{h} hours')
                {cam}
                GROUP BY d.person_id
                ORDER BY cnt DESC
                LIMIT {lim}
            """, prm)
            return [{"name": r[0], "count": r[1], "top_camera": r[2]}
                    for r in cur.fetchall()]

    def analytics_per_camera(self, hours=24):
        h = int(hours)
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(f"""
                SELECT camera_id, COUNT(*) AS cnt
                FROM   detections
                WHERE  timestamp >= datetime('now', '-{h} hours')
                GROUP  BY camera_id
                ORDER  BY cnt DESC
            """)
            return [{"camera_id": r[0], "count": r[1]} for r in cur.fetchall()]

    def analytics_identity_breakdown(self, camera_id=None, hours=24):
        h   = int(hours)
        cam = "AND camera_id = ?" if camera_id else ""
        prm = [camera_id] if camera_id else []
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(f"""
                SELECT
                    SUM(CASE WHEN person_id IS NOT NULL THEN 1 ELSE 0 END),
                    SUM(CASE WHEN person_id IS NULL     THEN 1 ELSE 0 END)
                FROM detections
                WHERE timestamp >= datetime('now', '-{h} hours') {cam}
            """, prm)
            row = cur.fetchone()
            return {"known": row[0] or 0, "unknown": row[1] or 0}
